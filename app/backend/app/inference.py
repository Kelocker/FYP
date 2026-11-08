from __future__ import annotations

from datetime import date as date_type, datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pvlib
import tensorflow as tf
from tensorflow import keras

from . import solar

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_PATH = Path(__file__).resolve().parents[2] / "FYP_Cleaned_Dataset.csv"

WINDOW_SIZE = 24

DATASET_FIRST_DAY = date_type(2016, 1, 1)
DATASET_LAST_REAL_DAY = date_type(2025, 12, 30)
MODEL_PREDICTED_DAY = DATASET_LAST_REAL_DAY + timedelta(days=1)

FEATURE_COLS = [
    "Latitude", "Cloud_Coverage_Pct", "Temp_Skin_C",
    "Relative_Humidity_Pct", "Wind_Speed_10m_m_s",
    "Precipitable_Water_cm", "Rainfall_mm_log", "Is_Daylight",
    "Hour_sin", "Hour_cos", "Wind_Sine", "Wind_Cosine",
    "GHI_lag_1h", "GHI_lag_24h",
]
TARGET_COL = "GHI_W_m2"

ENSEMBLE_WEIGHTS = {
    "lstm": 0.7908519622646556,
    "transformer": 0.11353298789926203,
    "linreg": 0.09561504983608249,
}
AVAILABLE_MODEL_IDS = ["ensemble", "lstm", "transformer", "linreg"]


@keras.utils.register_keras_serializable(package="Transformer")
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self._build_positional_encoding(sequence_length, d_model)

    @staticmethod
    def _build_positional_encoding(sequence_length, d_model):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((sequence_length, d_model), dtype="float32")
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pe[np.newaxis, ...], dtype="float32")

    def call(self, inputs):
        return inputs + self.pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return config


@keras.utils.register_keras_serializable(package="Transformer")
class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.attn_dropout = keras.layers.Dropout(dropout_rate)
        self.attn_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(d_model),
        ])
        self.ffn_dropout = keras.layers.Dropout(dropout_rate)
        self.ffn_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.attention.build(input_shape, input_shape)
        self.attn_dropout.build(input_shape)
        self.attn_norm.build(input_shape)
        self.ffn.build(input_shape)
        self.ffn_dropout.build(input_shape)
        self.ffn_norm.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn_out = self.attention(query=inputs, value=inputs, key=inputs, training=training)
        attn_out = self.attn_dropout(attn_out, training=training)
        x = self.attn_norm(inputs + attn_out)
        ffn_out = self.ffn(x, training=training)
        ffn_out = self.ffn_dropout(ffn_out, training=training)
        return self.ffn_norm(x + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package="Transformer")
class LastTokenPooling(keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]


_CUSTOM_OBJECTS = {
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
    "LastTokenPooling": LastTokenPooling,
}

_models: Optional[dict] = None
_state_windows: Optional[dict] = None
_full_ghi_by_state: Optional[dict] = None


def _load_models() -> dict:
    return {
        "lstm": {
            "model": keras.models.load_model(MODEL_DIR / "lstm_d_model.keras"),
            "x_scaler": joblib.load(MODEL_DIR / "lstm_d_x_scaler.joblib"),
            "y_scaler": joblib.load(MODEL_DIR / "lstm_d_y_scaler.joblib"),
            "kind": "sequence",
        },
        "transformer": {
            "model": keras.models.load_model(MODEL_DIR / "ghi_transformer_multistep_model.keras", custom_objects=_CUSTOM_OBJECTS),
            "x_scaler": joblib.load(MODEL_DIR / "transformer_multistep_x_scaler.joblib"),
            "y_scaler": joblib.load(MODEL_DIR / "transformer_multistep_y_scaler.joblib"),
            "kind": "sequence",
        },
        "linreg": {
            "model": joblib.load(MODEL_DIR / "linreg_h_model.joblib"),
            "x_scaler": joblib.load(MODEL_DIR / "linreg_h_x_scaler.joblib"),
            "y_scaler": joblib.load(MODEL_DIR / "linreg_h_y_scaler.joblib"),
            "kind": "flat",
        },
    }


def _load_state_windows() -> dict:
    usecols = ["State", "Datetime"] + FEATURE_COLS[:-2] + [TARGET_COL]
    df = pd.read_csv(DATA_PATH, usecols=usecols, parse_dates=["Datetime"])
    df = df.sort_values(["State", "Datetime"])

    windows = {}
    for state, group in df.groupby("State", sort=False):
        group = group.sort_values("Datetime")
        tail = group.tail(2 * WINDOW_SIZE).copy()
        tail["GHI_lag_1h"] = tail[TARGET_COL].shift(1)
        tail["GHI_lag_24h"] = tail[TARGET_COL].shift(WINDOW_SIZE)
        day_d = tail.tail(WINDOW_SIZE)
        if day_d[FEATURE_COLS].isna().any().any() or len(day_d) < WINDOW_SIZE:
            continue
        basis_date = day_d["Datetime"].max().date()
        windows[state] = {
            "window": day_d[FEATURE_COLS].to_numpy(dtype="float32"),
            "basis_date": basis_date,
            "target_date": basis_date + timedelta(days=1),
        }
    return windows


def _load_full_ghi_by_state() -> dict:
    df = pd.read_csv(DATA_PATH, usecols=["State", "Datetime", TARGET_COL], parse_dates=["Datetime"])
    result = {}
    for state, group in df.groupby("State", sort=False):
        group = group.sort_values("Datetime")
        result[state] = group.set_index("Datetime")[TARGET_COL]
    return result


def _ensure_loaded():
    global _models, _state_windows, _full_ghi_by_state
    if _models is None:
        _models = _load_models()
    if _state_windows is None:
        _state_windows = _load_state_windows()
    if _full_ghi_by_state is None:
        _full_ghi_by_state = _load_full_ghi_by_state()


def get_historical_ghi_day_w_m2(dataset_state: str, target_date: date_type) -> Optional[list]:
    _ensure_loaded()
    series = _full_ghi_by_state.get(dataset_state)
    if series is None:
        return None
    day_start = pd.Timestamp(target_date)
    hours = pd.date_range(day_start, periods=WINDOW_SIZE, freq="h")
    values = series.reindex(hours)
    if values.isna().any():
        return None
    return values.tolist()


def _predict_one(model_id: str, raw_window: np.ndarray) -> np.ndarray:
    spec = _models[model_id]
    x_scaler, y_scaler = spec["x_scaler"], spec["y_scaler"]
    scaled = x_scaler.transform(raw_window).astype("float32")

    if spec["kind"] == "sequence":
        pred_scaled = spec["model"].predict(scaled[np.newaxis, ...], verbose=0)[0]
    else:
        pred_scaled = spec["model"].predict(scaled.reshape(1, -1))[0]

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return np.clip(pred, 0, None)


class UnknownLocationForModel(Exception):
    pass

def predict_ghi_day_w_m2(dataset_state: str, model_id: str = "ensemble"):
    if model_id not in AVAILABLE_MODEL_IDS:
        raise ValueError(f"unknown model_id {model_id!r}; valid ids: {', '.join(AVAILABLE_MODEL_IDS)}")

    _ensure_loaded()
    info = _state_windows.get(dataset_state)
    if info is None:
        raise UnknownLocationForModel(dataset_state)

    raw_window = info["window"]
    if model_id == "ensemble":
        per_model = {name: _predict_one(name, raw_window) for name in ENSEMBLE_WEIGHTS}
        hourly = sum(ENSEMBLE_WEIGHTS[name] * per_model[name] for name in ENSEMBLE_WEIGHTS)
    else:
        hourly = _predict_one(model_id, raw_window)

    return hourly.tolist(), info["basis_date"], info["target_date"], f"model:{model_id}"


def predict_ghi_w_m2(latitude: float, longitude: float, timestamp: datetime) -> tuple[float, str]:
    solpos = solar.solar_position(latitude, longitude, timestamp)
    apparent_zenith = solpos["apparent_zenith_deg"]

    if apparent_zenith >= 90.0:
        return 0.0, "clearsky_placeholder"

    ghi_df = pvlib.clearsky.haurwitz(pd.Series([apparent_zenith]))
    ghi_w_m2 = float(ghi_df["ghi"].iloc[0])

    return ghi_w_m2, "clearsky_placeholder"
