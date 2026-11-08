#!/usr/bin/env python
"""Inference-only: computes TRAIN-split pooled metrics (stride-1 windowed,
all origins) for lstm_d, transformer_base, and linreg_h. Cross-checks each
result against the model's already-verified val_windowed number before
trusting it.
"""
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

WINDOW_SIZE = 24
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1

FEATURE_COLS_SUFFIX = [
    'Latitude', 'Cloud_Coverage_Pct', 'Temp_Skin_C',
    'Relative_Humidity_Pct', 'Wind_Speed_10m_m_s_log',
    'Precipitable_Water_cm', 'Rainfall_mm_log', 'Is_Daylight',
    'Hour_sin', 'Hour_cos', 'Wind_Sine', 'Wind_Cosine',
    'GHI_lag_1h', 'GHI_lag_24h',
]
FEATURE_COLS_RAW = [c if c != 'Wind_Speed_10m_m_s_log' else 'Wind_Speed_10m_m_s' for c in FEATURE_COLS_SUFFIX]
TARGET_COL = 'GHI_W_m2'


@keras.utils.register_keras_serializable(package='Transformer')
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
        pe = np.zeros((sequence_length, d_model), dtype='float32')
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pe[np.newaxis, ...], dtype='float32')

    def call(self, inputs):
        return inputs + self.pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({'sequence_length': self.sequence_length, 'd_model': self.d_model})
        return config


@keras.utils.register_keras_serializable(package='Transformer')
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
            keras.layers.Dense(ff_dim, activation='relu'),
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
        config.update({'d_model': self.d_model, 'num_heads': self.num_heads,
                        'ff_dim': self.ff_dim, 'dropout_rate': self.dropout_rate})
        return config


CUSTOM_OBJECTS = {'PositionalEncoding': PositionalEncoding, 'TransformerEncoderBlock': TransformerEncoderBlock}


def load_split(data_path, feature_cols):
    df = pd.read_csv(data_path, parse_dates=['Datetime'])
    df = df.sort_values(['State', 'Datetime']).reset_index(drop=True)
    df['GHI_lag_1h'] = df.groupby('State')[TARGET_COL].shift(1)
    df['GHI_lag_24h'] = df.groupby('State')[TARGET_COL].shift(24)
    df = df.dropna(subset=['GHI_lag_1h', 'GHI_lag_24h']).reset_index(drop=True)

    def split_state(group):
        n = len(group)
        train_end = int(n * TRAIN_FRAC)
        val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
        return group.iloc[:train_end], group.iloc[train_end:val_end], group.iloc[val_end:]

    train_parts, val_parts = [], []
    for _, group in df.groupby('State', sort=False):
        tr, va, _ = split_state(group)
        train_parts.append(tr); val_parts.append(va)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    return train_df, val_df


def windowed_predict(model, x_scaler, y_scaler, frame, feature_cols, kind):
    all_true, all_pred = [], []
    for _, g in frame.groupby('State', sort=False):
        g = g.sort_values('Datetime')
        n = len(g)
        if n <= 2 * WINDOW_SIZE:
            continue
        X = g[feature_cols].to_numpy(dtype='float32')
        y = g[TARGET_COL].to_numpy(dtype='float32')
        X_for_windowing = X[: n - WINDOW_SIZE]
        y_for_windowing = y[WINDOW_SIZE:]
        from numpy.lib.stride_tricks import sliding_window_view
        Xw = sliding_window_view(X_for_windowing, WINDOW_SIZE, axis=0).transpose(0, 2, 1)
        yw = sliding_window_view(y_for_windowing, WINDOW_SIZE, axis=0)
        flat = Xw.reshape(-1, len(feature_cols))
        scaled = x_scaler.transform(flat).reshape(Xw.shape).astype('float32')
        if kind == 'sequence':
            pred_scaled = model.predict(scaled, verbose=0)
        else:
            pred_scaled = model.predict(scaled.reshape(scaled.shape[0], -1))
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
        pred = np.clip(pred, 0, None)
        all_true.append(yw); all_pred.append(pred)
    yt = np.concatenate(all_true).flatten()
    yp = np.concatenate(all_pred).flatten()
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {
        'mae': float(mean_absolute_error(yt, yp)), 'rmse': rmse,
        'r2': float(r2_score(yt, yp)), 'mbe': float(np.mean(yp - yt)),
        'nrmse': float(rmse / np.mean(yt) * 100),
    }


MODELS = {
    'lstm_d': {
        'data_path': '/home/kelocker/FYP-project/submission/data/FYP_Cleaned_Dataset.csv',
        'feature_cols': FEATURE_COLS_SUFFIX,
        'model_path': '/home/kelocker/FYP-project/rolling_origin_eval/lstm_d_model.keras',
        'x_scaler_path': '/home/kelocker/FYP-project/rolling_origin_eval/lstm_d_x_scaler.joblib',
        'y_scaler_path': '/home/kelocker/FYP-project/rolling_origin_eval/lstm_d_y_scaler.joblib',
        'kind': 'sequence', 'custom_objects': None,
        'known_val_rmse': 65.7666,
    },
    'transformer_base': {
        'data_path': '/home/kelocker/FYP-project/FYP_Cleaned_Dataset.csv',
        'feature_cols': FEATURE_COLS_RAW,
        'model_path': '/home/kelocker/FYP-project/transformer_rolling_origin_eval/ghi_transformer_multistep_model.keras',
        'x_scaler_path': '/home/kelocker/FYP-project/transformer_rolling_origin_eval/transformer_multistep_x_scaler.joblib',
        'y_scaler_path': '/home/kelocker/FYP-project/transformer_rolling_origin_eval/transformer_multistep_y_scaler.joblib',
        'kind': 'sequence', 'custom_objects': CUSTOM_OBJECTS,
        'known_val_rmse': 68.2158,
    },
    'linreg_h': {
        'data_path': '/home/kelocker/FYP-project/submission/data/FYP_Cleaned_Dataset.csv',
        'feature_cols': FEATURE_COLS_SUFFIX,
        'model_path': '/home/kelocker/FYP-project/linear_regression_rolling_origin_eval/linreg_h_model.joblib',
        'x_scaler_path': '/home/kelocker/FYP-project/linear_regression_rolling_origin_eval/linreg_h_x_scaler.joblib',
        'y_scaler_path': '/home/kelocker/FYP-project/linear_regression_rolling_origin_eval/linreg_h_y_scaler.joblib',
        'kind': 'flat', 'custom_objects': None,
        'known_val_rmse': 70.9852,
    },
}

for name, spec in MODELS.items():
    print(f'=== {name} ===')
    train_df, val_df = load_split(spec['data_path'], spec['feature_cols'])
    if spec['kind'] == 'sequence':
        model = keras.models.load_model(spec['model_path'], custom_objects=spec['custom_objects'])
    else:
        model = joblib.load(spec['model_path'])
    x_scaler = joblib.load(spec['x_scaler_path'])
    y_scaler = joblib.load(spec['y_scaler_path'])

    val_metrics = windowed_predict(model, x_scaler, y_scaler, val_df, spec['feature_cols'], spec['kind'])
    match = abs(val_metrics['rmse'] - spec['known_val_rmse']) < 0.5
    print(f'  val check: computed={val_metrics["rmse"]:.4f} known={spec["known_val_rmse"]} MATCH={match}')
    if not match:
        print(f'  !! DATASET MISMATCH for {name} -- skipping train metrics, would be wrong !!')
        continue

    train_metrics = windowed_predict(model, x_scaler, y_scaler, train_df, spec['feature_cols'], spec['kind'])
    print(f'  TRAIN metrics: {train_metrics}')
    print(f'  VAL metrics:   {val_metrics}')
