#!/usr/bin/env python
"""GHI ensemble -- combines the LSTM, Transformer, and Linear Regression
predictions via weighted sum. Weights are fit by constrained least-squares
(SLSQP, minimizing pooled RMSE) on a validation-period rollout, then frozen
and applied once to the held-out test rollout.
"""
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

np.random.seed(42)

# ---------------------------------------------------------------------------
# 0. Custom Keras layers needed to reload the saved Transformer model.
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package='Transformer')
class PositionalEncoding(keras.layers.Layer):
    """Fixed (non-trainable) sinusoidal positional encoding, added to the input embedding."""

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
        config.update({
            'd_model': self.d_model, 'num_heads': self.num_heads,
            'ff_dim': self.ff_dim, 'dropout_rate': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='Transformer')
class LastTokenPooling(keras.layers.Layer):
    """Takes the encoder's output at the last input timestep only, instead
    of averaging across all timesteps."""

    def call(self, inputs):
        return inputs[:, -1, :]


CUSTOM_OBJECTS = {
    'PositionalEncoding': PositionalEncoding,
    'TransformerEncoderBlock': TransformerEncoderBlock,
    'LastTokenPooling': LastTokenPooling,
}

# ---------------------------------------------------------------------------
# 1. Load data + feature set -- identical to all three families' notebooks.
# ---------------------------------------------------------------------------

DATA_PATH = '../../app/FYP_Cleaned_Dataset.csv'
WINDOW_SIZE = 24
OUTPUT_HORIZON = 24
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1

df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'])
df = df.sort_values(['State', 'Datetime']).reset_index(drop=True)
df['GHI_lag_1h'] = df.groupby('State')['GHI_W_m2'].shift(1)
df['GHI_lag_24h'] = df.groupby('State')['GHI_W_m2'].shift(24)
df = df.dropna(subset=['GHI_lag_1h', 'GHI_lag_24h']).reset_index(drop=True)

FEATURE_COLS = [
    'Latitude', 'Cloud_Coverage_Pct', 'Temp_Skin_C',
    'Relative_Humidity_Pct', 'Wind_Speed_10m_m_s',
    'Precipitable_Water_cm', 'Rainfall_mm_log', 'Is_Daylight',
    'Hour_sin', 'Hour_cos', 'Wind_Sine', 'Wind_Cosine',
    'GHI_lag_1h', 'GHI_lag_24h',
]
TARGET_COL = 'GHI_W_m2'


def split_state(group):
    n = len(group)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return group.iloc[:train_end], group.iloc[train_end:val_end], group.iloc[val_end:]


val_parts, test_parts = [], []
for state, group in df.groupby('State', sort=False):
    _, va, te = split_state(group)
    val_parts.append(va); test_parts.append(te)
val_df = pd.concat(val_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

VAL_START = val_df['Datetime'].min()
TEST_START = test_df['Datetime'].min()
print('Val period starts: ', VAL_START)
print('Test period starts:', TEST_START, '(validation-rollout origins are capped before this -- no leakage)')

# ---------------------------------------------------------------------------
# 2. Load the three best models + their own scalers.
# ---------------------------------------------------------------------------

MODELS = {
    'lstm': {
        'model': keras.models.load_model('../LSTM/BestModel (Tuned)/lstm_d_model.keras'),
        'x_scaler': joblib.load('../LSTM/BestModel (Tuned)/lstm_d_x_scaler.joblib'),
        'y_scaler': joblib.load('../LSTM/BestModel (Tuned)/lstm_d_y_scaler.joblib'),
        'kind': 'sequence',
        'val_rmse': 65.51329033820161,
        'run': 'lstm_d',
    },
    'transformer': {
        'model': keras.models.load_model('../Transformer/BestModel (Tuned)/ghi_transformer_multistep_model.keras',
                                          custom_objects=CUSTOM_OBJECTS),
        'x_scaler': joblib.load('../Transformer/BestModel (Tuned)/transformer_multistep_x_scaler.joblib'),
        'y_scaler': joblib.load('../Transformer/BestModel (Tuned)/transformer_multistep_y_scaler.joblib'),
        'kind': 'sequence',
        'val_rmse': 67.8310,
        'run': 'transformer_base_lasttoken',
    },
    'linreg': {
        # Trained on raw Wind_Speed_10m_m_s (matches the current dataset).
        'model': joblib.load('../LinearRegression/BestModel (Tuned)/linreg_h_model.joblib'),
        'x_scaler': joblib.load('../LinearRegression/BestModel (Tuned)/linreg_h_x_scaler.joblib'),
        'y_scaler': joblib.load('../LinearRegression/BestModel (Tuned)/linreg_h_y_scaler.joblib'),
        'kind': 'flat',
        'val_rmse': 70.98520903923401,
        'run': 'linreg_h',
    },
}
NAMES = ['lstm', 'transformer', 'linreg']  # fixed column order for every P matrix below

inv = {k: 1.0 / v['val_rmse'] for k, v in MODELS.items()}
total_inv = sum(inv.values())
INV_RMSE_WEIGHTS = {k: v / total_inv for k, v in inv.items()}
print('Inverse-val-RMSE weights (heuristic, for comparison only):', INV_RMSE_WEIGHTS)


def predict_batch(name, raw_windows):
    """raw_windows: (n, WINDOW_SIZE, n_features) real units -> (n, OUTPUT_HORIZON) real units, clipped >=0."""
    spec = MODELS[name]
    x_scaler, y_scaler = spec['x_scaler'], spec['y_scaler']
    flat = raw_windows.reshape(-1, len(FEATURE_COLS))
    scaled = x_scaler.transform(flat).reshape(raw_windows.shape).astype('float32')

    if spec['kind'] == 'sequence':
        pred_scaled = spec['model'].predict(scaled, verbose=0)
    else:
        pred_scaled = spec['model'].predict(scaled.reshape(scaled.shape[0], -1))

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    return np.clip(pred, 0, None)


def build_rollout_origins(start, end=None):
    """Midnight-origin D -> D+1 rollout windows, restricted to origins whose
    day D falls in [start, end). Uses the full continuous per-state timeline
    (has correct lag features), same construction as each model's own
    notebook, so predictions align origin-for-origin across all three models."""
    state_arrays = {}
    for state, g in df.groupby('State', sort=False):
        g = g.sort_values('Datetime')
        g = g[g['Datetime'] >= start]
        if end is not None:
            g = g[g['Datetime'] < end]
        g = g.reset_index(drop=True)
        state_arrays[state] = {
            'dt': g['Datetime'].to_numpy(),
            'X': g[FEATURE_COLS].to_numpy(dtype='float32'),
            'y': g[TARGET_COL].to_numpy(dtype='float32'),
        }

    origin_windows, y_true_next_day, origin_meta = [], [], []
    for state, arr in state_arrays.items():
        dt, X, y = arr['dt'], arr['X'], arr['y']
        n = len(dt)
        midnight_idx = np.where(pd.DatetimeIndex(dt).hour == 0)[0]
        for p in midnight_idx:
            if p + 2 * WINDOW_SIZE > n:
                continue
            origin_windows.append(X[p:p + WINDOW_SIZE])
            y_true_next_day.append(y[p + WINDOW_SIZE:p + 2 * WINDOW_SIZE])
            origin_meta.append((state, dt[p]))

    return np.stack(origin_windows), np.stack(y_true_next_day), origin_meta


def pooled_metrics(y_true_flat, y_pred_flat):
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    mbe = float(np.mean(y_pred_flat - y_true_flat))
    nrmse = rmse / np.mean(y_true_flat) * 100
    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2), 'mbe': mbe, 'nrmse': float(nrmse)}


# ---------------------------------------------------------------------------
# 3. Validation-period rollout (weight fitting only -- never touches test).
# ---------------------------------------------------------------------------

val_origin_windows, val_y_true, val_origin_meta = build_rollout_origins(VAL_START, TEST_START)
print(f'\n{len(val_origin_meta)} VALIDATION rollout origins (for weight fitting)')

val_preds = {name: predict_batch(name, val_origin_windows) for name in NAMES}
y_val_flat = val_y_true.flatten()
P_val = np.column_stack([val_preds[name].flatten() for name in NAMES])  # (N, 3): lstm, transformer, linreg


def rmse_objective(weights):
    pred = P_val @ weights
    return np.sqrt(mean_squared_error(y_val_flat, pred))


opt_result = minimize(
    rmse_objective,
    x0=np.array([1 / 3, 1 / 3, 1 / 3]),
    method='SLSQP',
    bounds=[(0, 1)] * 3,
    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
)
OPT_WEIGHTS = dict(zip(NAMES, opt_result.x))
print('Optimized weights (fit on validation rollout, minimizing pooled RMSE):', OPT_WEIGHTS)
print(f'  converged: {opt_result.success}, val RMSE at optimum: {opt_result.fun:.4f}')
print(f'  val RMSE at equal weights (1/3,1/3,1/3):     {rmse_objective(np.array([1/3,1/3,1/3])):.4f}')
print(f'  val RMSE at inverse-RMSE heuristic weights:  '
      f'{rmse_objective(np.array([INV_RMSE_WEIGHTS[n] for n in NAMES])):.4f}')
for name in NAMES:
    onehot = np.zeros(3); onehot[NAMES.index(name)] = 1.0
    print(f'  val RMSE at {name} alone (weight=1):          {rmse_objective(onehot):.4f}')

# ---------------------------------------------------------------------------
# 4. Test rollout -- frozen weights applied ONCE, held-out data.
# ---------------------------------------------------------------------------

test_origin_windows, test_y_true, test_origin_meta = build_rollout_origins(TEST_START)
print(f'\n{len(test_origin_meta)} TEST rollout origins (final, held-out evaluation)')

test_preds = {name: predict_batch(name, test_origin_windows) for name in NAMES}
y_test_flat = test_y_true.flatten()
P_test = np.column_stack([test_preds[name].flatten() for name in NAMES])

pred_mean = P_test.mean(axis=1)
pred_inv_rmse = P_test @ np.array([INV_RMSE_WEIGHTS[n] for n in NAMES])
pred_optimized = P_test @ opt_result.x

metrics_mean = pooled_metrics(y_test_flat, pred_mean)
metrics_inv_rmse = pooled_metrics(y_test_flat, pred_inv_rmse)
metrics_optimized = pooled_metrics(y_test_flat, pred_optimized)

print('\n--- sanity check: individual model TEST-rollout metrics (should match summary_*.json) ---')
for name in NAMES:
    print(f'  {name:12s}', pooled_metrics(y_test_flat, test_preds[name].flatten()))

print('\n--- ensemble TEST-rollout metrics (weights frozen from validation, applied once) ---')
print('  simple mean:          ', metrics_mean)
print('  inverse-RMSE weighted:', metrics_inv_rmse)
print('  optimized (SLSQP):    ', metrics_optimized)

lstm_alone_rmse = pooled_metrics(y_test_flat, test_preds['lstm'].flatten())['rmse']
print(f'\nLSTM alone test-rollout RMSE: {lstm_alone_rmse:.4f}')
print(f'Optimized ensemble beats LSTM alone: {metrics_optimized["rmse"] < lstm_alone_rmse}')

# ---------------------------------------------------------------------------
# 5. Alignment sanity check -- one dataframe, eyeballed + NaN-checked, before
#    trusting any number above.
# ---------------------------------------------------------------------------

n_show_origins = 3
rows = []
for i in range(n_show_origins):
    state, origin_dt = test_origin_meta[i]
    for h in range(OUTPUT_HORIZON):
        rows.append({
            'state': state,
            'origin_dt': origin_dt,
            'horizon_h': h + 1,
            'actual': test_y_true[i, h],
            'lstm_pred': test_preds['lstm'][i, h],
            'transformer_pred': test_preds['transformer'][i, h],
            'linreg_pred': test_preds['linreg'][i, h],
        })
comparison = pd.DataFrame(rows)
print('\n--- alignment sanity check (first 3 test-rollout origins, all 24 horizons) ---')
print(comparison.head(20).to_string(index=False))
print('\nNaN check across full test-rollout arrays:')
print('  y_true_next_day NaNs:', np.isnan(test_y_true).sum())
for name in NAMES:
    print(f'  {name} pred NaNs:', np.isnan(test_preds[name]).sum())
print(f'shape check -- all equal to {test_y_true.shape}:',
      all(test_preds[n].shape == test_y_true.shape for n in NAMES))

# ---------------------------------------------------------------------------
# 6. Per-horizon / per-state breakdown for the optimized-weight ensemble.
# ---------------------------------------------------------------------------

y_pred_next_day = pred_optimized.reshape(test_y_true.shape)

horizon_rows = []
for h in range(WINDOW_SIZE):
    yt, yp = test_y_true[:, h], y_pred_next_day[:, h]
    horizon_rows.append({
        'horizon_h': h + 1,
        'mae': mean_absolute_error(yt, yp),
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'r2': r2_score(yt, yp),
    })
horizon_df = pd.DataFrame(horizon_rows)

state_of_origin = np.array([m[0] for m in test_origin_meta])
per_state_rows = []
for state in sorted(set(state_of_origin)):
    mask = state_of_origin == state
    yt = test_y_true[mask].flatten()
    yp = y_pred_next_day[mask].flatten()
    per_state_rows.append({
        'state': state, 'n_origins': int(mask.sum()),
        'mae': mean_absolute_error(yt, yp),
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'r2': r2_score(yt, yp),
    })
per_state_df = pd.DataFrame(per_state_rows)

horizon_df.to_csv('rolling_origin_direct_by_horizon.csv', index=False)
per_state_df.to_csv('rolling_origin_direct_by_state.csv', index=False)
print('\nsaved rolling_origin_direct_by_horizon.csv, rolling_origin_direct_by_state.csv (optimized-weight ensemble)')

summary = {
    'run_name': 'ensemble',
    'members': {name: {'run': spec['run'], 'val_rmse': spec['val_rmse']} for name, spec in MODELS.items()},
    'weight_fitting': {
        'method': 'SLSQP, minimize pooled RMSE on VALIDATION rollout (24h, midnight-origin), '
                  'weights in [0,1] summing to 1; frozen and applied once to the held-out test rollout.',
        'n_validation_rollout_origins': len(val_origin_meta),
        'converged': bool(opt_result.success),
        'validation_rmse_at_optimum': float(opt_result.fun),
        'optimized_weights': {n: float(w) for n, w in OPT_WEIGHTS.items()},
        'inverse_rmse_heuristic_weights': INV_RMSE_WEIGHTS,
    },
    'n_test_rollout_origins': len(test_origin_meta),
    'test_rollout_pooled': {
        'mean': metrics_mean,
        'inverse_rmse_weighted': metrics_inv_rmse,
        'optimized': metrics_optimized,
    },
    'per_model_test_rollout_pooled': {name: pooled_metrics(y_test_flat, test_preds[name].flatten()) for name in NAMES},
    'lstm_alone_beats_optimized_ensemble': bool(lstm_alone_rmse < metrics_optimized['rmse']),
}
with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('saved summary.json')
