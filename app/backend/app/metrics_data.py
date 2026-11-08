MODEL_METRICS = [
    {
        "id": "ensemble",
        "name": "Ensemble",
        "rmse": "69.37",
        "mae": "35.74",
        "r2": "0.9361",
        "mbe": "2.61",
        "nrmse": "34.16%",
        "selected": True,
        "note": (
            "Validation-optimized weighted combination of all three models below "
            "(LSTM 79%, Transformer 11%, Linear Regression 10% -- weights fit via "
            "SLSQP minimizing RMSE on a held-out validation rollout, then frozen). "
            "Rolling-origin rollout metric (not the test-split numbers below). "
            "Refit 2026-09-02 after the Transformer's pooling swap (average -> "
            "last-token); its weight grew from 6% to 11% on the strength of that "
            "improvement. Weights still lean heavily on LSTM because it's "
            "genuinely the strongest model once fairly evaluated -- beats the "
            "ensemble's other two members on every metric."
        ),
    },
    {
        "id": "lstm",
        "name": "LSTM",
        "rmse": "67.1257",
        "mae": "33.8836",
        "r2": "0.9402",
        "mbe": "0.2751",
        "nrmse": "33.06%",
        "selected": False,
        "note": (
            "2-layer LSTM (128/64 units), direct 24-hour-ahead multistep output. "
            "Retrained 2026-09-02 on current raw Wind_Speed_10m_m_s data (same "
            "config d hyperparameters). Best single model on RMSE, MAE, R2, and "
            "bias (near-zero MBE) -- standard test-split evaluation."
        ),
        "splits": {
            "train": {"r2": 0.9441, "rmse": 65.90, "mse": 4342.55, "mbe": 4.37, "nrmse": 31.99},
            "val":   {"r2": 0.9428, "rmse": 65.51, "mse": 4291.95, "mbe": 0.39, "nrmse": 32.34},
            "test":  {"r2": 0.9402, "rmse": 67.13, "mse": 4505.90, "mbe": 0.28, "nrmse": 33.06},
        },
    },
    {
        "id": "transformer",
        "name": "Encoder-Transformer",
        "rmse": "69.3780",
        "mae": "39.0789",
        "r2": "0.9361",
        "mbe": "5.9374",
        "nrmse": "34.17%",
        "selected": False,
        "note": (
            "Base config (d_model=64, 4 heads, ff_dim=128), encoder-only, direct "
            "24-hour-ahead multistep output -- last-token pooling (reads the "
            "final timestep's representation, x[:, -1, :]) + flat Dense head, no "
            "decoder. Swapped 2026-09-02 from GlobalAveragePooling1D: averaging "
            "over the full 168h window smeared the day/night recency signal the "
            "Dense(24) head needs at dawn/dusk boundaries, so this model "
            "under-predicted those hours; reading only the most recent token "
            "preserves that signal and measurably improves the boundary-hour "
            "error (see model_training_Transformer_multistep_rolling_origin.ipynb)."
        ),
        "splits": {
            "train": {"r2": 0.9401, "rmse": 68.21, "mse": 4652.51, "mbe": 10.21, "nrmse": 33.11},
            "val":   {"r2": 0.9387, "rmse": 67.83, "mse": 4601.05, "mbe": 6.10, "nrmse": 33.48},
            "test":  {"r2": 0.9361, "rmse": 69.38, "mse": 4813.30, "mbe": 5.94, "nrmse": 34.17},
        },
    },
    {
        "id": "linreg",
        "name": "Linear Regression",
        "rmse": "72.0799",
        "mae": "38.8734",
        "r2": "0.9310",
        "mbe": "2.7301",
        "nrmse": "35.50%",
        "selected": False,
        "note": (
            "Ordinary least squares on the same 24h-flattened feature window, native "
            "multi-output (no regularization needed -- an alpha sweep from 0.0001 to "
            "10000 found no benefit until alpha>=100, which only hurt). The simple "
            "baseline every other model is measured against."
        ),
        "splits": {
            "train": {"r2": 0.9325, "rmse": 72.42, "mse": 5244.89, "mbe": 7.45, "nrmse": 35.16},
            "val":   {"r2": 0.9321, "rmse": 71.41, "mse": 5099.99, "mbe": 6.28, "nrmse": 35.25},
            "test":  {"r2": 0.9310, "rmse": 72.08, "mse": 5195.51, "mbe": 2.73, "nrmse": 35.50},
        },
    },
]
