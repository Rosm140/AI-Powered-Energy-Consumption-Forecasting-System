# System Architecture — AI Energy Forecasting

## Data Flow

```
[Raw CSV Data]
      │
      ▼
[src/preprocess.py]
  • Parse Datetime index
  • Resample to hourly
  • Forward-fill gaps
  • Engineer 12 features
      │
      ▼
[src/train.py]
  • Chronological 80/20 split
  • StandardScaler (for MLP)
  • Train 3 models in parallel
  • Evaluate: MAE, RMSE, R², MAPE
  • Save best model + scaler
      │
      ├──────────────────────────────┐
      ▼                              ▼
[outputs/predictions.csv]    [models/best_model.pkl]
[outputs/metrics.json]       [models/scaler.pkl]
      │                              │
      ▼                              ▼
[src/visualize.py]           [src/app.py  ←  Flask API]
  6 plots → PNG                POST /predict
                               POST /predict/bulk
                               GET  /health
                               GET  /model-info
```

## Module Responsibilities

| Module | Role |
|---|---|
| `data/generate_dataset.py` | Synthetic smart-grid data generation |
| `src/preprocess.py` | ETL + feature engineering |
| `src/train.py` | ML training, evaluation, artifact export |
| `src/visualize.py` | 6 diagnostic + presentation plots |
| `src/app.py` | REST API for real-time inference |
| `main.py` | Pipeline orchestrator (CLI) |

## Model Selection Rationale

Three models were trained and compared:
- **Random Forest** — robust to noise, handles non-linear patterns, interpretable via feature importance
- **Gradient Boosting** — best overall performance (R² 0.929); sequential error correction improves accuracy
- **MLP Neural Network** — learns complex non-linear patterns; requires feature scaling

Gradient Boosting was selected as the production model based on lowest RMSE and highest R².
