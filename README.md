# Time Series Forecasting — Kaggle Competition

**End-to-end ML pipeline for the [Kaggle TS Forecasting](https://www.kaggle.com/competitions/ts-forecasting) competition: feature engineering, temporal validation, gradient boosting models, and submission generation.**

---

## Overview

- **Task**: Multi-horizon time series regression (predict `y_target` for horizons 1, 3, 10, 25).
- **Data**: ~5.3M training rows, ~1.4M test rows, 9,270 unique series; 86 feature columns + identifiers.
- **Approach**: Lag/rolling features, temporal train–validation split, Gradient Boosting (sklearn); optional LightGBM/XGBoost.
- **Deliverables**: Training scripts, prediction pipeline, `submission.csv` for Kaggle.

---

## Tech Stack

- **Language**: Python 3  
- **Data**: pandas, PyArrow (parquet)  
- **ML**: scikit-learn (GradientBoostingRegressor), optional LightGBM/XGBoost  
- **Features**: custom lag/rolling stats, horizon encoding, missing-value handling  

---

## Project Structure

```
├── ts-forecasting/           # Data (parquet; not in repo)
├── config.py                 # Paths, hyperparameters, feature config
├── utils.py                  # Helpers: lags, rolling, split, metrics
├── feature_engineering.py    # Feature pipeline
├── models.py                 # GradientBoosting / LightGBM / XGBoost
├── train.py                  # Full training
├── train_quick.py            # Quick training (sampled data)
├── predict.py                # Predict & write submission.csv
├── run_train_quick.sh        # Run quick training
├── run_train.sh              # Run full training
├── run_predict.sh            # Run prediction
├── requirements.txt
└── README.md
```

Generated artifacts: `lightgbm_model.pkl` with `feature_columns_lgbm.json`, or `gradient_boosting_model.pkl` with `feature_columns_gb.json`. `predict.py` aligns features using the schema for the loaded model only (no silent fallback).

---

## Setup

```bash
pip install -r requirements.txt
```

Or use system Python that already has pandas, numpy, scikit-learn, pyarrow.

---

## Usage

### Local

**Quick training (300k sampled rows, few minutes):**
```bash
python3 train.py
# train.py uses train.sample(300000, random_state=42) by default for fast iteration
```

**Full training (all data, hours):** Comment out the `train.sample(...)` line in `train.py`, then:
```bash
./run_train.sh
# or: python3 train.py
```

**Generate submission:**
```bash
./run_predict.sh
# or: python3 predict.py
```

Then upload `submission.csv` to the competition **Submit Predictions** page.

Data is read from `ts-forecasting/train.parquet` and `ts-forecasting/test.parquet` by default.

### Kaggle Notebook

- Add the competition dataset as an input to your notebook.
- Paths auto-detect: if `/kaggle/input` exists, the code uses that `DATA_DIR`; otherwise `ts-forecasting/`. No edits to `config.py` needed.
- Run `!python train.py` then `!python predict.py` (or import `train.main()` / `predict.main()`).
- Download `submission.csv` and submit on the **Submit Predictions** page.

---

## Engineering highlights

- **Model-specific feature schema**: Training saves the exact feature column list to a schema file that matches the model: `feature_columns_lgbm.json` for LightGBM, `feature_columns_gb.json` for Gradient Boosting (same run that writes `lightgbm_model.pkl` or `gradient_boosting_model.pkl`). At prediction time, `predict.py` detects the loaded model type and aligns test features using that schema only (no fallback to a different schema), so columns stay consistent and horizons can affect predictions.
- **Submission order and sanity checks**: Submission row order matches the original test `id` order (via `test_ids`). The pipeline validates horizon nunique after feature engineering and warns if median distinct predictions per series is ≤ 2 (often ~0 score); submission is still written.
- **No look-ahead**: Lag/rolling features use only past data (`shift(1)`); no `ts_index_norm` that would use future indices. Test-time target-derived features are filled from the training set tail per series (train-tail carryover), so the model receives a stable input distribution.
- **Kaggle + local one codebase**: Single `config.py` for paths; same scripts run locally and on Kaggle without branch-by-environment logic.

---

## Feature Engineering

- **Lags**: 1, 3, 7, 14, 30 on target (and selected features).
- **Rolling**: mean, std, min, max over windows 7, 14, 30.
- **Horizon**: single source of truth from `ensure_raw_horizon(df)` (parsed from `id`); dummies horizon_1, horizon_3, horizon_10, horizon_25; raw `horizon` kept and not excluded in `prepare_features`.
- **Other**: raw `ts_index`; median imputation for missing values.

---

## Validation & Metrics

- Temporal split: last 20% of each series as validation (no future leakage).
- Metrics: MAE, RMSE, MAPE; optional weighting via `weight` column.

---

## Configuration

Edit `config.py` for data paths, lag/rolling settings, validation fraction, and model hyperparameters.

---

## License

MIT.
