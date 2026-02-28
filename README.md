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

---

## Setup

```bash
pip install -r requirements.txt
```

Or use system Python that already has pandas, numpy, scikit-learn, pyarrow.

---

## Usage

**Quick training (sample data):**
```bash
./run_train_quick.sh
# or: python3 train_quick.py
```

**Full training (all data, ~30+ min):**
```bash
./run_train.sh
# or: python3 train.py
```

**Generate submission:**
```bash
./run_predict.sh
# or: python3 predict.py
```

Then upload `submission.csv` to the competition page.

---

## Feature Engineering

- **Lags**: 1, 3, 7, 14, 30 on target (and selected features).
- **Rolling**: mean, std, min, max over windows 7, 14, 30.
- **Horizon**: one-hot for 1, 3, 10, 25.
- **Other**: normalized time index per series; median imputation for missing values.

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
