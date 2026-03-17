# Time Series Forecasting — Kaggle Competition

Forecasting pipeline for the [Kaggle TS Forecasting](https://www.kaggle.com/competitions/ts-forecasting) competition using a direct LightGBM approach with hierarchy-aware categorical features and exogenous lag features.

## Overview

- Task: multi-horizon regression for `y_target` at horizons 1, 3, 10, and 25
- Data: approximately 5.3M training rows and 1.4M test rows
- Model: direct LightGBM with global time-based validation
- Output: `outputs/submission.csv`

## Repository Layout

```text
.
├── config.py
├── feature_engineering.py
├── models.py
├── predict.py
├── train.py
├── train_quick.py
├── utils.py
├── requirements.txt
├── scripts/
│   ├── make_kaggle_zip.sh
│   ├── run_predict.sh
│   ├── run_train.sh
│   └── run_train_quick.sh
└── README.md
```

Runtime directories such as `artifacts/`, `outputs/`, and `reports/` are created automatically and are not tracked.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Quick training:

```bash
./scripts/run_train_quick.sh
```

Full training:

```bash
./scripts/run_train.sh
```

Generate submission:

```bash
./scripts/run_predict.sh
```

The final file for Kaggle upload is `outputs/submission.csv`.

## Modeling Notes

- Global time split to better match leaderboard behavior
- Hierarchical categorical features: `code`, `sub_code`, `sub_category`, `horizon`
- Exogenous lag features selected from raw `feature_*` columns
- Competition-aligned weighted validation metric

## Configuration

Data paths, output paths, validation settings, and model hyperparameters are defined in `config.py`.
