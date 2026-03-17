# Hierarchical Time Series Forecasting

This project was built as a realistic applied machine learning exercise around a public hedge-fund-style forecasting task from [Kaggle TS Forecasting](https://www.kaggle.com/competitions/ts-forecasting). My motivation was not just to participate in a competition, but to use a noisy, large-scale, multi-horizon prediction problem to practice the parts of modeling work that matter in real teams: structuring an end-to-end pipeline, designing leakage-aware validation, engineering stable features, and iterating on models with clear evaluation logic.

It highlights how I approach ambiguous forecasting problems with both modeling and implementation discipline. Instead of treating the task as a notebook-only experiment, I organized it as a reproducible project with separate training, inference, configuration, and packaging steps. The result is a compact repository that demonstrates practical skills in time series feature engineering, hierarchical tabular modeling, experiment organization, and deployable prediction workflows.

## Overview

- Source problem: a hedge-fund-style forecasting task framed as a public Kaggle challenge
- Task: multi-horizon regression for `y_target` at horizons 1, 3, 10, and 25
- Data: approximately 5.3M training rows and 1.4M test rows
- Model: direct LightGBM with global time-based validation
- Evaluation focus: weighted error on future forecasts under a held-out test setting
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
