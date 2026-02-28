"""
Configuration file for time series forecasting project
"""
import os

# Data paths
DATA_DIR = "ts-forecasting"
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
SUBMISSION_PATH = "submission.csv"

# Model parameters
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2  # Use last 20% of data for validation (temporal split)

# Feature engineering
LAG_FEATURES = [1, 3, 7, 14, 30]  # Lag periods to create
ROLLING_WINDOWS = [7, 14, 30]  # Rolling window sizes
ROLLING_STATS = ['mean', 'std', 'min', 'max']  # Rolling statistics

# Model hyperparameters
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_STATE
}

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

# Training parameters
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
