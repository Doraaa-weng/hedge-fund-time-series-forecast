"""
Configuration file for time series forecasting project.
Paths auto-detect Kaggle (/kaggle/input) vs local project directories.
"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

def _get_data_dir():
    """Use Kaggle input path if present, else local ts-forecasting."""
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for name in os.listdir(kaggle_input):
            dirpath = os.path.join(kaggle_input, name)
            if os.path.isdir(dirpath):
                train_p = os.path.join(dirpath, "train.parquet")
                test_p = os.path.join(dirpath, "test.parquet")
                if os.path.isfile(train_p) and os.path.isfile(test_p):
                    return dirpath
    return os.path.join(ROOT_DIR, "ts-forecasting")

DATA_DIR = _get_data_dir()
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(OUTPUTS_DIR, "submission.csv")
TRAINING_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "training_results.csv")
MODEL_REGISTRY_PATH = os.path.join(ARTIFACTS_DIR, "model_registry.json")
GLOBAL_LGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_global_model.txt")
GLOBAL_LGB_META_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_global_meta.json")
GB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "gradient_boosting_model.pkl")
GB_FEATURE_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns_gb.json")
LEGACY_LGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.pkl")

for directory in (ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR):
    os.makedirs(directory, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2  # Use last 20% of data for validation (temporal split)
GLOBAL_TIME_VALID_FRACTION = 0.15

# Feature engineering
HORIZON_VALUES = [1, 3, 10, 25]
CATEGORICAL_COLUMNS = ["code", "sub_code", "sub_category", "horizon"]
MIN_HORIZON_NUNIQUE = 4  # Require at least this many distinct horizons (4 = competition default; use 10+ if raw horizon is 1–25)
LAG_FEATURES = [1, 3, 7, 14, 30]  # Lag periods to create
ROLLING_WINDOWS = [7, 14, 30]  # Rolling window sizes
ROLLING_STATS = ['mean', 'std', 'min', 'max']  # Rolling statistics
EXOGENOUS_LAGS = [3, 15]
MAX_LAG_BASE_FEATURES = 20
FEATURE_SELECTION_SAMPLE = 350000
FEATURE_SELECTION_MIN_NON_NULL = 1000
FEATURE_TOP_K = 134
FEATURE_SELECTION_VALID_FRACTION = 0.1
USE_RECENCY_WEIGHTING = True

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

LIGHTGBM_GLOBAL_PARAMS = {
    'objective': 'regression',
    'metric': 'None',
    'learning_rate': 0.02,
    'num_leaves': 1900,
    'min_data_in_leaf': 800,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.0,
    'lambda_l2': 1.0,
    'max_depth': 16,
    'max_bin': 255,
    'verbosity': -1,
    'seed': RANDOM_STATE,
}

LIGHTGBM_FS_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.06,
    'num_leaves': 255,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.0,
    'lambda_l2': 1.0,
    'max_depth': 10,
    'verbosity': -1,
    'seed': RANDOM_STATE,
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
