"""
Generate direct LightGBM predictions for the Kaggle submission file.
"""
import json
import os
import warnings

import pandas as pd

import config
from feature_engineering import build_global_lgb_features, ensure_raw_horizon
from utils import prepare_lightgbm_frame

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError as exc:
    raise RuntimeError("LightGBM is required for prediction.") from exc


def load_global_model():
    """Load the direct LightGBM model and its feature metadata."""
    if not os.path.exists(config.GLOBAL_LGB_MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{config.GLOBAL_LGB_MODEL_PATH}' not found. Run train.py first."
        )
    if not os.path.exists(config.GLOBAL_LGB_META_PATH):
        raise FileNotFoundError(
            f"Metadata file '{config.GLOBAL_LGB_META_PATH}' not found. Run train.py first."
        )
    model = lgb.Booster(model_file=config.GLOBAL_LGB_MODEL_PATH)
    with open(config.GLOBAL_LGB_META_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


def main():
    print("=" * 60, flush=True)
    print("Time Series Forecasting - Direct LightGBM Prediction", flush=True)
    print("=" * 60, flush=True)

    print("\n[0/4] Loading trained model...")
    model, metadata = load_global_model()
    print(f"Loaded model from '{config.GLOBAL_LGB_MODEL_PATH}'")

    print("\n[1/4] Loading test data...")
    test_df = pd.read_parquet(config.TEST_PATH)
    if "horizon" not in test_df.columns:
        test_df = ensure_raw_horizon(test_df)
    print(f"Test data shape: {test_df.shape}")

    print("\n[2/4] Building features...")
    test_df = build_global_lgb_features(
        test_df,
        lag_base_features=metadata["lag_base_features"],
    )

    print("\n[3/4] Preparing LightGBM frame...")
    X_test, _, _ = prepare_lightgbm_frame(
        test_df,
        feature_cols=metadata["feature_cols"],
        categorical_cols=metadata["categorical_cols"],
        numeric_fill_values=metadata["numeric_fill_values"],
        categorical_levels=metadata["categorical_levels"],
    )
    print(f"Test feature matrix: {X_test.shape}")

    print("\n[4/4] Predicting...")
    test_pred = model.predict(X_test)
    submission = pd.DataFrame(
        {
            "id": test_df["id"].values,
            "prediction": test_pred.astype("float64"),
        }
    )
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Saved: {config.SUBMISSION_PATH}")
    print(submission["prediction"].describe())

    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)
    return submission


if __name__ == "__main__":
    submission = main()
