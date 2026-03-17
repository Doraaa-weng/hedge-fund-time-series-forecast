"""
Inspect feature matrix X and (if available) model feature importance.
Run: python3 inspect_features.py
"""
import pandas as pd
import numpy as np
import config
from feature_engineering import engineer_features
from utils import prepare_features

def main():
    print("=" * 60)
    print("1. Loading a small sample of training data...")
    print("=" * 60)
    train = pd.read_parquet(config.TRAIN_PATH)
    train = train.head(5000).copy()
    train = engineer_features(train, is_train=True)
    sample_ts = train["ts_id"].iloc[0]
    df_one = train[train["ts_id"] == sample_ts].copy()

    print("\n2. Preparing features (X) for this sample...")
    X, y = prepare_features(df_one, target_col="y_target")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    print("\n3. Feature column names (first 30):")
    for i, col in enumerate(X.columns[:30]):
        print(f"   {i+1:2d}. {col}")
    if len(X.columns) > 30:
        print(f"   ... and {len(X.columns) - 30} more columns")

    print("\n4. First 3 rows of X (first 10 columns):")
    print(X.iloc[:3, :10].to_string())

    print("\n5. Feature importance (if model exists)...")
    import pickle
    model = None
    for pkl in ("lightgbm_model.pkl", "gradient_boosting_model.pkl"):
        try:
            with open(pkl, "rb") as f:
                model = pickle.load(f)
            print(f"   Loaded {pkl}")
            break
        except FileNotFoundError:
            continue
    if model is not None and hasattr(model, "get_feature_importance"):
        imp = model.get_feature_importance()
        print("   Top 15 features by importance:")
        print(imp.head(15).to_string(index=False))
    elif model is not None:
        print("   Model has no get_feature_importance().")
    else:
        print("   No model .pkl found. Run train_quick.py or train.py first.")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)

if __name__ == "__main__":
    main()
