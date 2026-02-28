"""
Quick training script with data sampling for faster iteration
"""
import pandas as pd
import numpy as np
import config
from utils import (
    prepare_features,
    calculate_metrics
)
from feature_engineering import engineer_features
from models import GradientBoostingModel
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("Time Series Forecasting - Quick Training (Sampled Data)")
    print("=" * 60)
    
    # Load training data
    print("\n[1/5] Loading training data...")
    train = pd.read_parquet(config.TRAIN_PATH)
    print(f"Full training data shape: {train.shape}")
    
    # Sample data for quick training (use 10% of time series)
    print("\n[2/5] Sampling data for quick training...")
    train = engineer_features(train, is_train=True)
    unique_ts = train['ts_id'].unique()
    sample_ts = np.random.choice(unique_ts, size=min(1000, len(unique_ts)), replace=False)
    train_sample = train[train['ts_id'].isin(sample_ts)].copy()
    print(f"Sampled data shape: {train_sample.shape}")
    
    # Prepare features and target
    print("\n[3/5] Preparing features...")
    X, y = prepare_features(train_sample, target_col='y_target')
    weights = train_sample['weight'].values if 'weight' in train_sample.columns else None
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {len(X.columns)}")
    
    # Simple train/validation split (80/20)
    print("\n[4/5] Creating train/validation split...")
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]
    weights_train = weights[:split_idx] if weights is not None else None
    weights_val = weights[split_idx:] if weights is not None else None
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train model
    print("\n[5/5] Training Gradient Boosting model...")
    gb_model = GradientBoostingModel(
        n_estimators=100,  # Reduced for speed
        max_depth=5,
        learning_rate=0.1
    )
    gb_model.train(
        X_train, y_train,
        X_val, y_val,
        weights_train, weights_val
    )
    
    # Make predictions
    y_train_pred = gb_model.predict(X_train)
    y_val_pred = gb_model.predict(X_val)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train.values, y_train_pred, weights_train)
    val_metrics = calculate_metrics(y_val.values, y_val_pred, weights_val)
    
    print(f"\nResults:")
    print(f"  Train - MAE: {train_metrics['MAE']:.4f}, RMSE: {train_metrics['RMSE']:.4f}")
    print(f"  Val   - MAE: {val_metrics['MAE']:.4f}, RMSE: {val_metrics['RMSE']:.4f}")
    
    # Feature importance
    print("\nFeature importance (Top 20):")
    importance = gb_model.get_feature_importance()
    print(importance.head(20).to_string(index=False))
    
    # Save model
    import pickle
    with open('gradient_boosting_model.pkl', 'wb') as f:
        pickle.dump(gb_model, f)
    print("\nModel saved to 'gradient_boosting_model.pkl'")
    
    print("\n" + "=" * 60)
    print("Quick training complete!")
    print("Run train.py for full training on all data.")
    print("=" * 60)
    
    return gb_model


if __name__ == "__main__":
    model = main()
