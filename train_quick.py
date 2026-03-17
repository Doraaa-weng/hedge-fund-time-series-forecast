"""
Quick training script with data sampling for faster iteration
"""
import pandas as pd
import numpy as np
import config
from utils import (
    prepare_features,
    calculate_metrics,
    save_feature_columns,
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
    
    # Sample data for quick training (keep complete top-level series to preserve temporal structure)
    print("\n[2/5] Sampling data for quick training...")
    train = engineer_features(train, is_train=True)
    sample_col = 'series_id' if 'series_id' in train.columns else 'ts_id'
    unique_series = train[sample_col].unique()
    sample_series = np.random.choice(unique_series, size=min(1000, len(unique_series)), replace=False)
    train_sample = train[train[sample_col].isin(sample_series)].copy().reset_index(drop=True)
    print(f"Sampled data shape: {train_sample.shape}")
    
    # Prepare features and target
    print("\n[3/5] Preparing features...")
    X, y = prepare_features(train_sample, target_col='y_target')
    weights = train_sample['weight'].values if 'weight' in train_sample.columns else None
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {len(X.columns)}")
    
    # Temporal train/validation split per horizon-specific series
    print("\n[4/5] Creating train/validation split...")
    train_indices = []
    val_indices = []
    for ts_id in train_sample['ts_id'].unique():
        ts_data = train_sample[train_sample['ts_id'] == ts_id].sort_values('ts_index')
        split_idx = int(len(ts_data) * 0.8)
        train_indices.extend(ts_data.index[:split_idx])
        val_indices.extend(ts_data.index[split_idx:])

    X_train = X.loc[train_indices]
    X_val = X.loc[val_indices]
    y_train = y.loc[train_indices]
    y_val = y.loc[val_indices]
    weights_train = weights[train_indices] if weights is not None else None
    weights_val = weights[val_indices] if weights is not None else None
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Save feature column names for predict/val alignment (GB-specific schema)
    save_feature_columns(list(X_train.columns), config.GB_FEATURE_COLUMNS_PATH)
    print(f"Saved {len(X_train.columns)} feature columns -> {config.GB_FEATURE_COLUMNS_PATH}")
    
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
    if 'OFFICIAL_SCORE' in train_metrics:
        print(f"          Official score: {train_metrics['OFFICIAL_SCORE']:.6f}")
    print(f"  Val   - MAE: {val_metrics['MAE']:.4f}, RMSE: {val_metrics['RMSE']:.4f}")
    if 'OFFICIAL_SCORE' in val_metrics:
        print(f"          Official score: {val_metrics['OFFICIAL_SCORE']:.6f}")
    
    # Feature importance
    print("\nFeature importance (Top 20):")
    importance = gb_model.get_feature_importance()
    print(importance.head(20).to_string(index=False))
    
    # Save model
    import pickle
    with open(config.GB_MODEL_PATH, 'wb') as f:
        pickle.dump(gb_model, f)
    print(f"\nModel saved to '{config.GB_MODEL_PATH}'")
    
    print("\n" + "=" * 60)
    print("Quick training complete!")
    print("Run train.py for full training on all data.")
    print("=" * 60)
    
    return gb_model


if __name__ == "__main__":
    model = main()
