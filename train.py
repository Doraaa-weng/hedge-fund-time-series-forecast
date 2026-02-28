"""
Main training script for time series forecasting
"""
import pandas as pd
import numpy as np
import config
from utils import (
    prepare_features,
    temporal_train_test_split,
    calculate_metrics
)
from feature_engineering import engineer_features
from models import LightGBMModel, XGBoostModel, GradientBoostingModel, LIGHTGBM_AVAILABLE
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("Time Series Forecasting - Training Pipeline")
    print("=" * 60)
    
    # Load training data
    print("\n[1/6] Loading training data...")
    train = pd.read_parquet(config.TRAIN_PATH)
    print(f"Training data shape: {train.shape}")
    
    # Feature engineering
    print("\n[2/6] Engineering features...")
    train = engineer_features(train, is_train=True)
    
    # Prepare features and target
    print("\n[3/6] Preparing features...")
    X, y = prepare_features(train, target_col='y_target')
    weights = train['weight'].values if 'weight' in train.columns else None
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {len(X.columns)}")
    
    # Temporal train/validation split
    print("\n[4/6] Creating temporal train/validation split...")
    train_df = train.copy()
    train_df['target'] = y
    train_df['features'] = [None] * len(train_df)  # Placeholder
    
    # Split by time series
    train_indices = []
    val_indices = []
    
    print("Splitting time series...")
    unique_ts = train_df['ts_id'].unique()
    for i, ts_id in enumerate(unique_ts):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(unique_ts)} time series...")
        ts_data = train_df[train_df['ts_id'] == ts_id].sort_values('ts_index')
        n = len(ts_data)
        split_idx = int(n * (1 - config.VALIDATION_SIZE))
        
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
    
    # Train models
    print("\n[5/6] Training models...")
    
    models = {}
    results = {}
    
    # Try LightGBM first, fallback to GradientBoosting
    if LIGHTGBM_AVAILABLE:
        try:
            print("\nTraining LightGBM...")
            lgb_model = LightGBMModel()
            lgb_model.train(
                X_train, y_train,
                X_val, y_val,
                weights_train, weights_val
            )
            models['lightgbm'] = lgb_model
            model_name = 'lightgbm'
        except Exception as e:
            print(f"LightGBM failed: {e}")
            print("Falling back to Gradient Boosting...")
            LIGHTGBM_AVAILABLE = False
    
    if not LIGHTGBM_AVAILABLE or 'lightgbm' not in models:
        print("\nTraining Gradient Boosting (scikit-learn)...")
        gb_model = GradientBoostingModel(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05
        )
        gb_model.train(
            X_train, y_train,
            X_val, y_val,
            weights_train, weights_val
        )
        models['gradient_boosting'] = gb_model
        model_name = 'gradient_boosting'
        lgb_model = gb_model  # Use for compatibility
    
    # Make predictions
    y_train_pred = lgb_model.predict(X_train)
    y_val_pred = lgb_model.predict(X_val)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train.values, y_train_pred, weights_train)
    val_metrics = calculate_metrics(y_val.values, y_val_pred, weights_val)
    
    results['lightgbm'] = {
        'train': train_metrics,
        'val': val_metrics
    }
    
    print(f"\nLightGBM Results:")
    print(f"  Train - MAE: {train_metrics['MAE']:.4f}, RMSE: {train_metrics['RMSE']:.4f}")
    print(f"  Val   - MAE: {val_metrics['MAE']:.4f}, RMSE: {val_metrics['RMSE']:.4f}")
    
    # Feature importance
    print("\n[6/6] Feature importance (Top 20):")
    importance = lgb_model.get_feature_importance()
    print(importance.head(20).to_string(index=False))
    
    # Save model
    import pickle
    model_filename = f'{model_name}_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"\nModel saved to '{model_filename}'")
    
    # Also save as lightgbm_model.pkl for compatibility with predict.py
    with open('lightgbm_model.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    
    # Save results
    results_df = pd.DataFrame({
        'model': ['lightgbm'],
        'train_mae': [train_metrics['MAE']],
        'train_rmse': [train_metrics['RMSE']],
        'val_mae': [val_metrics['MAE']],
        'val_rmse': [val_metrics['RMSE']]
    })
    results_df.to_csv('training_results.csv', index=False)
    print("Results saved to 'training_results.csv'")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return models, results


if __name__ == "__main__":
    models, results = main()
