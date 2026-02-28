"""
Feature engineering for time series forecasting
"""
import pandas as pd
import numpy as np
from utils import (
    create_ts_id, 
    create_lag_features, 
    create_rolling_features,
    handle_missing_values
)
import config


def engineer_features(df: pd.DataFrame, 
                     is_train: bool = True,
                     target_col: str = 'y_target') -> pd.DataFrame:
    """
    Main feature engineering function
    
    Args:
        df: Input dataframe
        is_train: Whether this is training data (has target)
        target_col: Name of target column
    
    Returns:
        DataFrame with engineered features
    """
    print("Starting feature engineering...")
    df = df.copy()
    
    # Create time series identifier
    df = create_ts_id(df)
    print(f"Created ts_id. Unique time series: {df['ts_id'].nunique()}")
    
    # Sort by time series and time index
    df = df.sort_values(['ts_id', 'ts_index']).reset_index(drop=True)
    
    # Create lag features from target (only for training data)
    if is_train and target_col in df.columns:
        print("Creating lag features from target...")
        df = create_lag_features(
            df, 
            target_col=target_col,
            lag_periods=config.LAG_FEATURES,
            group_col='ts_id'
        )
        
        # Create rolling features from target
        print("Creating rolling features from target...")
        df = create_rolling_features(
            df,
            target_col=target_col,
            windows=config.ROLLING_WINDOWS,
            stats=config.ROLLING_STATS,
            group_col='ts_id'
        )
    
    # Create lag features from other important features
    # Select a few key features for lagging (to avoid too many features)
    key_features = ['feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f']
    available_features = [f for f in key_features if f in df.columns]
    
    if len(available_features) > 0:
        print(f"Creating lag features from key features: {available_features[:3]}...")
        for feat in available_features[:3]:  # Limit to avoid too many features
            try:
                df = create_lag_features(
                    df,
                    target_col=feat,
                    lag_periods=[1, 3, 7],
                    group_col='ts_id'
                )
            except Exception as e:
                print(f"Warning: Could not create lag features for {feat}: {e}")
    
    # Create horizon-based features
    print("Creating horizon-based features...")
    df['horizon_1'] = (df['horizon'] == 1).astype(int)
    df['horizon_3'] = (df['horizon'] == 3).astype(int)
    df['horizon_10'] = (df['horizon'] == 10).astype(int)
    df['horizon_25'] = (df['horizon'] == 25).astype(int)
    
    # Create interaction features
    print("Creating interaction features...")
    if 'ts_index' in df.columns:
        df['ts_index_norm'] = df.groupby('ts_id')['ts_index'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
    
    # Handle missing values
    print("Handling missing values...")
    df = handle_missing_values(df, strategy='median')
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df
