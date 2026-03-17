"""
Feature engineering for time series forecasting
"""
import pandas as pd
import numpy as np
from utils import (
    create_ts_id,
    create_lag_features,
    create_rolling_features,
    handle_missing_values,
)
import config


def ensure_raw_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth: always parse raw horizon from id (format ...__<horizon>__...)."""
    if "id" not in df.columns:
        raise ValueError("Missing 'id' column required to parse horizon")
    h = df["id"].astype(str).str.split("__").str[3].astype(int)
    df["horizon"] = h
    return df


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
    df = ensure_raw_horizon(df)

    # Create time series identifier (no further assignment to df["horizon"] below)
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
    
    # Horizon dummies only (read df["horizon"]; no assignment to df["horizon"] here)
    print("Creating horizon-based features...")
    df["horizon_1"] = (df["horizon"] == 1).astype(int)
    df["horizon_3"] = (df["horizon"] == 3).astype(int)
    df["horizon_10"] = (df["horizon"] == 10).astype(int)
    df["horizon_25"] = (df["horizon"] == 25).astype(int)

    # No ts_index_norm: normalizing by (min, max) of full series would use future
    # ts_index, violating "predict t using only data 0..t". Use raw ts_index only.
    print("Creating interaction features...")
    
    # Handle missing values
    print("Handling missing values...")
    df = handle_missing_values(df, strategy='median')

    min_h = getattr(config, "MIN_HORIZON_NUNIQUE", 4)
    if df["horizon"].nunique() < min_h:
        raise RuntimeError(
            f"Horizon collapsed: nunique={df['horizon'].nunique()} (min required {min_h}). Raw horizon must be preserved."
        )
    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df


def build_global_lgb_features(
    df: pd.DataFrame,
    lag_base_features: list[str],
    key_cols=("code", "sub_code", "sub_category", "horizon"),
    time_col="ts_index",
    dtype_float=np.float32,
) -> pd.DataFrame:
    """
    Build a leakage-safe, non-autoregressive feature set.

    This mirrors the higher-scoring LightGBM approach more closely:
    keep hierarchy columns as categoricals, use raw factors, and only add
    lags of observable exogenous features (never y_target lags).
    """
    df = df.copy()
    if "horizon" not in df.columns:
        df = ensure_raw_horizon(df)

    sort_cols = [c for c in key_cols if c in df.columns] + [time_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    group_cols = [c for c in key_cols if c in df.columns]
    grouped = df.groupby(group_cols, sort=False) if group_cols else None

    tmin = float(df[time_col].min())
    tmax = float(df[time_col].max())
    denom = max(tmax - tmin, 1.0)
    ts_values = df[time_col].astype(dtype_float)
    df["ts_norm"] = ((ts_values - tmin) / denom).astype(dtype_float)
    df["ts_sin"] = np.sin(2.0 * np.pi * ts_values / 100.0).astype(dtype_float)
    df["ts_cos"] = np.cos(2.0 * np.pi * ts_values / 100.0).astype(dtype_float)

    if grouped is not None:
        for col in lag_base_features:
            if col not in df.columns:
                continue
            for lag in config.EXOGENOUS_LAGS:
                df[f"{col}_lag{lag}"] = grouped[col].shift(lag).astype(dtype_float)

    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("__missing__")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(dtype_float)

    return df
