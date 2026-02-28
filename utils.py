"""
Utility functions for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def create_ts_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a time series identifier from code, sub_code, and sub_category"""
    df = df.copy()
    df['ts_id'] = df['code'] + '__' + df['sub_code'] + '__' + df['sub_category']
    return df


def create_lag_features(df: pd.DataFrame, 
                       target_col: str,
                       lag_periods: List[int],
                       group_col: str = 'ts_id') -> pd.DataFrame:
    """
    Create lag features for a target column grouped by time series
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column to create lags for
        lag_periods: List of lag periods (e.g., [1, 3, 7])
        group_col: Column to group by (time series identifier)
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    df = df.sort_values([group_col, 'ts_index'])
    
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           target_col: str,
                           windows: List[int],
                           stats: List[str],
                           group_col: str = 'ts_id') -> pd.DataFrame:
    """
    Create rolling window statistics for a target column
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        windows: List of window sizes (e.g., [7, 14, 30])
        stats: List of statistics to compute (e.g., ['mean', 'std'])
        group_col: Column to group by
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    df = df.sort_values([group_col, 'ts_index'])
    
    for window in windows:
        for stat in stats:
            if stat == 'mean':
                df[f'{target_col}_rolling_{window}_{stat}'] = (
                    df.groupby(group_col)[target_col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                    )
                )
            elif stat == 'std':
                df[f'{target_col}_rolling_{window}_{stat}'] = (
                    df.groupby(group_col)[target_col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                    )
                )
            elif stat == 'min':
                df[f'{target_col}_rolling_{window}_{stat}'] = (
                    df.groupby(group_col)[target_col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
                    )
                )
            elif stat == 'max':
                df[f'{target_col}_rolling_{window}_{stat}'] = (
                    df.groupby(group_col)[target_col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
                    )
                )
    
    return df


def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'median',
                         fill_value: float = 0.0) -> pd.DataFrame:
    """
    Handle missing values in the dataframe
    
    Args:
        df: DataFrame with potential missing values
        strategy: 'median', 'mean', 'forward_fill', 'zero', or 'constant'
        fill_value: Value to use if strategy is 'constant'
    
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'forward_fill':
                df[col] = df[col].ffill()
                df[col].fillna(0, inplace=True)  # Fill remaining with 0
            elif strategy == 'zero':
                df[col].fillna(0, inplace=True)
            elif strategy == 'constant':
                df[col].fillna(fill_value, inplace=True)
    
    return df


def prepare_features(df: pd.DataFrame, 
                    target_col: str = 'y_target',
                    exclude_cols: List[str] = None) -> tuple:
    """
    Prepare features for modeling
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if exclude_cols is None:
        exclude_cols = ['id', 'y_target', 'weight', 'ts_id']
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y


def temporal_train_test_split(df: pd.DataFrame,
                              group_col: str = 'ts_id',
                              time_col: str = 'ts_index',
                              test_size: float = 0.2) -> tuple:
    """
    Split data temporally (use last portion of each time series for validation)
    
    Args:
        df: DataFrame with time series data
        group_col: Column to group by (time series identifier)
        time_col: Time index column
        test_size: Proportion of data to use for validation
    
    Returns:
        Tuple of (train_df, val_df)
    """
    df = df.copy()
    df = df.sort_values([group_col, time_col])
    
    train_dfs = []
    val_dfs = []
    
    for ts_id, group in df.groupby(group_col):
        n = len(group)
        split_idx = int(n * (1 - test_size))
        
        train_group = group.iloc[:split_idx]
        val_group = group.iloc[split_idx:]
        
        train_dfs.append(train_group)
        val_dfs.append(val_group)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    return train_df, val_df


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     weights: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        weights: Optional weights for weighted metrics
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Mean Absolute Error
    if weights is not None:
        mae = np.average(np.abs(y_true - y_pred), weights=weights)
        rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))
    else:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    metrics['MAE'] = mae
    metrics['RMSE'] = rmse
    
    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['MAPE'] = mape
    
    return metrics
