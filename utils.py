"""
Utility functions for time series forecasting
"""
import json
from collections import deque
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import config


def save_feature_columns(columns: List[str], path: str = config.GB_FEATURE_COLUMNS_PATH) -> None:
    """Save training feature column names so predict/val can align to the same space."""
    with open(path, "w") as f:
        json.dump(list(columns), f)


def align_features(X: pd.DataFrame,
                  feature_columns_path: str = config.GB_FEATURE_COLUMNS_PATH,
                  cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Align feature columns to training-time columns: add missing (fill 0), drop extra, reorder.
    Use feature_columns_path if no cols given; cols can be from model.feature_names as fallback.
    """
    if cols is None:
        try:
            with open(feature_columns_path, "r") as f:
                cols = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Feature columns file not found: {feature_columns_path}. "
                "Run training first to create it, or pass cols= explicitly."
            )
    missing = [c for c in cols if c not in X.columns]
    extra = [c for c in X.columns if c not in cols]
    if missing or extra:
        print(f"Aligning features: {len(missing)} cols filled with 0, {len(extra)} cols dropped.")
    X = X.reindex(columns=cols, fill_value=0.0)
    return X


def create_ts_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a horizon-specific series identifier used for autoregressive features."""
    df = df.copy()
    if 'horizon' not in df.columns:
        raise ValueError("Missing 'horizon' column required to build ts_id")
    df['series_id'] = df['code'] + '__' + df['sub_code'] + '__' + df['sub_category']
    df['ts_id'] = df['series_id'] + '__' + df['horizon'].astype(str)
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


def get_train_tail_y_features(train_df: pd.DataFrame,
                              target_col: str = 'y_target',
                              group_col: str = 'ts_id') -> pd.DataFrame:
    """
    For each ts_id in train, compute lag and rolling features using only the
    *last* part of the series. Used at predict time to fill test rows so the
    model gets train-history instead of zeros (no look-forward).
    Returns DataFrame with columns [ts_id, y_target_lag_1, ..., y_target_rolling_7_mean, ...].
    """
    df = train_df[[group_col, 'ts_index', target_col]].copy()
    df = df.sort_values([group_col, 'ts_index'])
    lag_periods = config.LAG_FEATURES
    windows = config.ROLLING_WINDOWS
    stats = config.ROLLING_STATS
    rows = []
    for ts_id, g in df.groupby(group_col):
        y = g[target_col].values
        n = len(y)
        if n == 0:
            continue
        row = {group_col: ts_id}
        for lag in lag_periods:
            row[f'{target_col}_lag_{lag}'] = y[-lag] if n >= lag else (y[0] if n else np.nan)
        for w in windows:
            tail = y[-w:] if n >= w else y
            for stat in stats:
                if stat == 'mean':
                    row[f'{target_col}_rolling_{w}_mean'] = np.nanmean(tail)
                elif stat == 'std':
                    row[f'{target_col}_rolling_{w}_std'] = np.nanstd(tail) if len(tail) > 1 else 0.0
                elif stat == 'min':
                    row[f'{target_col}_rolling_{w}_min'] = np.nanmin(tail)
                elif stat == 'max':
                    row[f'{target_col}_rolling_{w}_max'] = np.nanmax(tail)
        rows.append(row)
    return pd.DataFrame(rows)


def get_train_tail_target_history(train_df: pd.DataFrame,
                                  target_col: str = 'y_target',
                                  group_col: str = 'ts_id',
                                  history_len: Optional[int] = None) -> Dict[str, deque]:
    """
    Return the last `history_len` target values per time series for recursive inference.
    """
    if history_len is None:
        history_len = max(max(config.LAG_FEATURES), max(config.ROLLING_WINDOWS))
    df = train_df[[group_col, 'ts_index', target_col]].copy()
    df = df.sort_values([group_col, 'ts_index'])
    histories: Dict[str, deque] = {}
    for ts_id, group in df.groupby(group_col):
        tail = group[target_col].tail(history_len).tolist()
        histories[ts_id] = deque(tail, maxlen=history_len)
    return histories


def build_target_history_features(history: List[float],
                                  target_col: str = 'y_target') -> Dict[str, float]:
    """
    Build lag and rolling features from the available target history only.
    """
    values = np.asarray(list(history), dtype=float)
    features: Dict[str, float] = {}

    for lag in config.LAG_FEATURES:
        if len(values) >= lag:
            features[f'{target_col}_lag_{lag}'] = float(values[-lag])
        else:
            features[f'{target_col}_lag_{lag}'] = np.nan

    for window in config.ROLLING_WINDOWS:
        tail = values[-window:]
        for stat in config.ROLLING_STATS:
            col = f'{target_col}_rolling_{window}_{stat}'
            if len(tail) == 0:
                features[col] = np.nan
            elif stat == 'mean':
                features[col] = float(np.nanmean(tail))
            elif stat == 'std':
                features[col] = float(np.nanstd(tail)) if len(tail) > 1 else 0.0
            elif stat == 'min':
                features[col] = float(np.nanmin(tail))
            elif stat == 'max':
                features[col] = float(np.nanmax(tail))

    return features


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
        exclude_cols = ['id', 'y_target', 'weight', 'ts_id', 'series_id']
    # Exclude code/sub_code/sub_category so train and test have same features
    for skip in ['code', 'sub_code', 'sub_category']:
        if skip not in exclude_cols:
            exclude_cols.append(skip)
    # Do not exclude horizon or horizon_*; they are produced in engineer_features from ensure_raw_horizon (train and test consistent)
    assert "horizon" not in exclude_cols, "horizon must not be excluded so model can use it"
    feature_cols = [col for col in df.columns
                    if col not in exclude_cols and col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    # Handle any remaining categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Final safeguard: some raw features and early lags remain entirely/partially
    # missing after feature engineering. Tree models in this project expect a dense
    # numeric matrix, so clean up any residual NaN/inf here in one place.
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().values.any():
        numeric_medians = X.median(numeric_only=True)
        X = X.fillna(numeric_medians)
        X = X.fillna(0.0)
    
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
    if weights is not None:
        metrics['OFFICIAL_SCORE'] = weighted_rmse_score(y_true, y_pred, weights)
    
    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['MAPE'] = mape
    
    return metrics


def weighted_rmse_score(y_target: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """
    Official Kaggle competition metric (skill score).
    Score = sqrt(1 - clip(ratio)), ratio = sum(w*(y-yhat)^2) / sum(w*y^2).
    """
    denom = np.sum(w * y_target ** 2)
    if denom <= 0:
        return 0.0
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = float(np.minimum(np.maximum(ratio, 0.0), 1.0))
    return float(np.sqrt(1.0 - clipped))


def make_global_time_split(
    df: pd.DataFrame,
    valid_fraction: float = 0.15,
    time_col: str = "ts_index",
) -> tuple[pd.Index, pd.Index, int]:
    """Split by global time so validation is a true future holdout."""
    ts_values = np.sort(df[time_col].unique())
    cut_idx = int((1.0 - valid_fraction) * len(ts_values))
    cut_idx = min(max(cut_idx, 0), len(ts_values) - 1)
    cut_value = int(ts_values[cut_idx])
    train_idx = df.index[df[time_col] <= cut_value]
    valid_idx = df.index[df[time_col] > cut_value]
    return train_idx, valid_idx, cut_value


def prepare_lightgbm_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    numeric_fill_values: Optional[Dict[str, float]] = None,
    categorical_levels: Optional[Dict[str, List[str]]] = None,
) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, List[str]]]:
    """
    Prepare a DataFrame for LightGBM with stable numeric fills and categorical vocab.
    """
    X = df[feature_cols].copy()
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if numeric_fill_values is None:
        numeric_fill_values = {}
        for col in numeric_cols:
            series = X[col].dropna()
            numeric_fill_values[col] = float(series.median()) if not series.empty else 0.0
    for col in numeric_cols:
        X[col] = X[col].fillna(numeric_fill_values.get(col, 0.0)).astype(np.float32)

    if categorical_levels is None:
        categorical_levels = {}
        for col in categorical_cols:
            series = X[col].astype("string").fillna("__missing__")
            values = series.astype(str).tolist()
            categories = list(dict.fromkeys(values))
            if "__missing__" not in categories:
                categories.append("__missing__")
            categorical_levels[col] = categories
            X[col] = pd.Categorical(values, categories=categories)
    else:
        for col in categorical_cols:
            series = X[col].astype("string").fillna("__missing__").astype(str)
            categories = list(categorical_levels.get(col, []))
            if "__missing__" not in categories:
                categories.append("__missing__")
            series = series.where(series.isin(categories), "__missing__")
            X[col] = pd.Categorical(series, categories=categories)

    return X, numeric_fill_values, categorical_levels
