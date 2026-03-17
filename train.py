"""
Train a direct LightGBM model with hierarchy categoricals and exogenous lags.
"""
import json
import warnings

import numpy as np
import pandas as pd

import config
from feature_engineering import build_global_lgb_features, ensure_raw_horizon
from utils import (
    make_global_time_split,
    prepare_lightgbm_frame,
    weighted_rmse_score,
)

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError as exc:
    raise RuntimeError("LightGBM is required for this training pipeline.") from exc


def lgb_skill_metric(y_pred: np.ndarray, dataset: lgb.Dataset):
    """Competition metric for LightGBM early stopping."""
    y_true = dataset.get_label()
    weights = dataset.get_weight()
    if weights is None:
        weights = np.ones_like(y_true, dtype=np.float32)
    return "skill_score", weighted_rmse_score(y_true, y_pred, weights), True


def select_lag_base_features(train_df: pd.DataFrame, raw_feature_cols: list[str]) -> list[str]:
    """Pick raw features that have the strongest linear relationship with y_target."""
    sample_size = min(config.FEATURE_SELECTION_SAMPLE, len(train_df))
    sample = train_df.sample(sample_size, random_state=config.RANDOM_STATE)
    scores = []
    for col in raw_feature_cols:
        series = sample[col]
        valid = series.notna() & sample["y_target"].notna()
        if valid.sum() < config.FEATURE_SELECTION_MIN_NON_NULL:
            continue
        corr = np.corrcoef(series[valid].astype(float), sample.loc[valid, "y_target"].astype(float))[0, 1]
        if np.isnan(corr):
            continue
        scores.append((col, abs(float(corr))))
    scores.sort(key=lambda item: item[1], reverse=True)
    selected = [col for col, _ in scores[: config.MAX_LAG_BASE_FEATURES]]
    return selected


def select_top_features_lgb_gain(
    df: pd.DataFrame,
    model_features: list[str],
    categorical_columns: list[str],
    target_col: str,
    weight_col: str,
    top_k: int,
) -> list[str]:
    """Select top features by LightGBM gain on a sampled time split."""
    sample_size = min(config.FEATURE_SELECTION_SAMPLE, len(df))
    sample_df = df.sample(sample_size, random_state=config.RANDOM_STATE).copy()
    train_idx, valid_idx, _ = make_global_time_split(
        sample_df,
        valid_fraction=config.FEATURE_SELECTION_VALID_FRACTION,
    )

    X_train, fill_values, cat_levels = prepare_lightgbm_frame(
        sample_df.loc[train_idx],
        feature_cols=model_features,
        categorical_cols=categorical_columns,
    )
    X_valid, _, _ = prepare_lightgbm_frame(
        sample_df.loc[valid_idx],
        feature_cols=model_features,
        categorical_cols=categorical_columns,
        numeric_fill_values=fill_values,
        categorical_levels=cat_levels,
    )
    y_train = sample_df.loc[train_idx, target_col].astype(np.float32).values
    y_valid = sample_df.loc[valid_idx, target_col].astype(np.float32).values
    w_train = sample_df.loc[train_idx, weight_col].astype(np.float32).values
    w_valid = sample_df.loc[valid_idx, weight_col].astype(np.float32).values

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        categorical_feature=[c for c in categorical_columns if c in X_train.columns],
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        weight=w_valid,
        categorical_feature=[c for c in categorical_columns if c in X_valid.columns],
        free_raw_data=False,
    )
    fs_model = lgb.train(
        params=config.LIGHTGBM_FS_PARAMS.copy(),
        train_set=dtrain,
        valid_sets=[dvalid],
        valid_names=["valid"],
        num_boost_round=150,
        callbacks=[
            lgb.early_stopping(stopping_rounds=25, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    gains = pd.DataFrame(
        {
            "feature": model_features,
            "importance": fs_model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    selected = gains["feature"].head(top_k).tolist()
    for col in categorical_columns:
        if col not in selected and col in model_features:
            selected.append(col)
    return selected


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Use raw numeric features, engineered exogenous lags, time features, and categoricals."""
    exclude = {"id", "y_target", "weight", "ts_id", "series_id"}
    return [col for col in df.columns if col not in exclude]


def main():
    print("=" * 60)
    print("Time Series Forecasting - Direct LightGBM Pipeline")
    print("=" * 60)

    print("\n[1/6] Loading training data...")
    train_df = pd.read_parquet(config.TRAIN_PATH)
    print(f"Training data shape: {train_df.shape}")
    if "horizon" not in train_df.columns:
        train_df = ensure_raw_horizon(train_df)

    raw_feature_cols = [col for col in train_df.columns if col.startswith("feature_")]
    lag_base_features = select_lag_base_features(train_df, raw_feature_cols)
    print(f"Selected lag base features ({len(lag_base_features)}): {lag_base_features}")

    print("\n[2/6] Building features...")
    train_df = build_global_lgb_features(train_df, lag_base_features=lag_base_features)
    model_features = get_model_feature_columns(train_df)
    categorical_cols = [col for col in config.CATEGORICAL_COLUMNS if col in model_features]
    print(f"Candidate feature columns: {len(model_features)}")
    print(f"Categorical columns: {categorical_cols}")

    print("\n[2.5/6] Selecting top features by LightGBM gain...")
    feature_cols = select_top_features_lgb_gain(
        train_df,
        model_features=model_features,
        categorical_columns=categorical_cols,
        target_col="y_target",
        weight_col="weight",
        top_k=config.FEATURE_TOP_K,
    )
    print(f"Selected feature columns: {len(feature_cols)}")

    print("\n[3/6] Creating global time split...")
    train_idx, valid_idx, cut_ts = make_global_time_split(
        train_df,
        valid_fraction=config.GLOBAL_TIME_VALID_FRACTION,
    )
    print(f"time cutoff ts_index: {cut_ts}")
    print(f"train rows: {len(train_idx)} valid rows: {len(valid_idx)}")

    y_train = train_df.loc[train_idx, "y_target"].astype(np.float32).values
    y_valid = train_df.loc[valid_idx, "y_target"].astype(np.float32).values
    w_train = train_df.loc[train_idx, "weight"].astype(np.float32).values
    w_valid = train_df.loc[valid_idx, "weight"].astype(np.float32).values

    print("\n[4/6] Preparing LightGBM frames...")
    X_train, numeric_fill_values, categorical_levels = prepare_lightgbm_frame(
        train_df.loc[train_idx],
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )
    X_valid, _, _ = prepare_lightgbm_frame(
        train_df.loc[valid_idx],
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_fill_values=numeric_fill_values,
        categorical_levels=categorical_levels,
    )

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        weight=w_valid,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    print("\n[5/6] Training validation model...")
    model = lgb.train(
        params=config.LIGHTGBM_GLOBAL_PARAMS.copy(),
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        num_boost_round=config.N_ESTIMATORS,
        feval=lgb_skill_metric,
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )

    best_iter = model.best_iteration if model.best_iteration and model.best_iteration > 0 else config.N_ESTIMATORS
    valid_pred = model.predict(X_valid, num_iteration=best_iter)
    valid_score = weighted_rmse_score(y_valid, valid_pred, w_valid)
    print(f"Validation skill-like score: {valid_score}")

    print("\n[6/6] Refitting on full training data...")
    X_full, numeric_fill_values, categorical_levels = prepare_lightgbm_frame(
        train_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )
    y_full = train_df["y_target"].astype(np.float32).values
    w_full = train_df["weight"].astype(np.float32).values
    if config.USE_RECENCY_WEIGHTING:
        ts = train_df["ts_index"].astype(np.float32).values
        ts_min = float(ts.min())
        ts_max = float(ts.max())
        recency = (ts - ts_min) / (ts_max - ts_min + 1e-6)
        w_full = w_full * (0.5 + 42.0 * recency.astype(np.float32))

    dfull = lgb.Dataset(
        X_full,
        label=y_full,
        weight=w_full,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    final_model = lgb.train(
        params=config.LIGHTGBM_GLOBAL_PARAMS.copy(),
        train_set=dfull,
        num_boost_round=max(int(best_iter * 1.05), 20),
    )

    final_model.save_model(config.GLOBAL_LGB_MODEL_PATH)
    metadata = {
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_fill_values": numeric_fill_values,
        "categorical_levels": categorical_levels,
        "lag_base_features": lag_base_features,
        "cut_ts_index": cut_ts,
        "best_iteration": int(best_iter),
        "validation_score": float(valid_score),
    }
    with open(config.GLOBAL_LGB_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    results_df = pd.DataFrame(
        [
            {
                "model": "global_lightgbm_direct",
                "validation_score": valid_score,
                "best_iteration": best_iter,
                "feature_count": len(feature_cols),
                "categorical_count": len(categorical_cols),
                "cut_ts_index": cut_ts,
            }
        ]
    )
    results_df.to_csv("training_results.csv", index=False)
    print(f"Saved model -> {config.GLOBAL_LGB_MODEL_PATH}")
    print(f"Saved metadata -> {config.GLOBAL_LGB_META_PATH}")
    print("Results saved to 'training_results.csv'")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    return final_model, metadata


if __name__ == "__main__":
    model, metadata = main()
