"""
Model definitions and training functions
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config
from utils import calculate_metrics

# Try to import LightGBM and XGBoost, but make them optional
LIGHTGBM_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError, Exception):
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, OSError, Exception):
    XGBOOST_AVAILABLE = False
    xgb = None


class TimeSeriesModel:
    """Base class for time series models"""
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              weights_train: pd.Series = None, weights_val: pd.Series = None):
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        raise NotImplementedError


class LightGBMModel(TimeSeriesModel):
    """LightGBM model for time series forecasting"""
    
    def __init__(self, params: Dict = None):
        super().__init__('lightgbm')
        if not LIGHTGBM_AVAILABLE or lgb is None:
            raise ImportError("LightGBM is not available. Please install it or use another model.")
        self.params = params or config.LIGHTGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              weights_train: pd.Series = None, weights_val: pd.Series = None):
        """Train LightGBM model"""
        self.feature_names = X_train.columns.tolist()
        
        # Prepare training data
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            weight=weights_train,
            feature_name=self.feature_names
        )
        
        # Prepare validation data if provided
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                weight=weights_val,
                reference=train_data,
                feature_name=self.feature_names
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=config.N_ESTIMATORS,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        importance = self.model.feature_importance(importance_type='gain')
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class GradientBoostingModel(TimeSeriesModel):
    """Scikit-learn Gradient Boosting model (fallback)"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__('gradient_boosting')
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=config.RANDOM_STATE,
            verbose=1
        )
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              weights_train: pd.Series = None, weights_val: pd.Series = None):
        """Train Gradient Boosting model"""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train, sample_weight=weights_train)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class XGBoostModel(TimeSeriesModel):
    """XGBoost model for time series forecasting"""
    
    def __init__(self, params: Dict = None):
        super().__init__('xgboost')
        self.params = params or config.XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              weights_train: pd.Series = None, weights_val: pd.Series = None):
        """Train XGBoost model"""
        self.feature_names = X_train.columns.tolist()
        
        # Prepare training data
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
        
        # Prepare validation data if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val)
            evals.append((dval, 'valid'))
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=config.N_ESTIMATORS,
            evals=evals,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose_eval=100
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        importance = self.model.get_score(importance_type='gain')
        return pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)


def create_baseline_models() -> Dict[str, TimeSeriesModel]:
    """Create baseline models"""
    return {
        'lightgbm': LightGBMModel(),
        'xgboost': XGBoostModel()
    }
