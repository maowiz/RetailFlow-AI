# src/models/xgboost_model.py

"""
XGBoost Forecasting Model

XGBoost (eXtreme Gradient Boosting) is our PRIMARY model because:

1. Best performance on tabular data (consistently wins ML competitions)
2. Handles missing values natively
3. Feature importance built-in
4. Fast training with GPU support
5. Regularization prevents overfitting
6. Works well with our engineered features (especially lags)

How XGBoost works (simplified):
- Starts with a simple prediction (e.g., mean of all sales)
- Builds a decision tree to predict the ERRORS of that prediction
- Adds that tree to the model (weighted by learning rate)
- Builds another tree to predict the remaining errors
- Repeats 1000+ times, each tree correcting previous errors
- Final prediction = sum of all tree predictions

Key hyperparameters:
- n_estimators: Number of trees (more = potentially better but slower)
- max_depth: Tree depth (deeper = more complex patterns but risk overfit)
- learning_rate: How much each tree contributes (lower = more trees needed)
- subsample: Fraction of data per tree (< 1 adds randomness = less overfit)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import time
from typing import Dict, List, Optional

from src.models.base_model import BaseForecaster

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost-based sales forecasting model.
    """
    
    def __init__(
        self,
        params: Dict = None,
        model_dir: str = 'models/'
    ):
        """
        Initialize XGBoost forecaster.
        
        Args:
            params: XGBoost hyperparameters
            model_dir: Directory to save model
        """
        super().__init__('XGBoost', model_dir)
        
        # Default parameters (well-tuned for retail sales forecasting)
        self.params = params or {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,        # L1 regularization
            'reg_lambda': 1.0,       # L2 regularization
            'random_state': 42,
            'n_jobs': -1,            # Use all CPU cores
            'tree_method': 'hist',   # Fast histogram-based method
            'objective': 'reg:squarederror',
        }
        
        self.early_stopping_rounds = 50
        self.feature_importance_df: Optional[pd.DataFrame] = None
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_names: List[str],
        target_col: str = 'sales'
    ) -> Dict[str, float]:
        """
        Train XGBoost model with early stopping.
        
        Early stopping monitors validation loss and stops training
        when it hasn't improved for N rounds. This:
        1. Prevents overfitting
        2. Finds optimal number of trees automatically
        3. Saves training time
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {self.model_name}...")
        logger.info(f"{'='*50}")
        
        self.feature_names = feature_names
        
        # Prepare data
        X_train = train_df[feature_names].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_names].values
        y_val = val_df[target_col].values
        
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Validation set: {X_val.shape}")
        logger.info(f"  Features: {len(feature_names)}")
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Train with early stopping
        start_time = time.time()
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100  # Print every 100 rounds
        )
        
        training_time = time.time() - start_time
        
        # Get best iteration
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score
        
        logger.info(f"  Training time: {training_time:.1f}s")
        logger.info(f"  Best iteration: {best_iteration}")
        logger.info(f"  Best validation score: {best_score:.6f}")
        
        # Evaluate on training and validation sets
        train_pred = self.predict(train_df, feature_names)
        val_pred = self.predict(val_df, feature_names)
        
        train_metrics = self.evaluate(y_train, train_pred)
        val_metrics = self.evaluate(y_val, val_pred)
        
        logger.info(f"\n  Training Metrics:")
        for k, v in train_metrics.items():
            logger.info(f"    {k}: {v}")
        
        logger.info(f"\n  Validation Metrics:")
        for k, v in val_metrics.items():
            logger.info(f"    {k}: {v}")
        
        # Feature importance
        self._compute_feature_importance(feature_names)
        
        # Save metadata
        self.training_metadata = {
            'training_time_seconds': round(training_time, 1),
            'best_iteration': int(best_iteration),
            'n_features': len(feature_names),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'params': self.params
        }
        
        self.is_trained = True
        
        return val_metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        feature_names: List[str] = None
    ) -> np.ndarray:
        """
        Generate predictions using trained XGBoost model.
        
        Clips predictions to non-negative values since sales can't
        be negative.
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = feature_names or self.feature_names
        X = df[features].values
        
        predictions = self.model.predict(X)
        
        # Sales can't be negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def _compute_feature_importance(self, feature_names: List[str]) -> None:
        """
        Compute and log feature importance.
        
        XGBoost provides multiple importance types:
        - 'weight': Number of times feature is used in trees
        - 'gain': Average prediction improvement when feature is used
        - 'cover': Number of samples affected by the feature
        
        'gain' is usually the most informative for understanding 
        which features actually improve predictions.
        """
        importance = self.model.feature_importances_
        
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Log top 20 features
        logger.info(f"\n  Top 20 Most Important Features:")
        logger.info(f"  {'Feature':<40} {'Importance':>10}")
        logger.info(f"  {'-'*50}")
        
        for _, row in self.feature_importance_df.head(20).iterrows():
            logger.info(f"  {row['feature']:<40} {row['importance']:>10.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as DataFrame."""
        if self.feature_importance_df is None:
            raise ValueError("No feature importance available. Train model first.")
        return self.feature_importance_df.copy()
