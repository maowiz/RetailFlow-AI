
# src/models/random_forest_model.py

"""
Random Forest Forecasting Model

Random Forest serves as our ROBUST BASELINE because:

1. Resistant to overfitting (due to bagging/averaging)
2. No hyperparameter sensitivity (works well out of the box)
3. Provides feature importance (different perspective than XGBoost)
4. Handles non-linear relationships
5. Good performance even without extensive tuning

How Random Forest works:
- Creates N decision trees, each trained on a random subset of data
- Each tree also uses a random subset of features
- Final prediction = average of all tree predictions
- Randomness + averaging = low variance = robust predictions

vs XGBoost:
- XGBoost: Trees are built SEQUENTIALLY (each corrects previous errors)
- Random Forest: Trees are built INDEPENDENTLY (each sees different data)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import logging
import time
from typing import Dict, List, Optional

from src.models.base_model import BaseForecaster

logger = logging.getLogger(__name__)


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest-based sales forecasting model.
    """
    
    def __init__(
        self,
        params: Dict = None,
        model_dir: str = 'models/'
    ):
        super().__init__('Random_Forest', model_dir)
        
        self.params = params or {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',  # Use sqrt(n_features) per tree
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        self.feature_importance_df: Optional[pd.DataFrame] = None
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_names: List[str],
        target_col: str = 'sales'
    ) -> Dict[str, float]:
        """Train Random Forest model."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {self.model_name}...")
        logger.info(f"{'='*50}")
        
        self.feature_names = feature_names
        
        X_train = train_df[feature_names].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_names].values
        y_val = val_df[target_col].values
        
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Validation set: {X_val.shape}")
        
        # For memory efficiency on large datasets, sample if needed
        max_train_rows = 1_000_000
        if len(X_train) > max_train_rows:
            logger.info(f"  Sampling {max_train_rows:,} rows from "
                       f"{len(X_train):,} for memory efficiency")
            idx = np.random.choice(len(X_train), max_train_rows, replace=False)
            X_train_sample = X_train[idx]
            y_train_sample = y_train[idx]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        # Train
        self.model = RandomForestRegressor(**self.params)
        
        start_time = time.time()
        self.model.fit(X_train_sample, y_train_sample)
        training_time = time.time() - start_time
        
        logger.info(f"  Training time: {training_time:.1f}s")
        
        # Evaluate
        train_pred = np.maximum(self.model.predict(X_train_sample), 0)
        val_pred = np.maximum(self.model.predict(X_val), 0)
        
        train_metrics = self.evaluate(y_train_sample, train_pred)
        val_metrics = self.evaluate(y_val, val_pred)
        
        logger.info(f"\n  Training Metrics:")
        for k, v in train_metrics.items():
            logger.info(f"    {k}: {v}")
        
        logger.info(f"\n  Validation Metrics:")
        for k, v in val_metrics.items():
            logger.info(f"    {k}: {v}")
        
        # Feature importance
        self._compute_feature_importance(feature_names)
        
        self.training_metadata = {
            'training_time_seconds': round(training_time, 1),
            'n_features': len(feature_names),
            'train_rows_used': len(X_train_sample),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'params': {k: str(v) for k, v in self.params.items()}
        }
        
        self.is_trained = True
        return val_metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        feature_names: List[str] = None
    ) -> np.ndarray:
        """Generate predictions."""
        features = feature_names or self.feature_names
        X = df[features].values
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)
    
    def _compute_feature_importance(self, feature_names: List[str]) -> None:
        """Compute and log feature importance."""
        importance = self.model.feature_importances_
        
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n  Top 20 Most Important Features (Random Forest):")
        logger.info(f"  {'Feature':<40} {'Importance':>10}")
        logger.info(f"  {'-'*50}")
        
        for _, row in self.feature_importance_df.head(20).iterrows():
            logger.info(f"  {row['feature']:<40} {row['importance']:>10.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        return self.feature_importance_df.copy()
