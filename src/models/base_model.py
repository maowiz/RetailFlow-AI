# src/models/base_model.py

"""
Abstract Base Model Class
Defines the interface that all models must implement.
Ensures consistency across different model types.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    All models must implement:
    - train(): Fit model on training data
    - predict(): Generate forecasts
    - evaluate(): Calculate performance metrics
    - save() / load(): Persist model to disk
    
    This ensures we can swap models easily and create ensembles.
    """
    
    def __init__(self, model_name: str, model_dir: str = 'models/'):
        """
        Initialize base forecaster.
        
        Args:
            model_name: Human-readable model name
            model_dir: Directory to save model artifacts
        """
        self.model_name = model_name
        self.model_dir = os.path.join(model_dir, model_name.lower().replace(' ', '_'))
        self.model = None
        self.is_trained = False
        self.training_metadata: Dict = {}
        self.feature_names: list = []
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    @abstractmethod
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_names: list,
        target_col: str = 'sales'
    ) -> Dict[str, float]:
        """
        Train the model on training data.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (for early stopping, monitoring)
            feature_names: List of feature column names to use
            target_col: Name of the target column
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        feature_names: list = None
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            df: DataFrame to predict on
            feature_names: Feature columns (uses training features if None)
            
        Returns:
            Array of predictions
        """
        pass
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Metrics explained:
        - MAE: Average absolute error (easy to interpret, in sales units)
        - RMSE: Root mean squared error (penalizes large errors more)
        - MAPE: Mean absolute percentage error (scale-independent)
        - SMAPE: Symmetric MAPE (handles zero values better)
        - R²: Coefficient of determination (how much variance explained)
        """
        # Ensure non-negative predictions (sales can't be negative)
        y_pred = np.maximum(y_pred, 0)
        
        # MAE - Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # RMSE - Root Mean Squared Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE - Mean Absolute Percentage Error
        # Only compute where y_true > 0 to avoid division by zero
        mask = y_true > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')
        
        # SMAPE - Symmetric Mean Absolute Percentage Error
        # Handles zeros better than MAPE
        denominator = (np.abs(y_true) + np.abs(y_pred))
        denominator = np.where(denominator == 0, 1, denominator)
        smape = np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100
        
        # R² - Coefficient of Determination
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSLE - Root Mean Squared Log Error 
        # This is the actual Kaggle competition metric
        rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
        
        metrics = {
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'mape': round(float(mape), 4),
            'smape': round(float(smape), 4),
            'r2': round(float(r2), 4),
            'rmsle': round(float(rmsle), 6)
        }
        
        return metrics
    
    def save(self) -> str:
        """Save model and metadata to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Save model
        model_path = os.path.join(self.model_dir, 'model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        metadata = {
            'model_name': self.model_name,
            'trained_at': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'training_metadata': self.training_metadata
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"  Model saved to {self.model_dir}")
        return model_path
    
    def load(self) -> None:
        """Load model and metadata from disk."""
        model_path = os.path.join(self.model_dir, 'model.pkl')
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_metadata = metadata['training_metadata']
        self.is_trained = True
        
        logger.info(f"  Model loaded from {self.model_dir}")
