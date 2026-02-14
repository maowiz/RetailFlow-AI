# src/models/ensemble_model.py

"""
Ensemble Forecasting Model

Combines predictions from multiple models to create a 
stronger, more robust final forecast.

Why ensemble?
- XGBoost might capture non-linear feature interactions well
  but miss smooth seasonal trends
- Prophet captures seasonality and holidays beautifully
  but doesn't use rich tabular features
- Random Forest provides stable baseline predictions
  with natural variance estimation

By combining them intelligently, we get:
- Lower variance (averaging reduces individual model noise)
- Lower bias (different models capture different patterns)
- Better calibrated uncertainty estimates
- More robust predictions across different store/category segments

Ensemble Methods Implemented:
1. Simple Average - Equal weight to all models
2. Weighted Average - Weight by inverse validation error
3. Stacking (Meta-Learner) - Train a model on top of base model predictions
4. Dynamic Weighting - Different weights per store/category segment
"""

import numpy as np
import pandas as pd
import logging
import joblib
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """
    Container for a single model's predictions and metadata.
    
    Attributes:
        model_name: Identifier for the model (e.g., 'xgboost', 'prophet')
        predictions: Array of predicted values
        validation_metrics: Dictionary of validation performance metrics
        prediction_intervals: Optional tuple of (lower_bound, upper_bound)
        training_time: Time taken to train in seconds
        feature_importance: Optional dictionary of feature importances
    """
    model_name: str
    predictions: np.ndarray
    validation_metrics: Dict[str, float]
    prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    training_time: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None


class EnsembleForecaster:
    """
    Ensemble model that combines multiple forecasting models.
    
    This is the final prediction layer that stakeholders see.
    It takes predictions from Prophet, XGBoost, and Random Forest,
    and produces a single, optimized forecast with confidence intervals.
    
    Usage:
        ensemble = EnsembleForecaster(method='weighted_average')
        
        # Add base model predictions
        ensemble.add_model_prediction(xgb_prediction)
        ensemble.add_model_prediction(prophet_prediction)
        ensemble.add_model_prediction(rf_prediction)
        
        # Fit ensemble (calculates optimal weights)
        ensemble.fit(y_true_validation)
        
        # Generate final forecast
        final_forecast = ensemble.predict()
    """
    
    SUPPORTED_METHODS = [
        'simple_average',
        'weighted_average', 
        'inverse_error_weighted',
        'stacking_ridge',
        'stacking_linear',
        'dynamic_weighted'
    ]
    
    def __init__(
        self,
        method: str = 'inverse_error_weighted',
        clip_predictions: bool = True,
        min_prediction: float = 0.0,
        confidence_level: float = 0.95
    ):
        """
        Initialize EnsembleForecaster.
        
        Args:
            method: Ensemble combination method. One of:
                - 'simple_average': Equal weight to all models
                - 'weighted_average': Manual weights
                - 'inverse_error_weighted': Auto-weight by 1/RMSE
                - 'stacking_ridge': Ridge regression meta-learner
                - 'stacking_linear': Linear regression meta-learner
                - 'dynamic_weighted': Segment-specific weights
            clip_predictions: Whether to clip predictions to min_prediction
            min_prediction: Minimum allowed prediction value (0 for sales)
            confidence_level: Confidence level for prediction intervals
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Method '{method}' not supported. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        
        self.method = method
        self.clip_predictions = clip_predictions
        self.min_prediction = min_prediction
        self.confidence_level = confidence_level
        
        # Storage for base model predictions
        self.model_predictions: Dict[str, ModelPrediction] = {}
        
        # Ensemble parameters (fitted)
        self.weights: Optional[Dict[str, float]] = None
        self.meta_model: Optional[Any] = None
        self.is_fitted: bool = False
        
        # Performance tracking
        self.ensemble_metrics: Dict[str, float] = {}
        self.model_comparison: Optional[pd.DataFrame] = None
        
        # Dynamic weights storage (for segment-specific weighting)
        self.segment_weights: Optional[Dict[str, Dict[str, float]]] = None
        
        logger.info(f"EnsembleForecaster initialized with method: {method}")
    
    def add_model_prediction(self, prediction: ModelPrediction) -> None:
        """
        Add a base model's predictions to the ensemble.
        
        Args:
            prediction: ModelPrediction object containing predictions and metrics
        """
        if prediction.model_name in self.model_predictions:
            logger.warning(
                f"Overwriting existing predictions for '{prediction.model_name}'"
            )
        
        self.model_predictions[prediction.model_name] = prediction
        
        logger.info(
            f"Added model '{prediction.model_name}' to ensemble | "
            f"Val RMSE: {prediction.validation_metrics.get('rmse', 'N/A'):.4f} | "
            f"Val MAPE: {prediction.validation_metrics.get('mape', 'N/A'):.4f} | "
            f"Predictions shape: {prediction.predictions.shape}"
        )
    
    def fit(
        self,
        y_true: np.ndarray,
        validation_predictions: Optional[Dict[str, np.ndarray]] = None,
        segment_labels: Optional[np.ndarray] = None
    ) -> 'EnsembleForecaster':
        """
        Fit the ensemble by determining optimal combination weights/model.
        
        Args:
            y_true: True values for the validation period
            validation_predictions: Optional dict of {model_name: val_predictions}
                If not provided, uses the predictions stored in model_predictions
            segment_labels: Optional array of segment labels for dynamic weighting
                e.g., store_nbr or category names for per-segment weights
        
        Returns:
            self (for method chaining)
        """
        logger.info("=" * 60)
        logger.info(f"FITTING ENSEMBLE (method: {self.method})")
        logger.info("=" * 60)
        
        if len(self.model_predictions) < 2:
            raise ValueError(
                f"Need at least 2 models for ensemble, "
                f"got {len(self.model_predictions)}"
            )
        
        # Get validation predictions
        if validation_predictions is None:
            val_preds = {
                name: pred.predictions 
                for name, pred in self.model_predictions.items()
            }
        else:
            val_preds = validation_predictions
        
        # Verify all predictions have same length
        lengths = {name: len(preds) for name, preds in val_preds.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Prediction length mismatch: {lengths}. "
                f"All models must predict the same number of samples."
            )
        
        if len(y_true) != list(lengths.values())[0]:
            raise ValueError(
                f"y_true length ({len(y_true)}) doesn't match "
                f"predictions length ({list(lengths.values())[0]})"
            )
        
        # Fit based on selected method
        if self.method == 'simple_average':
            self._fit_simple_average(val_preds)
        
        elif self.method == 'weighted_average':
            self._fit_simple_average(val_preds)  # Same, weights set manually later
        
        elif self.method == 'inverse_error_weighted':
            self._fit_inverse_error_weighted(y_true, val_preds)
        
        elif self.method == 'stacking_ridge':
            self._fit_stacking(y_true, val_preds, regularized=True)
        
        elif self.method == 'stacking_linear':
            self._fit_stacking(y_true, val_preds, regularized=False)
        
        elif self.method == 'dynamic_weighted':
            if segment_labels is None:
                logger.warning(
                    "No segment_labels provided for dynamic weighting. "
                    "Falling back to inverse_error_weighted."
                )
                self._fit_inverse_error_weighted(y_true, val_preds)
            else:
                self._fit_dynamic_weighted(y_true, val_preds, segment_labels)
        
        # Calculate ensemble performance on validation
        ensemble_preds = self._combine_predictions(val_preds, segment_labels)
        self.ensemble_metrics = self._calculate_metrics(y_true, ensemble_preds)
        
        # Build comparison table
        self._build_comparison_table(y_true, val_preds, ensemble_preds)
        
        self.is_fitted = True
        
        logger.info(f"\nEnsemble fitted successfully!")
        logger.info(f"Ensemble Validation Metrics:")
        for metric, value in self.ensemble_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        return self
    
    def _fit_simple_average(self, val_preds: Dict[str, np.ndarray]) -> None:
        """
        Simple average: equal weight to each model.
        
        This is surprisingly effective and hard to beat.
        Research shows simple averaging often performs within 
        1-2% of optimal weighting.
        """
        n_models = len(val_preds)
        self.weights = {
            name: 1.0 / n_models 
            for name in val_preds.keys()
        }
        
        logger.info(f"Simple average weights: {self.weights}")
    
    def _fit_inverse_error_weighted(
        self,
        y_true: np.ndarray,
        val_preds: Dict[str, np.ndarray]
    ) -> None:
        """
        Weight each model by inverse of its validation RMSE.
        
        Logic: A model with half the error gets twice the weight.
        
        Formula:
            weight_i = (1 / RMSE_i) / sum(1 / RMSE_j for all j)
        
        This ensures:
        - Better models get higher weights
        - Weights sum to 1
        - No model gets zero weight (all contribute)
        """
        # Calculate RMSE for each model
        rmse_scores = {}
        for name, preds in val_preds.items():
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            rmse_scores[name] = rmse
            logger.info(f"  {name} validation RMSE: {rmse:.4f}")
        
        # Calculate inverse error weights
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        inverse_errors = {
            name: 1.0 / (rmse + epsilon) 
            for name, rmse in rmse_scores.items()
        }
        
        # Normalize to sum to 1
        total_inverse = sum(inverse_errors.values())
        self.weights = {
            name: inv_err / total_inverse 
            for name, inv_err in inverse_errors.items()
        }
        
        logger.info(f"\nInverse error weights:")
        for name, weight in sorted(
            self.weights.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(
                f"  {name}: {weight:.4f} "
                f"(RMSE: {rmse_scores[name]:.4f})"
            )
    
    def _fit_stacking(
        self,
        y_true: np.ndarray,
        val_preds: Dict[str, np.ndarray],
        regularized: bool = True
    ) -> None:
        """
        Stacking: Train a meta-learner on base model predictions.
        
        Instead of manually choosing weights, we let a regression 
        model learn the optimal combination.
        
        Ridge regression (regularized=True) is preferred because:
        - It prevents overfitting to validation data
        - It naturally shrinks weights toward equal weighting
        - It handles correlated predictions well (models often 
          make correlated errors)
        
        The meta-learner learns:
        - Which model is more accurate for different value ranges
        - How to correct systematic biases
        - Optimal combination that minimizes squared error
        """
        # Build meta-feature matrix
        # Each column is one model's predictions
        model_names = sorted(val_preds.keys())
        X_meta = np.column_stack([val_preds[name] for name in model_names])
        
        logger.info(f"Meta-feature matrix shape: {X_meta.shape}")
        logger.info(f"Model order: {model_names}")
        
        # Use time series cross-validation for the meta-learner too
        # This prevents the meta-learner from overfitting
        tscv = TimeSeriesSplit(n_splits=3)
        
        cv_scores = []
        
        if regularized:
            # Ridge with cross-validated alpha selection
            best_alpha = 1.0
            best_score = float('inf')
            
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                alpha_scores = []
                for train_idx, val_idx in tscv.split(X_meta):
                    model = Ridge(
                        alpha=alpha,
                        fit_intercept=True,
                        positive=True  # Force non-negative weights
                    )
                    model.fit(X_meta[train_idx], y_true[train_idx])
                    preds = model.predict(X_meta[val_idx])
                    score = np.sqrt(mean_squared_error(
                        y_true[val_idx], preds
                    ))
                    alpha_scores.append(score)
                
                avg_score = np.mean(alpha_scores)
                logger.info(f"  Ridge alpha={alpha}: CV RMSE={avg_score:.4f}")
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_alpha = alpha
            
            logger.info(f"  Best alpha: {best_alpha}")
            
            # Fit final meta-model with best alpha on all validation data
            self.meta_model = Ridge(
                alpha=best_alpha,
                fit_intercept=True,
                positive=True
            )
        else:
            self.meta_model = LinearRegression(
                fit_intercept=True,
                positive=True  # Force non-negative coefficients
            )
        
        self.meta_model.fit(X_meta, y_true)
        
        # Extract learned weights (coefficients)
        coefficients = self.meta_model.coef_
        intercept = self.meta_model.intercept_
        
        # Normalize coefficients to get interpretable weights
        total_coef = np.sum(np.abs(coefficients))
        if total_coef > 0:
            normalized_weights = np.abs(coefficients) / total_coef
        else:
            normalized_weights = np.ones(len(model_names)) / len(model_names)
        
        self.weights = {
            name: float(w) 
            for name, w in zip(model_names, normalized_weights)
        }
        
        # Store model order for prediction time
        self._stacking_model_order = model_names
        
        logger.info(f"\nStacking meta-learner fitted:")
        logger.info(f"  Intercept: {intercept:.4f}")
        for name, coef, weight in zip(
            model_names, coefficients, normalized_weights
        ):
            logger.info(
                f"  {name}: coef={coef:.4f}, "
                f"normalized_weight={weight:.4f}"
            )
    
    def _fit_dynamic_weighted(
        self,
        y_true: np.ndarray,
        val_preds: Dict[str, np.ndarray],
        segment_labels: np.ndarray
    ) -> None:
        """
        Dynamic weighting: Different weights for different segments.
        
        Why? Different models excel in different contexts:
        - Prophet might be best for high-volume stores with clear seasonality
        - XGBoost might be best for stores with complex promotion patterns
        - Random Forest might be best for stable, predictable categories
        
        This method calculates optimal weights PER SEGMENT
        (segment = store, category, store_type, or any grouping).
        """
        unique_segments = np.unique(segment_labels)
        logger.info(f"Fitting dynamic weights for {len(unique_segments)} segments")
        
        self.segment_weights = {}
        
        for segment in unique_segments:
            mask = segment_labels == segment
            
            if mask.sum() < 10:
                # Too few samples for reliable weight estimation
                # Fall back to equal weights
                self.segment_weights[segment] = {
                    name: 1.0 / len(val_preds) 
                    for name in val_preds.keys()
                }
                continue
            
            # Calculate per-segment RMSE for each model
            segment_rmse = {}
            for name, preds in val_preds.items():
                rmse = np.sqrt(mean_squared_error(
                    y_true[mask], preds[mask]
                ))
                segment_rmse[name] = rmse
            
            # Inverse error weighting per segment
            epsilon = 1e-10
            inverse_errors = {
                name: 1.0 / (rmse + epsilon)
                for name, rmse in segment_rmse.items()
            }
            total_inverse = sum(inverse_errors.values())
            
            self.segment_weights[segment] = {
                name: inv_err / total_inverse
                for name, inv_err in inverse_errors.items()
            }
        
        # Also set global weights as fallback
        self._fit_inverse_error_weighted(y_true, val_preds)
        
        # Log a few example segments
        for segment in list(unique_segments)[:5]:
            weights = self.segment_weights[segment]
            best_model = max(weights, key=weights.get)
            logger.info(
                f"  Segment '{segment}': "
                f"best={best_model} ({weights[best_model]:.3f})"
            )
    
    def predict(
        self,
        test_predictions: Optional[Dict[str, np.ndarray]] = None,
        segment_labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            test_predictions: Dict of {model_name: test_predictions}
                If None, uses stored predictions
            segment_labels: Segment labels for dynamic weighting
        
        Returns:
            Array of ensemble predictions
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Ensemble not fitted yet. Call .fit() first."
            )
        
        # Get predictions to combine
        if test_predictions is None:
            preds = {
                name: pred.predictions
                for name, pred in self.model_predictions.items()
            }
        else:
            preds = test_predictions
        
        # Combine predictions
        ensemble_preds = self._combine_predictions(preds, segment_labels)
        
        # Clip to non-negative (sales can't be negative)
        if self.clip_predictions:
            ensemble_preds = np.clip(ensemble_preds, self.min_prediction, None)
        
        return ensemble_preds
    
    def _combine_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        segment_labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Core combination logic. Routes to appropriate method.
        
        Args:
            predictions: Dict of model predictions
            segment_labels: Optional segment labels for dynamic weighting
            
        Returns:
            Combined prediction array
        """
        if self.method in ['stacking_ridge', 'stacking_linear']:
            return self._combine_stacking(predictions)
        
        elif self.method == 'dynamic_weighted' and segment_labels is not None:
            return self._combine_dynamic(predictions, segment_labels)
        
        else:
            return self._combine_weighted(predictions)
    
    def _combine_weighted(
        self, predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine predictions using pre-computed weights.
        
        Formula: y_ensemble = sum(weight_i * prediction_i)
        """
        result = np.zeros_like(list(predictions.values())[0], dtype=np.float64)
        
        for name, preds in predictions.items():
            weight = self.weights.get(name, 0.0)
            result += weight * preds
        
        return result
    
    def _combine_stacking(
        self, predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine predictions using the trained meta-learner.
        
        The meta-model was trained to learn the optimal linear
        combination of base model predictions, including an 
        intercept term that can correct systematic bias.
        """
        if self.meta_model is None:
            raise RuntimeError("Meta-model not trained")
        
        # Build meta-feature matrix in same order as training
        model_names = self._stacking_model_order
        X_meta = np.column_stack([predictions[name] for name in model_names])
        
        # Predict using meta-learner
        result = self.meta_model.predict(X_meta)
        
        return result
    
    def _combine_dynamic(
        self,
        predictions: Dict[str, np.ndarray],
        segment_labels: np.ndarray
    ) -> np.ndarray:
        """
        Combine predictions using segment-specific weights.
        
        Each data point gets weights based on its segment.
        Falls back to global weights for unseen segments.
        """
        result = np.zeros(len(segment_labels), dtype=np.float64)
        
        for i, segment in enumerate(segment_labels):
            # Get segment-specific weights, fall back to global
            weights = self.segment_weights.get(segment, self.weights)
            
            for name, preds in predictions.items():
                weight = weights.get(name, 0.0)
                result[i] += weight * preds[i]
        
        return result
    
    def predict_with_intervals(
        self,
        test_predictions: Optional[Dict[str, np.ndarray]] = None,
        segment_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Confidence intervals are calculated by:
        1. Computing the spread (disagreement) among base models
        2. Using this spread as a measure of prediction uncertainty
        3. Adding/subtracting a z-score-scaled spread for bounds
        
        This approach captures model uncertainty — when models 
        disagree strongly, we're less confident.
        
        Returns:
            Tuple of (point_forecast, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call .fit() first.")
        
        # Get predictions
        if test_predictions is None:
            preds = {
                name: pred.predictions
                for name, pred in self.model_predictions.items()
            }
        else:
            preds = test_predictions
        
        # Point forecast (weighted combination)
        point_forecast = self._combine_predictions(preds, segment_labels)
        
        # Calculate prediction uncertainty from model disagreement
        all_preds = np.column_stack(list(preds.values()))
        
        # Method 1: Standard deviation across models (model disagreement)
        model_std = np.std(all_preds, axis=1)
        
        # Method 2: If any model provides prediction intervals, use them
        intervals_available = False
        for name, pred in self.model_predictions.items():
            if pred.prediction_intervals is not None:
                intervals_available = True
                break
        
        if intervals_available:
            # Combine intervals from models that provide them
            # using the ensemble weights
            lower_components = []
            upper_components = []
            
            for name, pred in self.model_predictions.items():
                weight = self.weights.get(name, 0.0)
                if pred.prediction_intervals is not None:
                    lower_components.append(
                        weight * pred.prediction_intervals[0]
                    )
                    upper_components.append(
                        weight * pred.prediction_intervals[1]
                    )
                else:
                    # Use point prediction ± model_std as proxy
                    lower_components.append(
                        weight * (pred.predictions - model_std)
                    )
                    upper_components.append(
                        weight * (pred.predictions + model_std)
                    )
            
            lower_bound = np.sum(lower_components, axis=0)
            upper_bound = np.sum(upper_components, axis=0)
        else:
            # Use model disagreement for intervals
            # z-score for 95% confidence: 1.96
            z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z = z_score.get(self.confidence_level, 1.96)
            
            # Scale the uncertainty: use weighted std across models
            # Also factor in the individual model errors from validation
            weighted_uncertainty = model_std.copy()
            
            # Add validation error as a floor for uncertainty
            if self.ensemble_metrics:
                val_rmse = self.ensemble_metrics.get('rmse', 0)
                # Uncertainty should be at least as large as validation error
                weighted_uncertainty = np.maximum(
                    weighted_uncertainty, val_rmse * 0.5
                )
            
            lower_bound = point_forecast - z * weighted_uncertainty
            upper_bound = point_forecast + z * weighted_uncertainty
        
        # Clip lower bound to non-negative
        if self.clip_predictions:
            point_forecast = np.clip(point_forecast, self.min_prediction, None)
            lower_bound = np.clip(lower_bound, self.min_prediction, None)
        
        return point_forecast, lower_bound, upper_bound
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Metrics explained for finance context:
        - RMSE: Average magnitude of forecast errors (in sales units)
        - MAE: Average absolute error (more robust to outliers)
        - MAPE: Percentage error (intuitive for business stakeholders)
        - SMAPE: Symmetric MAPE (handles zero actuals better)
        - R²: Proportion of variance explained (1.0 = perfect)
        - WAPE: Weighted Absolute Percentage Error (better for retail)
        """
        # Mask for non-zero actual values (MAPE is undefined for zero)
        non_zero_mask = y_true > 0
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
        }
        
        # MAPE (only on non-zero actuals)
        if non_zero_mask.sum() > 0:
            metrics['mape'] = float(mean_absolute_percentage_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            ))
        else:
            metrics['mape'] = float('inf')
        
        # SMAPE: Symmetric Mean Absolute Percentage Error
        # Better than MAPE because it's bounded [0, 2] and handles
        # cases where actual is 0
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        denominator = np.where(denominator == 0, 1.0, denominator)
        smape = np.mean(np.abs(y_true - y_pred) / denominator)
        metrics['smape'] = float(smape)
        
        # WAPE: Weighted Absolute Percentage Error
        # Preferred in retail because it weights errors by actual sales
        # High-volume items contribute more to the metric
        total_actual = np.sum(np.abs(y_true))
        if total_actual > 0:
            metrics['wape'] = float(
                np.sum(np.abs(y_true - y_pred)) / total_actual
            )
        else:
            metrics['wape'] = float('inf')
        
        # Bias: Average signed error (positive = over-forecasting)
        metrics['bias'] = float(np.mean(y_pred - y_true))
        
        # Forecast Accuracy (1 - WAPE, capped at 0)
        metrics['forecast_accuracy'] = max(0.0, 1.0 - metrics['wape'])
        
        return metrics
    
    def _build_comparison_table(
        self,
        y_true: np.ndarray,
        val_preds: Dict[str, np.ndarray],
        ensemble_preds: np.ndarray
    ) -> None:
        """
        Build a comparison table showing all models vs ensemble.
        
        This is crucial for the dashboard — stakeholders need to see
        that the ensemble actually improves upon individual models.
        """
        rows = []
        
        # Individual models
        for name, preds in val_preds.items():
            metrics = self._calculate_metrics(y_true, preds)
            metrics['model'] = name
            metrics['weight'] = self.weights.get(name, 0.0)
            rows.append(metrics)
        
        # Ensemble
        ensemble_metrics = self._calculate_metrics(y_true, ensemble_preds)
        ensemble_metrics['model'] = f'ENSEMBLE ({self.method})'
        ensemble_metrics['weight'] = 1.0
        rows.append(ensemble_metrics)
        
        self.model_comparison = pd.DataFrame(rows)
        
        # Reorder columns for readability
        col_order = [
            'model', 'weight', 'rmse', 'mae', 'mape', 
            'smape', 'wape', 'r2', 'forecast_accuracy', 'bias'
        ]
        existing_cols = [c for c in col_order if c in self.model_comparison.columns]
        self.model_comparison = self.model_comparison[existing_cols]
        
        # Log the comparison
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON TABLE")
        logger.info("=" * 80)
        logger.info(
            self.model_comparison.to_string(index=False, float_format='%.4f')
        )
        logger.info("=" * 80)
        
        # Highlight if ensemble is best
        best_rmse_model = self.model_comparison.loc[
            self.model_comparison['rmse'].idxmin(), 'model'
        ]
        if 'ENSEMBLE' in best_rmse_model:
            logger.info("✅ Ensemble achieves BEST RMSE!")
        else:
            ensemble_rmse = self.model_comparison[
                self.model_comparison['model'].str.contains('ENSEMBLE')
            ]['rmse'].values[0]
            best_rmse = self.model_comparison['rmse'].min()
            gap = ((ensemble_rmse - best_rmse) / best_rmse) * 100
            logger.info(
                f"ℹ️ Best individual model: {best_rmse_model} "
                f"(ensemble is {gap:.1f}% behind)"
            )
    
    def get_model_contribution_analysis(self) -> pd.DataFrame:
        """
        Analyze how much each model contributes to the ensemble.
        
        Returns a DataFrame with:
        - Model name
        - Weight in ensemble
        - Standalone performance
        - Marginal contribution (how much removing it hurts)
        
        This is valuable for stakeholders who want to understand
        the "why" behind the ensemble decision.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted yet.")
        
        contributions = []
        
        for model_name in self.model_predictions.keys():
            pred = self.model_predictions[model_name]
            
            contribution = {
                'model': model_name,
                'weight': self.weights.get(model_name, 0.0),
                'standalone_rmse': pred.validation_metrics.get('rmse', None),
                'standalone_mape': pred.validation_metrics.get('mape', None),
                'training_time_s': pred.training_time,
            }
            
            # Top features if available
            if pred.feature_importance:
                top_features = sorted(
                    pred.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                contribution['top_features'] = [f[0] for f in top_features]
            
            contributions.append(contribution)
        
        return pd.DataFrame(contributions).sort_values(
            'weight', ascending=False
        )
    
    def save(self, directory: str) -> None:
        """
        Save the fitted ensemble to disk.
        
        Saves:
        - Ensemble configuration and weights
        - Meta-model (if stacking)
        - Segment weights (if dynamic)
        - Performance metrics
        - Comparison table
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save configuration and weights
        config = {
            'method': self.method,
            'weights': self.weights,
            'ensemble_metrics': self.ensemble_metrics,
            'clip_predictions': self.clip_predictions,
            'min_prediction': self.min_prediction,
            'confidence_level': self.confidence_level,
            'is_fitted': self.is_fitted,
            'n_models': len(self.model_predictions),
            'model_names': list(self.model_predictions.keys())
        }
        
        with open(os.path.join(directory, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save meta-model if stacking
        if self.meta_model is not None:
            joblib.dump(
                self.meta_model,
                os.path.join(directory, 'meta_model.joblib')
            )
            # Save model order
            with open(os.path.join(directory, 'stacking_model_order.json'), 'w') as f:
                json.dump(self._stacking_model_order, f)
        
        # Save segment weights if dynamic
        if self.segment_weights is not None:
            # Convert keys to strings for JSON serialization
            serializable_weights = {
                str(k): v for k, v in self.segment_weights.items()
            }
            with open(os.path.join(directory, 'segment_weights.json'), 'w') as f:
                json.dump(serializable_weights, f, indent=2)
        
        # Save comparison table
        if self.model_comparison is not None:
            self.model_comparison.to_csv(
                os.path.join(directory, 'model_comparison.csv'),
                index=False
            )
        
        logger.info(f"Ensemble saved to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'EnsembleForecaster':
        """
        Load a fitted ensemble from disk.
        
        Args:
            directory: Directory containing saved ensemble files
            
        Returns:
            Fitted EnsembleForecaster instance
        """
        # Load configuration
        with open(os.path.join(directory, 'ensemble_config.json'), 'r') as f:
            config = json.load(f)
        
        # Create instance
        ensemble = cls(
            method=config['method'],
            clip_predictions=config['clip_predictions'],
            min_prediction=config['min_prediction'],
            confidence_level=config['confidence_level']
        )
        
        ensemble.weights = config['weights']
        ensemble.ensemble_metrics = config['ensemble_metrics']
        ensemble.is_fitted = config['is_fitted']
        
        # Load meta-model if stacking
        meta_model_path = os.path.join(directory, 'meta_model.joblib')
        if os.path.exists(meta_model_path):
            ensemble.meta_model = joblib.load(meta_model_path)
            with open(os.path.join(directory, 'stacking_model_order.json'), 'r') as f:
                ensemble._stacking_model_order = json.load(f)
        
        # Load segment weights if dynamic
        segment_path = os.path.join(directory, 'segment_weights.json')
        if os.path.exists(segment_path):
            with open(segment_path, 'r') as f:
                ensemble.segment_weights = json.load(f)
        
        # Load comparison table
        comparison_path = os.path.join(directory, 'model_comparison.csv')
        if os.path.exists(comparison_path):
            ensemble.model_comparison = pd.read_csv(comparison_path)
        
        logger.info(f"Ensemble loaded from {directory}")
        logger.info(f"  Method: {ensemble.method}")
        logger.info(f"  Weights: {ensemble.weights}")
        
        return ensemble
    
    def get_forecast_summary(
        self,
        predictions: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        store_ids: Optional[np.ndarray] = None,
        categories: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate a high-level forecast summary for stakeholders.
        
        This is what appears on the executive dashboard.
        
        Returns:
            Dictionary with summary statistics ready for display
        """
        summary = {
            'total_forecasted_sales': float(np.sum(predictions)),
            'avg_daily_sales': float(np.mean(predictions)),
            'median_daily_sales': float(np.median(predictions)),
            'max_daily_sales': float(np.max(predictions)),
            'min_daily_sales': float(np.min(predictions)),
            'std_daily_sales': float(np.std(predictions)),
            'forecast_count': len(predictions),
            'ensemble_method': self.method,
            'model_weights': self.weights,
            'ensemble_accuracy': self.ensemble_metrics.get(
                'forecast_accuracy', None
            ),
        }
        
        # Add time-based aggregations if dates provided
        if dates is not None:
            forecast_df = pd.DataFrame({
                'date': dates,
                'forecast': predictions
            })
            
            # Weekly totals
            weekly = forecast_df.set_index('date').resample('W')['forecast']
            summary['weekly_forecast'] = {
                str(date.date()): float(val)
                for date, val in weekly.sum().items()
            }
            
            # Monthly totals
            monthly = forecast_df.set_index('date').resample('M')['forecast']
            summary['monthly_forecast'] = {
                str(date.date()): float(val)
                for date, val in monthly.sum().items()
            }
        
        # Add per-store aggregations if store_ids provided
        if store_ids is not None:
            forecast_df = pd.DataFrame({
                'store': store_ids,
                'forecast': predictions
            })
            store_totals = forecast_df.groupby('store')['forecast'].sum()
            summary['top_5_stores'] = {
                str(store): float(val)
                for store, val in store_totals.nlargest(5).items()
            }
            summary['bottom_5_stores'] = {
                str(store): float(val)
                for store, val in store_totals.nsmallest(5).items()
            }
        
        return summary


# =============================================================
# CONVENIENCE FUNCTION: Build and run ensemble in one call
# =============================================================

def build_ensemble(
    model_results: Dict[str, Dict],
    y_val: np.ndarray,
    val_predictions: Dict[str, np.ndarray],
    method: str = 'inverse_error_weighted',
    segment_labels: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None
) -> Tuple[EnsembleForecaster, np.ndarray]:
    """
    Convenience function to build a complete ensemble in one call.
    
    Args:
        model_results: Dictionary with model results, each containing:
            - 'predictions': np.ndarray of predictions
            - 'metrics': dict of validation metrics
            - 'training_time': float
            - 'feature_importance': optional dict
        y_val: True validation values
        val_predictions: Dict of {model_name: validation_predictions}
        method: Ensemble method
        segment_labels: Optional segment labels for dynamic weighting
        save_dir: Optional directory to save the fitted ensemble
    
    Returns:
        Tuple of (fitted EnsembleForecaster, ensemble predictions)
    
    Example:
        model_results = {
            'xgboost': {
                'predictions': xgb_test_preds,
                'val_predictions': xgb_val_preds,
                'metrics': {'rmse': 450.2, 'mape': 0.12},
                'training_time': 45.2,
                'feature_importance': {'lag_7': 0.15, 'month': 0.12}
            },
            'prophet': {
                'predictions': prophet_test_preds,
                'val_predictions': prophet_val_preds,
                'metrics': {'rmse': 520.1, 'mape': 0.15},
                'training_time': 120.5,
                'feature_importance': None
            },
            'random_forest': {
                'predictions': rf_test_preds,
                'val_predictions': rf_val_preds,
                'metrics': {'rmse': 480.7, 'mape': 0.13},
                'training_time': 30.1,
                'feature_importance': {'lag_7': 0.18, 'store_nbr': 0.10}
            }
        }
        
        ensemble, final_preds = build_ensemble(
            model_results=model_results,
            y_val=y_validation,
            val_predictions={
                'xgboost': xgb_val_preds,
                'prophet': prophet_val_preds,
                'random_forest': rf_val_preds
            },
            method='inverse_error_weighted'
        )
    """
    logger.info("\n" + "🎯" * 30)
    logger.info("BUILDING ENSEMBLE FORECAST")
    logger.info("🎯" * 30)
    
    # Create ensemble
    ensemble = EnsembleForecaster(method=method)
    
    # Add each model's predictions
    for name, results in model_results.items():
        prediction = ModelPrediction(
            model_name=name,
            predictions=results['predictions'],
            validation_metrics=results.get('metrics', {}),
            prediction_intervals=results.get('prediction_intervals', None),
            training_time=results.get('training_time', 0.0),
            feature_importance=results.get('feature_importance', None)
        )
        ensemble.add_model_prediction(prediction)
    
    # Fit ensemble
    ensemble.fit(
        y_true=y_val,
        validation_predictions=val_predictions,
        segment_labels=segment_labels
    )
    
    # Generate final predictions with intervals
    final_preds, lower_bound, upper_bound = ensemble.predict_with_intervals(
        segment_labels=segment_labels
    )
    
    logger.info(f"\n📊 Final Ensemble Forecast Summary:")
    logger.info(f"  Mean prediction: {np.mean(final_preds):.2f}")
    logger.info(f"  Median prediction: {np.median(final_preds):.2f}")
    logger.info(f"  Std prediction: {np.std(final_preds):.2f}")
    logger.info(f"  Avg interval width: {np.mean(upper_bound - lower_bound):.2f}")
    
    # Save if directory provided
    if save_dir:
        ensemble.save(save_dir)
    
    return ensemble, final_preds
