
# src/models/prophet_model.py

"""
Prophet Forecasting Model

Prophet (by Meta/Facebook) excels at:
1. Capturing multiple seasonalities (weekly, monthly, yearly)
2. Handling holidays automatically
3. Providing uncertainty intervals
4. Being robust to missing data and outliers
5. Decomposing forecast into trend + seasonality + holidays

LIMITATION: Prophet models one time series at a time.
With 54 stores × 33 families = 1,782 time series,
training all would take hours.

STRATEGY: We train Prophet on aggregated data:
- Total sales across all stores (captures overall trend)
- Top 5 store-family combinations (for store-level forecasts)
- Then use Prophet predictions as a FEATURE for XGBoost
  (this is called "feature stacking")

This gives us the best of both worlds:
- Prophet captures seasonality/trend
- XGBoost uses that as input alongside other features
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional
from prophet import Prophet

from src.models.base_model import BaseForecaster

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """
    Prophet-based sales forecasting model.
    
    Since Prophet is univariate (one series at a time),
    we use it strategically on aggregated series.
    """
    
    def __init__(
        self,
        params: Dict = None,
        model_dir: str = 'models/',
        top_n_series: int = 10
    ):
        super().__init__('Prophet', model_dir)
        
        self.params = params or {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05,
            'holidays_prior_scale': 10.0,
            'seasonality_prior_scale': 10.0,
            'interval_width': 0.95,
        }
        
        self.top_n_series = top_n_series
        self.models: Dict[str, Prophet] = {}
        self.aggregated_forecasts: Optional[pd.DataFrame] = None
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_names: List[str],
        target_col: str = 'sales'
    ) -> Dict[str, float]:
        """
        Train Prophet on aggregated time series.
        
        Prophet requires specific column names:
        - ds: datestamp column
        - y: target variable
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {self.model_name}...")
        logger.info(f"{'='*50}")
        
        self.feature_names = feature_names
        start_time = time.time()
        
        # ====== 1. TOTAL SALES FORECAST ======
        logger.info("  Training on total daily sales...")
        total_sales = (
            train_df.groupby('date')[target_col]
            .sum()
            .reset_index()
            .rename(columns={'date': 'ds', target_col: 'y'})
        )
        
        # Suppress Prophet's verbose output
        import logging as _logging
        _logging.getLogger('prophet').setLevel(_logging.WARNING)
        _logging.getLogger('cmdstanpy').setLevel(_logging.WARNING)
        
        model_total = Prophet(**self.params)
        model_total.fit(total_sales)
        self.models['total'] = model_total
        
        logger.info(f"    Total sales model trained on {len(total_sales)} days")
        
        # ====== 2. STORE-LEVEL FORECASTS (Top N) ======
        logger.info(f"  Training on top {self.top_n_series} store-family series...")
        
        # Find top store-family combinations by total sales
        store_family_sales = (
            train_df.groupby(['store_nbr', 'family'])[target_col]
            .sum()
            .sort_values(ascending=False)
        )
        top_combinations = store_family_sales.head(self.top_n_series).index.tolist()
        
        for store_nbr, family in top_combinations:
            key = f"store_{store_nbr}_{family}"
            
            series_data = (
                train_df[
                    (train_df['store_nbr'] == store_nbr) & 
                    (train_df['family'] == family)
                ]
                .groupby('date')[target_col]
                .sum()
                .reset_index()
                .rename(columns={'date': 'ds', target_col: 'y'})
            )
            
            if len(series_data) < 30:  # Need minimum data for Prophet
                continue
            
            model = Prophet(**self.params)
            model.fit(series_data)
            self.models[key] = model
        
        logger.info(f"    Trained {len(self.models)} Prophet models")
        
        # ====== 3. GENERATE FORECASTS FOR VALIDATION ======
        val_dates = val_df[['date']].drop_duplicates().rename(columns={'date': 'ds'})
        
        # Total forecast
        total_forecast = model_total.predict(val_dates)
        
        training_time = time.time() - start_time
        logger.info(f"  Training time: {training_time:.1f}s")
        
        # ====== 4. EVALUATE ON VALIDATION (aggregated level) ======
        val_total_actual = (
            val_df.groupby('date')[target_col]
            .sum()
            .reset_index()
            .rename(columns={'date': 'ds'})
        )
        
        merged = val_total_actual.merge(
            total_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds',
            how='left'
        )
        
        y_true = merged[target_col].values
        y_pred = np.maximum(merged['yhat'].values, 0)
        
        val_metrics = self.evaluate(y_true, y_pred)
        
        logger.info(f"\n  Validation Metrics (Total Sales):")
        for k, v in val_metrics.items():
            logger.info(f"    {k}: {v}")
        
        self.training_metadata = {
            'training_time_seconds': round(training_time, 1),
            'n_models': len(self.models),
            'total_training_days': len(total_sales),
            'val_metrics_total': val_metrics,
            'params': {k: str(v) for k, v in self.params.items()}
        }
        
        self.is_trained = True
        return val_metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        feature_names: List[str] = None
    ) -> np.ndarray:
        """
        Generate Prophet predictions.
        
        For row-level predictions, we use the total forecast
        and distribute it proportionally based on historical patterns.
        """
        if 'total' not in self.models:
            raise ValueError("Total model not trained.")
        
        # Get unique dates
        dates = df[['date']].drop_duplicates().rename(columns={'date': 'ds'})
        
        # Generate total forecast
        forecast = self.models['total'].predict(dates)
        
        # Map back to rows
        date_forecast = forecast[['ds', 'yhat']].rename(
            columns={'ds': 'date', 'yhat': 'prophet_total_forecast'}
        )
        
        # Merge to get per-row forecast
        df_temp = df[['date']].merge(date_forecast, on='date', how='left')
        
        predictions = np.maximum(df_temp['prophet_total_forecast'].values, 0)
        
        return predictions
    
    def get_prophet_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate Prophet-based features to use in XGBoost/RF.
        
        This is the "stacking" approach:
        Prophet predictions become features for other models.
        
        Features generated:
        - prophet_total_yhat: Total daily forecast
        - prophet_total_trend: Trend component
        - prophet_total_yearly: Yearly seasonality component
        - prophet_total_weekly: Weekly seasonality component
        """
        logger.info("  Generating Prophet features for stacking...")
        
        dates = df[['date']].drop_duplicates().rename(columns={'date': 'ds'})
        
        forecast = self.models['total'].predict(dates)
        
        prophet_features = forecast[['ds']].rename(columns={'ds': 'date'})
        prophet_features['prophet_yhat'] = forecast['yhat'].values
        prophet_features['prophet_trend'] = forecast['trend'].values
        
        if 'yearly' in forecast.columns:
            prophet_features['prophet_yearly'] = forecast['yearly'].values
        if 'weekly' in forecast.columns:
            prophet_features['prophet_weekly'] = forecast['weekly'].values
        
        # Merge to original df
        result = df.merge(prophet_features, on='date', how='left')
        
        # Fill NaN
        for col in ['prophet_yhat', 'prophet_trend', 'prophet_yearly', 'prophet_weekly']:
            if col in result.columns:
                result[col] = result[col].fillna(0).astype(np.float32)
        
        logger.info(f"    Added {len(prophet_features.columns)-1} Prophet features")
        
        return result
    
    def save(self) -> str:
        """Save all Prophet models."""
        import pickle
        
        for key, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{key}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        import json, os
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'n_models': len(self.models),
                'model_keys': list(self.models.keys()),
                'training_metadata': self.training_metadata
            }, f, indent=2, default=str)
        
        logger.info(f"  Saved {len(self.models)} Prophet models to {self.model_dir}")
        return self.model_dir
