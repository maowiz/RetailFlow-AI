# src/insights/anomaly_detector.py

"""
Sales Anomaly Detection
Identifies unusual deviations between forecast and actual sales.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SalesAnomalyDetector:
    """Detects anomalies using z-score and percentage deviation methods."""
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        pct_threshold: float = 0.5,
        min_samples: int = 30
    ):
        self.z_threshold = z_threshold
        self.pct_threshold = pct_threshold
        self.min_samples = min_samples
    
    def detect_forecast_anomalies(
        self,
        forecast_df: pd.DataFrame,
        actual_col: str = 'sales',
        forecast_col: str = 'forecast_ensemble',
        segment_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies where actuals deviate significantly from forecasts.
        """
        df = forecast_df.copy()
        
        # Calculate deviations
        df['deviation'] = df[actual_col] - df[forecast_col]
        df['abs_deviation'] = np.abs(df['deviation'])
        df['pct_deviation'] = np.where(
            df[forecast_col] > 0,
            df['deviation'] / df[forecast_col],
            0
        )
        df['abs_pct_deviation'] = np.abs(df['pct_deviation'])
        
        # Method 1: Percentage threshold
        pct_flag = df['abs_pct_deviation'] > self.pct_threshold
        
        # Method 2: Z-score
        residual_std = df['deviation'].std()
        if residual_std > 0:
            df['residual_zscore'] = (
                (df['deviation'] - df['deviation'].mean()) / residual_std
            )
            zscore_flag = np.abs(df['residual_zscore']) > self.z_threshold
        else:
            df['residual_zscore'] = 0
            zscore_flag = pd.Series(False, index=df.index)
        
        # Combine
        df['is_anomaly'] = pct_flag | zscore_flag
        
        # Severity score (0-100)
        df['severity'] = np.clip(
            df['abs_pct_deviation'] * 100 + np.abs(df['residual_zscore']) * 10,
            0, 100
        )
        
        # Type
        df['anomaly_type'] = 'Normal'
        df.loc[df['is_anomaly'] & (df['deviation'] > 0), 'anomaly_type'] = 'Spike'
        df.loc[df['is_anomaly'] & (df['deviation'] < 0), 'anomaly_type'] = 'Drop'
        
        # Filter anomalies only
        anomalies = df[df['is_anomaly']].copy()
        
        # Select columns
        cols = ['date', actual_col, forecast_col, 'deviation',
                'pct_deviation', 'severity', 'anomaly_type']
        if segment_cols:
            cols = [c for c in segment_cols if c in df.columns] + cols
        
        cols = [c for c in cols if c in anomalies.columns]
        anomalies = anomalies[cols].sort_values('severity', ascending=False)
        
        logger.info(f"Detected {len(anomalies)} anomalies "
                    f"({len(anomalies)/len(df)*100:.1f}%)")
        
        return anomalies
    
    def get_anomaly_summary(self, anomalies: pd.DataFrame) -> Dict:
        """Generate summary for dashboard display."""
        if len(anomalies) == 0:
            return {
                'total': 0,
                'message': 'No significant anomalies detected ✅',
                'top_anomalies': []
            }
        
        return {
            'total': len(anomalies),
            'spikes': int((anomalies['anomaly_type'] == 'Spike').sum()),
            'drops': int((anomalies['anomaly_type'] == 'Drop').sum()),
            'avg_severity': float(anomalies['severity'].mean()),
            'max_severity': float(anomalies['severity'].max()),
            'top_anomalies': anomalies.head(10).to_dict('records')
        }
