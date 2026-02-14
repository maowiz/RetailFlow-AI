# src/optimization/safety_stock.py

"""
Safety Stock Calculator

Calculates the optimal safety stock levels based on:
- Demand forecast uncertainty (from ensemble model)
- Supply lead time variability
- Desired service level
- Product/store characteristics

This is the foundation of inventory optimization.
Without proper safety stock, stockouts are inevitable.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class SafetyStockCalculator:
    """
    Calculates safety stock using multiple methods.
    
    Methods implemented:
    1. Standard method: Z × σ_demand × √(LT)
    2. Demand + lead time variability method
    3. Forecast error based method (uses ML model errors)
    4. Service-level differentiated method
    
    The forecast-error-based method (Method 3) is particularly 
    powerful because it directly uses the ensemble model's 
    prediction uncertainty to set safety stock levels.
    Better forecast = lower safety stock = lower costs.
    """
    
    # Z-scores for common service levels
    Z_SCORES = {
        0.85: 1.036,
        0.90: 1.282,
        0.95: 1.645,
        0.97: 1.881,
        0.98: 2.054,
        0.99: 2.326,
        0.995: 2.576,
        0.999: 3.090
    }
    
    def __init__(
        self,
        default_service_level: float = 0.95,
        default_lead_time_days: int = 7,
        lead_time_std_days: float = 2.0
    ):
        """
        Initialize SafetyStockCalculator.
        
        Args:
            default_service_level: Default probability of not stocking out
            default_lead_time_days: Average supplier lead time
            lead_time_std_days: Standard deviation of lead time
        """
        self.default_service_level = default_service_level
        self.default_lead_time_days = default_lead_time_days
        self.lead_time_std_days = lead_time_std_days
    
    def get_z_score(self, service_level: float) -> float:
        """
        Get z-score for a given service level.
        
        Uses lookup table for common values, scipy for others.
        
        Args:
            service_level: Desired probability (e.g., 0.95)
            
        Returns:
            Z-score value
        """
        if service_level in self.Z_SCORES:
            return self.Z_SCORES[service_level]
        
        # Use inverse normal CDF for custom service levels
        return float(stats.norm.ppf(service_level))
    
    def method_standard(
        self,
        demand_std: float,
        lead_time_days: Optional[int] = None,
        service_level: Optional[float] = None
    ) -> float:
        """
        Standard Safety Stock Method.
        
        Formula: SS = Z × σ_demand × √(Lead_Time)
        
        Assumptions:
        - Lead time is constant (no variability)
        - Demand follows normal distribution
        - Demand variability is stationary
        
        Best for: Stable products with reliable suppliers
        
        Args:
            demand_std: Standard deviation of daily demand
            lead_time_days: Supplier lead time in days
            service_level: Desired service level
            
        Returns:
            Safety stock quantity (units)
        """
        lt = lead_time_days or self.default_lead_time_days
        sl = service_level or self.default_service_level
        z = self.get_z_score(sl)
        
        safety_stock = z * demand_std * np.sqrt(lt)
        
        return max(0, np.ceil(safety_stock))
    
    def method_demand_and_lead_time(
        self,
        avg_daily_demand: float,
        demand_std: float,
        lead_time_days: Optional[int] = None,
        lead_time_std: Optional[float] = None,
        service_level: Optional[float] = None
    ) -> float:
        """
        Safety Stock with both demand AND lead time variability.
        
        Formula: SS = Z × √(LT × σ²_demand + d² × σ²_LT)
        
        This is more realistic because in practice, suppliers
        don't always deliver on time. Fashion/textile industry 
        especially has variable lead times due to:
        - Raw material availability
        - Production capacity constraints
        - Shipping delays
        - Customs clearance (for imports)
        
        This method accounts for BOTH sources of uncertainty.
        
        Args:
            avg_daily_demand: Mean daily demand
            demand_std: Standard deviation of daily demand
            lead_time_days: Average lead time
            lead_time_std: Standard deviation of lead time
            service_level: Desired service level
            
        Returns:
            Safety stock quantity (units)
        """
        lt = lead_time_days or self.default_lead_time_days
        lt_std = lead_time_std or self.lead_time_std_days
        sl = service_level or self.default_service_level
        z = self.get_z_score(sl)
        
        # Combined variance from demand and lead time uncertainty
        demand_variance_component = lt * (demand_std ** 2)
        lead_time_variance_component = (avg_daily_demand ** 2) * (lt_std ** 2)
        
        combined_std = np.sqrt(
            demand_variance_component + lead_time_variance_component
        )
        
        safety_stock = z * combined_std
        
        return max(0, np.ceil(safety_stock))
    
    def method_forecast_error(
        self,
        forecast_errors: np.ndarray,
        lead_time_days: Optional[int] = None,
        service_level: Optional[float] = None
    ) -> float:
        """
        Forecast Error Based Safety Stock — THE BEST METHOD.
        
        Instead of using historical demand variability, this uses
        the actual forecast errors from our ML model.
        
        Formula: SS = Z × σ_forecast_error × √(LT)
        
        WHY THIS IS SUPERIOR:
        - Directly measures HOW WRONG our forecast is
        - Better ML model → smaller errors → lower safety stock → lower costs
        - This creates a direct link: 
          Better AI = Lower Inventory Costs = Financial Impact
        
        This is the key insight for the Sapphire interview:
        "Our ML model doesn't just predict sales — it directly 
         reduces inventory holding costs by enabling more precise 
         safety stock calculations."
        
        Args:
            forecast_errors: Array of (actual - predicted) from validation
            lead_time_days: Supplier lead time
            service_level: Desired service level
            
        Returns:
            Safety stock quantity (units)
        """
        lt = lead_time_days or self.default_lead_time_days
        sl = service_level or self.default_service_level
        z = self.get_z_score(sl)
        
        # Standard deviation of forecast errors
        # Use absolute errors to be conservative
        error_std = np.std(forecast_errors)
        
        # Mean Absolute Deviation (more robust to outliers)
        mad = np.mean(np.abs(forecast_errors))
        
        # Use the larger of std and MAD×1.25 for conservatism
        # MAD × 1.25 ≈ std for normal distribution
        robust_std = max(error_std, mad * 1.25)
        
        safety_stock = z * robust_std * np.sqrt(lt)
        
        return max(0, np.ceil(safety_stock))
    
    def calculate_for_segments(
        self,
        forecast_df: pd.DataFrame,
        segment_col: str = 'store_nbr',
        actual_col: str = 'sales',
        forecast_col: str = 'forecast_ensemble',
        method: str = 'forecast_error',
        service_levels: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Calculate safety stock for each segment (store/category).
        
        This is the main method called by the inventory optimizer.
        It processes all segments and returns a DataFrame with
        safety stock levels.
        
        Args:
            forecast_df: DataFrame with actual and forecasted sales
            segment_col: Column to segment by
            actual_col: Column with actual sales
            forecast_col: Column with forecasted sales
            method: Which calculation method to use
            service_levels: Optional dict of {segment: service_level}
                for differentiated service levels
            
        Returns:
            DataFrame with safety stock per segment
        """
        logger.info(f"Calculating safety stock by {segment_col} "
                    f"using {method} method...")
        
        results = []
        
        for segment, group in forecast_df.groupby(segment_col):
            actuals = group[actual_col].values
            forecasts = group[forecast_col].values
            
            # Get service level for this segment
            sl = (
                service_levels.get(segment, self.default_service_level)
                if service_levels else self.default_service_level
            )
            
            # Calculate demand statistics
            avg_demand = np.mean(actuals)
            demand_std = np.std(actuals)
            
            # Calculate forecast errors
            errors = actuals - forecasts
            error_std = np.std(errors)
            
            # Calculate safety stock using selected method
            if method == 'standard':
                ss = self.method_standard(
                    demand_std=demand_std,
                    service_level=sl
                )
            elif method == 'demand_and_lead_time':
                ss = self.method_demand_and_lead_time(
                    avg_daily_demand=avg_demand,
                    demand_std=demand_std,
                    service_level=sl
                )
            elif method == 'forecast_error':
                ss = self.method_forecast_error(
                    forecast_errors=errors,
                    service_level=sl
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append({
                'segment': segment,
                'segment_col': segment_col,
                'avg_daily_demand': float(avg_demand),
                'demand_std': float(demand_std),
                'forecast_error_std': float(error_std),
                'forecast_mape': float(
                    np.mean(np.abs(errors[actuals > 0] / actuals[actuals > 0]))
                    if (actuals > 0).any() else 0
                ),
                'service_level': sl,
                'z_score': self.get_z_score(sl),
                'lead_time_days': self.default_lead_time_days,
                'safety_stock': float(ss),
                'safety_stock_days': float(
                    ss / avg_demand if avg_demand > 0 else 0
                ),
                'method': method,
                'n_observations': len(group)
            })
        
        results_df = pd.DataFrame(results)
        
        # Summary
        total_ss = results_df['safety_stock'].sum()
        avg_ss_days = results_df['safety_stock_days'].mean()
        
        logger.info(f"  Segments: {len(results_df)}")
        logger.info(f"  Total safety stock: {total_ss:,.0f} units")
        logger.info(f"  Avg safety stock days: {avg_ss_days:.1f}")
        
        return results_df
