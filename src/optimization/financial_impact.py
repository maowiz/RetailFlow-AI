# src/optimization/financial_impact.py

"""
Financial Impact Calculator

Quantifies the monetary value of using AI-driven forecasting
vs traditional methods.

THIS IS THE MOST IMPORTANT MODULE FOR THE SAPPHIRE INTERVIEW.

It answers: "What is the financial impact of your AI/ML solution?"

Key metrics calculated:
1. Inventory holding cost savings
2. Stockout prevention value
3. Working capital freed up
4. Forecast accuracy improvement value
5. ROI of the AI system
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FinancialImpactCalculator:
    """
    Calculates the financial impact of AI-driven inventory optimization
    vs traditional (naive/seasonal) methods.
    
    Demonstrates clear business value:
    - Cost reduction
    - Revenue protection
    - Capital efficiency
    """
    
    def __init__(
        self,
        avg_unit_price: float = 15.0,
        avg_unit_cost: float = 10.0,
        holding_cost_pct: float = 0.25,
        stockout_cost_multiplier: float = 1.5,
        working_capital_cost: float = 0.08
    ):
        """
        Initialize FinancialImpactCalculator.
        
        Args:
            avg_unit_price: Average selling price per unit ($)
            avg_unit_cost: Average cost per unit ($)
            holding_cost_pct: Annual holding cost as % of cost
            stockout_cost_multiplier: Multiplier for lost sale cost
            working_capital_cost: Annual cost of capital (interest rate)
        """
        self.avg_unit_price = avg_unit_price
        self.avg_unit_cost = avg_unit_cost
        self.avg_margin = avg_unit_price - avg_unit_cost
        self.holding_cost_pct = holding_cost_pct
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.working_capital_cost = working_capital_cost
    
    def compare_forecast_methods(
        self,
        actual_sales: np.ndarray,
        ai_forecast: np.ndarray,
        naive_forecast: np.ndarray,
        seasonal_naive_forecast: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """
        Compare AI forecast vs baseline methods.
        
        Creates a clear before/after comparison that demonstrates
        the value of the AI system.
        
        Args:
            actual_sales: True sales values
            ai_forecast: Our ensemble model's predictions
            naive_forecast: Simple baseline (e.g., last week's sales)
            seasonal_naive_forecast: Optional seasonal baseline
            
        Returns:
            Dictionary with comparison metrics
        """
        methods = {
            'AI Ensemble': ai_forecast,
            'Naive (Last Period)': naive_forecast
        }
        if seasonal_naive_forecast is not None:
            methods['Seasonal Naive'] = seasonal_naive_forecast
        
        comparison = {}
        
        for method_name, forecast in methods.items():
            errors = actual_sales - forecast
            abs_errors = np.abs(errors)
            
            # Forecast accuracy
            non_zero = actual_sales > 0
            mape = (
                np.mean(abs_errors[non_zero] / actual_sales[non_zero])
                if non_zero.any() else 1.0
            )
            
            # Over-forecasting (leads to excess inventory)
            over_forecast = np.where(forecast > actual_sales, 
                                      forecast - actual_sales, 0)
            total_over = np.sum(over_forecast)
            
            # Under-forecasting (leads to stockouts)
            under_forecast = np.where(forecast < actual_sales,
                                       actual_sales - forecast, 0)
            total_under = np.sum(under_forecast)
            
            # Financial impact
            holding_cost = (
                total_over * self.avg_unit_cost * 
                self.holding_cost_pct / 365 * len(actual_sales)
            )
            
            stockout_cost = (
                total_under * self.avg_margin * 
                self.stockout_cost_multiplier
            )
            
            comparison[method_name] = {
                'mape': float(mape),
                'forecast_accuracy': float(1 - mape),
                'total_over_forecast_units': float(total_over),
                'total_under_forecast_units': float(total_under),
                'excess_inventory_cost': float(holding_cost),
                'stockout_cost': float(stockout_cost),
                'total_cost_of_errors': float(holding_cost + stockout_cost),
                'rmse': float(np.sqrt(np.mean(errors**2)))
            }
        
        return comparison
    
    def calculate_savings(
        self,
        comparison: Dict[str, Dict],
        ai_method_name: str = 'AI Ensemble',
        baseline_method_name: str = 'Naive (Last Period)'
    ) -> Dict[str, float]:
        """
        Calculate savings from AI vs baseline.
        
        This is the headline number for stakeholders:
        "Our AI system saves $X per year in inventory costs."
        
        Args:
            comparison: Output from compare_forecast_methods
            ai_method_name: Name of AI method in comparison
            baseline_method_name: Name of baseline method
            
        Returns:
            Dictionary of savings metrics
        """
        ai = comparison[ai_method_name]
        baseline = comparison[baseline_method_name]
        
        # Direct cost savings
        holding_savings = (
            baseline['excess_inventory_cost'] - ai['excess_inventory_cost']
        )
        stockout_savings = (
            baseline['stockout_cost'] - ai['stockout_cost']
        )
        total_savings = (
            baseline['total_cost_of_errors'] - ai['total_cost_of_errors']
        )
        
        # Accuracy improvement
        accuracy_improvement = (
            ai['forecast_accuracy'] - baseline['forecast_accuracy']
        )
        
        # Working capital freed
        excess_inventory_reduction = (
            baseline['total_over_forecast_units'] - 
            ai['total_over_forecast_units']
        )
        capital_freed = excess_inventory_reduction * self.avg_unit_cost
        capital_cost_saved = capital_freed * self.working_capital_cost
        
        # Annualized savings estimate
        # If test period is N days, annualize
        annualization_factor = 365 / 16  # Our test period is 16 days
        
        savings = {
            'holding_cost_savings': float(holding_savings),
            'stockout_cost_savings': float(stockout_savings),
            'total_direct_savings': float(total_savings),
            'accuracy_improvement_pct': float(accuracy_improvement * 100),
            'excess_inventory_reduction_units': float(excess_inventory_reduction),
            'working_capital_freed': float(capital_freed),
            'working_capital_cost_saved': float(capital_cost_saved),
            'annualized_savings_estimate': float(
                total_savings * annualization_factor
            ),
            'annualized_capital_freed': float(
                capital_freed * annualization_factor
            ),
            'savings_pct': float(
                total_savings / baseline['total_cost_of_errors'] * 100
                if baseline['total_cost_of_errors'] > 0 else 0
            )
        }
        
        return savings
    
    def calculate_stockout_risk(
        self,
        forecast_df: pd.DataFrame,
        safety_stock_df: pd.DataFrame,
        forecast_col: str = 'forecast_ensemble',
        segment_col: str = 'store_nbr'
    ) -> pd.DataFrame:
        """
        Calculate stockout risk for each segment.
        
        Risk factors:
        - High forecast uncertainty (large prediction intervals)
        - Low safety stock relative to demand variability
        - Historical stockout frequency
        - Upcoming demand spikes (promotions, holidays)
        
        Args:
            forecast_df: DataFrame with forecasts
            safety_stock_df: Safety stock calculations
            forecast_col: Forecast column name
            segment_col: Segment column name
            
        Returns:
            DataFrame with risk scores and recommendations
        """
        logger.info("Calculating stockout risk...")
        
        risk_results = []
        
        for _, ss_row in safety_stock_df.iterrows():
            segment = ss_row['segment']
            
            # Get forecasts for this segment
            seg_mask = forecast_df[segment_col] == segment
            if not seg_mask.any():
                continue
            
            seg_data = forecast_df[seg_mask]
            
            forecasted_demand = seg_data[forecast_col].values
            safety_stock = ss_row['safety_stock']
            forecast_error_std = ss_row['forecast_error_std']
            avg_demand = ss_row['avg_daily_demand']
            
            # Risk Score Components (0-100 scale)
            
            # 1. Demand variability risk (higher CV = higher risk)
            cv = forecast_error_std / avg_demand if avg_demand > 0 else 1
            variability_risk = min(100, cv * 100)
            
            # 2. Safety stock adequacy risk
            # Compare safety stock to demand variability
            ss_ratio = safety_stock / (forecast_error_std + 1e-6)
            adequacy_risk = max(0, 100 - ss_ratio * 30)
            
            # 3. Demand trend risk (is demand increasing?)
            if len(forecasted_demand) >= 7:
                # Simple trend: compare last half to first half
                midpoint = len(forecasted_demand) // 2
                first_half = np.mean(forecasted_demand[:midpoint])
                second_half = np.mean(forecasted_demand[midpoint:])
                trend_pct = (
                    (second_half - first_half) / (first_half + 1e-6) * 100
                )
                trend_risk = max(0, min(100, trend_pct * 2))
            else:
                trend_risk = 50  # Unknown
            
            # 4. Volume risk (high-volume items are more critical)
            volume_risk = min(100, (avg_demand / 500) * 100)
            
            # Composite risk score (weighted average)
            risk_score = (
                variability_risk * 0.35 +
                adequacy_risk * 0.30 +
                trend_risk * 0.20 +
                volume_risk * 0.15
            )
            
            # Risk category
            if risk_score >= 70:
                risk_category = 'HIGH'
                recommendation = 'Increase safety stock by 20-30%'
            elif risk_score >= 40:
                risk_category = 'MEDIUM'
                recommendation = 'Monitor closely, consider 10% buffer'
            else:
                risk_category = 'LOW'
                recommendation = 'Current levels adequate'
            
            # Financial exposure
            daily_exposure = avg_demand * self.avg_margin
            
            risk_results.append({
                'segment': segment,
                'avg_daily_demand': float(avg_demand),
                'safety_stock': float(safety_stock),
                'variability_risk': float(variability_risk),
                'adequacy_risk': float(adequacy_risk),
                'trend_risk': float(trend_risk),
                'volume_risk': float(volume_risk),
                'composite_risk_score': float(risk_score),
                'risk_category': risk_category,
                'recommendation': recommendation,
                'daily_revenue_at_risk': float(daily_exposure),
                'weekly_revenue_at_risk': float(daily_exposure * 7),
                'demand_trend_pct': float(
                    trend_pct if len(forecasted_demand) >= 7 else 0
                )
            })
        
        risk_df = pd.DataFrame(risk_results).sort_values(
            'composite_risk_score', ascending=False
        )
        
        # Summary
        high_risk = (risk_df['risk_category'] == 'HIGH').sum()
        medium_risk = (risk_df['risk_category'] == 'MEDIUM').sum()
        low_risk = (risk_df['risk_category'] == 'LOW').sum()
        total_risk_exposure = risk_df['weekly_revenue_at_risk'].sum()
        
        logger.info(f"  Risk Distribution: "
                    f"HIGH={high_risk}, MEDIUM={medium_risk}, LOW={low_risk}")
        logger.info(f"  Total weekly revenue at risk: "
                    f"${total_risk_exposure:,.0f}")
        
        return risk_df
    
    def generate_executive_summary(
        self,
        savings: Dict[str, float],
        risk_df: pd.DataFrame,
        inventory_df: pd.DataFrame
    ) -> Dict:
        """
        Generate executive-level financial summary.
        
        This is what goes on the dashboard's main page.
        Designed for C-suite and finance leadership.
        
        Returns:
            Dictionary with executive summary data
        """
        summary = {
            'headline_metrics': {
                'annual_savings_estimate': savings['annualized_savings_estimate'],
                'forecast_accuracy_improvement': savings['accuracy_improvement_pct'],
                'working_capital_freed': savings['annualized_capital_freed'],
                'cost_reduction_pct': savings['savings_pct']
            },
            'inventory_health': {
                'avg_inventory_turnover': float(
                    inventory_df['inventory_turnover'].mean()
                ),
                'avg_days_of_supply': float(
                    inventory_df['days_of_supply'].mean()
                ),
                'total_safety_stock_cost': float(
                    inventory_df['safety_stock_cost'].sum()
                )
            },
            'risk_summary': {
                'high_risk_segments': int(
                    (risk_df['risk_category'] == 'HIGH').sum()
                ),
                'total_segments': len(risk_df),
                'total_weekly_exposure': float(
                    risk_df['weekly_revenue_at_risk'].sum()
                ),
                'top_risk_segments': risk_df.nlargest(
                    5, 'composite_risk_score'
                )[['segment', 'composite_risk_score', 'risk_category',
                   'weekly_revenue_at_risk']].to_dict('records')
            },
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        if savings['annualized_savings_estimate'] > 10000:
            summary['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Implement AI-driven reorder system across all stores',
                'impact': f"${savings['annualized_savings_estimate']:,.0f} annual savings"
            })
        
        high_risk_count = (risk_df['risk_category'] == 'HIGH').sum()
        if high_risk_count > 0:
            summary['recommendations'].append({
                'priority': 'HIGH',
                'action': f'Review safety stock levels for {high_risk_count} high-risk segments',
                'impact': 'Prevent potential stockouts and revenue loss'
            })
        
        if savings['accuracy_improvement_pct'] > 5:
            summary['recommendations'].append({
                'priority': 'MEDIUM',
                'action': 'Transition from manual to AI-based demand planning',
                'impact': f"{savings['accuracy_improvement_pct']:.1f}% accuracy improvement"
            })
        
        summary['recommendations'].append({
            'priority': 'MEDIUM',
            'action': 'Set up automated weekly model retraining pipeline',
            'impact': 'Maintain forecast accuracy as patterns evolve'
        })
        
        return summary
