# src/optimization/inventory_optimizer.py

"""
Master Inventory Optimizer

Orchestrates the complete inventory optimization pipeline:
1. Calculate safety stock per segment
2. Calculate reorder points and EOQ
3. Simulate inventory scenarios
4. Calculate financial impact
5. Generate risk assessments
6. Produce executive summary

This is the main entry point for inventory optimization.
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, Optional, Tuple

from src.optimization.safety_stock import SafetyStockCalculator
from src.optimization.reorder_point import ReorderPointCalculator
from src.optimization.financial_impact import FinancialImpactCalculator

logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """
    Complete inventory optimization system.
    
    Takes ML forecast outputs and produces actionable 
    inventory recommendations with financial quantification.
    """
    
    def __init__(
        self,
        lead_time_days: int = 7,
        service_level: float = 0.95,
        holding_cost_pct: float = 0.25,
        ordering_cost: float = 50.0,
        avg_unit_cost: float = 10.0,
        avg_unit_price: float = 15.0,
        stockout_cost_multiplier: float = 1.5
    ):
        """
        Initialize the complete optimization system.
        """
        self.ss_calculator = SafetyStockCalculator(
            default_service_level=service_level,
            default_lead_time_days=lead_time_days
        )
        
        self.rop_calculator = ReorderPointCalculator(
            lead_time_days=lead_time_days,
            ordering_cost=ordering_cost,
            holding_cost_pct=holding_cost_pct,
            avg_unit_cost=avg_unit_cost,
            stockout_cost_multiplier=stockout_cost_multiplier
        )
        
        self.financial_calculator = FinancialImpactCalculator(
            avg_unit_price=avg_unit_price,
            avg_unit_cost=avg_unit_cost,
            holding_cost_pct=holding_cost_pct,
            stockout_cost_multiplier=stockout_cost_multiplier
        )
    
    def optimize(
        self,
        forecast_df: pd.DataFrame,
        segment_col: str = 'store_nbr',
        actual_col: str = 'sales',
        forecast_col: str = 'forecast_ensemble',
        service_levels: Optional[Dict] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run complete inventory optimization pipeline.
        
        Args:
            forecast_df: DataFrame with actual sales and forecasts
            segment_col: Column to segment by
            actual_col: Actual sales column
            forecast_col: Forecast column
            service_levels: Optional per-segment service levels
            
        Returns:
            Dictionary with all optimization outputs
        """
        logger.info("=" * 60)
        logger.info("INVENTORY OPTIMIZATION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Safety Stock
        logger.info("\n📦 Step 1: Calculating Safety Stock...")
        safety_stock_df = self.ss_calculator.calculate_for_segments(
            forecast_df=forecast_df,
            segment_col=segment_col,
            actual_col=actual_col,
            forecast_col=forecast_col,
            method='forecast_error',
            service_levels=service_levels
        )
        
        # Step 2: Reorder Points & EOQ
        logger.info("\n📋 Step 2: Calculating Reorder Points & EOQ...")
        inventory_df = self.rop_calculator.calculate_for_segments(
            safety_stock_df=safety_stock_df
        )
        
        # Step 3: Financial Impact
        logger.info("\n💰 Step 3: Calculating Financial Impact...")
        
        # Create naive forecast (use mean of actuals as baseline)
        naive_forecast = np.full(
            len(forecast_df), 
            forecast_df[actual_col].mean()
        )
        
        # Create seasonal naive (same day last week)
        seasonal_naive = forecast_df.groupby(
            segment_col
        )[actual_col].transform(
            lambda x: x.shift(7).fillna(x.mean())
        ).values
        
        comparison = self.financial_calculator.compare_forecast_methods(
            actual_sales=forecast_df[actual_col].values,
            ai_forecast=forecast_df[forecast_col].values,
            naive_forecast=naive_forecast,
            seasonal_naive_forecast=seasonal_naive
        )
        
        savings = self.financial_calculator.calculate_savings(comparison)
        
        # Step 4: Risk Assessment
        logger.info("\n⚠️ Step 4: Assessing Stockout Risk...")
        risk_df = self.financial_calculator.calculate_stockout_risk(
            forecast_df=forecast_df,
            safety_stock_df=safety_stock_df,
            forecast_col=forecast_col,
            segment_col=segment_col
        )
        
        # Step 5: Executive Summary
        logger.info("\n📊 Step 5: Generating Executive Summary...")
        executive_summary = self.financial_calculator.generate_executive_summary(
            savings=savings,
            risk_df=risk_df,
            inventory_df=inventory_df
        )
        
        # Print key results
        self._print_optimization_results(
            savings, risk_df, inventory_df, executive_summary
        )
        
        return {
            'safety_stock': safety_stock_df,
            'inventory': inventory_df,
            'comparison': comparison,
            'savings': savings,
            'risk': risk_df,
            'executive_summary': executive_summary
        }
    
    def _print_optimization_results(
        self,
        savings: Dict,
        risk_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        executive_summary: Dict
    ):
        """Print formatted optimization results."""
        
        logger.info("\n" + "╔" + "═" * 58 + "╗")
        logger.info("║     INVENTORY OPTIMIZATION RESULTS                       ║")
        logger.info("╠" + "═" * 58 + "╣")
        
        logger.info(f"║  💰 FINANCIAL IMPACT                                     ║")
        logger.info(f"║  Annual Savings Estimate:   "
                    f"${savings['annualized_savings_estimate']:>12,.0f}          ║")
        logger.info(f"║  Working Capital Freed:     "
                    f"${savings['annualized_capital_freed']:>12,.0f}          ║")
        logger.info(f"║  Accuracy Improvement:      "
                    f"{savings['accuracy_improvement_pct']:>11.1f}%          ║")
        logger.info(f"║  Cost Reduction:            "
                    f"{savings['savings_pct']:>11.1f}%          ║")
        
        logger.info(f"║                                                          ║")
        logger.info(f"║  📦 INVENTORY HEALTH                                     ║")
        logger.info(f"║  Avg Turnover:              "
                    f"{inventory_df['inventory_turnover'].mean():>11.1f}x          ║")
        logger.info(f"║  Avg Days of Supply:        "
                    f"{inventory_df['days_of_supply'].mean():>11.1f}           ║")
        
        logger.info(f"║                                                          ║")
        logger.info(f"║  ⚠️  RISK ASSESSMENT                                     ║")
        high = (risk_df['risk_category'] == 'HIGH').sum()
        med = (risk_df['risk_category'] == 'MEDIUM').sum()
        low = (risk_df['risk_category'] == 'LOW').sum()
        logger.info(f"║  High Risk: {high}  |  Medium: {med}  |  Low: {low}"
                    + " " * (42 - len(f"  High Risk: {high}  |  Medium: {med}  |  Low: {low}")) + "║")
        
        logger.info("╚" + "═" * 58 + "╝")
    
    def save_results(
        self,
        results: Dict,
        output_dir: str
    ) -> None:
        """Save all optimization results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DataFrames
        results['safety_stock'].to_parquet(
            os.path.join(output_dir, 'safety_stock.parquet'), index=False
        )
        results['inventory'].to_parquet(
            os.path.join(output_dir, 'inventory_recommendations.parquet'), 
            index=False
        )
        results['risk'].to_parquet(
            os.path.join(output_dir, 'stockout_risk.parquet'), index=False
        )
        
        # Save financial results as JSON
        financial_output = {
            'comparison': results['comparison'],
            'savings': results['savings'],
            'executive_summary': results['executive_summary']
        }
        
        with open(os.path.join(output_dir, 'financial_impact.json'), 'w') as f:
            json.dump(financial_output, f, indent=2, default=str)
        
        logger.info(f"✅ All optimization results saved to {output_dir}")
