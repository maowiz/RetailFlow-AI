# src/optimization/reorder_point.py

"""
Reorder Point Calculator

Determines WHEN to place replenishment orders.

When inventory falls to the reorder point, a new order 
should be triggered to arrive before stockout occurs.

This integrates directly with the ML forecast because:
- Instead of using historical average demand for the ROP calculation,
  we use the FORECASTED demand for the upcoming lead time period.
- This makes reorder decisions forward-looking, not backward-looking.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class ReorderPointCalculator:
    """
    Calculates reorder points and order quantities.
    
    Integrates:
    - ML forecasted demand (forward-looking)
    - Safety stock (from SafetyStockCalculator)
    - Economic Order Quantity
    - Current inventory simulation
    
    Output: Actionable recommendations for procurement team
    """
    
    def __init__(
        self,
        lead_time_days: int = 7,
        review_period_days: int = 7,
        ordering_cost: float = 50.0,
        holding_cost_pct: float = 0.25,
        avg_unit_cost: float = 10.0,
        stockout_cost_multiplier: float = 1.5
    ):
        """
        Initialize ReorderPointCalculator.
        
        Args:
            lead_time_days: Days from order placement to delivery
            review_period_days: How often inventory is reviewed
            ordering_cost: Fixed cost per purchase order ($)
            holding_cost_pct: Annual holding cost as % of product value
            avg_unit_cost: Average cost per unit ($)
            stockout_cost_multiplier: Multiplier for lost sale cost
        """
        self.lead_time_days = lead_time_days
        self.review_period_days = review_period_days
        self.ordering_cost = ordering_cost
        self.holding_cost_pct = holding_cost_pct
        self.avg_unit_cost = avg_unit_cost
        self.stockout_cost_multiplier = stockout_cost_multiplier
        
        # Derived values
        self.annual_holding_cost = avg_unit_cost * holding_cost_pct
        self.daily_holding_cost = self.annual_holding_cost / 365
    
    def calculate_reorder_point(
        self,
        forecasted_daily_demand: float,
        safety_stock: float,
        lead_time_days: Optional[int] = None
    ) -> float:
        """
        Calculate the Reorder Point.
        
        Formula: ROP = (Forecasted_Daily_Demand × Lead_Time) + Safety_Stock
        
        KEY INSIGHT: We use FORECASTED demand, not historical average.
        This means the ROP adapts to upcoming trends:
        - Before a holiday season → ROP increases (higher expected demand)
        - During slow season → ROP decreases (lower expected demand)
        - When promotions are planned → ROP increases
        
        This is the "AI-driven" part of inventory optimization.
        
        Args:
            forecasted_daily_demand: ML model's forecast for daily demand
            safety_stock: Calculated safety stock
            lead_time_days: Supplier lead time
            
        Returns:
            Reorder point (units)
        """
        lt = lead_time_days or self.lead_time_days
        
        # Demand during lead time
        demand_during_lt = forecasted_daily_demand * lt
        
        # Reorder point
        rop = demand_during_lt + safety_stock
        
        return max(0, np.ceil(rop))
    
    def calculate_eoq(
        self,
        annual_demand: float,
        ordering_cost: Optional[float] = None,
        holding_cost: Optional[float] = None
    ) -> float:
        """
        Calculate Economic Order Quantity (EOQ).
        
        Formula: EOQ = √(2 × D × S / H)
        
        The EOQ balances two competing costs:
        - Ordering cost (fixed per order): Favors FEWER, LARGER orders
        - Holding cost (per unit per year): Favors MORE, SMALLER orders
        
        EOQ finds the sweet spot that minimizes total cost.
        
        Financial Impact Example:
        - Annual demand: 73,000 units
        - Ordering cost: $50/order
        - Holding cost: $2.50/unit/year
        - EOQ: 1,709 units
        - Orders per year: 73,000/1,709 = 43 orders
        - Total cost: 43 × $50 + (1,709/2) × $2.50 = $2,150 + $2,136 = $4,286
        
        Without EOQ (ordering monthly = 12 orders):
        - Order size: 6,083 units
        - Total cost: 12 × $50 + (6,083/2) × $2.50 = $600 + $7,604 = $8,204
        
        Savings: $8,204 - $4,286 = $3,918 per product per year!
        
        Args:
            annual_demand: Total expected annual demand
            ordering_cost: Fixed cost per order
            holding_cost: Annual holding cost per unit
            
        Returns:
            Economic Order Quantity (units)
        """
        s = ordering_cost or self.ordering_cost
        h = holding_cost or self.annual_holding_cost
        
        if annual_demand <= 0 or h <= 0:
            return 0
        
        eoq = np.sqrt((2 * annual_demand * s) / h)
        
        return max(1, np.ceil(eoq))
    
    def calculate_order_up_to_level(
        self,
        forecasted_daily_demand: float,
        safety_stock: float,
        lead_time_days: Optional[int] = None,
        review_period_days: Optional[int] = None
    ) -> float:
        """
        Calculate Order-Up-To Level (for periodic review systems).
        
        In practice, many retailers don't continuously monitor inventory.
        Instead, they review every R days and order enough to last 
        until the next review + lead time.
        
        Formula: S = d × (R + LT) + SS
        
        Where:
        - S = Order-up-to level
        - d = Forecasted daily demand
        - R = Review period (days)
        - LT = Lead time (days)
        - SS = Safety stock
        
        Order quantity = S - Current_Inventory
        
        Args:
            forecasted_daily_demand: ML forecast for daily demand
            safety_stock: Safety stock
            lead_time_days: Lead time
            review_period_days: Review period
            
        Returns:
            Order-up-to level (units)
        """
        lt = lead_time_days or self.lead_time_days
        rp = review_period_days or self.review_period_days
        
        # Demand during review period + lead time
        demand_during_rp_lt = forecasted_daily_demand * (rp + lt)
        
        # Order-up-to level
        s = demand_during_rp_lt + safety_stock
        
        return max(0, np.ceil(s))
    
    def calculate_total_costs(
        self,
        annual_demand: float,
        order_quantity: float,
        safety_stock: float,
        avg_unit_cost: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate total inventory costs.
        
        Components:
        1. Ordering cost: (Annual_Demand / Order_Quantity) × Cost_per_Order
        2. Holding cost: (Order_Quantity/2 + Safety_Stock) × Annual_Holding_Cost
        3. Product cost: Annual_Demand × Unit_Cost
        
        Returns:
            Dictionary of cost components
        """
        unit_cost = avg_unit_cost or self.avg_unit_cost
        
        if order_quantity <= 0:
            return {
                'ordering_cost': 0,
                'holding_cost': 0,
                'product_cost': 0,
                'total_cost': 0,
                'safety_stock_cost': 0
            }
        
        # Number of orders per year
        n_orders = annual_demand / order_quantity
        
        # Ordering cost
        ordering_cost = n_orders * self.ordering_cost
        
        # Holding cost
        # Average inventory = Order_Quantity/2 (cycle stock) + Safety_Stock
        avg_inventory = (order_quantity / 2) + safety_stock
        holding_cost = avg_inventory * self.annual_holding_cost
        
        # Safety stock holding cost specifically
        ss_holding_cost = safety_stock * self.annual_holding_cost
        
        # Product cost
        product_cost = annual_demand * unit_cost
        
        return {
            'n_orders_per_year': float(n_orders),
            'ordering_cost': float(ordering_cost),
            'holding_cost': float(holding_cost),
            'safety_stock_cost': float(ss_holding_cost),
            'product_cost': float(product_cost),
            'total_cost': float(ordering_cost + holding_cost + product_cost),
            'avg_inventory_units': float(avg_inventory),
            'inventory_turnover': float(
                annual_demand / avg_inventory if avg_inventory > 0 else 0
            ),
            'days_of_supply': float(
                avg_inventory / (annual_demand / 365) 
                if annual_demand > 0 else 0
            )
        }
    
    def calculate_for_segments(
        self,
        safety_stock_df: pd.DataFrame,
        forecast_horizon_days: int = 30
    ) -> pd.DataFrame:
        """
        Calculate reorder points and order quantities for all segments.
        
        Args:
            safety_stock_df: Output from SafetyStockCalculator
            forecast_horizon_days: Days to project demand forward
            
        Returns:
            DataFrame with complete inventory optimization results
        """
        logger.info("Calculating reorder points and order quantities...")
        
        results = []
        
        for _, row in safety_stock_df.iterrows():
            avg_demand = row['avg_daily_demand']
            ss = row['safety_stock']
            
            # Reorder point
            rop = self.calculate_reorder_point(
                forecasted_daily_demand=avg_demand,
                safety_stock=ss
            )
            
            # Annual demand projection
            annual_demand = avg_demand * 365
            
            # EOQ
            eoq = self.calculate_eoq(annual_demand=annual_demand)
            
            # Order-up-to level
            out_level = self.calculate_order_up_to_level(
                forecasted_daily_demand=avg_demand,
                safety_stock=ss
            )
            
            # Cost analysis
            costs = self.calculate_total_costs(
                annual_demand=annual_demand,
                order_quantity=eoq,
                safety_stock=ss
            )
            
            # Demand for forecast horizon
            demand_next_period = avg_demand * forecast_horizon_days
            
            result = {
                'segment': row['segment'],
                'avg_daily_demand': avg_demand,
                'safety_stock': ss,
                'reorder_point': float(rop),
                'eoq': float(eoq),
                'order_up_to_level': float(out_level),
                'demand_next_30d': float(demand_next_period),
                'annual_demand': float(annual_demand),
                'service_level': row['service_level'],
                **costs
            }
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Summary
        logger.info(f"  Total annual ordering cost: "
                    f"${results_df['ordering_cost'].sum():,.0f}")
        logger.info(f"  Total annual holding cost: "
                    f"${results_df['holding_cost'].sum():,.0f}")
        logger.info(f"  Avg inventory turnover: "
                    f"{results_df['inventory_turnover'].mean():.1f}")
        
        return results_df
