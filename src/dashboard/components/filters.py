# src/dashboard/components/filters.py

"""Sidebar filter components."""

import streamlit as st
import pandas as pd
from typing import Dict


def render_sidebar_filters(df: pd.DataFrame) -> Dict:
    """
    Render filters in sidebar and return selections.
    
    Args:
        df: DataFrame to extract filter options from
    
    Returns:
        Dictionary of filter selections
    """
    st.sidebar.title("🔍 Filters")
    st.sidebar.markdown("---")
    
    filters = {}
    
    # Date range
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        min_d = df['date'].min().date()
        max_d = df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "📅 Date Range", 
            value=(min_d, max_d),
            min_value=min_d, 
            max_value=max_d,
            help="Select date range for analysis"
        )
        if len(date_range) == 2:
            filters['date_start'], filters['date_end'] = date_range
        else:
            filters['date_start'], filters['date_end'] = min_d, max_d
    
    st.sidebar.markdown("---")
    
    # Stores
    if 'store_nbr' in df.columns:
        all_stores = sorted(df['store_nbr'].unique().tolist())
        
        select_all_stores = st.sidebar.checkbox(
            "Select All Stores", 
            value=True,
            help="Check to include all stores"
        )
        
        if select_all_stores:
            filters['stores'] = all_stores
        else:
            filters['stores'] = st.sidebar.multiselect(
                "🏪 Stores", 
                all_stores, 
                default=all_stores[:5] if len(all_stores) > 5 else all_stores,
                help="Select specific stores to analyze"
            )
    
    st.sidebar.markdown("---")
    
    # Categories
    if 'family' in df.columns:
        all_cats = sorted(df['family'].unique().tolist())
        
        select_all_cats = st.sidebar.checkbox(
            "Select All Categories", 
            value=True,
            help="Check to include all product categories"
        )
        
        if select_all_cats:
            filters['categories'] = all_cats
        else:
            filters['categories'] = st.sidebar.multiselect(
                "📦 Categories", 
                all_cats, 
                default=all_cats[:5] if len(all_cats) > 5 else all_cats,
                help="Select specific product categories"
            )
    
    st.sidebar.markdown("---")
    
    # Model selection
    available_models = ['Ensemble', 'XGBoost', 'Random Forest', 'Prophet']
    filters['model'] = st.sidebar.selectbox(
        "🤖 Forecast Model", 
        available_models,
        index=0,
        help="Select which model's forecasts to display"
    )
    
    # Map model name to column
    col_map = {
        'Ensemble': 'forecast_ensemble',
        'XGBoost': 'forecast_xgboost',
        'Random Forest': 'forecast_rf',
        'Prophet': 'forecast_prophet'
    }
    filters['forecast_col'] = col_map.get(filters['model'], 'forecast_ensemble')
    
    # Add info box
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip**: Use filters to drill down into specific stores, "
        "categories, or time periods for detailed analysis."
    )
    
    return filters


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of filter selections
    
    Returns:
        Filtered DataFrame
    """
    out = df.copy()
    
    # Date filter
    if 'date_start' in filters and 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'])
        out = out[
            (out['date'].dt.date >= filters['date_start']) &
            (out['date'].dt.date <= filters['date_end'])
        ]
    
    # Store filter
    if 'stores' in filters and 'store_nbr' in out.columns and filters['stores']:
        out = out[out['store_nbr'].isin(filters['stores'])]
    
    # Category filter
    if 'categories' in filters and 'family' in out.columns and filters['categories']:
        out = out[out['family'].isin(filters['categories'])]
    
    return out
