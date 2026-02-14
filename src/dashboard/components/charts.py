# src/dashboard/components/charts.py

"""Reusable Plotly chart components with premium dark theme."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Premium color palette
COLORS = {
    'primary': '#00d4ff',
    'secondary': '#7c3aed',
    'accent': '#a855f7',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'actual': '#10b981',
    'forecast': '#00d4ff',
    'bg_card': '#1a1a2e',
    'bg_dark': '#0d1117',
    'grid': 'rgba(255,255,255,0.06)',
    'text': '#e6edf3',
    'text_muted': '#8b949e',
}

FONT = dict(family='Inter, Segoe UI, sans-serif', color=COLORS['text'])

TEMPLATE = 'plotly_dark'

# Shared layout base
def _base_layout(title: str, height: int) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(size=20, family='Inter, Segoe UI, sans-serif', color='white'),
            x=0.02, xanchor='left'
        ),
        template=TEMPLATE,
        height=height,
        font=FONT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=30, t=70, b=50),
    )


def create_forecast_timeseries(
    df: pd.DataFrame,
    date_col: str = 'date',
    actual_col: str = 'sales',
    forecast_col: str = 'forecast_ensemble',
    lower_col: str = 'forecast_lower',
    upper_col: str = 'forecast_upper',
    title: str = 'Sales Forecast vs Actual',
    height: int = 480
) -> go.Figure:
    """Time series: actual vs forecast with confidence bands."""
    
    daily = df.groupby(date_col).agg({
        actual_col: 'sum',
        forecast_col: 'sum',
    }).reset_index()
    
    has_bounds = lower_col in df.columns and upper_col in df.columns
    if has_bounds:
        daily_lower = df.groupby(date_col)[lower_col].sum().values
        daily_upper = df.groupby(date_col)[upper_col].sum().values
    
    fig = go.Figure()
    
    # Confidence interval
    if has_bounds:
        fig.add_trace(go.Scatter(
            x=pd.concat([daily[date_col], daily[date_col][::-1]]),
            y=np.concatenate([daily_upper, daily_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.08)',
            line=dict(color='rgba(0, 212, 255, 0)'),
            name='95% CI', hoverinfo='skip'
        ))
    
    # Actual — solid vivid green
    fig.add_trace(go.Scatter(
        x=daily[date_col], y=daily[actual_col],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color=COLORS['actual'], width=3, shape='spline'),
        marker=dict(size=7, symbol='circle',
                    line=dict(width=1, color='rgba(16,185,129,0.5)'))
    ))
    
    # Forecast — dashed cyan
    fig.add_trace(go.Scatter(
        x=daily[date_col], y=daily[forecast_col],
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color=COLORS['forecast'], width=3, dash='dot', shape='spline'),
        marker=dict(size=7, symbol='diamond',
                    line=dict(width=1, color='rgba(0,212,255,0.5)'))
    ))
    
    layout = _base_layout(title, height)
    layout.update(
        hovermode='x unified',
        legend=dict(
            orientation='h', y=1.12, x=0.5, xanchor='center',
            bgcolor='rgba(13,17,23,0.85)',
            bordercolor='rgba(0,212,255,0.25)', borderwidth=1,
            font=dict(size=12)
        ),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'],
                     title_text='Date', title_font=dict(size=13, color=COLORS['text_muted']))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'],
                     title_text='Total Sales', title_font=dict(size=13, color=COLORS['text_muted']))
    return fig


def create_model_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str = 'rmse',
    title: str = 'Model Comparison',
    height: int = 420
) -> go.Figure:
    """Horizontal bar chart comparing models."""
    df = comparison_df.sort_values(metric, ascending=True)
    
    colors = [
        COLORS['success'] if 'ensemble' in str(m).lower() or 'ai' in str(m).lower()
        else COLORS['warning'] if 'naive' in str(m).lower()
        else COLORS['primary']
        for m in df['model']
    ]
    
    fig = go.Figure(go.Bar(
        y=df['model'], x=df[metric],
        orientation='h',
        marker=dict(color=colors, cornerradius=6,
                    line=dict(color='rgba(255,255,255,0.08)', width=1)),
        text=df[metric].apply(lambda x: f'{x:,.2f}'),
        textposition='outside',
        textfont=dict(size=13, family='Inter, sans-serif', color='white')
    ))
    
    layout = _base_layout(title, height)
    layout['margin'] = dict(l=140, r=80, t=70, b=40)
    fig.update_layout(**layout, showlegend=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    return fig


def create_risk_gauge(
    risk_score: float,
    title: str = 'Risk Score',
    height: int = 260
) -> go.Figure:
    """Radial gauge chart for risk."""
    color = (COLORS['danger'] if risk_score >= 70
             else COLORS['warning'] if risk_score >= 40
             else COLORS['success'])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number=dict(font=dict(size=36, color=color), suffix=''),
        title=dict(text=title, font=dict(size=14, color=COLORS['text_muted'])),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor='rgba(255,255,255,0.3)',
                      tickfont=dict(size=10, color=COLORS['text_muted'])),
            bar=dict(color=color, thickness=0.7),
            bgcolor='rgba(26,26,46,0.4)',
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color='rgba(16,185,129,0.12)'),
                dict(range=[40, 70], color='rgba(245,158,11,0.12)'),
                dict(range=[70, 100], color='rgba(239,68,68,0.12)'),
            ],
            threshold=dict(
                line=dict(color="white", width=3),
                thickness=0.8, value=risk_score
            )
        )
    ))
    
    fig.update_layout(
        template=TEMPLATE, height=height,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font=FONT
    )
    return fig


def create_savings_waterfall(
    savings: Dict[str, float],
    title: str = 'Financial Impact Breakdown',
    height: int = 450
) -> go.Figure:
    """Waterfall chart of financial savings."""
    categories = ['Holding Cost\nSavings', 'Stockout Cost\nSavings',
                   'Capital Cost\nSaved', 'Total Annual\nSavings']
    values = [
        savings.get('holding_cost_savings', 0),
        savings.get('stockout_cost_savings', 0),
        savings.get('working_capital_cost_saved', 0),
        savings.get('annualized_savings_estimate', 0)
    ]
    measures = ['relative', 'relative', 'relative', 'total']
    
    def fmt(v):
        if abs(v) >= 1e9: return f'${v/1e9:.1f}B'
        if abs(v) >= 1e6: return f'${v/1e6:.1f}M'
        if abs(v) >= 1e3: return f'${v/1e3:.0f}K'
        return f'${v:,.0f}'
    
    fig = go.Figure(go.Waterfall(
        measure=measures, x=categories, y=values,
        text=[fmt(v) for v in values],
        textposition='outside',
        textfont=dict(size=13, family='Inter, sans-serif', color='white'),
        increasing=dict(marker=dict(color=COLORS['success'],
                        line=dict(width=0))),
        totals=dict(marker=dict(color=COLORS['primary'],
                    line=dict(width=0))),
        connector=dict(line=dict(color='rgba(255,255,255,0.15)', width=2))
    ))
    
    layout = _base_layout(title, height)
    layout['showlegend'] = False
    layout['margin'] = dict(l=60, r=30, t=70, b=100)
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    return fig


def create_store_heatmap(
    df: pd.DataFrame,
    store_col: str = 'store_nbr',
    date_col: str = 'date',
    value_col: str = 'sales',
    title: str = 'Sales Heatmap by Store',
    height: int = 520
) -> go.Figure:
    """Heatmap of sales by store and date — fixed for Plotly 6.x."""
    pivot = df.pivot_table(values=value_col, index=store_col,
                           columns=date_col, aggfunc='sum').fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    if len(pivot) > 20:
        pivot = pivot.head(20)
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(d)[:10] for d in pivot.columns],
        y=[f'Store {s}' for s in pivot.index],
        colorscale=[
            [0, '#0d1117'], [0.2, '#1a1a2e'], [0.4, '#0e4d6b'],
            [0.6, '#0891b2'], [0.8, '#06b6d4'], [1, '#22d3ee']
        ],
        colorbar=dict(
            title=dict(text='Sales', side='right'),
            thickness=14,
            outlinewidth=0,
            tickfont=dict(size=10, color=COLORS['text_muted'])
        ),
        hovertemplate='Store %{y}<br>Date: %{x}<br>Sales: %{z:,.0f}<extra></extra>'
    ))
    
    layout = _base_layout(title, height)
    layout['margin'] = dict(l=100, r=30, t=70, b=90)
    fig.update_layout(**layout, xaxis_tickangle=-45)
    return fig


def create_inventory_treemap(
    inventory_df: pd.DataFrame,
    title: str = 'Inventory Cost Distribution',
    height: int = 500
) -> go.Figure:
    """Treemap of inventory costs."""
    df = inventory_df.copy()
    df['segment'] = df['segment'].astype(str)
    
    value_col = 'holding_cost' if 'holding_cost' in df.columns else 'safety_stock'
    df = df[df[value_col] > 0]
    
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No positive inventory costs to display",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['text_muted'])
        )
    else:
        fig = px.treemap(
            df, path=['segment'], values=value_col,
            color='inventory_turnover' if 'inventory_turnover' in df.columns else value_col,
            color_continuous_scale='tealgrn',
            title=title
        )
    
    fig.update_layout(
        template=TEMPLATE, height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)', font=FONT
    )
    return fig


def create_donut_risk(risk_df: pd.DataFrame, height: int = 350) -> go.Figure:
    """Donut chart of risk distribution."""
    if risk_df.empty or 'risk_category' not in risk_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No risk data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=COLORS['text_muted']))
        fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)')
        return fig
    
    counts = risk_df['risk_category'].value_counts()
    labels = ['HIGH', 'MEDIUM', 'LOW']
    values = [counts.get(l, 0) for l in labels]
    colors_map = [COLORS['danger'], COLORS['warning'], COLORS['success']]
    
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors_map,
                    line=dict(color=COLORS['bg_dark'], width=3)),
        textinfo='label+value',
        textfont=dict(size=13, color='white'),
        hovertemplate='%{label}: %{value} stores<extra></extra>'
    ))
    fig.update_layout(
        template=TEMPLATE, height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False, font=FONT,
        annotations=[dict(
            text='Risk', x=0.5, y=0.5, font=dict(size=18, color='white'),
            showarrow=False
        )]
    )
    return fig
