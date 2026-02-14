# src/dashboard/components/metrics.py

"""KPI metric card components."""

import streamlit as st
from typing import Optional, List, Dict


def render_kpi_row(metrics: List[Dict]):
    """
    Render a row of KPI cards with enhanced styling.
    
    Args:
        metrics: List of dicts with keys: title, value, delta, icon, help
    """
    if not metrics:
        return
    
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=f"{m.get('icon', '📊')} {m['title']}",
                value=m['value'],
                delta=m.get('delta', None),
                delta_color=m.get('delta_color', 'normal'),
                help=m.get('help', None)
            )


def render_stat_card(
    title: str,
    value: str,
    icon: str = "📊",
    subtitle: Optional[str] = None,
    help_text: Optional[str] = None
):
    """
    Render a single stat card with custom HTML.
    
    Args:
        title: Card title
        value: Main value to display
        icon: Emoji icon
        subtitle: Optional subtitle text
        help_text: Optional help tooltip
    """
    subtitle_html = f'<p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">{subtitle}</p>' if subtitle else ''
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span style="font-size: 2rem;">{icon}</span>
            <span style="color: #9ca3af; font-size: 0.9rem; font-weight: 500;">{title}</span>
        </div>
        <div style="font-size: 2rem; font-weight: 700; 
                    background: linear-gradient(135deg, #00d4ff, #7c3aed); 
                    -webkit-background-clip: text; 
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 4px;">
            {value}
        </div>
        {subtitle_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
