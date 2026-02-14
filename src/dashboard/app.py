# src/dashboard/app.py

"""
AI Sales Forecasting & Inventory Optimization Dashboard
Run:  streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, sys

# ── Page config (MUST be first st call) ───────────────────────
st.set_page_config(
    page_title="AI Sales Forecast & Inventory Optimizer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths / env ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except Exception:
    pass

css_path = os.path.join(os.path.dirname(__file__), 'styles', 'custom.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
def _fmt(v, prefix='$', suffix=''):
    """Format large numbers: $35.9B, $1.88B, $12.5M, $350K ..."""
    if v is None or v == 0:
        return f'{prefix}0{suffix}'
    av = abs(v)
    if av >= 1e9:
        return f'{prefix}{v/1e9:.2f}B{suffix}'
    if av >= 1e6:
        return f'{prefix}{v/1e6:.1f}M{suffix}'
    if av >= 1e3:
        return f'{prefix}{v/1e3:.0f}K{suffix}'
    return f'{prefix}{v:,.0f}{suffix}'


def _pct(v, decimals=1):
    """Format percentage."""
    if v is None:
        return '0%'
    return f'{v:.{decimals}f}%'


# ── Data loading ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    data_dir = os.environ.get(
        'DATA_DIR', os.path.join(PROJECT_ROOT, 'data', 'output'))
    data = {}

    fp = os.path.join(data_dir, 'forecasts.parquet')
    if os.path.exists(fp):
        data['forecasts'] = pd.read_parquet(fp)
        data['forecasts']['date'] = pd.to_datetime(data['forecasts']['date'])
    else:
        data['forecasts'] = pd.DataFrame()

    fp = os.path.join(data_dir, 'financial_impact.json')
    if os.path.exists(fp):
        with open(fp) as f:
            data['financial'] = json.load(f)
    else:
        data['financial'] = {}

    fp = os.path.join(data_dir, 'inventory_recommendations.parquet')
    data['inventory'] = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()

    fp = os.path.join(data_dir, 'stockout_risk.parquet')
    data['risk'] = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()

    return data


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — EXECUTIVE OVERVIEW                                ║
# ╚══════════════════════════════════════════════════════════════╝
def page_overview(data):
    from src.dashboard.components.charts import (
        create_forecast_timeseries, create_risk_gauge, create_donut_risk
    )

    st.title("Executive Overview")
    st.caption("AI-powered sales forecasting & inventory intelligence")
    st.markdown("---")

    forecast_df = data['forecasts']
    fin = data['financial']
    risk_df = data['risk']

    summary  = fin.get('executive_summary', {})
    headline = summary.get('headline_metrics', {})
    comp     = fin.get('comparison', {})
    ai_comp  = comp.get('AI Ensemble', comp.get('ai_ensemble', {}))
    mape     = ai_comp.get('mape', 0.15)

    # ── KPI row ───────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Annual Savings",
        _fmt(headline.get('annual_savings_estimate', 0)),
        f"{headline.get('cost_reduction_pct', 0):.1f}% reduction",
        help="Estimated annual savings from AI optimization"
    )
    k2.metric(
        "Forecast Accuracy",
        _pct((1 - mape) * 100),
        f"+{headline.get('forecast_accuracy_improvement', 0):.1f}% vs baseline",
        help="AI ensemble model accuracy"
    )
    k3.metric(
        "Capital Freed",
        _fmt(headline.get('working_capital_freed', 0)),
        help="Working capital released from inventory optimization"
    )
    k4.metric(
        "High-Risk Stores",
        str(summary.get('risk_summary', {}).get('high_risk_segments', 0)),
        f"of {summary.get('risk_summary', {}).get('total_segments', 0)} total",
        delta_color='inverse',
        help="Stores with HIGH stockout risk"
    )

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    col_chart, col_risk = st.columns([5, 2])

    with col_chart:
        if not forecast_df.empty:
            st.plotly_chart(
                create_forecast_timeseries(forecast_df),
                use_container_width=True)
        else:
            st.info("No forecast data available")

    with col_risk:
        if not risk_df.empty and 'composite_risk_score' in risk_df.columns:
            avg = risk_df['composite_risk_score'].mean()
            st.plotly_chart(
                create_risk_gauge(avg, 'System Risk Score'),
                use_container_width=True)
            st.plotly_chart(
                create_donut_risk(risk_df, height=270),
                use_container_width=True)

    # ── Recommendations ───────────────────────────────────────
    recs = summary.get('recommendations', [])
    if recs:
        st.markdown("---")
        st.subheader("Priority Actions")
        for rec in recs:
            p = rec.get('priority', 'MEDIUM')
            icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(p, '⚪')
            with st.expander(
                f"{icon} **[{p}]** {rec.get('action', '')}",
                expanded=(p == 'HIGH')
            ):
                st.markdown(f"**Impact:** {rec.get('impact', 'N/A')}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — SALES FORECASTS                                   ║
# ╚══════════════════════════════════════════════════════════════╝
def page_forecasts(data):
    from src.dashboard.components.charts import (
        create_forecast_timeseries, create_store_heatmap
    )
    from src.dashboard.components.filters import (
        render_sidebar_filters, apply_filters
    )

    st.title("Sales Forecast Analysis")
    st.markdown("---")

    forecast_df = data['forecasts']
    if forecast_df.empty:
        st.warning("No forecast data available"); return

    filters  = render_sidebar_filters(forecast_df)
    filtered = apply_filters(forecast_df, filters)
    fc = filters.get('forecast_col', 'forecast_ensemble')
    if fc not in filtered.columns:
        fc = 'forecast_ensemble'
    if fc not in filtered.columns:
        st.error(f"Column {fc} not found."); return
    if len(filtered) == 0:
        st.warning("No data matches current filters"); return

    total_a = filtered['sales'].sum()
    total_f = filtered[fc].sum()
    err = abs(total_a - total_f) / total_a * 100 if total_a > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual Sales", _fmt(total_a))
    c2.metric("Forecasted",   _fmt(total_f))
    c3.metric("Error", _pct(err))
    c4.metric("Records", f"{len(filtered):,}")

    st.markdown("---")
    st.plotly_chart(
        create_forecast_timeseries(
            filtered, forecast_col=fc,
            title=f'Forecast vs Actual  ({filters["model"]})'),
        use_container_width=True)

    st.subheader("Store Performance Heatmap")
    st.plotly_chart(
        create_store_heatmap(filtered),
        use_container_width=True)

    with st.expander("Raw Forecast Data"):
        st.dataframe(
            filtered.sort_values('date', ascending=False).head(500),
            use_container_width=True, height=400)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — INVENTORY                                         ║
# ╚══════════════════════════════════════════════════════════════╝
def page_inventory(data):
    from src.dashboard.components.charts import (
        create_risk_gauge, create_inventory_treemap
    )

    st.title("Inventory Optimization")
    st.markdown("---")

    inv  = data['inventory']
    risk = data['risk']
    if inv.empty:
        st.warning("No inventory data available"); return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Turnover",
              f"{inv['inventory_turnover'].mean():.1f}x"
              if 'inventory_turnover' in inv.columns else "N/A")
    c2.metric("Avg Days Supply",
              f"{inv['days_of_supply'].mean():.0f} days"
              if 'days_of_supply' in inv.columns else "N/A")
    c3.metric("Total Holding Cost",
              _fmt(inv['holding_cost'].sum())
              if 'holding_cost' in inv.columns else "N/A")
    c4.metric("Safety Stock Cost",
              _fmt(inv['safety_stock_cost'].sum())
              if 'safety_stock_cost' in inv.columns else "N/A")

    st.markdown("---")
    st.subheader("Cost Distribution")
    st.plotly_chart(create_inventory_treemap(inv), use_container_width=True)

    if not risk.empty:
        st.markdown("---")
        st.subheader("Stockout Risk Analysis")
        g1, g2, g3 = st.columns(3)
        for col_w, cat in zip([g1, g2, g3], ['HIGH', 'MEDIUM', 'LOW']):
            sub = (risk[risk['risk_category'] == cat]
                   if 'risk_category' in risk.columns else pd.DataFrame())
            with col_w:
                s = sub['composite_risk_score'].mean() if len(sub) > 0 else 0
                st.plotly_chart(
                    create_risk_gauge(s, f'{cat} ({len(sub)})'),
                    use_container_width=True)

        display_cols = [c for c in [
            'segment', 'composite_risk_score', 'risk_category',
            'recommendation', 'weekly_revenue_at_risk'] if c in risk.columns]
        if display_cols:
            st.dataframe(
                risk[display_cols].sort_values(
                    'composite_risk_score', ascending=False),
                use_container_width=True, height=400)

    with st.expander("Detailed Inventory Metrics"):
        st.dataframe(inv, use_container_width=True, height=400)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — FINANCIAL IMPACT                                  ║
# ╚══════════════════════════════════════════════════════════════╝
def page_financial(data):
    from src.dashboard.components.charts import (
        create_savings_waterfall, create_model_comparison_bar
    )

    st.title("Financial Impact Analysis")
    st.markdown("---")

    fin     = data['financial']
    savings = fin.get('savings', {})
    comp    = fin.get('comparison', {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual Savings",
              _fmt(savings.get('annualized_savings_estimate', 0)),
              f"{savings.get('savings_pct', 0):.1f}% cost reduction")
    c2.metric("Capital Freed",
              _fmt(savings.get('annualized_capital_freed', 0)))
    c3.metric("Accuracy Gain",
              f"+{savings.get('accuracy_improvement_pct', 0):,.0f}%")
    c4.metric("Direct Savings",
              _fmt(savings.get('total_direct_savings', 0)))

    st.markdown("---")
    st.subheader("Savings Breakdown")
    st.plotly_chart(
        create_savings_waterfall(savings),
        use_container_width=True)

    if comp:
        st.markdown("---")
        st.subheader("AI vs Traditional Forecasting")
        rows = [{'model': m, **v} for m, v in comp.items()]
        comp_df = pd.DataFrame(rows)

        c1, c2 = st.columns(2)
        with c1:
            if 'mape' in comp_df.columns:
                st.plotly_chart(
                    create_model_comparison_bar(
                        comp_df, 'mape', 'MAPE  (Lower = Better)'),
                    use_container_width=True)
        with c2:
            if 'total_cost_of_errors' in comp_df.columns:
                st.plotly_chart(
                    create_model_comparison_bar(
                        comp_df, 'total_cost_of_errors',
                        'Error Cost  (Lower = Better)'),
                    use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 5 — AI INSIGHTS (Groq)                                ║
# ╚══════════════════════════════════════════════════════════════╝
def page_ai_insights(data):
    st.title("AI-Powered Insights")
    st.caption("Powered by Groq  +  Llama 3.3 70B")
    st.markdown("---")

    groq_key = os.environ.get('GROQ_API_KEY', '')
    if not groq_key:
        st.warning("GROQ_API_KEY not set in environment.")
        groq_key = st.text_input(
            "Enter Groq API Key:", type="password",
            help="Get free key at console.groq.com")
        if groq_key:
            os.environ['GROQ_API_KEY'] = groq_key
    if not groq_key:
        st.info("Enter your free Groq API key above to enable AI insights.")
        st.markdown("""
        ### How to get your free API key
        1. Visit [console.groq.com/keys](https://console.groq.com/keys)
        2. Sign in (no credit card required)
        3. Click **Create API Key**
        4. Paste it above
        """)
        return

    try:
        from src.insights.groq_insights import GroqInsightsEngine
        engine = GroqInsightsEngine(api_key=groq_key)
        if engine.client is None:
            st.error("Failed to init Groq client. Check API key."); return
    except Exception as e:
        st.error(f"Groq init failed: {e}"); return

    fin     = data['financial']
    fc_df   = data['forecasts']
    inv_df  = data['inventory']
    risk_df = data['risk']

    tab1, tab2, tab3, tab4 = st.tabs([
        "Forecast Analysis", "Inventory Insights",
        "Ask AI", "Weekly Report"])

    with tab1:
        st.subheader("AI Forecast Analysis")
        st.markdown("Get executive-ready insights on forecasting performance")
        if st.button("Generate Forecast Insights", key="fc_btn", type="primary"):
            with st.spinner("Analyzing forecast data ..."):
                res = engine.generate_forecast_insights(
                    forecast_summary={
                        'total_sales': float(fc_df['sales'].sum()) if not fc_df.empty else 0,
                        'stores': int(fc_df['store_nbr'].nunique()) if not fc_df.empty else 0,
                        'period': (f"{fc_df['date'].min()} to {fc_df['date'].max()}"
                                   if not fc_df.empty else "N/A"),
                        'records': len(fc_df)},
                    model_comparison=fin.get('comparison', {}))
                st.markdown("---"); st.markdown(res)

    with tab2:
        st.subheader("AI Inventory Analysis")
        st.markdown("Understand your inventory optimization opportunities")
        if st.button("Generate Inventory Insights", key="inv_btn", type="primary"):
            with st.spinner("Analyzing inventory data ..."):
                es = fin.get('executive_summary', {})
                res = engine.generate_inventory_insights(
                    savings=fin.get('savings', {}),
                    risk_summary=es.get('risk_summary', {}),
                    inventory_health=es.get('inventory_health', {}),
                    recommendations=es.get('recommendations', []))
                st.markdown("---"); st.markdown(res)

    with tab3:
        st.subheader("Ask AI Anything")
        st.markdown("Get instant answers to your business questions")
        examples = [
            "Which stores need immediate inventory attention?",
            "How much money are we saving with AI vs manual forecasting?",
            "What are the main risk factors we should monitor?"]
        for q in examples:
            if st.button(q, key=f"eq_{hash(q)}"):
                st.session_state['user_q'] = q

        user_q = st.text_input(
            "Your question:",
            value=st.session_state.get('user_q', ''),
            placeholder="Type your question here ...")
        if user_q and st.button("Ask AI", key="ask_btn", type="primary"):
            with st.spinner("Thinking ..."):
                ctx = {
                    'savings': fin.get('savings', {}),
                    'comparison': fin.get('comparison', {}),
                    'risk_summary': fin.get('executive_summary', {}).get('risk_summary', {}),
                    'total_stores': (int(fc_df['store_nbr'].nunique())
                                     if not fc_df.empty else 0)}
                ans = engine.answer_stakeholder_question(user_q, ctx)
                st.markdown("---")
                st.markdown(f"**Q:** {user_q}")
                st.markdown(ans)

    with tab4:
        st.subheader("Weekly Executive Report")
        st.markdown("Generate a comprehensive weekly summary for leadership")
        if st.button("Generate Weekly Report", key="rpt_btn", type="primary"):
            with st.spinner("Writing weekly report ..."):
                rpt = engine.generate_weekly_report(
                    forecast_data=fin.get('comparison', {}),
                    inventory_data=fin.get('executive_summary', {}).get('inventory_health', {}),
                    financial_data=fin.get('savings', {}),
                    week_date=(str(fc_df['date'].min())[:10]
                               if not fc_df.empty else "N/A"))
                st.markdown("---"); st.markdown(rpt)
                st.download_button(
                    "Download Report", rpt,
                    file_name=f"weekly_report_{pd.Timestamp.now():%Y%m%d}.md",
                    mime="text/markdown")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.markdown(
        "<h2 style='text-align:center; margin-bottom:0;'>"
        "🔮 AI Forecast</h2>"
        "<p style='text-align:center; color:#8b949e; font-size:0.82rem; "
        "margin-top:4px;'>Powered by ML + Groq AI</p>",
        unsafe_allow_html=True)
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", [
        "Executive Overview",
        "Sales Forecasts",
        "Inventory",
        "Financial Impact",
        "AI Insights"],
        label_visibility='collapsed')

    data = load_data()

    pages = {
        "Executive Overview": page_overview,
        "Sales Forecasts":    page_forecasts,
        "Inventory":          page_inventory,
        "Financial Impact":   page_financial,
        "AI Insights":        page_ai_insights,
    }
    pages[page](data)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This dashboard analyses sales forecasts and inventory "
        "optimisation powered by AI/ML models.")


if __name__ == "__main__":
    main()
