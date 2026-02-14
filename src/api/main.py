# src/api/main.py

"""
FastAPI Application — REST API for Sales Forecasting System

Endpoints:
  GET  /          → System info
  GET  /health    → Health check
  POST /predict   → Sales forecast
  POST /optimize-inventory → Inventory recommendations
  POST /insights  → AI-powered insights (Groq)

Run locally:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except Exception:
    pass

from src.api.schemas import (
    ForecastRequest, ForecastResponse,
    InventoryRequest, InventoryResponse,
    InsightRequest, InsightResponse,
    HealthResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================================================================
# CREATE APP
# ===================================================================
app = FastAPI(
    title="AI Sales Forecast & Inventory Optimizer API",
    description=(
        "REST API for AI-driven sales forecasting and inventory optimization.\n\n"
        "**Features:**\n"
        "- Sales forecasting with ensemble ML models\n"
        "- Inventory optimization (safety stock, reorder points)\n"
        "- AI-powered insights via Groq LLM\n\n"
        "Built for Sapphire Group AI/ML assessment."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allow requests from any origin (needed for dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# APPLICATION STATE
# ===================================================================
class AppState:
    """Holds loaded data. Initialized on startup."""

    def __init__(self):
        self.forecast_df: Optional[pd.DataFrame] = None
        self.financial_data: Optional[dict] = None
        self.inventory_df: Optional[pd.DataFrame] = None
        self.risk_df: Optional[pd.DataFrame] = None
        self.groq_engine = None
        self.is_loaded = False
        self.load_errors: list = []

    def load_all(self):
        """Load all data files on startup."""
        data_dir = os.environ.get(
            'DATA_DIR',
            os.path.join(PROJECT_ROOT, 'data', 'output')
        )

        logger.info(f"Loading data from: {data_dir}")

        # ---- Forecasts ----
        try:
            path = os.path.join(data_dir, 'forecasts.parquet')
            if os.path.exists(path):
                self.forecast_df = pd.read_parquet(path)
                self.forecast_df['date'] = pd.to_datetime(
                    self.forecast_df['date'])
                logger.info(
                    f"Forecasts loaded: {len(self.forecast_df):,} rows")
            else:
                self.load_errors.append(
                    f"forecasts.parquet not found at {path}")
                logger.warning(f"{path} not found")
        except Exception as e:
            self.load_errors.append(f"forecasts load error: {e}")
            logger.error(f"Forecasts error: {e}")

        # ---- Financial impact ----
        try:
            path = os.path.join(data_dir, 'financial_impact.json')
            if os.path.exists(path):
                with open(path) as f:
                    self.financial_data = json.load(f)
                logger.info("Financial data loaded")
            else:
                self.load_errors.append("financial_impact.json not found")
        except Exception as e:
            self.load_errors.append(f"financial load error: {e}")

        # ---- Inventory recommendations ----
        try:
            path = os.path.join(data_dir, 'inventory_recommendations.parquet')
            if os.path.exists(path):
                self.inventory_df = pd.read_parquet(path)
                logger.info(f"Inventory loaded: {len(self.inventory_df)} rows")
        except Exception as e:
            self.load_errors.append(f"inventory load error: {e}")

        # ---- Risk data ----
        try:
            path = os.path.join(data_dir, 'stockout_risk.parquet')
            if os.path.exists(path):
                self.risk_df = pd.read_parquet(path)
                logger.info(f"Risk loaded: {len(self.risk_df)} rows")
        except Exception as e:
            self.load_errors.append(f"risk load error: {e}")

        # ---- Groq engine ----
        try:
            from src.insights.groq_insights import GroqInsightsEngine
            self.groq_engine = GroqInsightsEngine()
            if self.groq_engine.client:
                logger.info("Groq engine ready")
            else:
                logger.warning("Groq initialized without API key")
        except Exception as e:
            logger.warning(f"Groq not available: {e}")

        self.is_loaded = self.forecast_df is not None

        if self.is_loaded:
            logger.info("API data loading complete")
        else:
            logger.error(f"API data loading failed: {self.load_errors}")


state = AppState()


# ===================================================================
# STARTUP
# ===================================================================
@app.on_event("startup")
async def startup():
    """Load data when server starts."""
    logger.info("Starting API server...")
    state.load_all()


# ===================================================================
# ENDPOINTS
# ===================================================================

# ---------- Root ----------
@app.get("/", tags=["System"])
async def root():
    """API root — system information."""
    return {
        "name": "AI Sales Forecast & Inventory Optimizer API",
        "version": "1.0.0",
        "status": "running" if state.is_loaded else "loading",
        "documentation": "/docs",
        "endpoints": {
            "GET /health": "System health check",
            "POST /predict": "Generate sales forecasts",
            "POST /optimize-inventory": "Get inventory recommendations",
            "POST /insights": "Generate AI insights"
        }
    }


# ---------- Health ----------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and data availability."""

    data_stats = None
    if state.forecast_df is not None:
        data_stats = {
            "forecast_rows": len(state.forecast_df),
            "forecast_stores": int(
                state.forecast_df['store_nbr'].nunique()),
            "forecast_date_range": {
                "start": str(state.forecast_df['date'].min())[:10],
                "end": str(state.forecast_df['date'].max())[:10]
            }
        }
        if state.inventory_df is not None:
            data_stats["inventory_segments"] = len(state.inventory_df)
        if state.risk_df is not None:
            data_stats["risk_segments"] = len(state.risk_df)

    return HealthResponse(
        status="healthy" if state.is_loaded else "degraded",
        version="1.0.0",
        models_loaded=state.is_loaded,
        data_available=state.forecast_df is not None,
        groq_available=(
            state.groq_engine is not None
            and state.groq_engine.client is not None
        ),
        last_updated=datetime.now().isoformat(),
        data_stats=data_stats
    )


# ---------- Predict ----------
@app.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
async def predict_sales(request: ForecastRequest):
    """
    Generate sales forecasts.

    Returns predicted sales by store for the requested parameters.
    Supports ensemble, XGBoost, Random Forest, and Prophet models.
    """
    if state.forecast_df is None:
        raise HTTPException(
            status_code=503,
            detail="Forecast data not loaded. Check /health for status."
        )

    df = state.forecast_df.copy()

    # Filter by stores
    if request.store_ids:
        df = df[df['store_nbr'].isin(request.store_ids)]

    # Filter by categories
    if request.categories and 'family' in df.columns:
        df = df[df['family'].isin(request.categories)]

    if len(df) == 0:
        raise HTTPException(
            status_code=404,
            detail="No data found for the requested filters."
        )

    # Select forecast column
    col_map = {
        'ensemble': 'forecast_ensemble',
        'xgboost': 'forecast_xgboost',
        'random_forest': 'forecast_rf',
        'prophet': 'forecast_prophet'
    }
    fc_col = col_map.get(request.model, 'forecast_ensemble')

    if fc_col not in df.columns:
        available = [c for c in col_map.values() if c in df.columns]
        fc_col = available[0] if available else 'forecast_ensemble'

    if fc_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model}' not available. "
                f"Columns: {df.columns.tolist()}"
            )
        )

    # Limit rows for response
    max_rows = request.horizon_days * 54  # ~54 stores
    subset = df.head(max_rows)

    # Build forecasts list
    forecasts = []
    for _, row in subset.iterrows():
        entry = {
            'date': str(row['date'])[:10],
            'store_nbr': int(row['store_nbr']),
            'predicted_sales': round(float(row[fc_col]), 2),
            'actual_sales': round(float(row['sales']), 2),
        }
        if 'family' in row.index:
            entry['category'] = str(row['family'])
        forecasts.append(entry)

    # Prediction intervals
    intervals = None
    if request.include_intervals:
        lower_col = 'forecast_lower'
        upper_col = 'forecast_upper'
        if lower_col in df.columns and upper_col in df.columns:
            intervals = {
                'confidence_level': 0.95,
                'avg_lower': round(float(df[lower_col].mean()), 2),
                'avg_upper': round(float(df[upper_col].mean()), 2),
                'avg_width': round(float(
                    (df[upper_col] - df[lower_col]).mean()
                ), 2)
            }

    return ForecastResponse(
        status="success",
        model_used=request.model,
        forecast_count=len(forecasts),
        total_forecasted_sales=round(float(df[fc_col].sum()), 2),
        avg_daily_forecast=round(float(
            df.groupby('date')[fc_col].sum().mean()
        ), 2),
        forecasts=forecasts[:1000],  # cap at 1000 rows
        prediction_intervals=intervals,
        metadata={
            'date_range': {
                'start': str(df['date'].min())[:10],
                'end': str(df['date'].max())[:10]
            },
            'stores_included': int(df['store_nbr'].nunique()),
            'categories_included': (
                int(df['family'].nunique())
                if 'family' in df.columns else 0
            ),
            'generated_at': datetime.now().isoformat()
        }
    )


# ---------- Inventory ----------
@app.post(
    "/optimize-inventory",
    response_model=InventoryResponse,
    tags=["Inventory"]
)
async def optimize_inventory(request: InventoryRequest):
    """
    Get inventory optimization recommendations.

    Returns safety stock levels, reorder points, EOQ,
    and stockout risk assessments per segment.
    """
    if state.inventory_df is None:
        raise HTTPException(
            status_code=503, detail="Inventory data not loaded.")

    if state.risk_df is None:
        raise HTTPException(
            status_code=503, detail="Risk data not loaded.")

    inv = state.inventory_df.copy()
    risk = state.risk_df.copy()

    # Filter by stores if requested
    if request.store_ids and 'segment' in inv.columns:
        inv = inv[inv['segment'].isin(request.store_ids)]
        risk = risk[risk['segment'].isin(request.store_ids)]

    # Build recommendations
    recommendations = []
    for _, row in inv.iterrows():
        rec = {}
        for col in inv.columns:
            val = row[col]
            if pd.notna(val):
                if isinstance(val, (np.integer, int)):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    rec[col] = round(float(val), 2)
                else:
                    rec[col] = str(val)
        recommendations.append(rec)

    # Build risk summary
    risk_summary: dict = {
        'total_segments': len(risk)
    }
    if 'risk_category' in risk.columns:
        risk_summary['high_risk'] = int(
            (risk['risk_category'] == 'HIGH').sum())
        risk_summary['medium_risk'] = int(
            (risk['risk_category'] == 'MEDIUM').sum())
        risk_summary['low_risk'] = int(
            (risk['risk_category'] == 'LOW').sum())
    if 'weekly_revenue_at_risk' in risk.columns:
        risk_summary['total_weekly_exposure'] = round(
            float(risk['weekly_revenue_at_risk'].sum()), 2)
    if 'composite_risk_score' in risk.columns:
        risk_summary['avg_risk_score'] = round(
            float(risk['composite_risk_score'].mean()), 2)

    return InventoryResponse(
        status="success",
        segments_optimized=len(recommendations),
        total_safety_stock=(
            round(float(inv['safety_stock'].sum()), 2)
            if 'safety_stock' in inv.columns else 0
        ),
        total_annual_holding_cost=(
            round(float(inv['holding_cost'].sum()), 2)
            if 'holding_cost' in inv.columns else 0
        ),
        recommendations=recommendations,
        risk_summary=risk_summary
    )


# ---------- Insights ----------
@app.post("/insights", response_model=InsightResponse, tags=["AI Insights"])
async def generate_insights(request: InsightRequest):
    """
    Generate AI-powered business insights using Groq LLM.

    Types:
    - "forecast": Analyze forecast performance
    - "inventory": Analyze inventory optimization
    - "question": Answer a specific business question
    """
    if state.groq_engine is None or state.groq_engine.client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI insights not available. "
                "Set GROQ_API_KEY environment variable. "
                "Get free key at console.groq.com"
            )
        )

    try:
        if request.insight_type == "question" and request.question:
            context = request.context or {}
            if state.financial_data:
                context['financial_summary'] = state.financial_data.get(
                    'savings', {})

            text = state.groq_engine.answer_stakeholder_question(
                question=request.question,
                data_context=context
            )

        elif request.insight_type == "forecast":
            text = state.groq_engine.generate_forecast_insights(
                forecast_summary=request.context or {},
                model_comparison=(
                    state.financial_data.get('comparison', {})
                    if state.financial_data else {}
                )
            )

        elif request.insight_type == "inventory":
            savings = {}
            risk_summary = {}
            inventory_health = {}
            recommendations = []

            if state.financial_data:
                savings = state.financial_data.get('savings', {})
                exec_sum = state.financial_data.get(
                    'executive_summary', {})
                risk_summary = exec_sum.get('risk_summary', {})
                inventory_health = exec_sum.get('inventory_health', {})
                recommendations = exec_sum.get('recommendations', [])

            text = state.groq_engine.generate_inventory_insights(
                savings=savings,
                risk_summary=risk_summary,
                inventory_health=inventory_health,
                recommendations=recommendations
            )

        else:
            text = (
                f"Unknown insight type: '{request.insight_type}'. "
                "Use: forecast, inventory, question"
            )

        return InsightResponse(
            status="success",
            insight_type=request.insight_type,
            insight_text=text,
            generated_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Insight generation failed: {str(e)}"
        )
