# src/api/schemas.py

"""
Pydantic schemas for request/response validation.

Pydantic ensures:
- Incoming requests have correct data types
- Responses follow a consistent format
- API documentation is auto-generated
- Invalid requests get clear error messages
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


class ForecastRequest(BaseModel):
    """What the client sends to get a forecast."""

    store_ids: Optional[List[int]] = Field(
        None,
        description="Store IDs to forecast. None = all stores.",
        examples=[[1, 2, 3]]
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Product categories. None = all.",
        examples=[["BEVERAGES", "GROCERY"]]
    )
    horizon_days: int = Field(
        16,
        ge=1,
        le=90,
        description="How many days to forecast"
    )
    model: str = Field(
        "ensemble",
        description="Model: ensemble, xgboost, random_forest, prophet"
    )
    include_intervals: bool = Field(
        True,
        description="Include prediction intervals?"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "store_ids": [1, 2, 3],
                "horizon_days": 16,
                "model": "ensemble",
                "include_intervals": True
            }]
        }
    }


class ForecastResponse(BaseModel):
    """What the API returns for a forecast request."""

    status: str
    model_used: str
    forecast_count: int
    total_forecasted_sales: float
    avg_daily_forecast: float
    forecasts: List[Dict]
    prediction_intervals: Optional[Dict] = None
    metadata: Dict


class InventoryRequest(BaseModel):
    """Request for inventory optimization."""

    store_ids: Optional[List[int]] = None
    service_level: float = Field(
        0.95,
        ge=0.5,
        le=0.999,
        description="Target service level (0.95 = 95%)"
    )
    lead_time_days: int = Field(
        7,
        ge=1,
        le=60,
        description="Supplier lead time in days"
    )


class InventoryResponse(BaseModel):
    """Response for inventory optimization."""

    status: str
    segments_optimized: int
    total_safety_stock: float
    total_annual_holding_cost: float
    recommendations: List[Dict]
    risk_summary: Dict


class InsightRequest(BaseModel):
    """Request for AI insight generation."""

    insight_type: str = Field(
        "forecast",
        description="Type: forecast, inventory, question"
    )
    question: Optional[str] = Field(
        None,
        description="Your question (for type='question')"
    )
    context: Optional[Dict] = None


class InsightResponse(BaseModel):
    """Response for AI insights."""

    status: str
    insight_type: str
    insight_text: str
    generated_at: str


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    version: str
    models_loaded: bool
    data_available: bool
    groq_available: bool
    last_updated: str
    data_stats: Optional[Dict] = None
