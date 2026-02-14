# test_api.py — API endpoint tests

"""
Run this to test all API endpoints.
Make sure the API server is running first:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("API ENDPOINT TESTS")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Test 1: Root
# ──────────────────────────────────────────────────────────────
print("\n1️⃣  Root Endpoint — GET /")
try:
    r = requests.get(f"{BASE_URL}/")
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Name: {data.get('name', 'N/A')}")
        print(f"   Version: {data.get('version', 'N/A')}")
        print(f"   Status: {data.get('status', 'N/A')}")
    else:
        print(f"   Error: {r.text}")
except Exception as e:
    print(f"   ❌ Connection failed: {e}")
    print(f"   Make sure API server is running!")
    exit(1)

# ──────────────────────────────────────────────────────────────
# Test 2: Health
# ──────────────────────────────────────────────────────────────
print("\n2️⃣  Health Check — GET /health")
try:
    r = requests.get(f"{BASE_URL}/health")
    if r.status_code == 200:
        health = r.json()
        print(f"   Status: {health.get('status', 'N/A')}")
        print(f"   Data Available: {health.get('data_available', False)}")
        print(f"   Models Loaded: {health.get('models_loaded', False)}")
        print(f"   Groq Available: {health.get('groq_available', False)}")
        if health.get('data_stats'):
            stats = health['data_stats']
            print(f"   Forecast Rows: {stats.get('forecast_rows', 0):,}")
            print(f"   Stores: {stats.get('forecast_stores', 0)}")
    else:
        print(f"   ❌ Status: {r.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
# Test 3: Predict
# ──────────────────────────────────────────────────────────────
print("\n3️⃣  Forecast Endpoint — POST /predict")
try:
    payload = {
        "store_ids": [1, 2, 3],
        "horizon_days": 7,
        "model": "ensemble",
        "include_intervals": True
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    if r.status_code == 200:
        data = r.json()
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Model: {data.get('model_used', 'N/A')}")
        print(f"   Forecast Count: {data.get('forecast_count', 0):,}")
        print(f"   Total Forecasted: ${data.get('total_forecasted_sales', 0):,.0f}")
        print(f"   Avg Daily: ${data.get('avg_daily_forecast', 0):,.0f}")
        if data.get('prediction_intervals'):
            pi = data['prediction_intervals']
            print(f"   Confidence: {pi.get('confidence_level', 0)*100:.0f}%")
    else:
        print(f"   ❌ Status: {r.status_code}")
        print(f"   Detail: {r.json().get('detail', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
# Test 4: Inventory
# ──────────────────────────────────────────────────────────────
print("\n4️⃣  Inventory Endpoint — POST /optimize-inventory")
try:
    payload = {
        "service_level": 0.95,
        "lead_time_days": 7
    }
    r = requests.post(f"{BASE_URL}/optimize-inventory", json=payload)
    if r.status_code == 200:
        data = r.json()
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Segments Optimized: {data.get('segments_optimized', 0)}")
        print(f"   Total Safety Stock: {data.get('total_safety_stock', 0):,.0f}")
        print(f"   Annual Holding Cost: ${data.get('total_annual_holding_cost', 0):,.0f}")
        risk = data.get('risk_summary', {})
        h = risk.get('high_risk', 0)
        m = risk.get('medium_risk', 0)
        l = risk.get('low_risk', 0)
        print(f"   Risk: HIGH={h} MEDIUM={m} LOW={l}")
    else:
        print(f"   ❌ Status: {r.status_code}")
        print(f"   Detail: {r.json().get('detail', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
# Test 5: AI Insights
# ──────────────────────────────────────────────────────────────
print("\n5️⃣  AI Insights Endpoint — POST /insights")
try:
    payload = {
        "insight_type": "question",
        "question": "Summarize the forecast performance in one sentence"
    }
    r = requests.post(f"{BASE_URL}/insights", json=payload)
    if r.status_code == 200:
        data = r.json()
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Type: {data.get('insight_type', 'N/A')}")
        insight = data.get('insight_text', '')
        print(f"   Insight: {insight[:150]}...")
    elif r.status_code == 503:
        print(f"   ⚠️  Groq not available (expected if no API key set)")
    else:
        print(f"   ❌ Status: {r.status_code}")
        print(f"   Detail: {r.json().get('detail', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ All endpoint tests complete!")
print("=" * 60)
print("\nNext steps:")
print("  1. Open http://localhost:8000/docs for Swagger UI")
print("  2. Test endpoints interactively")
print("  3. Check http://localhost:8000/redoc for API documentation")
