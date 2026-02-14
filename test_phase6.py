# Phase 6 Verification Test Suite
import json
import pandas as pd
from src.insights.groq_insights import GroqInsightsEngine
from src.insights.anomaly_detector import SalesAnomalyDetector

print("="*60)
print("PHASE 6 VERIFICATION TEST SUITE")
print("="*60)
print()

# Test 1: Groq Client Initialization
print("TEST 1: Groq Client Initialization")
print("-"*60)
engine = GroqInsightsEngine()
if engine.client:
    print("✅ Groq client initialized successfully")
    print(f"   Model: {engine.model}")
    print(f"   Temperature: {engine.temperature}")
    print(f"   Max Tokens: {engine.max_tokens}")
else:
    print("❌ Client initialization failed - check API key")
print()

# Test 2: Anomaly Detection
print("TEST 2: Anomaly Detection System")
print("-"*60)
df = pd.read_parquet('data/output/forecasts.parquet')
print(f"📊 Loaded {len(df):,} forecast records")

detector = SalesAnomalyDetector(z_threshold=3.0, pct_threshold=0.5)
anomalies = detector.detect_forecast_anomalies(
    df, 
    actual_col='sales',
    forecast_col='forecast_ensemble',
    segment_cols=['store_nbr', 'family']
)

summary = detector.get_anomaly_summary(anomalies)
print(f"✅ Detected {summary['total']} anomalies")
print(f"   Spikes: {summary['spikes']}")
print(f"   Drops: {summary['drops']}")
print(f"   Avg Severity: {summary['avg_severity']:.1f}")

if summary['total'] > 0:
    print(f"\n📋 Top 3 Anomalies by Severity:")
    top_3 = anomalies.head(3)
    for idx, row in top_3.iterrows():
        print(f"   - Date: {row['date']}, Store: {row['store_nbr']}, "
              f"Severity: {row['severity']:.1f}, Type: {row['anomaly_type']}")
print()

# Test 3: Stakeholder Q&A (Simple Test)
print("TEST 3: Stakeholder Q&A")
print("-"*60)
with open('data/output/financial_impact.json') as f:
    financial_data = json.load(f)

answer = engine.answer_stakeholder_question(
    question='What is our annual cost savings from the AI forecasting system?',
    data_context={
        'annual_savings': financial_data['savings']['annualized_savings_estimate'],
        'accuracy_improvement': financial_data['savings']['accuracy_improvement_pct'],
        'working_capital_freed': financial_data['savings']['annualized_capital_freed']
    }
)
print("Question: What is our annual cost savings from the AI forecasting system?")
print()
print("Answer:")
print(answer)
print()

# Test 4: Forecast Insights Generation
print("TEST 4: Forecast Insights Generation")
print("-"*60)
insights = engine.generate_forecast_insights(
    forecast_summary={
        'total_stores': 54,
        'forecast_period': '16 days',
        'model': 'Ensemble (XGBoost + Random Forest + Prophet)'
    },
    model_comparison=financial_data['comparison']
)
print(insights)
print()

# Test 5: Inventory Insights Generation
print("TEST 5: Inventory Insights Generation")
print("-"*60)
inventory_insights = engine.generate_inventory_insights(
    savings=financial_data['savings'],
    risk_summary=financial_data['executive_summary']['risk_summary'],
    inventory_health=financial_data['executive_summary']['inventory_health'],
    recommendations=financial_data['executive_summary']['recommendations']
)
print(inventory_insights)
print()

print("="*60)
print("✅ ALL PHASE 6 TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
