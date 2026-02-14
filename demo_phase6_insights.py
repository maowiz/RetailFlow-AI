# Demo: Generate Real AI Insights from Your Data
import json
from src.insights import GroqInsightsEngine

print("="*70)
print(" PHASE 6 DEMO: AI-Generated Business Insights")
print("="*70)
print()

# Load actual financial data
with open('data/output/financial_impact.json') as f:
    financial_data = json.load(f)

# Initialize Groq engine
engine = GroqInsightsEngine()

print("🤖 Generating insights using Groq's Llama 3.3 70B...")
print()

# Generate comprehensive forecast insights
print("📊 FORECAST ANALYSIS (AI-Generated)")
print("-"*70)
insights = engine.generate_forecast_insights(
    forecast_summary={
        'total_stores': 54,
        'forecast_period': '16 days',
        'model': 'Ensemble (XGBoost + Random Forest + Prophet)',
        'data_points': '26,730 forecasts'
    },
    model_comparison=financial_data['comparison']
)
print(insights)
print()
print()

# Generate inventory insights
print("📦 INVENTORY OPTIMIZATION INSIGHTS (AI-Generated)")
print("-"*70)
inventory_insights = engine.generate_inventory_insights(
    savings=financial_data['savings'],
    risk_summary=financial_data['executive_summary']['risk_summary'],
    inventory_health=financial_data['executive_summary']['inventory_health'],
    recommendations=financial_data['executive_summary']['recommendations']
)
print(inventory_insights)
print()
print()

# Q&A Demonstration
print("💬 STAKEHOLDER Q&A (AI-Generated)")
print("-"*70)

questions = [
    "What is the total annual savings from implementing this AI system?",
    "How does our AI model compare to traditional forecasting methods?",
    "What are the main sources of cost savings?"
]

for i, question in enumerate(questions, 1):
    print(f"\nQuestion {i}: {question}")
    print()
    
    answer = engine.answer_stakeholder_question(
        question=question,
        data_context={
            'annual_savings': financial_data['savings']['annualized_savings_estimate'],
            'accuracy_improvement': financial_data['savings']['accuracy_improvement_pct'],
            'working_capital_freed': financial_data['savings']['annualized_capital_freed'],
            'comparison': financial_data['comparison'],
            'savings_breakdown': financial_data['savings']
        }
    )
    
    print(f"Answer: {answer}")
    print("-"*70)

print()
print("="*70)
print("✅ Phase 6 Demo Complete! Your insights are ready for production use.")
print("="*70)
