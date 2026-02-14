# src/insights/groq_insights.py

"""
Groq LLM Integration Module

Uses Groq's free API to generate natural language insights
from forecasting and optimization results.

This module reads YOUR ACTUAL data files and creates 
prompts specifically matched to YOUR data structure.
"""

import os
import json
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq package not installed. Run: pip install groq")


class GroqInsightsEngine:
    """
    Generates AI-powered business insights using Groq LLM.
    
    All methods:
    1. Take structured data (dicts, DataFrames)
    2. Format into clear prompts
    3. Send to Groq
    4. Return formatted text for dashboard display
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        self.client = None  # SDK client (optional)
        self._use_direct_http = False  # fallback flag
        
        if not self.api_key:
            logger.warning("❌ No GROQ_API_KEY found")
            return
        
        if GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.api_key)
                # Test the client with a minimal call
                logger.info(f"Groq SDK client created | Model: {model}")
            except Exception as e:
                logger.warning(f"Groq SDK init failed: {e}, using direct HTTP")
                self._use_direct_http = True
        else:
            self._use_direct_http = True
        
        logger.info(f"✅ Groq ready | Model: {model} | Direct HTTP: {self._use_direct_http}")
    
    def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Make a call to Groq API with error handling.
        Uses direct HTTP requests as fallback if SDK fails.
        """
        if not self.api_key:
            return self._fallback_response()
        
        temp = temperature or self.temperature
        tokens = max_tokens or self.max_tokens
        
        # Try SDK first if available and not already flagged for HTTP
        if self.client and not self._use_direct_http:
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=0.9,
                    stream=False
                )
                result = response.choices[0].message.content
                logger.info(f"Groq SDK call successful | Tokens: {response.usage.total_tokens}")
                return result
            except Exception as e:
                logger.warning(f"Groq SDK call failed: {e}, switching to direct HTTP")
                self._use_direct_http = True
        
        # Direct HTTP fallback (works everywhere)
        return self._call_groq_http(system_prompt, user_prompt, temp, tokens)
    
    def _call_groq_http(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Direct HTTP call to Groq API — bypasses SDK entirely."""
        import requests as req
        
        try:
            resp = req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.9,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", "?")
            logger.info(f"Groq HTTP call successful | Tokens: {tokens_used}")
            return result
        except Exception as e:
            logger.error(f"Groq HTTP API error: {str(e)}")
            return f"⚠️ AI insight generation failed: {str(e)}"
    
    def _fallback_response(self) -> str:
        return (
            "🤖 **AI insights unavailable.** \n\n"
            "Please set your GROQ_API_KEY:\n"
            "1. Get free key at [console.groq.com](https://console.groq.com/keys)\n"
            "2. Enter it in the sidebar or set as environment variable"
        )
    
    def generate_forecast_insights(
        self,
        forecast_summary: Dict,
        model_comparison: Dict,
        segment_performance: Optional[Dict] = None
    ) -> str:
        """Generate forecast analysis insights."""
        
        system_prompt = """You are a senior financial analyst at a major retail 
and textile company (like Sapphire Group with 50+ stores across Pakistan).

Your audience is the CFO and VP of Operations. Write in a professional 
tone suitable for a board presentation.

Rules:
- Be specific — use exact numbers from the data
- Quantify everything in dollars or percentages
- Prioritize actionable recommendations
- Keep total response under 400 words
- Use markdown formatting with headers and bullet points
- Flag risks with ⚠️ emoji
- Highlight wins with ✅ emoji"""

        # Build segment analysis section if provided
        segment_section = ""
        if segment_performance:
            segment_section = "\n\n## SEGMENT ANALYSIS\n" + json.dumps(segment_performance, indent=2, default=str)
        
        user_prompt = f"""Analyze these sales forecasting results:

## FORECAST SUMMARY
{json.dumps(forecast_summary, indent=2, default=str)}

## MODEL PERFORMANCE
{json.dumps(model_comparison, indent=2, default=str)}{segment_section}

Provide:
1. **Executive Summary** (2 sentences)
2. **Key Findings** (3-4 bullet points with specific numbers)
3. **Risk Factors** (what should leadership watch?)
4. **Recommendations** (3 specific actions for next quarter)
5. **Financial Impact** (quantify the business value)"""

        return self._call_groq(system_prompt, user_prompt)
    
    def generate_inventory_insights(
        self,
        savings: Dict,
        risk_summary: Dict,
        inventory_health: Dict,
        recommendations: List[Dict]
    ) -> str:
        """Generate inventory optimization insights."""
        
        system_prompt = """You are a supply chain and inventory expert at a 
retail/textile company. Interpret inventory optimization results for the 
CFO and VP of Operations.

Focus on:
- Financial impact (cost savings, capital freed)
- Risk mitigation (stockout prevention)  
- Operational improvements
- Keep response under 400 words
- Use markdown formatting"""

        user_prompt = f"""Analyze these inventory optimization results:

## FINANCIAL SAVINGS
{json.dumps(savings, indent=2, default=str)}

## INVENTORY HEALTH
{json.dumps(inventory_health, indent=2, default=str)}

## RISK ASSESSMENT
{json.dumps(risk_summary, indent=2, default=str)}

## SYSTEM RECOMMENDATIONS
{json.dumps(recommendations, indent=2, default=str)}

Provide:
1. **Bottom Line** (one paragraph for the CFO)
2. **Savings Breakdown** (where does each dollar come from?)
3. **Risk Actions** (prioritized for high-risk segments)
4. **ROI Summary** (is the AI system worth it?)
5. **90-Day Roadmap** (implementation steps)"""

        return self._call_groq(system_prompt, user_prompt)
    
    def explain_anomaly(
        self,
        store: str,
        category: str,
        date: str,
        expected_sales: float,
        actual_sales: float,
        context: Optional[Dict] = None
    ) -> str:
        """Explain a sales anomaly."""
        
        deviation = (
            (actual_sales - expected_sales) / expected_sales * 100
            if expected_sales > 0 else 0
        )
        direction = "higher" if actual_sales > expected_sales else "lower"
        
        system_prompt = """You are a retail analytics expert. Provide a brief 
explanation for a sales anomaly. Give 3-4 plausible causes and one 
recommended action. Keep it under 150 words."""

        user_prompt = f"""Sales anomaly detected:
- Store: {store}
- Category: {category}
- Date: {date}
- Expected: {expected_sales:.0f} units
- Actual: {actual_sales:.0f} units
- Deviation: {deviation:+.1f}% ({direction} than expected)
{f'- Context: {json.dumps(context)}' if context else ''}

What caused this? What should we do?"""

        return self._call_groq(
            system_prompt, user_prompt,
            max_tokens=500, temperature=0.4
        )
    
    def answer_stakeholder_question(
        self,
        question: str,
        data_context: Dict
    ) -> str:
        """Answer a free-form stakeholder question."""
        
        system_prompt = """You are an AI assistant in a sales forecasting 
dashboard at a retail/textile company. Answer based ONLY on the data 
provided. If you don't have enough info, say so. Be specific and concise.
Use numbers from the data context. Keep response under 200 words."""

        user_prompt = f"""Available Data:
{json.dumps(data_context, indent=2, default=str)}

Question: {question}

Provide a clear, data-driven answer."""

        return self._call_groq(
            system_prompt, user_prompt,
            temperature=0.2, max_tokens=800
        )
    
    def generate_weekly_report(
        self,
        forecast_data: Dict,
        inventory_data: Dict,
        financial_data: Dict,
        week_date: str
    ) -> str:
        """Generate a complete weekly report."""
        
        system_prompt = """Generate a professional weekly business intelligence 
report for retail/textile company leadership. Use markdown formatting with 
clear sections. Keep under 500 words."""

        user_prompt = f"""Weekly Report for week of {week_date}:

## FORECAST PERFORMANCE
{json.dumps(forecast_data, indent=2, default=str)}

## INVENTORY STATUS
{json.dumps(inventory_data, indent=2, default=str)}

## FINANCIAL METRICS
{json.dumps(financial_data, indent=2, default=str)}

Create a report with:
- Executive Summary
- Key Metrics
- Risks & Alerts
- Action Items for Next Week"""

        return self._call_groq(
            system_prompt, user_prompt,
            max_tokens=2500, temperature=0.25
        )
