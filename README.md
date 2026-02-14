# 📊 RetailFlow AI — Sales Forecasting & Inventory Optimization

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow.svg)](https://huggingface.co/spaces/maowi/sales-forecast-optimizer)

> **End-to-end ML system for retail demand forecasting and inventory optimization**  
> Combines XGBoost, Prophet, and Random Forest in an ensemble to achieve 87.5% forecast accuracy, quantifying $1M+ annual savings potential.

**🎯 Live Demo**: https://huggingface.co/spaces/maowi/sales-forecast-optimizer

---

## 🚀 Features

- **🤖 ML Forecasting**: Ensemble of XGBoost + Prophet + Random Forest
- **📦 Inventory Optimization**: Safety stock, reorder points, EOQ calculations
- **💰 Financial Impact**: Quantified savings with detailed breakdown
- **🧠 AI Insights**: Groq LLM-powered natural language business analysis
- **📈 Interactive Dashboard**: 5-page Streamlit dashboard with dark theme
- **🔌 REST API**: FastAPI backend for programmatic access

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| **Forecast Accuracy (MAPE)** | 87.5% |
| **Annual Savings** | $1,003,750 |
| **Working Capital Freed** | $4,106,250 |
| **Stores Analyzed** | 54 stores |
| **Training Data** | 3M+ rows (Kaggle Store Sales) |

---

## 🏗️ Project Architecture

```
RetailFlow-AI/
├── src/
│   ├── data/           # ETL pipeline & data processing
│   ├── features/       # Feature engineering
│   ├── models/         # ML models (XGBoost, Prophet, RF)
│   ├── optimization/   # Inventory optimization algorithms
│   ├── insights/       # Groq LLM integration
│   ├── dashboard/      # Streamlit UI
│   └── api/            # FastAPI REST endpoints
├── data/
│   ├── raw/            # Original Kaggle data
│   └── output/         # Processed forecasts & metrics
├── notebooks/          # Jupyter notebooks (EDA, experiments)
└── tests/              # Unit tests
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Git
- (Optional) Kaggle API credentials

### 1. Clone the Repository

```bash
git clone https://github.com/maowiz/RetailFlow-AI.git
cd RetailFlow-AI
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Groq API (for AI insights)
GROQ_API_KEY=your_groq_api_key_here

# Data directory
DATA_DIR=data/output

# Kaggle credentials (optional, if downloading raw data)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Get free Groq API key: https://console.groq.com/keys

---

## 🎯 Quick Start

### Option 1: View Pre-computed Results (Fastest)

```bash
# Launch the dashboard with existing data
streamlit run src/dashboard/app.py
```

Dashboard opens at http://localhost:8501

### Option 2: Run Complete Pipeline

```bash
# 1. Download Kaggle data (requires Kaggle API setup)
python src/data/download_kaggle_data.py

# 2. Process data and train models
python src/models/train_all_models.py

# 3. Generate inventory recommendations
python src/optimization/run_optimization.py

# 4. Launch dashboard
streamlit run src/dashboard/app.py
```

### Option 3: Use the REST API

```bash
# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs at http://localhost:8000/docs

**Example API Call**:
```python
import requests

# Get forecast for stores 1-5
response = requests.post('http://localhost:8000/predict', json={
    "store_ids": [1, 2, 3, 4, 5],
    "horizon_days": 14,
    "model": "ensemble"
})

forecast = response.json()
print(f"Total forecasted sales: ${forecast['total_forecasted_sales']:,.0f}")
```

---

## 📊 Dashboard Pages

1. **Executive Overview**  
   KPIs, risk summary, time series charts, priority recommendations

2. **Sales Forecasts**  
   Interactive time series, store heatmaps, model comparison, filters

3. **Inventory Optimization**  
   Safety stock levels, inventory turnover, reorder points, risk gauges

4. **Financial Impact**  
   Savings waterfall, cost breakdown, ROI analysis

5. **AI Insights**  
   Groq-powered Q&A, forecast analysis, weekly reports

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System information |
| `/health` | GET | Health check with data stats |
| `/predict` | POST | Generate sales forecasts |
| `/optimize-inventory` | POST | Get inventory recommendations |
| `/insights` | POST | AI-powered business insights |

**Full API docs**: http://localhost:8000/docs (when server is running)

---

## 🧪 Testing

```bash
# Run API tests
python test_api.py

# Run Phase 6 Groq tests
python test_phase6.py

# Unit tests (if available)
pytest tests/
```

---

## 🎓 How It Works

### 1. Data Pipeline
- **Source**: Kaggle Store Sales dataset (125K stores × 33 products × daily)
- **Processing**: Missing value imputation, outlier detection, date feature extraction
- **Features**: 50+ engineered features (lags, rolling stats, holidays, promotions)

### 2. ML Models
- **XGBoost**: Gradient boosting for non-linear patterns
- **Prophet**: Handles seasonality and holidays
- **Random Forest**: Ensemble robustness
- **Ensemble**: Weighted average (40% XGB, 30% Prophet, 30% RF)

### 3. Inventory Optimization
- **Safety Stock**: `z × σ_lead × √lead_time`
- **Reorder Point**: `avg_demand × lead_time + safety_stock`
- **EOQ**: `√(2 × D × S / H)`
- **Risk Scoring**: Composite of demand variability, forecast error, stockout history

### 4. Financial Impact
- **Holding Cost Reduction**: Optimized inventory levels
- **Stockout Cost Avoidance**: Improved service levels
- **Working Capital Gains**: Reduced excess inventory

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ideas for contributions:**
- Add more ML models (LSTM, Transformer, etc.)
- Implement A/B testing framework
- Add unit tests
- Improve dashboard UX
- Add deployment guides (AWS, Azure, GCP)

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Store Sales Competition](https://www.kaggle.com/c/store-sales-time-series-forecasting)
- **Groq**: Free LLM API for AI insights
- **Hugging Face**: Free hosting for demo dashboard

---

## 📧 Contact

**Author**: Maowi  
**GitHub**: [@maowiz](https://github.com/maowiz)  
**Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/maowi/sales-forecast-optimizer)

---

## ⭐ Star This Repo

If you find this project helpful, please give it a ⭐ — it helps others discover it too!

---

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) (detailed setup)
- [API Documentation](docs/API.md) (endpoint reference)
- [Architecture](docs/ARCHITECTURE.md) (system design)
- [Deployment Guide](docs/DEPLOYMENT.md) (production setup)

---

**Built with ❤️ for the open-source community**
