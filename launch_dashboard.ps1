# Quick Launch Script for Dashboard
# Run this to start the Streamlit dashboard

$env:PYTHONPATH = "."
$env:DATA_DIR = "data/output"

Write-Host "Launching AI Forecasting Dashboard..." -ForegroundColor Cyan
Write-Host "Data directory: data/output" -ForegroundColor Yellow
Write-Host "Dashboard will open at http://localhost:8501" -ForegroundColor Green
Write-Host ""

streamlit run src/dashboard/app.py
