
# ============================================================================
# COPY THIS CODE TO YOUR NOTEBOOK AND RUN IT
# ============================================================================
# After replacing the variable names with yours!

import os
import joblib
import json

# Create directories
os.makedirs('/kaggle/working/data/output', exist_ok=True)
os.makedirs('/kaggle/working/models/xgboost', exist_ok=True)
os.makedirs('/kaggle/working/models/random_forest', exist_ok=True)
os.makedirs('/kaggle/working/models/prophet', exist_ok=True)
os.makedirs('/kaggle/working/models/ensemble', exist_ok=True)

# -----------------------------------------------
# SAVE DATA FILES
# -----------------------------------------------
# Replace 'forecasts' with YOUR dataframe name!

forecasts.to_csv('/kaggle/working/data/output/sales_forecasts.csv', index=False)
print(f"✓ Saved sales_forecasts.csv")

# If you have inventory recommendations:
# inventory_df.to_csv('/kaggle/working/data/output/inventory_recommendations.csv', index=False)

# If you have financial impact:
# financial_df.to_csv('/kaggle/working/data/output/financial_impact.csv', index=False)

# -----------------------------------------------
# SAVE MODELS
# -----------------------------------------------
# Replace 'xgb_model', 'rf_model', etc. with YOUR variable names!

joblib.dump(xgb_model, '/kaggle/working/models/xgboost/model.joblib')
print(f"✓ Saved XGBoost model")

joblib.dump(rf_model, '/kaggle/working/models/random_forest/model.joblib')
print(f"✓ Saved Random Forest model")

joblib.dump(prophet_model, '/kaggle/working/models/prophet/model.joblib')
print(f"✓ Saved Prophet model")

joblib.dump(ensemble_model, '/kaggle/working/models/ensemble/meta_model.joblib')
print(f"✓ Saved Ensemble model")

# If you have scalers/encoders:
# joblib.dump(scaler, '/kaggle/working/models/scalers/scaler.joblib')

# -----------------------------------------------
# SAVE METRICS
# -----------------------------------------------
# If you have performance metrics:
# metrics = {
#     'xgboost_mape': 12.5,
#     'rf_mape': 13.2,
#     'prophet_mape': 14.8,
#     'ensemble_mape': 11.9
# }
# with open('/kaggle/working/data/output/forecast_metrics.json', 'w') as f:
#     json.dump(metrics, f, indent=2)

print("\n✅ ALL FILES SAVED!")
