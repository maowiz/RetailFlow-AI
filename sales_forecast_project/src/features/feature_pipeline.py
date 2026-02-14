import pandas as pd
import numpy as np
import os
import logging
from src.features.time_features import TimeFeatureEngineer
from src.features.store_features import StoreFeatureEngineer
from src.features.external_features import ExternalFeatureEngineer
from src.features.lag_features import LagFeatureEngineer
from src.features.interaction_features import InteractionFeatureEngineer

logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def run(self, df):
        print("  > Time Features...")
        df = TimeFeatureEngineer().create_all_time_features(df)
        
        print("  > Store Features...")
        df = StoreFeatureEngineer().create_all_store_features(df)
        
        print("  > External Features...")
        df = ExternalFeatureEngineer().create_all_external_features(df)
        
        print("  > Lag Features (Heavy)...")
        # Using standard lags
        df = LagFeatureEngineer(lag_days=[7, 14, 28], rolling_windows=[7, 28]).create_all_lag_features(df)
        
        # --- SMART FILLNA FIX ---
        print("  > Handling Missing Values (Smart Fill)...")
        
        # 1. Fill NUMERIC columns with 0 (Lags, rolling means, etc.)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        
        # 2. Fill CATEGORICAL/STRING columns with "None"
        # This prevents the ArrowTypeError (String vs Int conflict)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            # Convert to object type to allow string filling if it was category
            df[col] = df[col].astype(object).fillna("None")
            
        # ------------------------

        print("  > Saving to Parquet...")
        output_path = os.path.join(self.output_dir, 'features_engineered.parquet')
        df.to_parquet(output_path)
        return df
