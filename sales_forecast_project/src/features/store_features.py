
import pandas as pd
from sklearn.preprocessing import LabelEncoder
class StoreFeatureEngineer:
    def create_all_store_features(self, df):
        for col in ['city', 'state', 'store_type']:
            if col in df.columns:
                df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))
        return df
