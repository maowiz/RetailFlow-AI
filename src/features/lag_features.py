
import pandas as pd
class LagFeatureEngineer:
    def __init__(self, lag_days, rolling_windows):
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        
    def create_all_lag_features(self, df):
        df = df.sort_values(['store_nbr', 'family', 'date'])
        grouped = df.groupby(['store_nbr', 'family'])['sales']
        for lag in self.lag_days:
            df[f'sales_lag_{lag}'] = grouped.shift(lag).astype('float32')
        for w in self.rolling_windows:
            df[f'sales_rolling_mean_{w}'] = grouped.shift(1).rolling(w).mean().astype('float32')
        return df
