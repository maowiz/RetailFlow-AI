
import pandas as pd
import numpy as np
class TimeFeatureEngineer:
    def create_all_time_features(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek.astype('int8')
        df['month'] = df['date'].dt.month.astype('int8')
        df['year'] = df['date'].dt.year.astype('int16')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_payday'] = df['date'].dt.day.isin([15, 30, 31]).astype('int8')
        return df
