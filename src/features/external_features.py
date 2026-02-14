import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

class ExternalFeatureEngineer:
    def __init__(self):
        self.created_features = []

    def create_all_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating external features...")
        df = df.copy()
        
        if 'oil_price' in df.columns:
            df = self._oil_price_features(df)
        if 'is_holiday' in df.columns:
            df = self._holiday_features(df)
        if 'onpromotion' in df.columns:
            df = self._promotion_features(df)
        if 'transactions' in df.columns:
            df = self._transaction_features(df)
            
        logger.info(f"  ✓ Created {len(self.created_features)} external features")
        return df

    def _oil_price_features(self, df):
        df_sorted = df.sort_values('date')
        unique_dates = df_sorted[['date', 'oil_price']].drop_duplicates('date').sort_values('date')
        
        unique_dates['oil_price_lag_7'] = unique_dates['oil_price'].shift(7)
        unique_dates['oil_price_ma_14'] = unique_dates['oil_price'].rolling(14, min_periods=1).mean()
        unique_dates['oil_price_change_7'] = unique_dates['oil_price'].pct_change(7).fillna(0).clip(-0.5, 0.5)
        unique_dates['oil_price_volatility_14'] = unique_dates['oil_price'].rolling(14, min_periods=1).std().fillna(0)
        
        oil_33 = unique_dates['oil_price'].quantile(0.33)
        oil_66 = unique_dates['oil_price'].quantile(0.66)
        unique_dates['oil_price_level'] = pd.cut(
            unique_dates['oil_price'],
            bins=[-np.inf, oil_33, oil_66, np.inf],
            labels=[0, 1, 2]
        ).astype(np.int8)
        
        oil_features = ['oil_price_lag_7', 'oil_price_ma_14', 'oil_price_change_7', 'oil_price_volatility_14', 'oil_price_level']
        
        df = df.merge(unique_dates[['date'] + oil_features], on='date', how='left')
        for col in oil_features:
            df[col] = df[col].ffill().bfill().astype(np.float32)
            
        self.created_features.extend(oil_features)
        return df

    def _holiday_features(self, df):
        # Work with unique dates for efficiency
        date_df = df[['date', 'is_holiday']].drop_duplicates('date').sort_values('date').copy()
        holiday_dates = date_df[date_df['is_holiday'] == 1]['date'].values
        
        if len(holiday_dates) == 0:
            return df

        # --- THE FIX: Use .days for scalars ---
        def days_since_last_holiday(date, holidays):
            past = holidays[holidays <= date]
            if len(past) == 0: return 999
            return (date - past[-1]).days

        def days_to_next_holiday(date, holidays):
            future = holidays[holidays > date]
            if len(future) == 0: return 999
            return (future[0] - date).days
        
        # Apply
        date_df['days_since_holiday'] = date_df['date'].apply(lambda d: days_since_last_holiday(d, holiday_dates)).clip(0, 60).astype(np.int8)
        date_df['days_to_holiday'] = date_df['date'].apply(lambda d: days_to_next_holiday(d, holiday_dates)).clip(0, 60).astype(np.int8)
        
        date_df['holiday_proximity'] = (1.0 / (1.0 + date_df[['days_since_holiday', 'days_to_holiday']].min(axis=1))).astype(np.float32)
        date_df['is_holiday_eve'] = (date_df['days_to_holiday'] == 1).astype(np.int8)
        date_df['is_post_holiday'] = (date_df['days_since_holiday'] == 1).astype(np.int8)
        date_df['is_holiday_week'] = ((date_df['days_since_holiday'] <= 3) | (date_df['days_to_holiday'] <= 3)).astype(np.int8)
        
        holiday_features = ['days_since_holiday', 'days_to_holiday', 'holiday_proximity', 'is_holiday_eve', 'is_post_holiday', 'is_holiday_week']
        
        df = df.merge(date_df[['date'] + holiday_features], on='date', how='left')
        self.created_features.extend(holiday_features)
        return df

    def _promotion_features(self, df):
        df['has_promotion'] = (df['onpromotion'] > 0).astype(np.int8)
        df['log_onpromotion'] = np.log1p(df['onpromotion']).astype(np.float32)
        
        store_daily_promo = df.groupby(['date', 'store_nbr'])['onpromotion'].transform('sum').astype(np.float32)
        df['store_total_promotions'] = store_daily_promo
        
        store_avg_promo = df.groupby('store_nbr')['onpromotion'].transform('mean')
        df['promo_vs_store_avg'] = (df['onpromotion'] / store_avg_promo.replace(0, np.nan)).fillna(0).clip(0, 10).astype(np.float32)
        
        self.created_features.extend(['has_promotion', 'log_onpromotion', 'store_total_promotions', 'promo_vs_store_avg'])
        return df

    def _transaction_features(self, df):
        df['log_transactions'] = np.log1p(df['transactions']).astype(np.float32)
        store_avg_txn = df.groupby('store_nbr')['transactions'].transform('mean')
        df['txn_vs_store_avg'] = (df['transactions'] / store_avg_txn.replace(0, np.nan)).fillna(1.0).clip(0, 5).astype(np.float32)
        
        self.created_features.extend(['log_transactions', 'txn_vs_store_avg'])
        return df
