
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self): self.stats = {}
    def transform_all(self, datasets):
        train = datasets['train']
        train['sales'] = train['sales'].clip(lower=0)
        train['is_zero_sales'] = (train['sales'] == 0).astype('int8')
        
        # Merge
        merged = train.merge(datasets['stores'], on='store_nbr', how='left')
        
        # Oil
        oil = datasets['oil'].set_index('date').reindex(pd.date_range(datasets['oil'].date.min(), datasets['oil'].date.max())).ffill().bfill().reset_index().rename(columns={'index': 'date', 'dcoilwtico': 'oil_price'})
        merged = merged.merge(oil, on='date', how='left')
        
        # Holidays
        holidays = datasets['holidays'][datasets['holidays']['transferred'] == False]
        holidays = holidays[['date', 'type', 'locale']].drop_duplicates('date').rename(columns={'type': 'holiday_type'})
        holidays['is_holiday'] = 1
        merged = merged.merge(holidays, on='date', how='left')
        merged['is_holiday'] = merged['is_holiday'].fillna(0).astype('int8')
        
        # Transactions
        merged = merged.merge(datasets['transactions'], on=['date', 'store_nbr'], how='left')
        merged['transactions'] = merged['transactions'].fillna(0)
        merged['oil_price'] = merged['oil_price'].fillna(method='ffill').fillna(method='bfill')
        
        return merged
