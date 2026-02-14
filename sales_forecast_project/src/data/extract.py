
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.datasets = {}
        
    def extract_all(self) -> Dict[str, pd.DataFrame]:
        logger.info("Extracting data...")
        self.datasets['train'] = pd.read_csv(os.path.join(self.data_path, 'train.csv'), dtype={'store_nbr': 'int8', 'sales': 'float32', 'onpromotion': 'int16'}, parse_dates=['date'])
        self.datasets['stores'] = pd.read_csv(os.path.join(self.data_path, 'stores.csv'), dtype={'store_nbr': 'int8', 'cluster': 'int8'})
        self.datasets['oil'] = pd.read_csv(os.path.join(self.data_path, 'oil.csv'), parse_dates=['date'])
        self.datasets['holidays'] = pd.read_csv(os.path.join(self.data_path, 'holidays_events.csv'), parse_dates=['date'])
        self.datasets['transactions'] = pd.read_csv(os.path.join(self.data_path, 'transactions.csv'), parse_dates=['date'])
        return self.datasets

def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns: df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns: df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
