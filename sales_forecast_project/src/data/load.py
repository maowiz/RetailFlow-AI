
import pandas as pd
import sqlite3
import os
import json

class DataLoader:
    def __init__(self, output_dir, db_path):
        self.output_dir = output_dir
        self.db_path = db_path
        os.makedirs(output_dir, exist_ok=True)

    def load_to_parquet(self, df, filename):
        path = os.path.join(self.output_dir, f"{filename}.parquet")
        df.to_parquet(path, index=False)
        return path

    def load_to_sqlite(self, df, table_name):
        conn = sqlite3.connect(self.db_path)
        df.drop(columns=['date'], errors='ignore').to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    
    def create_indexes(self): pass
    def save_metadata(self, meta): pass
