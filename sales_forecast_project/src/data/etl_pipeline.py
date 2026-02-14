
import time
import logging
from src.data.extract import DataExtractor, optimize_memory
from src.data.transform import DataTransformer
from src.data.load import DataLoader
logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self, raw_data_path, output_dir, db_path):
        self.extractor = DataExtractor(raw_data_path)
        self.transformer = DataTransformer()
        self.loader = DataLoader(output_dir, db_path)

    def run(self):
        print("  > Extracting...")
        raw = self.extractor.extract_all()
        print("  > Transforming...")
        clean = self.transformer.transform_all(raw)
        print("  > Optimizing...")
        clean = optimize_memory(clean)
        print("  > Loading...")
        self.loader.load_to_parquet(clean, 'sales_clean')
        self.loader.load_to_sqlite(clean, 'sales_clean')
        return clean
