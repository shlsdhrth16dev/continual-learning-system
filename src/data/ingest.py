# src/data/ingest.py
import pandas as pd
from pathlib import Path
import logging
from typing import List, Optional

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")

def load_data(file_path: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads data from a CSV file and performs basic validation.
    
    Args:
        file_path: Path to the CSV file.
        required_columns: List of columns that must be present.
    
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Loaded dataset is empty")
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def save_raw_data(df: pd.DataFrame, filename: str, output_dir: Path = RAW_DATA_DIR):
    """
    Saves the dataframe as a raw snapshot.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        logger.info(f"Saving raw snapshot to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info("Save successful.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    # Ensure this path exists or provide a valid one
    source_path = "data/raw/credit_default.csv" 
    
    # Define expected schema if known, e.g., ["feature1", "target"]
    # required_cols = ["target"] 
    
    try:
        # Note: This will fail if source_path doesn't exist, which is expected behavior
        if Path(source_path).exists():
            df = load_data(source_path)
            save_raw_data(df, "raw_snapshot.csv")
        else:
            logger.warning(f"Demo file {source_path} not found. Skipped ingestion.")
    except Exception as e:
        logger.warning(f"Script failed: {e}")
