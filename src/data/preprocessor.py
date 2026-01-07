import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

def get_preprocessor_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Creates a scikit-learn preprocessing pipeline."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def preprocess(df: pd.DataFrame, target_col: str = None, is_training: bool = True):
    """
    Preprocesses the data.
    If is_training=True, fits and saves the preprocessor.
    If is_training=False, loads the preprocessor and transforms.
    """
    target = None
    if target_col and target_col in df.columns:
        target = df[target_col]
        features = df.drop(columns=[target_col])
    else:
        features = df
        if target_col:
            logger.warning(f"Target column '{target_col}' not found in dataframe.")

    # Identify columns
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = MODELS_DIR / "preprocessor.joblib"

    if is_training:
        logger.info("Fitting new preprocessor...")
        preprocessor = get_preprocessor_pipeline(numeric_features, categorical_features)
        X_processed = preprocessor.fit_transform(features)
        
        joblib.dump(preprocessor, pipeline_path)
        logger.info(f"Preprocessor saved to {pipeline_path}")
    else:
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {pipeline_path}. Run training first.")
        logger.info("Loading existing preprocessor...")
        preprocessor = joblib.load(pipeline_path)
        X_processed = preprocessor.transform(features)

    # Convert back to DataFrame for easier handling downstream
    # We lose original column names with OHE, so generic or reconstructed names are used
    # For simplicity, we return a DataFrame with auto-generated names
    X_df = pd.DataFrame(X_processed)
    
    return X_df, target

def save_processed(X: pd.DataFrame, y: pd.Series, output_dir: Path = PROCESSED_DATA_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(output_dir / "X.csv", index=False)
    if y is not None:
        y.to_csv(output_dir / "y.csv", index=False)
    logger.info(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    try:
        # Check if raw data exists
        input_path = Path("data/raw/raw_snapshot.csv")
        if input_path.exists():
            df = pd.read_csv(input_path)
            target_col = "default.payment.next.month"
            if target_col not in df.columns:
                # If the exact column isn't found, try to find it with spaces (common in this dataset)
                # and rename it to the requested dot-notation
                alt_col = "default payment next month"
                if alt_col in df.columns:
                    logger.info(f"Renaming '{alt_col}' to '{target_col}'")
                    df = df.rename(columns={alt_col: target_col})
            
            if target_col not in df.columns and not df.empty:
                # Fallback purely for demonstration if neither is there
                logger.warning(f"'{target_col}' not found. Using last column as target for demo.")
                target_col = df.columns[-1]

            X, y = preprocess(df, target_col=target_col, is_training=True)
            save_processed(X, y)
        else:
            logger.warning("No raw data found at data/raw/raw_snapshot.csv. Run ingest.py first or ensure data exists.")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
