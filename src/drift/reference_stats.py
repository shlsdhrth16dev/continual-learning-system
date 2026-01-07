from typing import Dict, Any
import pandas as pd

def compute_reference_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes statistical baseline for features to enable drift detection.
    Returns a dictionary with a strict schema:
    {
        "column_name": {
            "type": "numeric" | "categorical",
            "stats": { ... }
        }
    }
    """
    stats = {}

    for col in df.columns:
        col_data = df[col]
        
        if col_data.dtype.kind in "if":  # numeric (int or float)
            stats[col] = {
                "type": "numeric",
                "stats": {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "p25": float(col_data.quantile(0.25)),
                    "p50": float(col_data.quantile(0.50)),
                    "p75": float(col_data.quantile(0.75)),
                    "count": int(col_data.count())
                }
            }
        else:
            stats[col] = {
                "type": "categorical",
                "stats": {
                    "value_counts": col_data.value_counts().head(20).to_dict(),
                    "unique_count": int(col_data.nunique()),
                    "count": int(col_data.count())
                }
            }

    return stats
