import pandas as pd


TARGET_COL = "default.payment.next.month"

def load_data(path="data/raw/credit_default.csv"):
    df = pd.read_csv(path)


    
    # Handle potential column name discrepancies
    if TARGET_COL not in df.columns:
        alt_col = "default payment next month"
        if alt_col in df.columns:
            df = df.rename(columns={alt_col: TARGET_COL})
        else:
            # Fallback or error
            raise KeyError(f"Target column '{TARGET_COL}' (or '{alt_col}') not found in dataset columns: {df.columns.tolist()}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]


    return X, y
