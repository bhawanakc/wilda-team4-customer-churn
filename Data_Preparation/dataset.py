"""
Customer Churn Analysis — Data Preprocessing Pipeline
Telecommunications Company Dataset

Steps covered:
  1. Load the dataset
  2. Data integrity checks (duplicates, dtypes, invalid values)
  3. Handle missing data
  4. Encode categorical variables
  5. Feature scaling / normalisation
  6. Save the cleaned, model-ready dataset
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ---------------------------------------------------------------------------
# CONFIG — change this path to point at your actual file
# ---------------------------------------------------------------------------
INPUT_PATH = "Dataset_ATS_v2.xlsx"   # put this file in the same folder as this script,
                                       # or paste the full path e.g. r"C:\Users\rohan\Downloads\Dataset_ATS_v2.xlsx"
OUTPUT_PATH = "telco_churn_preprocessed.csv"
TARGET_COL = "Churn"
ID_COL = "customerID"  # not present in this dataset — code handles that gracefully


def load_data(path: str) -> pd.DataFrame:
    """Step 1: Load the dataset (CSV or Excel) and report basic shape/info."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\nCould not find the file at: {os.path.abspath(path)}\n"
            f"Fix: either move 'Dataset_ATS_v2.xlsx' into the same folder as this script,\n"
            f"or edit INPUT_PATH at the top of this script to the file's full path."
        )
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")
    return df


def check_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Data integrity & consistency checks."""
    print("--- Data Integrity Check ---")

    # Duplicate rows (based on customer ID if present, else full row)
    if ID_COL in df.columns:
        dup_count = df.duplicated(subset=[ID_COL]).sum()
        if dup_count > 0:
            print(f"Found {dup_count} duplicate customer IDs — dropping duplicates.")
            df = df.drop_duplicates(subset=[ID_COL], keep="first")
    else:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            print(f"Found {dup_count} fully duplicate rows — dropping.")
            df = df.drop_duplicates()

    # Known Telco quirk: TotalCharges is often read as object/string with blank entries
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        print("Converted 'TotalCharges' to numeric (invalid/blank entries -> NaN).")

    # Standardise inconsistent category text (whitespace/case issues are common)
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != ID_COL]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan})

    # Validate SeniorCitizen is binary if present
    if "SeniorCitizen" in df.columns:
        invalid = ~df["SeniorCitizen"].isin([0, 1])
        if invalid.any():
            print(f"Warning: {invalid.sum()} invalid SeniorCitizen values found.")

    # Validate tenure and charges are non-negative
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                print(f"Warning: {neg} negative values found in '{col}'.")

    print(f"Shape after integrity checks: {df.shape}\n")
    return df.reset_index(drop=True)


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Handle missing values with column-type-appropriate strategies."""
    print("--- Missing Data Handling ---")
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    print("Missing values per column before handling:")
    print(missing_summary if not missing_summary.empty else "None found.")

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if df[col].dtype in ["float64", "int64"]:
            # Numeric: impute with median (robust to outliers, common for charges/tenure)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled numeric column '{col}' with median ({median_val:.2f}).")
        else:
            # Categorical: impute with mode (most frequent category)
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            print(f"Filled categorical column '{col}' with mode ('{fill_val}').")

    print(f"Remaining missing values: {df.isnull().sum().sum()}\n")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Step 4: Encode categorical variables.
      - Binary Yes/No-style columns -> label encoded (0/1)
      - Multi-category columns -> one-hot encoded
      - Target column -> label encoded separately (kept interpretable)
    """
    print("--- Categorical Encoding ---")
    encoders = {}

    df = df.drop(columns=[ID_COL], errors="ignore")  # ID has no predictive value

    # Encode target separately
    if TARGET_COL in df.columns:
        le_target = LabelEncoder()
        df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL])
        encoders[TARGET_COL] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
        print(f"Target '{TARGET_COL}' encoded as: {encoders[TARGET_COL]}")

    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
    multi_cols = [c for c in cat_cols if df[c].nunique() > 2]

    # Binary columns -> label encode
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    if binary_cols:
        print(f"Label-encoded binary columns: {binary_cols}")

    # Multi-category columns -> one-hot encode
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        print(f"One-hot encoded multi-category columns: {multi_cols}")

    print(f"Shape after encoding: {df.shape}\n")
    return df, encoders


def scale_features(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Step 5: Feature scaling / normalisation.
      method='standard' -> z-score standardisation (mean=0, std=1)
      method='minmax'   -> normalise to [0, 1] range
    Applied only to continuous numeric columns, not one-hot/binary flags.
    """
    print(f"--- Feature Scaling ({method}) ---")

    # Identify continuous numeric columns (exclude binary 0/1 and target)
    candidate_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    candidate_cols = [c for c in candidate_cols if c != TARGET_COL]
    scale_cols = [c for c in candidate_cols if df[c].nunique() > 2]

    if not scale_cols:
        print("No continuous columns found to scale.\n")
        return df

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"Scaled columns: {scale_cols}\n")
    return df


def run_pipeline():
    df = load_data(INPUT_PATH)
    df = check_integrity(df)
    df = handle_missing_data(df)
    df, encoders = encode_categorical(df)
    df = scale_features(df, method="standard")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessed dataset saved to: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")
    return df, encoders


if __name__ == "__main__":
    processed_df, label_encoders = run_pipeline()
    print("\nPreview of processed data:")
    print(processed_df.head())