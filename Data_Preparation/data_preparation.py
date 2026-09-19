"""
Customer Churn Analysis - Data Preparation Pipeline
Telecommunications Company Dataset

Steps covered:
  1. Load the dataset
  2. Data integrity checks (duplicates, whitespace, invalid values)
  3. Handle missing data (median / mode imputation)
  4. Encode categorical variables (label + one-hot)
  5. Save the preprocessed (encoded, unscaled) dataset
  6. Stratified train/test split
  7. Feature scaling - StandardScaler fitted on the TRAINING set only
  8. Save train/test sets, the fitted scaler, the encoding map and a summary
  9. Save the before/after scaling figure

Run from the repository root:
    python Data_Preparation/data_preparation.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# CONFIG - every path is resolved relative to this script, so the pipeline
# runs from anywhere with no edits.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # Data_Preparation/
ROOT = HERE.parent                              # repository root
INPUT_PATH = ROOT / "data" / "Dataset_ATS_v2.csv"
OUT_DIR = HERE
FIG_DIR = HERE / "figures"

TARGET_COL = "Churn"
TEST_SIZE = 0.2                                 # 20% held out for testing
RANDOM_STATE = 42                               # fixed seed, reproducible split
SCALE_COLS = ["tenure", "MonthlyCharges"]       # the only continuous features

# Final column order - target last, so X = df[:-1] and y = df[-1].
FINAL_COLS = [
    "gender", "SeniorCitizen", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "MonthlyCharges",
    "Contract_One_year", "Contract_Two_year", TARGET_COL,
]


def load_data(path: Path) -> pd.DataFrame:
    """Step 1: Load the raw dataset and report basic shape/info."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n\nCould not find the raw dataset at: {path}\n"
            f"Fix: the file 'Dataset_ATS_v2.csv' must sit in the repository's data/ folder."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")
    return df


def check_integrity(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Step 2: Data integrity & consistency checks. No rows are dropped."""
    print("--- Data Integrity Check ---")

    # Full-row duplicates. There is no customer ID in this dataset and only
    # 10 coarse columns, so identical rows are most likely different customers
    # who happen to share the same profile. Team decision: keep all of them.
    dup_count = int(df.duplicated().sum())
    print(
        f"Found {dup_count} identical rows, KEPT: no customer ID, "
        f"so identical profiles are treated as different customers."
    )

    # Standardise inconsistent category text (stray whitespace is common).
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan})
    print(f"Stripped whitespace from text columns: {cat_cols}")

    # Validate SeniorCitizen is binary.
    invalid = ~df["SeniorCitizen"].isin([0, 1])
    if invalid.any():
        print(f"Warning: {invalid.sum()} invalid SeniorCitizen values found.")
    else:
        print("SeniorCitizen check: all values are 0 or 1.")

    # Validate tenure and charges are non-negative.
    for col in ["tenure", "MonthlyCharges"]:
        neg = int((df[col] < 0).sum())
        if neg > 0:
            print(f"Warning: {neg} negative values found in '{col}'.")
        else:
            print(f"'{col}' check: no negative values.")

    print(f"Shape after integrity checks: {df.shape}\n")
    return df.reset_index(drop=True), dup_count


def handle_missing_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Step 3: Handle missing values with column-type-appropriate strategies."""
    print("--- Missing Data Handling ---")
    missing_total = int(df.isnull().sum().sum())
    print(f"Missing values found: {missing_total}")

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric: median is robust to outliers (charges/tenure).
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled numeric column '{col}' with median ({median_val:.2f}).")
        else:
            # Categorical: most frequent category.
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            print(f"Filled categorical column '{col}' with mode ('{fill_val}').")

    print(f"Remaining missing values: {int(df.isnull().sum().sum())}\n")
    return df, missing_total


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Step 4: Encode categorical variables.
      - Target Churn        -> label encoded (No=0, Yes=1)
      - Binary text columns -> label encoded, alphabetical (0/1)
      - Contract            -> one-hot, drop_first (Month-to-month = baseline)
      - SeniorCitizen       -> already 0/1, left as it is
    Everything ends up as int, so there are no True/False values in the output.
    """
    print("--- Categorical Encoding ---")
    encoders: dict = {}

    # Target, encoded separately so it stays interpretable.
    le_target = LabelEncoder()
    df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL])
    encoders[TARGET_COL] = {c: int(v) for c, v in
                            zip(le_target.classes_, le_target.transform(le_target.classes_))}
    print(f"Target '{TARGET_COL}' encoded as: {encoders[TARGET_COL]}")

    # Binary text columns -> label encode (alphabetical: Female=0/Male=1, No=0/Yes=1).
    binary_cols = [c for c in df.columns
                   if not pd.api.types.is_numeric_dtype(df[c])
                   and df[c].nunique() == 2 and c != "Contract"]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = {c: int(v) for c, v in zip(le.classes_, le.transform(le.classes_))}
    print(f"Label-encoded binary columns: {binary_cols}")

    # Contract -> one-hot with the first level dropped as the baseline.
    df = pd.get_dummies(df, columns=["Contract"], drop_first=True)
    df = df.rename(columns={"Contract_One year": "Contract_One_year",
                            "Contract_Two year": "Contract_Two_year"})
    encoders["Contract"] = {
        "Month-to-month": "baseline (both 0)",
        "One year": "Contract_One_year=1",
        "Two year": "Contract_Two_year=1",
    }
    print("One-hot encoded 'Contract' (Month-to-month = baseline): "
          "Contract_One_year, Contract_Two_year")

    # Cast every encoded column to int - no True/False anywhere.
    for col in df.columns:
        if col not in SCALE_COLS:
            df[col] = df[col].astype(int)

    df = df[FINAL_COLS]
    print(f"Shape after encoding: {df.shape}")
    print(f"Column order: {list(df.columns)}\n")
    return df, encoders


def split_data(df: pd.DataFrame):
    """Step 6: Stratified 80/20 split, so both sets keep the same churn ratio."""
    print("--- Train/Test Split ---")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set: {len(X_train)} rows ({len(X_train) / len(df):.1%})")
    print(f"Testing set:  {len(X_test)} rows ({len(X_test) / len(df):.1%})")
    print(f"Stratified on '{TARGET_COL}' (random_state={RANDOM_STATE})\n")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Step 7: Feature scaling - z-score standardisation.

    The scaler is FITTED ON THE TRAINING SET ONLY and then used to transform
    both sets. Fitting on the full dataset would let the test set's mean and
    standard deviation influence the training data: that is data leakage.
    Only the continuous columns are scaled; the 0/1 flags are left alone.
    """
    print("--- Feature Scaling (StandardScaler, fitted on train only) ---")
    X_train, X_test = X_train.copy(), X_test.copy()

    before = {
        "train_mean": X_train[SCALE_COLS].mean().to_dict(),
        "train_std": X_train[SCALE_COLS].std(ddof=0).to_dict(),
        "test_mean": X_test[SCALE_COLS].mean().to_dict(),
        "test_std": X_test[SCALE_COLS].std(ddof=0).to_dict(),
    }

    scaler = StandardScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])   # fit on TRAIN
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])         # apply to TEST

    after = {
        "train_mean": X_train[SCALE_COLS].mean().to_dict(),
        "train_std": X_train[SCALE_COLS].std(ddof=0).to_dict(),
        "test_mean": X_test[SCALE_COLS].mean().to_dict(),
        "test_std": X_test[SCALE_COLS].std(ddof=0).to_dict(),
    }

    print(f"Scaled columns: {SCALE_COLS}")
    for i, col in enumerate(SCALE_COLS):
        print(f"  {col}: training mean {scaler.mean_[i]:.4f}, std {scaler.scale_[i]:.4f}")
    print("Binary/encoded columns and the target were left unscaled.\n")
    return X_train, X_test, scaler, before, after


def save_scaling_figure(train_before: pd.DataFrame, train_after: pd.DataFrame) -> Path:
    """Step 9: 2x2 histogram grid of the training set before and after scaling."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "scaling_before_after.png"

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    units = {"tenure": "months", "MonthlyCharges": "dollars"}

    for j, col in enumerate(SCALE_COLS):
        axes[0, j].hist(train_before[col], bins=30, color="#4269d0", edgecolor="white")
        axes[0, j].set_title(f"{col} - before scaling (original units)")
        axes[0, j].set_xlabel(f"{col} ({units[col]})")
        axes[0, j].set_ylabel("Number of customers")

        axes[1, j].hist(train_after[col], bins=30, color="#efb118", edgecolor="white")
        axes[1, j].set_title(f"{col} - after StandardScaler (z-score)")
        axes[1, j].set_xlabel(f"{col} (standard deviations from the mean)")
        axes[1, j].set_ylabel("Number of customers")

    fig.suptitle("Training set before and after standardisation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved scaling figure to: {out_path.relative_to(ROOT)}")
    return out_path


def save_outputs(preprocessed, train_df, test_df, scaler, encoders,
                 before, after, dup_count, missing_total, raw_shape) -> None:
    """Steps 5 & 8: write every file the rest of the project depends on."""
    print("--- Saving Outputs ---")

    train_df.to_csv(OUT_DIR / "train_set.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_set.csv", index=False)
    print(f"Training set saved to: {(OUT_DIR / 'train_set.csv').relative_to(ROOT)}")
    print(f"Testing set saved to:  {(OUT_DIR / 'test_set.csv').relative_to(ROOT)}")

    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    print(f"Fitted scaler saved to: {(OUT_DIR / 'scaler.pkl').relative_to(ROOT)}")

    with open(OUT_DIR / "encoding_map.json", "w", encoding="utf-8") as f:
        json.dump(encoders, f, indent=2)
    print(f"Encoding map saved to: {(OUT_DIR / 'encoding_map.json').relative_to(ROOT)}")

    contract_mix = {
        "full": _contract_mix(preprocessed),
        "train": _contract_mix(train_df),
        "test": _contract_mix(test_df),
    }
    summary = {
        "source_file": str(INPUT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "raw_shape": {"rows": int(raw_shape[0]), "columns": int(raw_shape[1])},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "scaled_columns": SCALE_COLS,
        "columns": FINAL_COLS,
        "duplicate_rows_found_and_kept": dup_count,
        "missing_values_found": missing_total,
        "rows": {"full": len(preprocessed), "train": len(train_df), "test": len(test_df)},
        "share_of_total": {
            "train": round(len(train_df) / len(preprocessed), 4),
            "test": round(len(test_df) / len(preprocessed), 4),
        },
        "churners": {
            "full": int(preprocessed[TARGET_COL].sum()),
            "train": int(train_df[TARGET_COL].sum()),
            "test": int(test_df[TARGET_COL].sum()),
        },
        "churn_rate": {
            "full": round(float(preprocessed[TARGET_COL].mean()), 6),
            "train": round(float(train_df[TARGET_COL].mean()), 6),
            "test": round(float(test_df[TARGET_COL].mean()), 6),
        },
        "contract_mix": contract_mix,
        "scaler": {
            "type": "StandardScaler",
            "fitted_on": "training set only",
            "mean_": {c: float(v) for c, v in zip(SCALE_COLS, scaler.mean_)},
            "scale_": {c: float(v) for c, v in zip(SCALE_COLS, scaler.scale_)},
        },
        "scaling_before": _round_nested(before),
        "scaling_after": _round_nested(after),
    }
    with open(OUT_DIR / "preparation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {(OUT_DIR / 'preparation_summary.json').relative_to(ROOT)}\n")


def _contract_mix(df: pd.DataFrame) -> dict:
    """Counts of the three original Contract levels, back out of the one-hot pair."""
    one = int(df["Contract_One_year"].sum())
    two = int(df["Contract_Two_year"].sum())
    return {"Month-to-month": int(len(df) - one - two), "One year": one, "Two year": two}


def _round_nested(d: dict) -> dict:
    return {k: {c: round(float(v), 6) for c, v in inner.items()} for k, inner in d.items()}


def run_pipeline():
    df = load_data(INPUT_PATH)
    raw_shape = df.shape
    df, dup_count = check_integrity(df)
    df, missing_total = handle_missing_data(df)
    df, encoders = encode_categorical(df)

    # Step 5: the preprocessed dataset is saved ENCODED BUT NOT SCALED, so the
    # raw values stay readable and the scaling can be reproduced from the split.
    preprocessed_path = OUT_DIR / "preprocessed_dataset.csv"
    df.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed dataset saved to: {preprocessed_path.relative_to(ROOT)}")
    print(f"Final shape: {df.shape}\n")

    X_train, X_test, y_train, y_test = split_data(df)
    train_before = X_train[SCALE_COLS].copy()
    X_train_s, X_test_s, scaler, before, after = scale_features(X_train, X_test)

    train_df = pd.concat([X_train_s, y_train], axis=1)[FINAL_COLS]
    test_df = pd.concat([X_test_s, y_test], axis=1)[FINAL_COLS]

    save_outputs(df, train_df, test_df, scaler, encoders,
                 before, after, dup_count, missing_total, raw_shape)
    save_scaling_figure(train_before, X_train_s[SCALE_COLS])

    # ---- Final summary -----------------------------------------------------
    print("\n=== Data Preparation Complete ===")
    print(f"Full dataset:  {len(df):>5} rows, churn rate {df[TARGET_COL].mean():.2%} "
          f"({int(df[TARGET_COL].sum())} churners)")
    print(f"Training set:  {len(train_df):>5} rows, churn rate {train_df[TARGET_COL].mean():.2%} "
          f"({int(train_df[TARGET_COL].sum())} churners)")
    print(f"Testing set:   {len(test_df):>5} rows, churn rate {test_df[TARGET_COL].mean():.2%} "
          f"({int(test_df[TARGET_COL].sum())} churners)")
    print("Scaler (fitted on training data only):")
    for i, col in enumerate(SCALE_COLS):
        print(f"  {col}: mean {scaler.mean_[i]:.4f}, std {scaler.scale_[i]:.4f}")
    return df, train_df, test_df, scaler, encoders


if __name__ == "__main__":
    preprocessed_df, train_set, test_set, fitted_scaler, label_encoders = run_pipeline()
    print("\nPreview of the training set:")
    print(train_set.head())
