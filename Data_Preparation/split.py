"""
Train/Test Split — Customer Churn Dataset
Splits the preprocessed dataset into training and testing sets,
ready for model building in Scikit-learn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_PATH = "telco_churn_preprocessed.csv"   # put in same folder, or use full path
TARGET_COL = "Churn"
TEST_SIZE = 0.2        # 20% held out for testing, 80% for training
RANDOM_STATE = 42      # fixed seed so the split is reproducible every run


def split_data():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded preprocessed dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate features (X) from the target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # stratify=y ensures both train and test sets keep the same churn ratio
    # as the full dataset -- important since churn is imbalanced (~26% churned)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("train_dataset.csv", index=False)
    test_df.to_csv("test_dataset.csv", index=False)

    print(f"\nTraining set: {train_df.shape[0]} rows -> saved to train_dataset.csv")
    print(f"Testing set:  {test_df.shape[0]} rows -> saved to test_dataset.csv")
    print(f"\nChurn rate in training set: {y_train.mean():.4f}")
    print(f"Churn rate in testing set:  {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()