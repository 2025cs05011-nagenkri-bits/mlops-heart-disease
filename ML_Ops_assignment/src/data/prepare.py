"""
prepare.py

Load the raw Heart Disease (Cleveland) dataset, binarize the target,
perform a stratified train/test split, and save the splits as CSVs.

No preprocessing (imputation/scaling/encoding) happens here — that
lives in preprocess.py and is fitted inside the training Pipeline to
avoid data leakage.

Usage:
  python -m src.data.prepare
  python -m src.data.prepare --test-size 0.2 --random-state 42
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = "Raw_data/heart+disease/processed.cleveland.data"
DEFAULT_OUTPUT_DIR = "data/processed"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_raw(path=DEFAULT_DATA_PATH):
    """Load raw UCI Cleveland CSV and binarize target (0 vs 1-4)."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    return df


def split(df, test_size=0.2, random_state=42):
    """Stratified train/test split on the target column."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )


def save_splits(train_df, test_df, output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Prepare Heart Disease train/test splits")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to the processed.cleveland.data file")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write train.csv and test.csv")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set proportion (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    args = parser.parse_args()

    df = load_raw(args.data)
    train_df, test_df = split(df, args.test_size, args.random_state)
    train_path, test_path = save_splits(train_df, test_df, args.output_dir)

    print("Heart Disease split complete")
    print("=" * 50)
    print(f"Source        : {args.data}")
    print(f"Total samples : {len(df)}")
    print(f"Train samples : {len(train_df)}  -> {train_path}")
    print(f"Test samples  : {len(test_df)}  -> {test_path}")
    print(f"Train target  : {train_df['target'].value_counts().to_dict()}")
    print(f"Test target   : {test_df['target'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
