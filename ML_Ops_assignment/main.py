"""
heart_dataset.py

Simple utility to:
  - Load Heart Disease (Cleveland) dataset from a CSV file
  - Show dataset info
  - Show first N rows
  - Show class distribution
  - Show missing-value summary

Usage:
  python main.py
  python main.py --head 10
  python main.py --stats
  python main.py --save heart.csv
  python main.py --data Raw_data/heart+disease/processed.cleveland.data
"""

import argparse
import pandas as pd

DEFAULT_DATA_PATH = "Raw_data/heart+disease/processed.cleveland.data"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_heart_dataset(path=DEFAULT_DATA_PATH):
    """Load Heart Disease (Cleveland) dataset and return as Pandas DataFrame.

    The raw UCI file has no header row and uses '?' for missing values.
    The original target is 0-4 (severity); we binarize to 0 (no disease)
    or 1 (disease present), which is the standard classification setup.
    """
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Heart Disease Dataset Utility")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to the processed.cleveland.data file")
    parser.add_argument("--head", type=int, help="Show first N rows")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--save", type=str, help="Save dataset to CSV file")
    args = parser.parse_args()

    df = load_heart_dataset(args.data)

    print("\nLoaded Heart Disease Dataset (Cleveland)")
    print("=" * 50)
    print(f"Source  : {args.data}")
    print(f"Samples : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"Features: {list(df.columns[:-1])}")
    print("Target  : 0 = no disease, 1 = disease present")
    print("=" * 50)

    if args.head:
        print(f"\nFirst {args.head} rows:\n")
        print(df.head(args.head))

    if args.stats:
        print("\nStatistical Summary:\n")
        print(df.describe())

        print("\nClass Distribution:\n")
        print(df["target"].value_counts())

        print("\nMissing Values per Column:\n")
        print(df.isna().sum())

    if args.save:
        df.to_csv(args.save, index=False)
        print(f"\nDataset saved to: {args.save}")


if __name__ == "__main__":
    main()