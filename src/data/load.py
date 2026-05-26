"""
Load and preprocess the IEEE-CIS Fraud Detection dataset.

Dataset: https://www.kaggle.com/competitions/ieee-fraud-detection
Two files:
  - train_transaction.csv  (590k rows, 394 features)
  - train_identity.csv     (144k rows, 41 features)
Join on TransactionID. ~3.5% of transactions are fraudulent.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

DATA_DIR = Path("data/raw")


def load_raw(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw transaction and identity CSVs."""
    print("Loading transaction data...")
    train_txn = pd.read_csv(data_dir / "train_transaction.csv")

    print("Loading identity data...")
    train_id = pd.read_csv(data_dir / "train_identity.csv")

    print(f"Transactions: {train_txn.shape} | Identity: {train_id.shape}")
    return train_txn, train_id


def merge_tables(
    txn: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    """Left join transactions with identity on TransactionID."""
    df = txn.merge(identity, on="TransactionID", how="left")
    print(f"Merged shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.4f} ({df['isFraud'].sum()} fraud / {len(df)} total)")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode object columns and fill NaNs.
    Categorical columns in IEEE-CIS: ProductCD, card4, card6,
    P_emaildomain, R_emaildomain, M1-M9, DeviceType, DeviceInfo, id_12-id_38.
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"Encoding {len(cat_cols)} categorical columns...")

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING")
        df[col] = le.fit_transform(df[col])

    # Fill remaining numeric NaNs with -999 (signals missingness to tree models)
    df = df.fillna(-999)

    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Return named groups of features — useful for graph construction.

    Groups:
      - card:       card1-card6 (payment card attributes)
      - email:      P_emaildomain, R_emaildomain (purchaser / recipient)
      - device:     DeviceType, DeviceInfo
      - time:       TransactionDT
      - amount:     TransactionAmt
      - identity:   id_01 through id_38
      - vesta:      V1-V339 (Vesta engineered features, mostly anonymous)
    """
    all_cols = df.columns.tolist()
    return {
        "card":     [c for c in all_cols if c.startswith("card")],
        "email":    [c for c in all_cols if "emaildomain" in c],
        "device":   [c for c in all_cols if c.startswith("Device")],
        "time":     ["TransactionDT"],
        "amount":   ["TransactionAmt"],
        "identity": [c for c in all_cols if c.startswith("id_")],
        "vesta":    [c for c in all_cols if c.startswith("V")],
        "match":    [c for c in all_cols if c.startswith("M")],
    }


def load_and_preprocess(data_dir: Path = DATA_DIR, max_rows: int = None) -> pd.DataFrame:
    """Full pipeline: load → merge → encode → return clean DataFrame."""
    txn, identity = load_raw(data_dir)
    df = merge_tables(txn, identity)
    if max_rows and len(df) > max_rows:
        # Stratify on isFraud to preserve fraud rate in the sample
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0]
        fraud_rate = len(fraud) / len(df)
        n_fraud = int(max_rows * fraud_rate)
        n_legit = max_rows - n_fraud
        df = pd.concat([
            fraud.sample(n=min(n_fraud, len(fraud)), random_state=42),
            legit.sample(n=min(n_legit, len(legit)), random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows (fraud rate: {df['isFraud'].mean():.4f})")
    df = encode_categoricals(df)
    return df


if __name__ == "__main__":
    df = load_and_preprocess()
    print("\nSample:")
    print(df[["TransactionID", "TransactionAmt", "isFraud"]].head())
    print(f"\nFeature groups: {list(get_feature_groups(df).keys())}")
