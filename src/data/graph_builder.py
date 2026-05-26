"""
graph_builder.py — Convert IEEE-CIS tabular data into a PyTorch Geometric graph.

This is the most important file in the project.

Graph design:
  Node types:
    - transaction  : each row is a transaction node (label = isFraud)
    - card         : unique card1 values (card identity)
    - email        : unique P_emaildomain values
    - device       : unique DeviceInfo values

  Edge types:
    - transaction → card      : transaction was made with this card
    - transaction → email     : transaction used this email domain
    - transaction → device    : transaction made from this device
    (all edges added as undirected by also adding the reverse)

Why this matters:
  A fraudster reuses the same card, email, or device across multiple transactions.
  The GNN propagates labels through these shared connections — which a flat
  XGBoost model cannot do.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
from pathlib import Path


# ── Node feature columns ─────────────────────────────────────────────────────

# Features used to describe each transaction node
TXN_FEATURES = [
    "TransactionAmt",
    "ProductCD",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "D11", "D12", "D13", "D14", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
] + [f"V{i}" for i in range(1, 340)]  # Vesta-engineered features — most predictive in this dataset

# Identity features (exist only for ~25% of transactions — fill others with 0)
IDENTITY_FEATURES = [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType"]


def _build_node_index(series: pd.Series) -> Dict:
    """Map unique non-missing values of a Series to integer node IDs."""
    unique_vals = [v for v in series.dropna().unique() if v != -999]
    return {val: idx for idx, val in enumerate(unique_vals)}


def _get_txn_features(df: pd.DataFrame) -> torch.Tensor:
    """Extract and normalise transaction node features."""
    cols = [c for c in TXN_FEATURES + IDENTITY_FEATURES if c in df.columns]
    X = df[cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return torch.tensor(X, dtype=torch.float)


def _get_entity_features(df: pd.DataFrame, col: str) -> Tuple[Dict, torch.Tensor]:
    """
    Build simple node features for a categorical entity (card / email / device).
    Features: fraud rate, transaction count, mean transaction amount.
    Returns (node_index dict, feature tensor).
    """
    node_index = _build_node_index(df[col])
    n = len(node_index)

    feat = np.zeros((n, 3), dtype=np.float32)

    grp = df[df[col].isin(node_index)].groupby(col).agg(
        fraud_rate=("isFraud", "mean"),
        count=(col, "count"),
        mean_amt=("TransactionAmt", "mean"),
    )
    mapped = grp.index.map(node_index).values
    feat[mapped, 0] = grp["fraud_rate"].values
    feat[mapped, 1] = np.log1p(grp["count"].values)
    feat[mapped, 2] = np.log1p(grp["mean_amt"].values)

    return node_index, torch.tensor(feat, dtype=torch.float)


def _build_edges(
    df: pd.DataFrame,
    col: str,
    node_index: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index for (transaction → entity) edges.
    Returns (src_txn_ids, dst_entity_ids) — both as 1-D tensors.
    Skips rows where the column value is missing or not in node_index.
    """
    valid = df[col].isin(node_index)
    src = torch.tensor(df.index[valid].tolist(), dtype=torch.long)
    dst = torch.tensor(df[col][valid].map(node_index).tolist(), dtype=torch.long)
    return src, dst


def build_hetero_graph(df: pd.DataFrame) -> HeteroData:
    """
    Main function: build a HeteroData graph from the merged IEEE-CIS DataFrame.

    Args:
        df: output of src.data.load.load_and_preprocess()

    Returns:
        HeteroData with node types: transaction, card, email, device
        and edge types connecting them bidirectionally.
    """
    print("Building heterogeneous graph...")
    df = df.reset_index(drop=True)  # ensure integer index = transaction position

    data = HeteroData()

    # ── Transaction nodes ────────────────────────────────────────────────────
    print("  Building transaction nodes...")
    data["transaction"].x = _get_txn_features(df)
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)
    data["transaction"].num_nodes = len(df)

    # ── Entity nodes + edges ─────────────────────────────────────────────────
    entity_cols = {
        "card":   "card1",          # most discriminative card feature
        "email":  "P_emaildomain",
        "device": "DeviceInfo",
    }

    for entity_name, col in entity_cols.items():
        if col not in df.columns:
            print(f"  Skipping {entity_name} — column {col} not found")
            continue

        print(f"  Building {entity_name} nodes from '{col}'...")
        node_index, feat = _get_entity_features(df, col)

        data[entity_name].x = feat
        data[entity_name].num_nodes = len(node_index)

        # Build edges: transaction → entity
        src, dst = _build_edges(df, col, node_index)
        edge_type = ("transaction", f"uses_{entity_name}", entity_name)
        rev_edge_type = (entity_name, f"used_in", "transaction")

        data[edge_type].edge_index = torch.stack([src, dst], dim=0)
        # Add reverse edges so message passing goes both ways
        data[rev_edge_type].edge_index = torch.stack([dst, src], dim=0)

        print(f"    {entity_name}: {len(node_index)} nodes, {len(src)} edges")

    print(f"\nGraph summary:")
    print(f"  Transaction nodes : {data['transaction'].num_nodes:,}")
    for name in entity_cols:
        if name in data.node_types:
            print(f"  {name:16s} nodes : {data[name].num_nodes:,}")
    print(f"  Edge types        : {data.edge_types}")

    return data


def add_train_val_test_masks(
    data: HeteroData,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> HeteroData:
    """
    Add train/val/test boolean masks to transaction nodes.
    Uses a time-aware split: earlier transactions → train, later → test.
    (Assumes df was sorted by TransactionDT before building the graph.)
    """
    n = data["transaction"].num_nodes
    indices = torch.arange(n)

    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[:train_end]        = True
    val_mask[train_end:val_end]   = True
    test_mask[val_end:]           = True

    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask   = val_mask
    data["transaction"].test_mask  = test_mask

    print(f"Masks: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    return data


if __name__ == "__main__":
    from src.data.load import load_and_preprocess

    df = load_and_preprocess()
    # Sort by time for temporal split
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    graph = build_hetero_graph(df)
    graph = add_train_val_test_masks(graph)

    # Save for reuse (graph construction is slow)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    torch.save(graph, "data/processed/hetero_graph.pt")
    print("\nSaved to data/processed/hetero_graph.pt")
