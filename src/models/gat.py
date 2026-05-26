"""
GAT — Graph Attention Network for fraud detection.

Architecture:
  - HeteroData input (transaction, card, email, device nodes)
  - HANConv (Heterogeneous Attention Network) layers to handle multiple node types
  - Two-layer design with dropout + residual connection
  - Output: binary classification on transaction nodes (fraud / not-fraud)

Why GAT over plain GCN:
  GCN averages ALL neighbor messages equally.
  GAT learns which neighbors are MORE important via attention weights.
  In fraud: a card node connected to many fraud transactions should
  have higher attention weight than a card with one suspicious transaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, Linear
from torch_geometric.data import HeteroData


class FraudGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network for fraud detection.

    Args:
        metadata:    output of data.metadata() — node types + edge types
        in_channels: dict mapping node_type → input feature dim
        hidden_dim:  hidden dimension (default 64)
        out_dim:     output classes (2 for binary fraud)
        heads:       number of attention heads
        dropout:     dropout probability
    """

    def __init__(
        self,
        metadata,
        in_channels: dict,
        hidden_dim: int = 64,
        out_dim: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection: different node types may have different feature dims
        self.input_proj = nn.ModuleDict({
            node_type: Linear(dim, hidden_dim)
            for node_type, dim in in_channels.items()
        })

        # HANConv: heterogeneous attention — one attention head per edge type
        self.conv1 = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
        )

        self.conv2 = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            metadata=metadata,
            heads=1,
            dropout=dropout,
        )

        # Classification head — only applied to transaction nodes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // heads, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """
        Args:
            x_dict:          {node_type: feature_tensor}
            edge_index_dict: {edge_type: edge_index_tensor}

        Returns:
            logits for transaction nodes — shape (n_transactions, 2)
        """
        # Project all node types to common hidden_dim
        h = {
            node_type: F.relu(self.input_proj[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.input_proj
        }

        # Graph attention message passing
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}

        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}

        # Classify only transaction nodes
        out = self.classifier(h["transaction"])
        return out


def build_gat(data: HeteroData, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.3) -> FraudGAT:
    """Convenience constructor that infers dims from a HeteroData object."""
    in_channels = {
        node_type: data[node_type].x.shape[1]
        for node_type in data.node_types
        if hasattr(data[node_type], 'x')
    }
    return FraudGAT(
        metadata=data.metadata(),
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_dim=2,
        heads=heads,
        dropout=dropout,
    )
