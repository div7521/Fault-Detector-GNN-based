"""
GraphSAGE — Inductive graph learning for fraud detection.

Key advantage over GCN/GAT:
  GraphSAGE SAMPLES a fixed-size neighborhood at each layer instead of
  using all neighbors. This means:
    1. Scales to massive graphs (millions of nodes)
    2. Works on UNSEEN nodes at inference time (inductive)
    3. Faster training via mini-batching with NeighborLoader

In production fraud detection, new cards and emails appear every day —
inductive learning is essential.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero, Linear
from torch_geometric.data import HeteroData


class _HomoSAGE(nn.Module):
    """
    Homogeneous GraphSAGE backbone — GNN layers only.
    Classifier is intentionally excluded so to_hetero() only wraps the
    message-passing layers, giving each node type its own SAGE weights.
    """

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x


class FraudSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE using PyG's to_hetero conversion.

    to_hetero wraps a homogeneous GNN to apply separate weight matrices
    per node type and edge type — cleanest way to handle HeteroData.
    """

    def __init__(
        self,
        metadata,
        in_channels: dict,
        hidden_dim: int = 64,
        out_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Project each node type to the same hidden_dim
        self.input_proj = nn.ModuleDict({
            node_type: Linear(dim, hidden_dim)
            for node_type, dim in in_channels.items()
        })

        homo_model = _HomoSAGE(hidden_dim, hidden_dim, dropout)
        self.gnn = to_hetero(homo_model, metadata, aggr="sum")

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        # Project to common dim
        h = {
            node_type: F.relu(self.input_proj[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.input_proj
        }

        out_dict = self.gnn(h, edge_index_dict)
        return self.classifier(out_dict["transaction"])


def build_sage(data: HeteroData, hidden_dim: int = 64, dropout: float = 0.3) -> FraudSAGE:
    """Convenience constructor that infers dims from a HeteroData object."""
    in_channels = {
        node_type: data[node_type].x.shape[1]
        for node_type in data.node_types
        if hasattr(data[node_type], 'x')
    }
    return FraudSAGE(
        metadata=data.metadata(),
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_dim=2,
        dropout=dropout,
    )
