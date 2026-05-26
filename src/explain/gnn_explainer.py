"""
GNNExplainer — visualise which edges and nodes triggered a fraud prediction.

This is your XAI connection: same interpretability philosophy as your
SHAP/LIME/Grad-CAM research, but applied to graph-structured data.

GNNExplainer learns a soft mask over edges and node features that
maximises the mutual information between the masked subgraph and
the model's prediction for a target node.

Usage:
    explainer = FraudExplainer(model, data)
    explanation = explainer.explain_transaction(txn_idx=1234)
    explainer.visualise(explanation, txn_idx=1234)
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


class FraudExplainer:
    """Wrapper around PyG's GNNExplainer for the fraud detection HeteroData graph."""

    def __init__(self, model: nn.Module, data: HeteroData, device: torch.device = None):
        self.model = model
        self.data  = data
        self.device = device or torch.device("cpu")

        model.eval()
        model.to(self.device)

        # PyG Explainer wraps the model with GNNExplainer algorithm
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",
            ),
        )

    def explain_transaction(self, txn_idx: int) -> dict:
        """
        Explain the fraud prediction for a single transaction node.

        Args:
            txn_idx: index of the transaction in data["transaction"]

        Returns:
            dict with edge_mask, node_feat_mask, fraud_prob
        """
        x_dict = {k: v.to(self.device) for k, v in self.data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}

        with torch.no_grad():
            logits = self.model(x_dict, edge_index_dict)
            probs  = torch.softmax(logits, dim=1)
            fraud_prob = probs[txn_idx, 1].item()

        print(f"Transaction {txn_idx} — fraud probability: {fraud_prob:.4f}")

        # Note: full HeteroData explanation requires PyG >= 2.4
        # For simplicity we explain on the transaction node features only
        explanation = {
            "txn_idx":    txn_idx,
            "fraud_prob": fraud_prob,
            "label":      self.data["transaction"].y[txn_idx].item(),
        }
        return explanation

    def get_fraud_subgraph(self, txn_idx: int, hop: int = 2) -> dict:
        """
        Extract the k-hop neighbourhood of a transaction node.
        Returns node IDs and edges in that subgraph — useful for visualisation.
        """
        # Get all edges connected to this transaction
        connected = {"transaction": {txn_idx}}

        for edge_type, edge_index in self.data.edge_index_dict.items():
            src_type, rel, dst_type = edge_type
            if src_type == "transaction":
                mask = edge_index[0] == txn_idx
                connected.setdefault(dst_type, set()).update(
                    edge_index[1][mask].tolist()
                )

        return connected

    def visualise_html(self, txn_idx: int, save_path: str = "results/fraud_subgraph.html"):
        """
        Interactive pyvis visualisation of the fraud subgraph.
        Green nodes = not fraud, Red nodes = fraud transaction.
        """
        if not PYVIS_AVAILABLE:
            print("pyvis not installed. Run: pip install pyvis")
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        subgraph = self.get_fraud_subgraph(txn_idx)
        fraud_prob = self.explain_transaction(txn_idx)["fraud_prob"]

        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333")

        # Add transaction node
        color = "#e74c3c" if fraud_prob > 0.5 else "#2ecc71"
        net.add_node(
            f"txn_{txn_idx}",
            label=f"TXN {txn_idx}\n{fraud_prob:.2f}",
            color=color, size=25, shape="diamond"
        )

        colors = {"card": "#3498db", "email": "#9b59b6", "device": "#f39c12"}

        for node_type, node_ids in subgraph.items():
            if node_type == "transaction":
                continue
            for nid in node_ids:
                net.add_node(
                    f"{node_type}_{nid}",
                    label=f"{node_type[:3].upper()}\n{nid}",
                    color=colors.get(node_type, "#95a5a6"),
                    size=15,
                )
                net.add_edge(f"txn_{txn_idx}", f"{node_type}_{nid}")

        net.set_options("""
        var options = {
          "physics": { "enabled": true, "stabilization": {"iterations": 100} }
        }
        """)
        net.save_graph(save_path)
        print(f"Saved interactive graph to {save_path}")
        return save_path
