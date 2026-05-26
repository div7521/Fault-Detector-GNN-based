"""
Training loop for GNN fraud detection models.

Key design decisions:
  - Focal loss instead of cross-entropy (fraud is only ~3.5% of data)
  - Trains only on transaction nodes (not card/email/device nodes)
  - Saves best model by validation AUC
  - Logs train/val metrics every epoch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from pathlib import Path
import time
from typing import Optional

from src.training.evaluate import compute_metrics


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced binary classification.
    Down-weights easy negatives so the model focuses on hard fraud cases.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard from the original paper (Lin et al., 2017).
    alpha handles class imbalance (set to inverse class frequency).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Single training epoch. Returns training loss."""
    model.train()

    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    y = data["transaction"].y.to(device)
    mask = data["transaction"].train_mask.to(device)

    optimizer.zero_grad()
    logits = model(x_dict, edge_index_dict)
    loss = criterion(logits[mask], y[mask])
    loss.backward()

    # Gradient clipping — important for stability with attention layers
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: HeteroData,
    mask_name: str,
    device: torch.device,
) -> dict:
    """Evaluate on val or test split. Returns metrics dict."""
    model.eval()

    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    y = data["transaction"].y.to(device)
    mask = getattr(data["transaction"], mask_name).to(device)

    logits = model(x_dict, edge_index_dict)
    probs = torch.softmax(logits, dim=1)[:, 1]

    return compute_metrics(
        y[mask].cpu().numpy(),
        probs[mask].cpu().numpy(),
    )


def train(
    model: nn.Module,
    data: HeteroData,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    save_path: str = "models/saved/best_gnn.pt",
    device: Optional[torch.device] = None,
) -> dict:
    """
    Full training loop with early stopping.

    Args:
        model:        FraudGAT or FraudSAGE instance
        data:         HeteroData with train/val/test masks
        epochs:       max training epochs
        lr:           learning rate
        weight_decay: L2 regularisation
        patience:     early stopping patience (epochs without val AUC improvement)
        save_path:    where to save the best model weights
        device:       cpu / mps (Apple Silicon) / cuda

    Returns:
        dict with training history and best val metrics
    """
    if device is None:
        # MPS disabled: HANConv produces silently wrong gradients on MPS with
        # Python 3.14 due to torch.jit.script deprecation in PyG internals.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"Training on: {device}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Weighted cross-entropy — equivalent to XGBoost's scale_pos_weight.
    # Focal loss caused mode collapse at extreme imbalance (3.5% fraud);
    # weighted CE is simpler and converges reliably.
    y_train = data["transaction"].y[data["transaction"].train_mask]
    n_neg = (y_train == 0).sum().float()
    n_pos = (y_train == 1).sum().float()
    class_weight = torch.tensor([1.0, n_neg / n_pos]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_auc": [], "val_ap": []}
    best_val_auc = 0.0
    no_improve = 0

    print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Val AUC':>8} | {'Val AP':>8} | {'Time':>6}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        loss = train_one_epoch(model, data, optimizer, criterion, device)
        val_metrics = evaluate(model, data, "val_mask", device)
        scheduler.step()

        history["train_loss"].append(loss)
        history["val_auc"].append(val_metrics["auc"])
        history["val_ap"].append(val_metrics["ap"])

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {loss:>8.4f} | {val_metrics['auc']:>8.4f} | {val_metrics['ap']:>8.4f} | {elapsed:>5.1f}s")

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best weights and evaluate on test set
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, data, "test_mask", device)

    print(f"\n{'='*50}")
    print(f"Best Val AUC : {best_val_auc:.4f}")
    print(f"Test AUC     : {test_metrics['auc']:.4f}")
    print(f"Test AP      : {test_metrics['ap']:.4f}")
    print(f"Test F1      : {test_metrics['f1']:.4f}")

    return {"history": history, "best_val_auc": best_val_auc, "test_metrics": test_metrics}
