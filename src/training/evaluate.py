"""
Evaluation metrics for fraud detection.

Important: don't just report ROC-AUC.
Fraud datasets are heavily imbalanced — ROC-AUC can look great even if
the model flags very few fraud cases correctly.

Report ALL of these:
  - ROC-AUC   : overall discrimination ability
  - PR-AUC    : precision-recall — more informative for imbalanced data
  - F1        : balance of precision and recall at 0.5 threshold
  - Precision@K: of the top-K flagged transactions, how many are actually fraud?
                 (most relevant for ops teams who review a fixed number of alerts)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute all fraud detection metrics.

    Args:
        y_true:    ground truth labels (0/1)
        y_proba:   predicted fraud probabilities
        threshold: classification threshold

    Returns:
        dict with auc, ap, f1, precision, recall, precision@100, precision@500
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "auc":       roc_auc_score(y_true, y_proba),
        "ap":        average_precision_score(y_true, y_proba),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }

    # Precision@K — sort by predicted probability, take top K
    sorted_idx = np.argsort(y_proba)[::-1]
    for k in [100, 500, 1000]:
        if k <= len(y_true):
            top_k_labels = y_true[sorted_idx[:k]]
            metrics[f"precision@{k}"] = top_k_labels.mean()

    return metrics


def print_metrics(metrics: dict, split_name: str = "Test"):
    print(f"\n{'='*40}")
    print(f"{split_name} metrics")
    print(f"{'='*40}")
    print(f"  ROC-AUC        : {metrics['auc']:.4f}")
    print(f"  PR-AUC         : {metrics['ap']:.4f}")
    print(f"  F1             : {metrics['f1']:.4f}")
    if "precision" in metrics:
        print(f"  Precision      : {metrics['precision']:.4f}")
    if "recall" in metrics:
        print(f"  Recall         : {metrics['recall']:.4f}")
    for k in [100, 500, 1000]:
        key = f"precision@{k}"
        if key in metrics:
            print(f"  Precision@{k:<4} : {metrics[key]:.4f}")


def compare_models(results: dict, save_path: str = "results/model_comparison.png"):
    """
    Plot a bar chart comparing AUC and PR-AUC across models.

    Args:
        results: {model_name: metrics_dict}
        save_path: where to save the figure
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_names = list(results.keys())
    aucs = [results[m]["auc"] for m in model_names]
    aps  = [results[m]["ap"]  for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, aucs, width, label="ROC-AUC",  color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width/2, aps,  width, label="PR-AUC",   color="tomato",    alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — ROC-AUC vs PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")
    return fig
