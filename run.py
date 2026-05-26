import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

"""
run.py — End-to-end training pipeline.

Usage:
    python run.py --model gat        # train GAT
    python run.py --model sage       # train GraphSAGE
    python run.py --model baseline   # train XGBoost baseline only
    python run.py --model all        # run all three and compare
"""

import argparse
import torch
from pathlib import Path

from src.data.load import load_and_preprocess
from src.data.graph_builder import build_hetero_graph, add_train_val_test_masks
from src.models.gat import build_gat
from src.models.graphsage import build_sage
from src.models.xgb_baseline import train_baseline
from src.training.train import train
from src.training.evaluate import print_metrics, compare_models


def main(args):
    print("=" * 60)
    print("GNN Fraud Detection — IEEE-CIS Dataset")
    print("=" * 60)

    # ── Step 1: Load and preprocess ──────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df = load_and_preprocess()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # ── Step 2: Build graph (or load cached) ─────────────────────────────────
    data = None
    if args.model != "baseline":
        graph_path = Path("data/processed/hetero_graph.pt")
        if graph_path.exists() and not args.rebuild_graph:
            print(f"\n[2/4] Loading cached graph from {graph_path}")
            data = torch.load(graph_path, weights_only=False)
        else:
            print("\n[2/4] Building heterogeneous graph (this takes ~5 mins)...")
            data = build_hetero_graph(df)
            data = add_train_val_test_masks(data)
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, graph_path)
            print(f"Cached graph to {graph_path}")
    else:
        print("\n[2/4] Skipping graph load (baseline only)")

    # ── Step 3: Train models ──────────────────────────────────────────────────
    all_results = {}

    if args.model in ("baseline", "all"):
        print("\n[3a/4] Training XGBoost baseline...")
        xgb_results = train_baseline(df)
        from sklearn.metrics import f1_score as _f1
        oof_binary = (xgb_results["oof_preds"] >= 0.5).astype(int)
        all_results["XGBoost"] = {
            "auc": xgb_results["oof_auc"],
            "ap":  xgb_results["oof_ap"],
            "f1":  _f1(df["isFraud"].values, oof_binary, zero_division=0),
        }
        print_metrics(all_results["XGBoost"], split_name="XGBoost OOF")

    if args.model in ("gat", "all"):
        print("\n[3b/4] Training GAT...")
        gat_model = build_gat(data, hidden_dim=128, heads=4, dropout=0.2)
        gat_results = train(
            gat_model, data,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            save_path="models/saved/gat_best.pt",
        )
        all_results["GAT"] = gat_results["test_metrics"]
        print_metrics(gat_results["test_metrics"], split_name="GAT Test")

    if args.model in ("sage", "all"):
        print("\n[3c/4] Training GraphSAGE...")
        sage_model = build_sage(data, hidden_dim=64, dropout=0.3)
        sage_results = train(
            sage_model, data,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            save_path="models/saved/sage_best.pt",
        )
        all_results["GraphSAGE"] = sage_results["test_metrics"]
        print_metrics(sage_results["test_metrics"], split_name="GraphSAGE Test")

    # ── Step 4: Compare ───────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n[4/4] Comparing models...")
        compare_models(all_results, save_path="results/model_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default="all",
                        choices=["gat", "sage", "baseline", "all"])
    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int, default=15)
    parser.add_argument("--rebuild_graph", action="store_true",
                        help="Rebuild graph even if cached version exists")
    args = parser.parse_args()
    main(args)
