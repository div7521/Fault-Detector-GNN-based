# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio   # install PyTorch first
pip install torch_geometric                # then PyG
pip install -r requirements.txt
```

On Apple Silicon (M1/M2/M3), MPS GPU is used automatically — no extra config needed.

## Dataset

Raw CSVs go in `data/raw/`. Download from Kaggle (`ieee-fraud-detection` competition): `train_transaction.csv` and `train_identity.csv`. These are gitignored, as are `data/processed/` and `models/saved/`.

```bash
# Fastest way to get data
kaggle competitions download -c ieee-fraud-detection -p data/raw/
unzip data/raw/ieee-fraud-detection.zip -d data/raw/
```

## Running the Pipeline

```bash
# Explore data first
jupyter notebook notebooks/01_eda.ipynb

# Train a specific model
python run.py --model baseline   # XGBoost only (fast, good for sanity check)
python run.py --model gat        # Graph Attention Network
python run.py --model sage       # GraphSAGE (inductive)
python run.py --model all        # all three + comparison chart

# Key flags
python run.py --model gat --epochs 50 --lr 0.001 --patience 10
python run.py --model gat --rebuild_graph   # force graph rebuild even if cached
```

Graph construction takes ~5 minutes the first time; the result is cached at `data/processed/hetero_graph.pt` and reloaded on subsequent runs.

## Architecture

### Data flow

`load_and_preprocess()` → `build_hetero_graph()` → `add_train_val_test_masks()` → `train()`

1. **`src/data/load.py`** — loads `train_transaction.csv` + `train_identity.csv`, left-joins on `TransactionID`, label-encodes categoricals, fills numeric NaNs with `-999`.

2. **`src/data/graph_builder.py`** — the most important file. Converts the tabular DataFrame into a PyG `HeteroData` graph with four node types:
   - `transaction` — one node per row, features from `TXN_FEATURES` + `IDENTITY_FEATURES`, label = `isFraud`
   - `card` — unique `card1` values; features: fraud rate, log count, log mean amount
   - `email` — unique `P_emaildomain` values; same 3 features
   - `device` — unique `DeviceInfo` values; same 3 features

   Edge types are bidirectional: `transaction → uses_card → card` and `card → used_in → transaction` (same pattern for email and device). Train/val/test split is time-aware (sorted by `TransactionDT`): 70/15/15.

3. **`src/models/gat.py`** — `FraudGAT` uses `HANConv` (Heterogeneous Attention Network) for two message-passing layers across all node and edge types, then classifies only `transaction` nodes. Use `build_gat(data)` to construct.

4. **`src/models/graphsage.py`** — `FraudSAGE` builds a homogeneous `_HomoSAGE` backbone (two `SAGEConv` layers) then calls `to_hetero()` to convert it to handle the heterogeneous graph. Inductive — works on unseen nodes. Use `build_sage(data)` to construct.

5. **`src/models/xgb_baseline.py`** — 5-fold stratified CV XGBoost with `scale_pos_weight` for class imbalance. `tune_with_optuna()` runs 30-trial Optuna hyperparameter search.

6. **`src/training/train.py`** — training loop using `FocalLoss` (alpha set from actual fraud rate, gamma=2). AdamW optimizer + cosine LR schedule. Gradient clipping at `max_norm=1.0`. Saves best checkpoint by val AUC. Auto-detects MPS / CUDA / CPU.

7. **`src/training/evaluate.py`** — `compute_metrics()` returns ROC-AUC, PR-AUC, F1, Precision, Recall, and Precision@{100,500,1000}. `compare_models()` saves a bar chart to `results/model_comparison.png`.

8. **`src/explain/gnn_explainer.py`** — `FraudExplainer` wraps PyG's `GNNExplainer`. `visualise_html()` generates an interactive pyvis subgraph saved to `results/fraud_subgraph.html`.

### Key design choices

- **Focal loss over cross-entropy** — fraud is ~3.5% of data; focal loss down-weights easy negatives.
- **GAT vs GraphSAGE** — GAT (transductive, learns attention weights) vs SAGE (inductive, fixed-size neighborhood sampling — preferred for production where new cards/emails appear daily).
- **Heterogeneous graph** — fraudsters reuse the same card/email/device; the GNN propagates labels through shared entity nodes, which flat XGBoost cannot do.
- **NaN encoding** — categoricals filled with `"MISSING"` before label encoding; numeric NaNs filled with `-999` to signal missingness to tree models.
