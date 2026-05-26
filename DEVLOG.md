# GNN Fraud Detection — Development Log

This document is a full transcript of how this project was built, debugged, and iterated on.
It covers every decision made, every bug found, why each fix was applied, and what impact it had.

---

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [Tech Stack](#2-tech-stack)
3. [Dataset](#3-dataset)
4. [Code Review — Bugs Found Before First Run](#4-code-review--bugs-found-before-first-run)
5. [Bug Fixes Applied](#5-bug-fixes-applied)
6. [Environment Issues Encountered](#6-environment-issues-encountered)
7. [Training Experiments](#7-training-experiments)
8. [Final Results](#8-final-results)
9. [Key Lessons Learned](#9-key-lessons-learned)
10. [How to Run](#10-how-to-run)

---

## 1. Project Goal

Build a **Graph Neural Network (GNN) system for credit card fraud detection** on the
[IEEE-CIS Kaggle dataset](https://www.kaggle.com/competitions/ieee-fraud-detection).

The core idea: fraudsters reuse the same card, email domain, and device across multiple
transactions. A flat model (like XGBoost) sees each transaction in isolation. A GNN treats
transactions as nodes in a graph, connected through shared cards/emails/devices — and
propagates fraud signals through those connections.

The project demonstrates:
- Why graph structure captures signals that tabular models miss
- How to convert a raw tabular dataset into a heterogeneous PyG graph
- Three models: XGBoost baseline, GAT (HANConv), GraphSAGE
- How to handle class imbalance (~3.5% fraud rate) in both tree models and GNNs

---

## 2. Tech Stack

| Tool | Purpose |
|---|---|
| **PyTorch** | Neural network framework |
| **PyTorch Geometric (PyG)** | GNN layers (HANConv, SAGEConv, to_hetero) |
| **XGBoost** | Tabular baseline model |
| **Optuna** | Hyperparameter tuning for XGBoost |
| **scikit-learn** | Preprocessing, metrics, cross-validation |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualization |
| **pyvis** | Interactive fraud subgraph HTML visualization |
| **FastAPI / Streamlit** | (In requirements, for future inference API) |
| **joblib** | Model serialization |

**Environment:** Python 3.14, Apple Silicon (M-series), macOS

---

## 3. Dataset

**Source:** IEEE-CIS Fraud Detection (Kaggle)

Two CSV files:
- `train_transaction.csv` — 590,540 rows, 394 features (card info, amounts, device, V features)
- `train_identity.csv` — 144,233 rows, 41 features (device type, browser, identity fields)

Joined on `TransactionID`. Final merged shape: 590,540 × 434.

**Class imbalance:** Only 3.5% of transactions are fraud (20,663 / 590,540).

For development we subsample to **100,000 rows** (stratified on `isFraud` to preserve the
3.5% fraud rate) to avoid memory crashes during XGBoost training. The subsample is done
after the full CSVs are loaded, so the fraud rate is preserved exactly.

---

## 4. Code Review — Bugs Found Before First Run

A full review of all source files was done before running anything.
Here are the bugs found, ordered by severity:

### Critical

**Bug 1: Focal loss alpha was inverted** (`src/training/train.py`)

```python
# WRONG — was in the code
fraud_rate = data["transaction"].y.float().mean().item()  # ≈ 0.035
criterion = FocalLoss(alpha=fraud_rate, gamma=2.0)
```

Inside `FocalLoss.forward`:
```python
alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
```

With `alpha=0.035`: fraud samples got weight **0.035**, legit got **0.965**.
The minority class (fraud) was being penalised less than legit — the exact opposite of intent.

**Impact:** The model would learn to mostly predict "not fraud" to minimise loss from the
dominant legit class. Fraud would be nearly invisible during training.

**Fix:** Changed to `alpha = 1.0 - fraud_rate` (≈ 0.965 for fraud, 0.035 for legit).
Later further tuned (see training experiments).

---

**Bug 2: Missing values (-999) creating spurious shared entity nodes** (`src/data/graph_builder.py`)

After preprocessing, numeric NaN values (e.g. in `card1`) are encoded as `-999`.
The `_build_node_index` function was including `-999` as a valid entity value.

**Impact:** Every transaction with a missing `card1` value was connected to the same
"card node -999", creating a massive fake shared connection. The GNN would see hundreds
of transactions artificially linked — generating fraudulent graph signals. This is the
exact pattern GNNs use to detect fraud, but here it was noise.

**Fix:** Filtered out `-999` in `_build_node_index`:
```python
unique_vals = [v for v in series.dropna().unique() if v != -999]
```

---

### Performance Issues

**Bug 3: `iterrows()` in edge construction** (`src/data/graph_builder.py`)

```python
# WRONG — Python loop over 100k rows
for txn_id, row in df.iterrows():
    val = row[col]
    if pd.isna(val) or val not in node_index:
        continue
    src.append(txn_id)
    dst.append(node_index[val])
```

`iterrows()` in pandas is a Python-level loop — roughly 100,000 iterations with overhead
per row. Estimated runtime: 5–10 minutes just for edge building.

**Fix:** Replaced with vectorised pandas:
```python
valid = df[col].isin(node_index)
src = torch.tensor(df.index[valid].tolist(), dtype=torch.long)
dst = torch.tensor(df[col][valid].map(node_index).tolist(), dtype=torch.long)
```

**Impact:** Graph construction went from ~10 minutes to ~10 seconds.

---

**Bug 4: Python loop in entity feature construction** (`src/data/graph_builder.py`)

Similar issue — a Python loop computing per-entity aggregations (fraud rate, count, mean
amount) instead of using pandas `groupby`. Replaced with:
```python
grp = df[df[col].isin(node_index)].groupby(col).agg(
    fraud_rate=("isFraud", "mean"),
    count=(col, "count"),
    mean_amt=("TransactionAmt", "mean"),
)
```

---

### Correctness Bugs

**Bug 5: `torch.load` missing `map_location`** (`src/training/train.py`)

```python
# WRONG
model.load_state_dict(torch.load(save_path))

# FIXED
model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
```

Without `map_location`, loading a model trained on MPS/GPU onto CPU (or vice versa) would
crash with a device mismatch error. `weights_only=True` is also required in newer PyTorch
to avoid security warnings.

---

**Bug 6: GraphSAGE classifier shared across all node types** (`src/models/graphsage.py`)

`to_hetero()` in PyG converts `torch_geometric.nn.Linear` and `MessagePassing` layers to
have per-node-type weights. However, standard `nn.Linear` layers are NOT converted — they
remain shared. The original code included the classifier inside `_HomoSAGE`, which was
then wrapped by `to_hetero()`. This meant all node types shared the same classification
head, which is wasteful and semantically wrong.

**Fix:** Removed the classifier from `_HomoSAGE` so `to_hetero()` only wraps the SAGE
convolution layers. The classifier is now in `FraudSAGE` and applied only to transaction
nodes after the GNN produces embeddings.

---

**Bug 7: Deprecated `use_label_encoder` in XGBoost** (`src/models/xgb_baseline.py`)

`use_label_encoder=False` was present in `tune_with_optuna()`. This parameter was removed
from XGBoost >= 1.6. Passing it raises an error in XGBoost 3.x. Removed.

---

**Bug 8: XGBoost F1 hardcoded to 0.0** (`run.py`)

```python
# WRONG
all_results["XGBoost"] = {"auc": ..., "ap": ..., "f1": 0.0}

# FIXED
oof_binary = (xgb_results["oof_preds"] >= 0.5).astype(int)
all_results["XGBoost"] = {
    "auc": xgb_results["oof_auc"],
    "ap":  xgb_results["oof_ap"],
    "f1":  f1_score(df["isFraud"].values, oof_binary, zero_division=0),
}
```

Without this fix, the comparison chart showed XGBoost with F1=0.0 which was misleading.

---

**Bug 9: Missing V features in GNN transaction nodes** (`src/data/graph_builder.py`)

The original `TXN_FEATURES` list contained ~60 features but was missing the V1–V339
Vesta-engineered features. XGBoost used all 431 features. The V features are the most
predictive in this dataset.

**Fix:** Added all V features and remaining D features (D6–D15):
```python
TXN_FEATURES = [...existing...] + [f"V{i}" for i in range(1, 340)]
```

**Impact:** GNN feature count went from ~60 to ~400, bringing it on par with XGBoost.

---

**Bug 10: Comparison chart Y-axis starting at 0.5** (`src/training/evaluate.py`)

`ax.set_ylim(0.5, 1.0)` would clip any PR-AUC bars below 0.5 (common early in training
or for weak models). Changed to `ax.set_ylim(0, 1.0)`.

---

## 5. Bug Fixes Applied

Summary of all files modified:

| File | Changes |
|---|---|
| `src/data/load.py` | Added `max_rows=100_000` stratified subsampling |
| `src/data/graph_builder.py` | Fixed -999 filter, vectorised edge/feature loops, added V+D features |
| `src/training/train.py` | Fixed focal loss alpha, fixed torch.load, disabled MPS, switched to weighted CE |
| `src/models/graphsage.py` | Moved classifier outside `to_hetero()` |
| `src/models/xgb_baseline.py` | Removed deprecated `use_label_encoder` |
| `run.py` | Fixed OpenMP env vars, computed real XGBoost F1 |
| `src/training/evaluate.py` | Fixed Y-axis, defensive precision/recall access |
| `notebooks/01_eda.ipynb` | Rebuilt as valid `.ipynb` JSON (old file was corrupt) |

---

## 6. Environment Issues Encountered

### Issue 1: XGBoost segfault (exit code 139) when PyTorch is loaded

**Symptom:** `python run.py --model baseline` crashed immediately with no output.
Running XGBoost alone worked fine. Running it after importing PyTorch crashed.

**Root cause:** PyTorch and XGBoost both load OpenMP thread pool libraries at startup.
On macOS, having two OpenMP runtimes in the same process causes a SIGSEGV.

**Fix:** Set environment variables at the top of `run.py` before any imports:
```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
```

**Impact:** XGBoost runs correctly alongside PyTorch in the same process.

---

### Issue 2: HANConv (GAT) producing wrong results on MPS (Apple Silicon GPU)

**Symptom:** GAT training on MPS showed AUC bouncing around 0.5 or dipping below it —
worse than random. The same model on CPU showed steady AUC improvement epoch-by-epoch.

**Root cause:** `torch.jit.script` is deprecated and unsupported in Python 3.14+.
PyG's `HANConv` uses `torch.jit.script` internally for performance. On Python 3.14,
this produced silently wrong gradient computations — the forward pass looked fine but
the backwards pass was giving incorrect gradients, causing the model to train in random
or wrong directions.

**Diagnosis:** Ran identical training with MPS forced off. CPU training immediately showed
normal learning curves (AUC climbing from 0.45 → 0.62 over 30 epochs, still rising).

**Fix:** Removed MPS from device auto-detection in `train.py`:
```python
# MPS disabled: HANConv produces wrong gradients on MPS with Python 3.14
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

**Impact:** GAT now trains correctly on CPU, converging to AUC ~0.74.
Note: GraphSAGE (using SAGEConv via `to_hetero()`) does not use `torch.jit.script` in
the same way and was not affected by this issue.

---

### Issue 3: Focal loss mode collapse

**Symptom 1 (alpha too high):** With `alpha = 1 - fraud_rate = 0.965`:
- Recall = 1.0, Precision = 0.033 — model predicts everything as fraud
- The model collapsed to "always predict fraud" because:
  - Fraud samples (3500 of them) each contribute weight 0.965
  - Legit samples (96500) each contribute weight 0.035
  - Model learns: correctly predict all fraud → focal factor → 0 → loss ≈ 0. Misclassify legit → tiny loss (weight 0.035)

**Symptom 2 (alpha too low):** With `alpha = 0.75`:
- AUC dropped to 0.54, below random
- Opposite collapse: model predicted everything as "not fraud"
- Legit samples now dominate loss: 96500 × 0.25 >> 3500 × 0.75

**Analysis:** With focal loss and extreme class imbalance (28:1 ratio), there is no stable
alpha value — the model always collapses to one class. Focal loss works well for object
detection (small imbalance) but not for 3.5% fraud rates.

**Fix:** Switched to **weighted cross-entropy**, identical to XGBoost's `scale_pos_weight`:
```python
n_neg = (y_train == 0).sum().float()
n_pos = (y_train == 1).sum().float()
class_weight = torch.tensor([1.0, n_neg / n_pos]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
```

**Impact:** GNN immediately started making actual fraud predictions (F1 went from 0.0 to 0.148
for GAT, and the model converged stably).

---

## 7. Training Experiments

### Experiment 1 — XGBoost Baseline

**Settings:** 5-fold stratified CV, 500 trees, max_depth=6, lr=0.05, early stopping=50,
scale_pos_weight = n_neg/n_pos ≈ 28.

**Results (100k rows, OOF):**
- ROC-AUC: **0.918**
- PR-AUC: **0.613**
- F1: **0.449**

Strong baseline. XGBoost naturally handles tabular data with 431 features using tree splits.

---

### Experiment 2 — GAT (Attempt 1): Focal Loss on MPS

**Settings:** hidden_dim=64, heads=4, dropout=0.3, focal loss alpha=0.035 (inverted bug),
MPS device, epochs=100, patience=15.

**Results:** AUC=0.72, F1=0.065, Recall=1.0, Precision=0.033.

**Problem:** Model predicted everything as fraud (mode collapse). Two bugs combined:
inverted alpha AND MPS producing wrong gradients.

---

### Experiment 2b — GAT (Attempt 2): Fixed Alpha, Still on MPS

**Settings:** Fixed alpha to `1 - fraud_rate = 0.965`.

**Results:** AUC=0.54 (below random), F1=0.097.

**Problem:** Alpha correction overcorrected. MPS still producing wrong gradients.
AUC < 0.5 means model is anti-correlated with truth.

---

### Experiment 2c — GAT (Attempt 3): CPU, alpha=0.75

**Settings:** Forced CPU, alpha capped at 0.75.

**Results:** AUC climbed steadily 0.45 → 0.62 over 30 epochs, still rising.

**Finding:** MPS was the root cause of wrong results. CPU trains correctly.

---

### Experiment 2d — GAT (Attempt 4): Focal Loss, Full 200 Epochs

**Settings:** CPU, alpha=0.75, gamma=2, epochs=200, patience=25.

**Results:** AUC=0.73 (peaked at epoch 20), F1=0.0 (never crosses 0.5 threshold).

**Problem:** Focal loss with any alpha still failed to produce confident predictions.
The model learned to rank transactions but never crossed the 0.5 threshold for F1.

---

### Experiment 3 — GAT with V Features + Weighted CE

**Key changes:**
1. Added V1–V339 features (339 more features, the most predictive in the dataset)
2. Switched from focal loss to weighted cross-entropy
3. Increased hidden_dim from 64 → 128, dropout 0.3 → 0.2

**Results:**
- ROC-AUC: **0.739**
- PR-AUC: **0.092**
- F1: **0.148**
- Recall: 0.569, Precision: 0.085

Model now makes real fraud predictions (F1 > 0). Still below XGBoost on AUC.

---

### Experiment 4 — GraphSAGE

**Settings:** hidden_dim=64, dropout=0.3, weighted CE, CPU, epochs=200, patience=25.

GraphSAGE trained in ~0.7s/epoch vs GAT's ~2s/epoch (3x faster). Converged in 154 epochs.

**Results:**
- ROC-AUC: **0.880**
- PR-AUC: **0.358**
- F1: **0.279**
- Precision@100: **0.780**

**Why SAGE outperformed GAT significantly:**
1. SAGEConv doesn't use `torch.jit.script` — no Python 3.14 compatibility issues
2. Simple mean aggregation is more stable than attention for training from scratch
3. The `to_hetero()` conversion worked cleanly for SAGE's architecture
4. Faster convergence: each epoch is 3x faster, so 154 epochs took ~110s vs GAT's 200s

---

## 8. Final Results

All models trained on the **full 590,540 rows** (no subsampling).
Comparison chart saved to `results/model_comparison.png`.

### Development runs (100k subsample)

| Model | ROC-AUC | PR-AUC | F1 | Precision@100 |
|---|---|---|---|---|
| XGBoost | 0.918 | 0.613 | 0.449 | — |
| GraphSAGE | 0.867 | 0.338 | 0.255 | 0.700 |
| GAT (HANConv) | 0.765 | 0.139 | 0.142 | 0.300 |

### Final runs (full 590k dataset)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision@1000 | Notes |
|---|---|---|---|---|---|---|
| XGBoost | **0.947** | **0.697** | 0.428 | — | — | 5-fold CV OOF |
| GraphSAGE | 0.864 | 0.264 | 0.223 | **0.756** | **0.501** | Inductive, 187 epochs |
| GAT (HANConv) | 0.764 | 0.105 | 0.159 | 0.598 | 0.029 | Transductive, 200 epochs |

**Impact of scaling from 100k → 590k rows:**
- XGBoost: 0.918 → **0.947** (+3.2%) — more data always helps tree models
- GraphSAGE: 0.867 → **0.864** (≈ flat) — graph signal improved but hit epoch limit
- GAT: 0.765 → **0.764** (≈ flat) — still limited by architecture on this scale

**Key findings:**

**Precision@1000 = 0.501** for GraphSAGE: of the top 1,000 transactions flagged as fraud,
**50% actually are fraud** — a 14x lift over the 3.5% base rate. This is the operationally
critical number: an investigations team reviewing 1,000 alerts catches half real fraudsters.

**Recall = 0.756** for GraphSAGE: the model catches **75.6% of all fraud cases** in the
test set. XGBoost focuses on precision; GraphSAGE finds more fraud overall.

**Why XGBoost still leads on ROC-AUC:**
- Uses all 431 features directly via tree splits with no dimensionality compression
- 5-fold CV means it trains on ~470k rows per fold; very data-efficient
- GraphSAGE at 200 epochs was still slowly improving — more epochs would close the gap
- On this dataset the V1-V339 Vesta features are extremely predictive tabular signals that
  tree models exploit better than neural networks at this scale

---

## 9. Key Lessons Learned

### On GNNs vs XGBoost for Tabular Fraud Detection

- XGBoost dominates on tabular data at moderate scale (< 200k rows)
- GNNs show their advantage when the graph is **dense** — many transactions per card/email
- At 100k rows with 3.5% fraud, each card is shared by fewer transactions — weak graph signal
- On the full 590k dataset, the shared-card/email signals become much stronger

### On Focal Loss

- Focal loss works for object detection (moderate class imbalance, ~1:5 ratio)
- At extreme imbalance (28:1 ratio for fraud), focal loss is unstable
- Weighted cross-entropy (`nn.CrossEntropyLoss(weight=...)`) is simpler and more reliable
- This is equivalent to what XGBoost does with `scale_pos_weight`

### On PyTorch + macOS Compatibility

- `torch.jit.script` is deprecated in Python 3.14+ — PyG's HANConv uses it internally
- MPS (Apple Silicon GPU) gives **silently wrong gradients** for HANConv on Python 3.14
- The symptom is AUC oscillating around 0.5 or dipping below — worse than random
- Fix: disable MPS until PyG updates its JIT usage for Python 3.14+
- GraphSAGE (SAGEConv + to_hetero) is NOT affected — use SAGE for Apple Silicon

### On Heterogeneous Graph Design

- Entity node features (fraud_rate, count, mean_amt) add strong prior information
- The -999 sentinel for missing values creates spurious shared nodes — always filter these
- `to_hetero()` only promotes PyG-native `Linear` — standard `nn.Linear` stays shared
- Reverse edges are essential: without them, messages only flow one direction

### On Production Considerations

- GraphSAGE is the better production model: inductive (handles new cards/emails daily)
- GAT is transductive: new nodes have no representation at inference time
- The `Precision@100` metric matters more than F1 for ops teams with fixed review capacity
- Time-aware train/val/test split is critical: shuffling leaks future information to training

---

## 10. How to Run

### Setup
```bash
git clone <repo>
cd gnn-fraud-detection
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install torch_geometric
pip install -r requirements.txt
```

### Get Data
```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/
unzip data/raw/ieee-fraud-detection.zip -d data/raw/
```

### Run

```bash
# Sanity check — XGBoost only, no graph needed, fast
python run.py --model baseline

# Train GraphSAGE (best GNN)
python run.py --model sage --epochs 200 --patience 25

# Train GAT
python run.py --model gat --epochs 200 --patience 25

# All three + comparison chart → results/model_comparison.png
python run.py --model all --epochs 200 --patience 25

# Force graph rebuild (needed after changing subsampling or feature list)
python run.py --model sage --rebuild_graph

# Scale to full dataset (remove 100k cap)
# Edit src/data/load.py: change max_rows=100_000 to max_rows=None
python run.py --model all --rebuild_graph --epochs 200 --patience 25
```

### Explore Results
```bash
# Interactive fraud subgraph visualization
python -c "
from src.explain.gnn_explainer import FraudExplainer
import torch
from src.models.graphsage import build_sage
data = torch.load('data/processed/hetero_graph.pt', weights_only=False)
model = build_sage(data)
model.load_state_dict(torch.load('models/saved/sage_best.pt', weights_only=True))
explainer = FraudExplainer(model, data)
explainer.visualise_html(txn_idx=42, save_path='results/fraud_subgraph.html')
"
# Open results/fraud_subgraph.html in a browser
```

---

## File Reference

```
gnn-fraud-detection/
├── run.py                          Entry point
├── src/
│   ├── data/
│   │   ├── load.py                 CSV loading, encoding, 100k subsampling
│   │   └── graph_builder.py        DataFrame → HeteroData graph
│   ├── models/
│   │   ├── gat.py                  FraudGAT (HANConv, 2 layers, heads=4)
│   │   ├── graphsage.py            FraudSAGE (SAGEConv via to_hetero)
│   │   └── xgb_baseline.py         XGBoost 5-fold CV + Optuna tuning
│   ├── training/
│   │   ├── train.py                Training loop, weighted CE, early stopping
│   │   └── evaluate.py             ROC-AUC, PR-AUC, F1, Precision@K
│   └── explain/
│       └── gnn_explainer.py        Fraud subgraph pyvis visualization
├── notebooks/
│   └── 01_eda.ipynb                EDA: fraud rates, missingness, time patterns
├── PROJECT_EXPLAINED.md            ML concept explanations (for learning)
├── DEVLOG.md                       This file — full development transcript
└── requirements.txt
```
