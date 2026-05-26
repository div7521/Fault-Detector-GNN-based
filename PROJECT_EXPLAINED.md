# GNN Fraud Detection — Full Project Explanation

This document explains everything about this project: the ML concepts, the code structure,
why each decision was made, and how everything connects. Paste this into a new Claude chat
to ask follow-up questions about any part of it.

---

## The Problem

We're trying to detect fraudulent credit card transactions in the IEEE-CIS Kaggle dataset.

- ~590,000 transactions
- Only ~3.5% are fraud (heavily imbalanced)
- Each transaction has ~400 features: card info, device, email domain, amounts, timestamps,
  and hundreds of anonymized "V" features engineered by Vesta Corporation

The challenge: a flat table of features misses a key signal — **fraudsters reuse the same
card, email, and device across multiple transactions**. A standard model (like XGBoost)
treats each row independently and can't see these connections.

---

## Why Graphs?

Imagine card number 1234 was used in 10 transactions, 7 of which were fraud. If a new
transaction comes in on card 1234, that shared history is a huge red flag — but XGBoost
sees the new row in isolation and has to re-learn this from the card's encoded numeric ID.

A graph makes this explicit:
- Each card is a **node**
- Each transaction that used that card has an **edge** to the card node
- When the GNN processes a transaction, it aggregates messages from the card node, which
  has already aggregated information from ALL transactions that used that card
- Fraud "leaks" through shared nodes — a transaction near many fraud transactions gets
  a fraud-like representation even if its own features look clean

This is the core insight. The rest is engineering to make it work.

---

## The Dataset

Two CSV files joined on `TransactionID`:

**train_transaction.csv** — one row per transaction:
- `TransactionAmt` — dollar amount
- `ProductCD` — product category (W/H/C/S/R)
- `card1`–`card6` — card attributes (type, bank, country, etc.)
- `addr1`, `addr2` — billing/zip addresses
- `dist1`, `dist2` — distances (likely billing to shipping)
- `C1`–`C14` — counts (how many addresses/cards associated with this user, etc.)
- `D1`–`D15` — time deltas (days since last transaction, etc.)
- `M1`–`M9` — match flags (name matches, address matches, etc.)
- `V1`–`V339` — Vesta-engineered features (mostly anonymous)
- `TransactionDT` — seconds offset from a reference point (not a real timestamp)
- `isFraud` — label (0 or 1)

**train_identity.csv** — ~144k rows (only ~25% of transactions have identity data):
- `DeviceType`, `DeviceInfo` — mobile/desktop, browser/OS
- `id_01`–`id_38` — various identity features

---

## Graph Structure

The graph has **4 node types** and **6 edge types** (3 forward + 3 reverse):

```
transaction ──uses_card──► card
transaction ──uses_email──► email
transaction ──uses_device──► device

card ──used_in──► transaction       (reverse)
email ──used_in──► transaction      (reverse)
device ──used_in──► transaction     (reverse)
```

This is called a **heterogeneous graph** — different node types have different feature
dimensions and semantics.

**Transaction node features** (~60 features):
- All the TXN_FEATURES + IDENTITY_FEATURES from the raw data
- Normalized with StandardScaler

**Entity node features** (3 features each — fraud_rate, log_count, log_mean_amount):
- `card`: built from `card1` values
- `email`: built from `P_emaildomain` values
- `device`: built from `DeviceInfo` values

The reverse edges (card → transaction) are critical: without them, messages only flow
one direction. With them, a transaction can receive information from its card, but the
card can also receive information from transactions — meaning the card node itself
becomes a summary of all transactions on it.

---

## Preprocessing Pipeline

```
load_raw() → merge_tables() → [subsample 100k] → encode_categoricals() → sort by TransactionDT
    → build_hetero_graph() → add_train_val_test_masks() → train()
```

**encode_categoricals()** (`src/data/load.py`):
- Object columns (card4, DeviceInfo, P_emaildomain, etc.) → LabelEncoder → integers
- Remaining numeric NaNs → -999 (signals missingness to tree models)

**Why -999?** Tree models can split on "is this feature == -999?" and learn that
missingness itself is informative. Neural nets don't benefit from this trick as much.

**Time-aware split** (`graph_builder.py:add_train_val_test_masks`):
- Sort by `TransactionDT` first, then take first 70% as train, next 15% as val, last 15% as test
- This is critical: shuffling would let the model see "future" transactions during training,
  artificially inflating metrics. Real fraud detection always predicts into the future.

---

## Model 1: XGBoost Baseline

**File:** `src/models/xgb_baseline.py`

XGBoost is a gradient boosted decision tree ensemble. It's the standard strong baseline
for tabular data and regularly wins Kaggle competitions.

**How it works:**
1. Build a tree that predicts residuals of the current ensemble
2. Add that tree to the ensemble with a small learning rate
3. Repeat 500 times
4. Each tree looks at a random subset of features (colsample_bytree=0.8) and rows (subsample=0.8)

**Key param: `scale_pos_weight`**
- Set to `(# negatives) / (# positives)` ≈ 28 for this dataset
- Tells XGBoost to weight each fraud sample 28x more than a legit sample
- Without this, the model would just predict "not fraud" for everything and achieve 96.5% accuracy

**Why use it?**
- Establishes a baseline AUC to beat with the GNN
- Fast to train (minutes vs hours for GNN)
- Explainable via feature importance
- Cannot use graph structure — that's the GNN's advantage

**5-fold stratified CV:**
- Splits data into 5 folds, keeping ~3.5% fraud rate in each fold
- Trains on 4 folds, evaluates on 1, rotates
- Out-of-fold (OOF) predictions cover all rows without leakage
- Final AUC is computed on all OOF predictions together

---

## Model 2: GAT with HANConv

**File:** `src/models/gat.py`

### What is a GNN?

A Graph Neural Network passes "messages" between connected nodes. Each node updates its
representation by aggregating information from its neighbors. After L layers, each node
has "seen" its L-hop neighborhood.

For fraud: after 2 layers, a transaction node has seen:
- Layer 1: its card/email/device nodes
- Layer 2: all other transactions that share that card/email/device

### What is Attention?

Plain GCN (Graph Convolutional Network) averages all neighbor messages equally. GAT
(Graph Attention Network) learns weights: "how much should I attend to each neighbor?"

This matters for fraud because:
- A card connected to 100 legit transactions and 2 fraud transactions should probably
  attend more to those 2 fraud transactions than to the 100 legit ones
- Attention weights are learned during training

### What is HANConv?

**HAN = Heterogeneous Attention Network**

Since we have multiple node/edge types, a plain GAT can't handle it directly — it
assumes all nodes have the same feature dimension and semantics.

HANConv extends GAT to heterogeneous graphs by:
1. For each edge type (e.g., transaction→card), learning a separate attention mechanism
2. Computing node-type-level attention ("how important is the card channel vs email channel?")
3. Combining messages from all edge types via a second level of attention

This is called **two-level hierarchical attention**:
- Level 1 (node-level): which specific card/email/device neighbors matter most?
- Level 2 (semantic/edge-type-level): which type of connection matters most overall?

### Architecture (FraudGAT)

```
Input: x_dict (transaction: 60 features, card: 3 features, email: 3, device: 3)
    ↓
input_proj: Linear per node type → all projected to hidden_dim=64
    ↓
HANConv layer 1: hidden_dim=64, heads=4
    - For each edge type: compute attention-weighted neighbor aggregation
    - Combine across edge types with semantic attention
    Output: hidden_dim=64 per node
    ↓
ReLU + Dropout(0.3)
    ↓
HANConv layer 2: out_channels=64//4=16, heads=1
    ↓
ReLU
    ↓
classifier (transaction nodes only):
    Linear(16 → 32) → ReLU → Dropout → Linear(32 → 2)
    ↓
Output: logits shape (n_transactions, 2) — [not-fraud score, fraud score]
```

**Why project to common dim first?** HANConv needs all node types to have the same
dimension going in. Since transaction nodes have 60 features and card/email/device
nodes have only 3, we project everything to 64 first.

**Why only classify transaction nodes?** We only have labels for transactions. Card/email/
device nodes are auxiliary — they exist to pass information between transactions.

---

## Model 3: GraphSAGE (Inductive)

**File:** `src/models/graphsage.py`

### Why GraphSAGE?

HANConv is **transductive**: it learns representations for the specific nodes seen during
training. New nodes (new cards, new email domains) at inference time have no representation.

GraphSAGE is **inductive**: it learns a function that generates representations from
a node's features and neighborhood. New nodes just need their features and edges.

In production fraud detection, new cards appear every day. An inductive model is essential.

### How SAGE differs from GAT

Instead of attention weights, SAGE uses **fixed aggregation functions** (mean, max, LSTM)
over a **sampled** neighborhood. At each layer:

```
h_v^(k) = σ(W · CONCAT(h_v^(k-1), MEAN(h_u^(k-1) for u in sample(neighbors(v)))))
```

"My new representation is a function of my old representation plus the mean of a random
sample of my neighbors' old representations."

Sampling is the key: instead of aggregating ALL neighbors (which can be millions for
popular cards), SAGE samples a fixed number. This makes training scalable and inference
on new nodes possible.

### to_hetero() trick

Rather than writing a heterogeneous SAGE from scratch, PyG's `to_hetero()` wraps a
homogeneous GNN and creates separate weight matrices per (node_type, edge_type) combination.

```python
homo_model = _HomoSAGE(...)         # plain SAGEConv layers
self.gnn = to_hetero(homo_model, metadata, aggr="sum")
# now self.gnn handles HeteroData automatically
```

The classifier is kept **outside** `to_hetero()` — this is intentional. We only need
to classify transaction nodes, not every node type.

---

## Training Details

**File:** `src/training/train.py`

### Focal Loss

Cross-entropy loss on its own would give ~97% of samples a loss of nearly 0 (the legit
transactions are easy to classify) and only ~3% meaningful gradient. The model never
learns to detect fraud well.

**Focal Loss** (Lin et al., 2017, originally for object detection) fixes this:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

- `p_t` = model's predicted probability for the correct class
- `(1 - p_t)^γ` = **focusing factor**: if the model is confident (p_t close to 1), this
  term is near 0 and the loss contribution is small. If the model is wrong (p_t near 0),
  this term is near 1 and the loss is full strength.
- `γ=2` is the standard value from the paper
- `α_t` = class weight: `1 - fraud_rate` (≈0.965) for fraud class, `fraud_rate` (≈0.035)
  for legit class — upweights the minority fraud class

The combined effect: easy legit transactions (correctly predicted with high confidence)
contribute almost nothing to the loss. Hard fraud cases get full gradient signal.

### AdamW Optimizer

AdamW = Adam + proper weight decay (L2 regularization). Adam adapts the learning rate
per parameter based on gradient history. AdamW fixes a subtle bug in Adam's weight decay
implementation that was causing overfitting in practice.

### Cosine LR Schedule

The learning rate starts at `lr=0.001` and follows a cosine curve down to near 0 by
the final epoch. This is better than a fixed LR because:
- High LR early: escape local minima, explore the loss landscape
- Low LR later: fine-tune, converge precisely

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Attention mechanisms can produce unstable gradients (exploding gradients problem). Clipping
caps the gradient norm at 1.0 before the optimizer step, preventing NaN weights.

### Early Stopping

If validation AUC doesn't improve for `patience=15` epochs, training stops. The best
model weights are saved to disk and reloaded at the end. This prevents overfitting and
avoids wasting time training past the optimal point.

---

## Evaluation Metrics

**File:** `src/training/evaluate.py`

With 3.5% fraud, accuracy is meaningless (predicting all-legit gives 96.5% accuracy).
We use:

**ROC-AUC (Area Under the ROC Curve)**
- Probability that a randomly chosen fraud transaction scores higher than a randomly
  chosen legit transaction
- 0.5 = random, 1.0 = perfect
- The most common metric but can be misleading for imbalanced data

**PR-AUC (Precision-Recall AUC)**
- Precision = of everything flagged as fraud, what fraction actually is fraud?
- Recall = of all actual fraud, what fraction did we catch?
- PR-AUC averages precision across all recall levels
- More informative than ROC-AUC when positive class is rare
- A random classifier achieves PR-AUC ≈ fraud_rate ≈ 0.035

**F1 Score**
- Harmonic mean of precision and recall at threshold=0.5
- 2 * (precision * recall) / (precision + recall)

**Precision@K**
- Of the top K transactions the model is most confident are fraud, what fraction are?
- This is what matters for operations teams: "we can review 100 alerts per day — how
  many of those will be real fraud?"

---

## Full Code Structure

```
gnn-fraud-detection/
│
├── run.py                          ← entry point, orchestrates everything
│
├── src/
│   ├── data/
│   │   ├── load.py                 ← CSV loading, merging, encoding, subsampling
│   │   └── graph_builder.py       ← converts DataFrame → HeteroData graph
│   │
│   ├── models/
│   │   ├── gat.py                  ← FraudGAT (HANConv-based)
│   │   ├── graphsage.py            ← FraudSAGE (to_hetero SAGEConv)
│   │   └── xgb_baseline.py        ← XGBoost with 5-fold CV + Optuna tuning
│   │
│   ├── training/
│   │   ├── train.py                ← training loop, FocalLoss, early stopping
│   │   └── evaluate.py            ← metrics (AUC, PR-AUC, F1, Precision@K)
│   │
│   └── explain/
│       └── gnn_explainer.py       ← subgraph extraction + pyvis HTML visualization
│
├── notebooks/
│   └── 01_eda.ipynb               ← exploratory data analysis
│
├── data/
│   ├── raw/                        ← train_transaction.csv, train_identity.csv (gitignored)
│   └── processed/                  ← hetero_graph.pt cache (gitignored)
│
├── models/saved/                   ← best model checkpoints (gitignored)
├── results/                        ← model_comparison.png, fraud_subgraph.html
└── requirements.txt
```

---

## How a Forward Pass Works (Step by Step)

1. **Input:** `x_dict = {"transaction": tensor(100k, 60), "card": tensor(8k, 3), ...}`
   and `edge_index_dict = {("transaction","uses_card","card"): tensor(2, 95k), ...}`

2. **Input projection:** Each node type's features are linearly projected to `hidden_dim=64`.
   Now all node types have 64-dimensional representations.

3. **HANConv layer 1:**
   - For each edge type, compute attention scores between source and destination node pairs
   - Each transaction aggregates messages from its connected card/email/device nodes
   - Each card/email/device aggregates from connected transactions (via reverse edges)
   - Semantic attention weights determine which edge type contributes most to each node

4. **HANConv layer 2:** Same process, now with richer representations from layer 1.
   A transaction node now "knows about" the 2-hop neighborhood: not just its own card,
   but also all other transactions that share that card.

5. **Classifier:** Applied only to transaction nodes.
   `Linear(16 → 32) → ReLU → Dropout → Linear(32 → 2)`
   Outputs 2 logits: [score for not-fraud, score for fraud]

6. **Loss:** Focal loss between logits and ground-truth labels, computed only on training
   mask nodes (70% of transactions).

7. **Backprop:** Gradients flow through the classifier → through HANConv layers → through
   the input projection → weights update.

---

## Key Design Decisions and Trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Heterogeneous graph | Different node types (card vs email) have different semantics | More complex than homogeneous |
| HANConv vs plain GCN | Attention lets the model focus on important neighbors | Slower, more parameters |
| GraphSAGE as alternative | Inductive — works on new cards/emails at inference | Slightly lower accuracy than transductive GAT |
| Focal Loss | Down-weights easy negatives so model focuses on hard fraud cases | Hyperparameters (α, γ) need tuning |
| Time-aware split | Prevents data leakage from future into training | Can't shuffle; later transactions harder to predict |
| `card1` for card node | Most discriminative card feature | Ignores other card dimensions |
| 2 GNN layers | Each layer = 1 hop; 2 layers = see 2-hop neighborhood | More layers = oversmoothing |

---

## Bugs That Were Fixed Before Running

1. **Focal loss alpha was inverted**: `alpha=fraud_rate` (0.035) meant fraud got weight
   0.035 and legit got 0.965 — opposite of intended. Fixed to `alpha=1-fraud_rate`.

2. **-999 creating spurious graph edges**: After encoding, missing `card1` values become
   -999. Previously all missing-card transactions connected to the same "card -999" node,
   creating false shared connections. Fixed by excluding -999 from entity node indexes.

3. **`iterrows()` in edge building**: Was doing a Python loop over 100k rows — would take
   5-10 minutes. Replaced with vectorized pandas `.isin()` and `.map()`.

4. **`_get_entity_features` loop**: Same issue — replaced with `groupby().agg()`.

5. **GraphSAGE classifier shared across all node types**: `to_hetero()` only promotes
   PyG-native layers. The `nn.Linear` classifier was shared. Fixed by moving the
   classifier outside `to_hetero()` and applying it only to transaction nodes.

6. **`torch.load` missing `map_location`**: Could crash if model was trained on MPS but
   loaded on CPU. Fixed to pass `map_location=device`.

7. **XGBoost F1 hardcoded to 0.0**: Computed from OOF predictions now.

---

## Running the Pipeline

```bash
source venv/bin/activate

# Sanity check (fast — XGBoost only, no graph)
python run.py --model baseline

# Full GAT training
python run.py --model gat --epochs 100 --lr 0.001 --patience 15

# Force graph rebuild (needed after changing subsampling)
python run.py --model gat --rebuild_graph

# Compare all three models
python run.py --model all
```

Expected results (on 100k subsample):
- XGBoost OOF AUC: ~0.88–0.92
- GAT Test AUC: ~0.90–0.94 (if it beats XGBoost, the graph is helping)
- GraphSAGE Test AUC: ~0.88–0.92

On the full 590k dataset, XGBoost typically reaches ~0.926 and well-tuned GNNs reach
~0.94–0.96 on this specific dataset.
