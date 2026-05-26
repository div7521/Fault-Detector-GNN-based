# Setup Guide — GNN Fraud Detection

## 1. Create the project in VS Code

Open Terminal and run these commands one by one:

```bash
# Clone or create the project folder
cd ~/Desktop
mkdir gnn-fraud-detection
cd gnn-fraud-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # you'll need this every time you open a new terminal

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio   # install PyTorch first
pip install torch_geometric                # then PyG
pip install -r requirements.txt
```

> Apple Silicon (M1/M2/M3): PyTorch will use the MPS GPU automatically.
> The training script detects this — no extra setup needed.

## 2. Get the dataset from Kaggle

Option A — Kaggle API (fastest):
```bash
pip install kaggle

# Put your kaggle.json API key in ~/.kaggle/
# (Download from kaggle.com → Account → Create New API Token)
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
mkdir -p data/raw
cd data/raw
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip
cd ../..
```

Option B — Manual download:
Go to https://www.kaggle.com/competitions/ieee-fraud-detection/data
Download train_transaction.csv and train_identity.csv into data/raw/

## 3. Open in VS Code

```bash
code .   # opens VS Code in current folder
```

In VS Code:
- Press Cmd+Shift+P → "Python: Select Interpreter" → choose the venv you created
- Install extensions: Python, Pylance, Jupyter

## 4. Run in order

```bash
# Step 1: Explore the data in the notebook
jupyter notebook notebooks/01_eda.ipynb

# Step 2: Train XGBoost baseline first
python run.py --model baseline

# Step 3: Build graph + train GAT
python run.py --model gat

# Step 4: Train all models and compare
python run.py --model all
```

## 5. Project structure

```
gnn-fraud-detection/
├── src/
│   ├── data/
│   │   ├── load.py          ← load + preprocess IEEE-CIS
│   │   └── graph_builder.py ← tabular data → PyG HeteroData graph
│   ├── models/
│   │   ├── gat.py           ← Graph Attention Network
│   │   ├── graphsage.py     ← GraphSAGE (inductive)
│   │   └── xgb_baseline.py  ← XGBoost baseline
│   ├── training/
│   │   ├── train.py         ← training loop + focal loss + early stopping
│   │   └── evaluate.py      ← AUC, PR-AUC, F1, Precision@K
│   └── explain/
│       └── gnn_explainer.py ← GNNExplainer + pyvis subgraph visualisation
├── notebooks/
│   └── 01_eda.ipynb         ← start here
├── data/
│   ├── raw/                 ← put CSVs here (gitignored)
│   └── processed/           ← cached graph (gitignored)
├── models/saved/            ← trained weights (gitignored)
├── results/                 ← plots and comparison charts
├── run.py                   ← end-to-end pipeline
└── requirements.txt
```

## Common issues

**ImportError: torch_geometric not found**
→ Make sure you activated venv: `source venv/bin/activate`

**MPS not available**
→ Update macOS to 12.3+ and PyTorch to 2.0+

**Graph construction takes too long**
→ It only runs once. After that, the cached graph loads instantly.
→ If RAM is an issue, subsample: `df = df.sample(100_000, random_state=42)`
