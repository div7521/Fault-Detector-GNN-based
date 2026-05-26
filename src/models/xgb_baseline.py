"""
XGBoost baseline — establish this BEFORE training any GNN.

The goal: get AUC > 0.92 with XGBoost + your existing skills.
Then show GNN beats it. A resume bullet that says
"GNN improved AUC by 3% over strong XGBoost baseline" is far
more compelling than just reporting GNN numbers in isolation.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import optuna

# Suppress XGBoost verbosity
import warnings
warnings.filterwarnings("ignore")


FEATURE_COLS_TO_DROP = ["TransactionID", "isFraud", "TransactionDT"]


def get_X_y(df: pd.DataFrame):
    drop = [c for c in FEATURE_COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=drop).values.astype(np.float32)
    y = df["isFraud"].values.astype(np.int32)
    return X, y


def train_baseline(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Train XGBoost with cross-validation.
    Returns dict with oof predictions and feature importances.
    """
    X, y = get_X_y(df)
    oof_preds = np.zeros(len(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),  # handle imbalance
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",  # fast on CPU
    }

    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")
        models.append(model)

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ap  = average_precision_score(y, oof_preds)
    print(f"\nOOF AUC: {oof_auc:.4f}  |  OOF AP (PR-AUC): {oof_ap:.4f}")

    return {"models": models, "oof_preds": oof_preds, "oof_auc": oof_auc, "oof_ap": oof_ap}


def tune_with_optuna(df: pd.DataFrame, n_trials: int = 30) -> dict:
    """
    Use Optuna to tune XGBoost hyperparameters.
    You already know Optuna from your recommendation system project — same pattern.
    """
    X, y = get_X_y(df)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
            "eval_metric": "auc",
            "random_state": 42,
            "tree_method": "hist",
        }

        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx], verbose=False)
            preds = model.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], preds))

        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def save_model(models: list, path: str = "models/saved/xgb_baseline.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    from src.data.load import load_and_preprocess
    df = load_and_preprocess()
    print("Training XGBoost baseline (5-fold CV)...")
    results = train_baseline(df)
    save_model(results["models"])
