"""
ann_model.py

Changes in this version:
  1. weight_decay=1e-4 added to Adam optimizer — L2 regularisation prevents
     overfitting on small repo datasets (50-200 files).

  2. StratifiedKFold via target binning — bug counts are zero-inflated, so
     plain KFold can produce folds with zero bug-prone files. We bin targets
     into [0, 1, 2+] and stratify on those bins.
     Safe fold count guard added: n_folds is clamped to the minimum class
     count so StratifiedKFold never raises ValueError on tiny datasets.

  3. Flexible MaintainabilityANN — accepts hidden1, hidden2, dropout as params
     so tune.py / Optuna can search the architecture space.

  4. load_hyperparams() — reads data/results/best_hyperparams.json if it exists
     (written by tune.py).

  5. Fixed redundant second train_test_split — replaced with single split that
     keeps y_model and y_all partitioned together.

  6. verbose=False removed from ReduceLROnPlateau in get_predictions —
     this kwarg was removed in PyTorch 2.2 and raises TypeError on recent installs.

  7. best_preds None fallback in get_predictions — best_preds is initialised
     to None and only set when mse improves. If predictions are always NaN,
     best_preds stays None and np.maximum(best_preds, 0) raises TypeError.
     Fixed with a zeros fallback.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

_HYPERPARAMS_PATH = "data/results/best_hyperparams.json"


def load_hyperparams() -> dict:
    """Load tuned hyperparameters if tune.py has been run, else return safe defaults."""
    defaults = {
        "lr": 0.001, "hidden1": 32, "hidden2": 16,
        "dropout": 0.3, "weight_decay": 1e-4, "batch_size": 16,
    }
    if os.path.exists(_HYPERPARAMS_PATH):
        with open(_HYPERPARAMS_PATH) as f:
            tuned = json.load(f)
        defaults.update(tuned)
        print(f"  [ANN] Loaded tuned hyperparams from {_HYPERPARAMS_PATH}")
    return defaults


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaintainabilityDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class MaintainabilityANN(nn.Module):
    """
    3-layer feed-forward: Input -> hidden1 -> hidden2 -> 1.
    Sizes are tunable via tune.py / Optuna.
    Dropout after each hidden layer.
    """
    def __init__(self, input_dim: int, hidden1: int = 32, hidden2: int = 16,
                 dropout: float = 0.3):
        super().__init__()
        self.layer1       = nn.Linear(input_dim, hidden1)
        self.relu1        = nn.ReLU()
        self.dropout1     = nn.Dropout(dropout)
        self.layer2       = nn.Linear(hidden1, hidden2)
        self.relu2        = nn.ReLU()
        self.dropout2     = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden2, 1)

    def forward(self, x):
        out = self.dropout1(self.relu1(self.layer1(x)))
        out = self.dropout2(self.relu2(self.layer2(out)))
        return self.output_layer(out)


# ---------------------------------------------------------------------------
# Internal: train + eval on a single (pre-split, pre-scaled) fold
# ---------------------------------------------------------------------------

def _train_fold(
    X_train, y_train_transformed,
    X_val,   y_val_original,
    epochs, batch_size, patience, log_transform,
    lr: float = 0.001, hidden1: int = 32, hidden2: int = 16,
    dropout: float = 0.3, weight_decay: float = 1e-4,
) -> float:
    """
    Train on (X_train, y_train_transformed) and evaluate on (X_val, y_val_original).
    If log_transform=True, predictions are back-transformed with expm1 before MSE.
    Returns validation MSE on original scale.
    """
    input_dim = X_train.shape[1]
    model     = MaintainabilityANN(input_dim, hidden1=hidden1, hidden2=hidden2,
                                   dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_loader = DataLoader(
        MaintainabilityDataset(X_train, y_train_transformed),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        MaintainabilityDataset(X_val, y_val_original),
        batch_size=batch_size, shuffle=False
    )

    best_mse      = float('inf')
    epochs_no_imp = 0

    for _ in range(epochs):
        model.train()
        for bX, by in train_loader:
            pred  = model(bX)
            loss  = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bX, by in val_loader:
                raw_preds = model(bX)
                if log_transform:
                    raw_preds = torch.expm1(raw_preds.clamp(min=0))
                val_loss += criterion(raw_preds, by).item()

        epoch_mse = val_loss / len(val_loader)
        scheduler.step(epoch_mse)

        if epoch_mse < best_mse:
            best_mse      = epoch_mse
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                break

    return best_mse


# ---------------------------------------------------------------------------
# Public: train and evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate_ann(
    csv_file: str,
    feature_mask = None,
    epochs: int       = 100,
    batch_size: int   = None,
    patience: int     = 15,
    split_seed: int   = 42,
    use_kfold: bool   = False,
    n_folds: int      = 5,
    log_transform: bool = True,
    hyperparams: dict = None,
) -> float:
    """
    Returns validation MSE (on original bug-count scale).

    use_kfold=False  — single 80/20 split with fixed seed (fast, used by GA)
    use_kfold=True   — stratified k-fold CV, returns mean MSE (reliable, for final evals)
    log_transform    — train on log1p(y), evaluate on expm1(pred) vs y
    hyperparams      — if None, loads from best_hyperparams.json (written by tune.py)
    """
    hp = load_hyperparams()
    if hyperparams:
        hp.update(hyperparams)
    if batch_size is None:
        batch_size = hp['batch_size']

    df    = pd.read_csv(csv_file)
    X_all = df.iloc[:, 1:-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values.astype(np.float32)

    if feature_mask is not None:
        mask_indices = [i for i, v in enumerate(feature_mask) if v == 1]
        X_all = X_all[:, mask_indices]

    y_model = np.log1p(y_all) if log_transform else y_all.copy()

    if use_kfold:
        y_bins = np.clip(y_all.astype(int), 0, 2)

        # FIX: StratifiedKFold raises ValueError when any class has fewer
        # samples than n_splits (common on small repos with zero-inflated bug
        # counts).  Clamp n_folds to the minimum class count, floor at 2.
        min_class_count = int(np.bincount(y_bins).min())
        n_folds_safe    = max(2, min(n_folds, len(X_all), min_class_count))
        if n_folds_safe < n_folds:
            print(f"  [ANN] Warning: reduced n_folds {n_folds} -> {n_folds_safe} "
                  f"(min class count = {min_class_count})")

        skf       = StratifiedKFold(n_splits=n_folds_safe, shuffle=True,
                                    random_state=split_seed)
        fold_mses = []

        for train_idx, val_idx in skf.split(X_all, y_bins):
            X_tr, X_va = X_all[train_idx], X_all[val_idx]
            y_tr_m     = y_model[train_idx]
            y_va_orig  = y_all[val_idx]

            scaler = MinMaxScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_va   = scaler.transform(X_va)

            fold_mses.append(_train_fold(
                X_tr, y_tr_m, X_va, y_va_orig,
                epochs, batch_size, patience, log_transform,
                lr=hp['lr'], hidden1=hp['hidden1'], hidden2=hp['hidden2'],
                dropout=hp['dropout'], weight_decay=hp['weight_decay'],
            ))

        return float(np.mean(fold_mses))

    else:
        idx = np.arange(len(X_all))
        tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=split_seed)

        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr_m     = y_model[tr_idx]
        y_va_orig  = y_all[va_idx]

        scaler = MinMaxScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_va   = scaler.transform(X_va)

        return _train_fold(
            X_tr, y_tr_m, X_va, y_va_orig,
            epochs, batch_size, patience, log_transform,
            lr=hp['lr'], hidden1=hp['hidden1'], hidden2=hp['hidden2'],
            dropout=hp['dropout'], weight_decay=hp['weight_decay'],
        )


# ---------------------------------------------------------------------------
# Public: get per-sample predictions for visualisation
# ---------------------------------------------------------------------------

def get_predictions(
    csv_file: str,
    feature_mask = None,
    split_seed: int     = 42,
    log_transform: bool = True,
    hyperparams: dict   = None,
) -> tuple:
    """
    Trains a final model on 80% of data and returns predictions on the 20% holdout.
    Returns: (file_names, y_true, y_pred, mse)
    Used by the Streamlit dashboard to display per-file risk scores.
    """
    hp = load_hyperparams()
    if hyperparams:
        hp.update(hyperparams)

    df    = pd.read_csv(csv_file)
    files = df.iloc[:, 0].values
    X_all = df.iloc[:, 1:-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values.astype(np.float32)

    if feature_mask is not None:
        mask_indices = [i for i, v in enumerate(feature_mask) if v == 1]
        X_all = X_all[:, mask_indices]

    y_model = np.log1p(y_all) if log_transform else y_all.copy()

    idx = np.arange(len(X_all))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=split_seed)

    X_tr, X_va  = X_all[tr_idx], X_all[va_idx]
    y_tr_m      = y_model[tr_idx]
    y_va_orig   = y_all[va_idx]
    files_val   = files[va_idx]

    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_va   = scaler.transform(X_va)

    model     = MaintainabilityANN(X_tr.shape[1], hp['hidden1'], hp['hidden2'],
                                   hp['dropout'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'],
                           weight_decay=hp['weight_decay'])
    # FIX: removed verbose=False — kwarg was dropped in PyTorch 2.2, raises TypeError.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    train_loader = DataLoader(
        MaintainabilityDataset(X_tr, y_tr_m),
        batch_size=hp['batch_size'], shuffle=True
    )

    best_mse, no_imp, best_preds = float('inf'), 0, None

    for _ in range(150):
        model.train()
        for bX, by in train_loader:
            loss = criterion(model(bX), by)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_va_t = torch.tensor(X_va, dtype=torch.float32)
            raw    = model(X_va_t)
            preds  = (torch.expm1(raw.clamp(min=0)).squeeze().numpy()
                      if log_transform else raw.squeeze().numpy())
            mse    = float(np.mean((preds - y_va_orig) ** 2))

        scheduler.step(mse)
        if mse < best_mse:
            best_mse, no_imp, best_preds = mse, 0, preds.copy()
        else:
            no_imp += 1
            if no_imp >= 15:
                break

    # FIX: best_preds stays None if every epoch produced NaN predictions
    # (bad data / extreme LR). np.maximum(None, 0) raises TypeError.
    if best_preds is None:
        best_preds = np.zeros_like(y_va_orig)

    return files_val, y_va_orig, np.maximum(best_preds, 0), best_mse


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv = "data/flask_dataset.csv"
    print("Single split MSE:", train_and_evaluate_ann(csv, use_kfold=False))
    print("5-fold CV MSE:  ", train_and_evaluate_ann(csv, use_kfold=True))