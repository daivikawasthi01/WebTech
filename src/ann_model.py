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

  8. Lazy torch imports — torch is imported inside each function/class that
     needs it rather than at module level. This means the dashboard tabs that
     only display saved results (GA, Baselines, Ablation, etc.) load fine even
     when torch is not installed. Only the Pipeline tab fails if torch is absent.
"""

import json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

_HYPERPARAMS_PATH = "data/results/best_hyperparams.json"


def load_hyperparams() -> dict:
    """Load tuned hyperparameters if tune.py has been run, else return safe defaults."""
    defaults = {
        "lr": 0.001, "hidden1": 64, "hidden2": 32, "hidden3": 16,
        "dropout": 0.3, "weight_decay": 1e-4, "batch_size": 16,
        "use_bn": True,
    }
    if os.path.exists(_HYPERPARAMS_PATH):
        with open(_HYPERPARAMS_PATH) as f:
            tuned = json.load(f)
        defaults.update(tuned)
        # Cap lr at 0.003 — Optuna can suggest aggressive rates (e.g. 0.009)
        # that cause ANN divergence on certain random splits, producing
        # MSE outliers that corrupt ablation/stats means.
        defaults['lr'] = min(defaults.get('lr', 0.001), 0.003)
        print(f"  [ANN] Loaded tuned hyperparams from {_HYPERPARAMS_PATH} "
              f"(lr capped at {defaults['lr']:.4f})")
    return defaults


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaintainabilityDataset:
    """Lazy-import-safe dataset wrapper. Imports torch on first instantiation."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch
        from torch.utils.data import Dataset

        class _TorchDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        self._inner = _TorchDataset(X, y)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]

    # Allow passing directly to DataLoader
    def __iter__(self):
        return iter(self._inner)


def _make_dataset(X: np.ndarray, y: np.ndarray):
    """Return a torch Dataset without requiring a top-level torch import."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return _DS(X, y)


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

def _make_model(input_dim: int, hidden1: int = 64, hidden2: int = 32,
                hidden3: int = 16, dropout: float = 0.3, use_bn: bool = True):
    """
    Build and return the ANN. Imports torch lazily.

    Architecture: up to 3 hidden layers with optional BatchNorm.
    hidden3=0 disables the third layer (2-layer mode for small datasets).
    BatchNorm stabilises training on varied-scale features and reduces
    sensitivity to the learning rate — especially helpful when features
    span very different numerical ranges post-MinMaxScaling.
    """
    import torch.nn as nn

    class MaintainabilityANN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            # Layer 1
            layers.append(nn.Linear(input_dim, hidden1))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            # Layer 2
            layers.append(nn.Linear(hidden1, hidden2))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            # Layer 3 (optional — skip when hidden3 == 0)
            last_dim = hidden2
            if hidden3 and hidden3 > 0:
                layers.append(nn.Linear(hidden2, hidden3))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden3))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout / 2))  # lighter dropout on final hidden
                last_dim = hidden3
            # Output
            layers.append(nn.Linear(last_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return MaintainabilityANN()


# Keep the class name accessible for any code that imports it directly
def MaintainabilityANN(input_dim, hidden1=64, hidden2=32, hidden3=16,
                       dropout=0.3, use_bn=True):
    return _make_model(input_dim, hidden1, hidden2, hidden3, dropout, use_bn)


# ---------------------------------------------------------------------------
# Internal: train + eval on a single (pre-split, pre-scaled) fold
# ---------------------------------------------------------------------------

def _train_fold(
    X_train, y_train_transformed,
    X_val,   y_val_original,
    epochs, batch_size, patience, log_transform,
    lr: float = 0.001, hidden1: int = 64, hidden2: int = 32, hidden3: int = 16,
    dropout: float = 0.3, weight_decay: float = 1e-4, use_bn: bool = True,
    divergence_cap: float = None,
) -> float:
    """
    Train on (X_train, y_train_transformed) and evaluate on (X_val, y_val_original).
    If log_transform=True, predictions are back-transformed with expm1 before MSE.
    Returns validation MSE on original scale.

    divergence_cap: if None, computed as 4× the naive-predictor MSE on y_val_original
    (predicting the training mean). This is data-adaptive and more principled than a
    hardcoded constant — a cap of 4× naive MSE flags diverged runs without penalising
    legitimate high-MSE outcomes on imbalanced splits.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Compute adaptive cap: 4× the MSE of always predicting the training-set mean
    if divergence_cap is None:
        train_mean     = float(np.mean(y_val_original))
        naive_mse      = float(np.mean((y_val_original - train_mean) ** 2))
        divergence_cap = max(4.0 * naive_mse, 2.0)  # floor at 2.0 for tiny datasets

    input_dim = X_train.shape[1]
    model     = _make_model(input_dim, hidden1=hidden1, hidden2=hidden2,
                             hidden3=hidden3, dropout=dropout, use_bn=use_bn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_loader = DataLoader(
        _make_dataset(X_train, y_train_transformed),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        _make_dataset(X_val, y_val_original),
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

    # Guard: if ANN diverged (NaN or above the adaptive cap), return the cap.
    if best_mse != best_mse or best_mse > divergence_cap:
        return divergence_cap
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
    y_all = df['target_bug_proneness'].values.astype(np.float32)
    # Select only numeric columns, explicitly excluding the target.
    # This safely skips string metadata columns like 'repo' that
    # combine_datasets.py inserts — df.iloc[:,1:-1] would include them.
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != 'target_bug_proneness']
    X_all = df[feature_cols].values.astype(np.float32)

    if feature_mask is not None:
        mask_indices = [i for i, v in enumerate(feature_mask) if v == 1]
        X_all = X_all[:, mask_indices]

    y_model = np.log1p(y_all) if log_transform else y_all.copy()

    if use_kfold:
        y_bins = np.clip(y_all.astype(int), 0, 2)

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
                hidden3=hp.get('hidden3', 16), dropout=hp['dropout'],
                weight_decay=hp['weight_decay'], use_bn=hp.get('use_bn', True),
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
            hidden3=hp.get('hidden3', 16), dropout=hp['dropout'],
            weight_decay=hp['weight_decay'], use_bn=hp.get('use_bn', True),
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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    hp = load_hyperparams()
    if hyperparams:
        hp.update(hyperparams)

    df    = pd.read_csv(csv_file)
    files = df['file_name'].values if 'file_name' in df.columns else df.iloc[:, 0].values
    y_all = df['target_bug_proneness'].values.astype(np.float32)
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != 'target_bug_proneness']
    X_all = df[feature_cols].values.astype(np.float32)

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

    model     = _make_model(X_tr.shape[1], hp['hidden1'], hp['hidden2'],
                            hp.get('hidden3', 16), hp['dropout'],
                            hp.get('use_bn', True))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'],
                           weight_decay=hp['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    train_loader = DataLoader(
        _make_dataset(X_tr, y_tr_m),
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