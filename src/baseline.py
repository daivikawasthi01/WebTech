"""
baseline.py — Automated baseline comparison for research credibility.

Runs four parallel evaluations:
  ALL      — ANN trained on all features (mirrors base paper approach)
  RANDOM   — ANN trained on a random subset the same size as GA-selected
  GA       — ANN trained on GA-selected features
  XGB_GA   — XGBoost trained on GA-selected features (model comparison)

The XGBoost baseline answers a secondary research question:
  "Is the ANN the right model for this task, or would a tree-based model
   generalise better on this small tabular dataset?"
Including it strengthens the paper: either the ANN wins (justifying the
architecture) or XGBoost wins (which is a finding in itself).

Results saved to data/results/baseline_results.json.
"""

import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from src.ann_model import train_and_evaluate_ann


def _xgb_cv_mse(
    csv_file: str,
    feature_mask: list,
    n_folds: int        = 5,
    split_seed: int     = 42,
    log_transform: bool = True,
) -> float:
    """
    5-fold stratified CV MSE for XGBoost on the given feature mask.
    Returns MSE on original bug-count scale (back-transformed if log_transform).
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return float('nan')   # graceful skip if xgboost not installed

    df    = pd.read_csv(csv_file)
    X_all = df.iloc[:, 1:-1].values.astype(float)
    y_all = df.iloc[:, -1].values.astype(float)

    mask_idx = [i for i, v in enumerate(feature_mask) if v == 1]
    X_all    = X_all[:, mask_idx]
    y_model  = np.log1p(y_all) if log_transform else y_all.copy()

    y_bins = np.clip(y_all.astype(int), 0, 2)
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    mses   = []

    for tr_idx, va_idx in skf.split(X_all, y_bins):
        scaler   = MinMaxScaler()
        X_tr     = scaler.fit_transform(X_all[tr_idx])
        X_va     = scaler.transform(X_all[va_idx])
        y_tr_m   = y_model[tr_idx]
        y_va_orig = y_all[va_idx]

        model = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=split_seed, verbosity=0,
            early_stopping_rounds=20,
            eval_metric='rmse',
        )
        model.fit(X_tr, y_tr_m,
                  eval_set=[(X_va, y_va_orig if not log_transform else np.log1p(y_va_orig))],
                  verbose=False)

        preds = model.predict(X_va)
        if log_transform:
            preds = np.expm1(np.maximum(preds, 0))
        mses.append(float(np.mean((preds - y_va_orig) ** 2)))

    return float(np.mean(mses))


def _build_mask(n_total: int, selected_indices: list) -> list:
    mask = [0] * n_total
    for i in selected_indices:
        mask[i] = 1
    return mask


def run_baselines(
    csv_file: str,
    ga_chromosome: tuple,
    n_trials: int       = 20,
    log_transform: bool = True,
    output_path: str    = "data/results/baseline_results.json",
) -> dict:
    """
    Runs n_trials of each baseline and returns MSE distributions.

    Args:
        csv_file:      Path to cleaned dataset CSV.
        ga_chromosome: Best chromosome tuple from GA.evolve().
        n_trials:      Number of independent ANN runs per method (different seeds).
        log_transform: Must match what was used during GA training.
        output_path:   Where to save the JSON results.

    Returns:
        {
          'all_features':    {'mses': [...], 'mean': x, 'std': x},
          'random_subset':   {'mses': [...], 'mean': x, 'std': x},
          'ga_selected':     {'mses': [...], 'mean': x, 'std': x},
          'n_trials':        n_trials,
          'ga_n_features':   k,
          'total_features':  n,
        }
    """
    df          = pd.read_csv(csv_file)
    n_total     = len(df.columns) - 2
    n_ga        = int(sum(ga_chromosome))
    all_mask    = [1] * n_total
    ga_mask     = list(ga_chromosome)


    results = {
        'all_features':  {'mses': []},
        'random_subset': {'mses': []},
        'ga_selected':   {'mses': []},
        'xgb_ga':        {'mses': []},   # XGBoost on GA features
    }

    print(f"\n[BASELINES] Running {n_trials} trials × 4 methods "
          f"(GA uses {n_ga}/{n_total} features)")

    for trial in range(n_trials):
        seed = trial * 7 + 13

        mse_all = train_and_evaluate_ann(
            csv_file, feature_mask=all_mask,
            use_kfold=True, split_seed=seed, log_transform=log_transform
        )
        results['all_features']['mses'].append(mse_all)

        rand_indices = random.sample(range(n_total), n_ga)
        rand_mask    = _build_mask(n_total, rand_indices)
        mse_rand     = train_and_evaluate_ann(
            csv_file, feature_mask=rand_mask,
            use_kfold=True, split_seed=seed, log_transform=log_transform
        )
        results['random_subset']['mses'].append(mse_rand)

        mse_ga = train_and_evaluate_ann(
            csv_file, feature_mask=ga_mask,
            use_kfold=True, split_seed=seed, log_transform=log_transform
        )
        results['ga_selected']['mses'].append(mse_ga)

        mse_xgb = _xgb_cv_mse(csv_file, ga_mask, split_seed=seed,
                               log_transform=log_transform)
        results['xgb_ga']['mses'].append(mse_xgb)

        print(f"  Trial {trial+1:02d}/{n_trials} — "
              f"All: {mse_all:.4f}  Rand: {mse_rand:.4f}  "
              f"GA-ANN: {mse_ga:.4f}  GA-XGB: {mse_xgb:.4f}")

    # Compute summary stats
    for key in results:
        mses                  = results[key]['mses']
        results[key]['mean']  = float(np.mean(mses))
        results[key]['std']   = float(np.std(mses))
        results[key]['min']   = float(np.min(mses))
        results[key]['max']   = float(np.max(mses))

    results['n_trials']       = n_trials
    results['ga_n_features']  = n_ga
    results['total_features'] = n_total

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    print(f"  Saved to: {output_path}")
    return results


def _print_summary(results: dict):
    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │  BASELINE COMPARISON SUMMARY                     │")
    print("  ├──────────────────────┬─────────────┬────────────┤")
    print("  │  Method              │  Mean MSE   │  Std Dev   │")
    print("  ├──────────────────────┼─────────────┼────────────┤")
    for label, key in [("All Features    ", "all_features"),
                       ("Random Subset   ", "random_subset"),
                       ("GA + ANN        ", "ga_selected"),
                       ("GA + XGBoost    ", "xgb_ga")]:
        data = results.get(key, {})
        m    = data.get('mean', float('nan'))
        s    = data.get('std',  float('nan'))
        print(f"  │  {label}  │  {m:.4f}     │  {s:.4f}    │")
    print("  └──────────────────────┴─────────────┴────────────┘")


if __name__ == "__main__":
    import json
    with open("data/results/ga_results.json") as f:
        ga_results = json.load(f)
    run_baselines(
        csv_file     = "data/flask_dataset.csv",
        ga_chromosome = tuple(ga_results['chromosome']),
        n_trials     = 10,
    )