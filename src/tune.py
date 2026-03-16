"""
tune.py — Optuna-based hyperparameter tuning for the ANN component.

Runs once to find the best ANN configuration for your specific repo.
Result saved to data/results/best_hyperparams.json and automatically
loaded by ann_model.py for all subsequent GA runs, baselines, and ablation.

Search space:
  lr           : log-uniform [1e-4, 1e-2]
  hidden1      : categorical [16, 32, 64]
  hidden2      : categorical [8, 16, 32]
  dropout      : uniform [0.1, 0.5]
  weight_decay : log-uniform [1e-5, 1e-2]
  batch_size   : categorical [8, 16, 32]

Objective: minimise 5-fold stratified CV MSE on all features.

Usage:
  python -m src.tune --csv data/flask_dataset_clean.csv --trials 50
"""

import argparse
import json
import os
# Removed unused `import numpy as np` and `import pandas as pd`.

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial, csv_file: str, log_transform: bool):
    """Optuna objective: returns 5-fold CV MSE for a given hyperparameter set."""
    from src.ann_model import train_and_evaluate_ann

    hp = {
        "lr":           trial.suggest_float("lr",           1e-4, 1e-2, log=True),
        "hidden1":      trial.suggest_categorical("hidden1", [16, 32, 64]),
        "hidden2":      trial.suggest_categorical("hidden2", [8, 16, 32]),
        "dropout":      trial.suggest_float("dropout",      0.1,  0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [8, 16, 32]),
    }

    if hp["hidden2"] > hp["hidden1"]:
        raise optuna.exceptions.TrialPruned()

    mse = train_and_evaluate_ann(
        csv_file,
        feature_mask  = None,
        epochs        = 100,
        patience      = 15,
        use_kfold     = True,
        n_folds       = 5,
        log_transform = log_transform,
        hyperparams   = hp,
    )
    return mse


def run_tuning(
    csv_file: str,
    n_trials: int       = 50,
    log_transform: bool = True,
    output_path: str    = "data/results/best_hyperparams.json",
) -> dict:
    """
    Runs Optuna study and saves best hyperparameters.

    Returns best hyperparameter dict.
    """
    print(f"\n[TUNE] Optuna hyperparameter search — {n_trials} trials")
    print(f"  Dataset: {csv_file}")
    print(f"  Objective: minimise 5-fold CV MSE (all features)\n")

    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=42),
        pruner    = optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda trial: _objective(trial, csv_file, log_transform),
        n_trials          = n_trials,
        show_progress_bar = True,
        catch             = (Exception,),
    )

    best     = study.best_params
    best_mse = study.best_value

    print(f"\n[TUNE] Best hyperparameters found:")
    for k, v in best.items():
        print(f"  {k:15s}: {v}")
    print(f"  Best CV MSE: {best_mse:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {**best, "best_cv_mse": float(best_mse)}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {output_path}")
    print("  ann_model.py will automatically load these on next run.\n")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna ANN hyperparameter tuning")
    parser.add_argument("--csv",    type=str, default="data/flask_dataset_clean.csv")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--no-log", action="store_true",
                        help="Disable log-transform (not recommended)")
    args = parser.parse_args()

    run_tuning(
        csv_file      = args.csv,
        n_trials      = args.trials,
        log_transform = not args.no_log,
    )