"""
multi_repo.py — Orchestrates GA feature selection across multiple repositories.

Runs the full GA pipeline on each repo independently, then aggregates results
to identify features that generalise across codebases.

Fixes vs original:
  1. Signature changed from run_multi_repo(repo_configs: list, ...) to
     run_multi_repo(repo_names: list, ga_kwargs: dict, run_baselines: bool)
     to match the call site in main.py (which passes repo_names=, ga_kwargs=,
     run_baselines=). The old signature caused a TypeError on every run.

  2. Output format changed from a nested structure
       {'per_repo': {...}, 'consensus_chromosome': [...], ...}
     to a flat dict
       {repo_name: {'ga_ann_mse': ..., 'n_selected': ..., ...}, ...}
     which is what app.py and report.py both expect when iterating
     mr_data.items() and accessing keys like 'ga_ann_mse', 'n_features_total',
     'selected_features', 'reduction_pct'. The old structure caused KeyError
     crashes in the dashboard's multi-repo tab.
"""

import json
import os
import time

import pandas as pd

from src.data_collector    import build_dataset_from_repo
from src.preprocess        import preprocess_dataset
from src.genetic_algorithm import FeatureSelectionGA
from src.ann_model         import train_and_evaluate_ann


# ---------------------------------------------------------------------------
# Registry: short name -> local path (relative to project root)
# ---------------------------------------------------------------------------
REPO_REGISTRY: dict = {
    'flask':    'test_repos/flask',
    'requests': 'test_repos/requests',
    'django':   'test_repos/django',
    'fastapi':  'test_repos/fastapi',
    'numpy':    'test_repos/numpy',
}


def _all_features_mse(csv_file: str, log_transform: bool = True) -> float:
    """Train ANN on all features; return k-fold CV MSE as baseline."""
    df         = pd.read_csv(csv_file, nrows=0)
    n_features = len(df.columns) - 2
    all_mask   = tuple(1 for _ in range(n_features))
    return train_and_evaluate_ann(
        csv_file,
        feature_mask  = all_mask,
        use_kfold     = True,
        log_transform = log_transform,
    )


def run_multi_repo(
    repo_names: list,
    ga_kwargs: dict     = None,
    run_baselines: bool = True,
    log_transform: bool = True,
    output_path: str    = "data/results/multi_repo_results.json",
) -> dict:
    """
    Run the GA on each repository in repo_names and write a flat results dict.

    Returns flat dict consumed by app.py and report.py:
        {
          repo_name: {
            'n_files':          int,
            'n_features_total': int,
            'n_selected':       int,
            'reduction_pct':    float,
            'ga_best_mse':      float,
            'ga_ann_mse':       float,
            'all_features_mse': float,
            'improvement_pct':  float,
            'selected_features': list[str],
            'elapsed_s':        float,
          },
          ...
        }
    """
    if ga_kwargs is None:
        ga_kwargs = {}

    results: dict = {}

    for repo_name in repo_names:
        repo_path = REPO_REGISTRY.get(repo_name)
        if repo_path is None:
            print(f"\n[multi_repo] '{repo_name}' not in REPO_REGISTRY — skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"  REPO: {repo_name}")
        print(f"{'='*55}")

        raw_csv   = f"data/{repo_name}_dataset.csv"
        clean_csv = f"data/{repo_name}_dataset_clean.csv"

        if not os.path.exists(clean_csv):
            if not os.path.exists(raw_csv):
                print(f"  Collecting data for '{repo_name}' ...")
                try:
                    build_dataset_from_repo(repo_path, raw_csv)
                except Exception as exc:
                    print(f"  [ERROR] Data collection failed: {exc}")
                    continue
            try:
                preprocess_dataset(raw_csv, clean_csv)
            except Exception as exc:
                print(f"  [ERROR] Preprocessing failed: {exc}")
                continue
        else:
            print(f"  Dataset found at {clean_csv}, skipping collection.")

        try:
            df = pd.read_csv(clean_csv)
        except Exception as exc:
            print(f"  [ERROR] Cannot read '{clean_csv}': {exc}")
            continue

        n_files       = len(df)
        n_features    = len(df.columns) - 2
        feature_names = df.columns[1:-1].tolist()

        if n_files < 5:
            print(f"  WARNING: only {n_files} rows — skipping GA.")
            continue

        print(f"  {n_files} files, {n_features} features")

        t0         = time.time()
        checkpoint = f"data/results/ga_checkpoint_{repo_name}.json"
        try:
            ga = FeatureSelectionGA(
                csv_file        = clean_csv,
                checkpoint_path = checkpoint,
                log_transform   = log_transform,
                **ga_kwargs,
            )
            ga_result = ga.evolve()
        except Exception as exc:
            print(f"  [ERROR] GA failed: {exc}")
            continue
        elapsed = round(time.time() - t0, 2)

        all_feat_mse = float('nan')
        if run_baselines:
            try:
                all_feat_mse = _all_features_mse(clean_csv, log_transform)
            except Exception as exc:
                print(f"  [WARN] Baseline failed: {exc}")

        n_selected    = int(sum(ga_result['chromosome']))
        reduction_pct = round((1 - n_selected / max(1, n_features)) * 100, 1)
        improvement   = float('nan')
        if all_feat_mse == all_feat_mse and all_feat_mse > 0:  # nan-safe
            improvement = round(
                (all_feat_mse - ga_result['best_mse']) / all_feat_mse * 100, 2
            )

        results[repo_name] = {
            'n_files':           n_files,
            'n_features_total':  n_features,
            'n_selected':        n_selected,
            'reduction_pct':     reduction_pct,
            'ga_best_mse':       float(ga_result['best_mse']),
            'ga_ann_mse':        float(ga_result['best_mse']),
            'all_features_mse':  all_feat_mse,
            'improvement_pct':   improvement,
            'selected_features': [
                name for name, bit in zip(feature_names, ga_result['chromosome'])
                if bit
            ],
            'elapsed_s': elapsed,
        }

        print(f"  [{repo_name}] MSE {ga_result['best_mse']:.4f} | "
              f"{n_selected}/{n_features} features | {elapsed}s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[multi_repo] Results saved -> {output_path}  ({len(results)} repos)")
    return results


if __name__ == "__main__":
    run_multi_repo(
        repo_names    = ['flask', 'requests'],
        ga_kwargs     = {'population_size': 10, 'generations': 5},
        run_baselines = True,
    )