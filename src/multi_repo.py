"""
multi_repo.py — Multi-repository testing harness.

Runs the full mine → clean → GA pipeline on each configured repository
and collects results into a single comparison table.

Why this is essential for publication:
  A single-repo result (Flask only) cannot support generalisable claims.
  Testing across Flask, Django, and Requests — three stylistically different
  Python codebases — demonstrates that the GA consistently selects a
  cross-dimensional feature subset and outperforms baselines regardless of
  project size or domain.

Output: data/results/multi_repo_results.json  +  printed comparison table.

Default repos (cloned automatically if not present):
  flask    — micro web framework,   ~250 .py files
  django   — full-stack framework,  ~1000+ .py files
  requests — HTTP library,          ~50 .py files

Usage:
  python -m src.multi_repo
  python main.py --multi-repo --repos flask django requests
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO_REGISTRY = {
    'flask':    'https://github.com/pallets/flask.git',
    'django':   'https://github.com/django/django.git',
    'requests': 'https://github.com/psf/requests.git',
    'fastapi':  'https://github.com/tiangolo/fastapi.git',
    'httpx':    'https://github.com/encode/httpx.git',
}


def _ensure_repo(name: str, base_dir: str = 'test_repos') -> str:
    """Clone repo if not already present. Returns local path."""
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        url = REPO_REGISTRY.get(name)
        if not url:
            raise ValueError(f"Unknown repo '{name}'. Add it to REPO_REGISTRY.")
        os.makedirs(base_dir, exist_ok=True)
        print(f"  Cloning {name} from {url} ...")
        subprocess.run(
            ['git', 'clone', '--depth', '500', url, path],
            check=True, capture_output=True
        )
        print(f"  Cloned to {path}")
    else:
        print(f"  Using existing clone: {path}")
    return path


def run_multi_repo(
    repo_names: list         = None,
    base_dir: str            = 'test_repos',
    data_dir: str            = 'data',
    ga_kwargs: dict          = None,
    run_baselines: bool      = True,
    output_path: str         = 'data/results/multi_repo_results.json',
) -> dict:
    """
    Runs the complete pipeline on each repository independently.
    Each repo gets its own raw CSV, clean CSV, and GA results.
    All results are aggregated into a single comparison table.

    Args:
        repo_names:   List of repo names from REPO_REGISTRY.
                      Defaults to ['flask', 'requests', 'django'].
        base_dir:     Where repos are cloned.
        data_dir:     Where per-repo CSVs are saved.
        ga_kwargs:    GA constructor kwargs (same for all repos).
        run_baselines: Whether to run baseline comparison per repo.
        output_path:  Where to save aggregated results JSON.

    Returns:
        Dict keyed by repo name with pipeline results per repo.
    """
    from src.data_collector  import build_dataset_from_repo
    from src.preprocess      import preprocess_dataset
    from src.genetic_algorithm import FeatureSelectionGA

    if repo_names is None:
        repo_names = ['flask', 'requests', 'django']

    ga_defaults = dict(
        population_size  = 15,
        generations      = 10,
        mutation_rate    = 0.20,
        min_mutation_rate= 0.03,
        alpha            = 1.0,
        beta             = 0.5,
        stagnation_limit = 5,
        log_transform    = True,
    )
    if ga_kwargs:
        ga_defaults.update(ga_kwargs)

    os.makedirs(os.path.join(data_dir, 'results'), exist_ok=True)
    all_results = {}

    for name in repo_names:
        print(f"\n{'='*60}")
        print(f"  REPO: {name.upper()}")
        print(f"{'='*60}")
        t0 = time.time()

        # Paths
        repo_path   = _ensure_repo(name, base_dir)
        raw_csv     = os.path.join(data_dir, f'{name}_dataset.csv')
        clean_csv   = os.path.join(data_dir, f'{name}_dataset_clean.csv')
        ga_out      = os.path.join(data_dir, 'results', f'{name}_ga_results.json')
        ckpt_path   = os.path.join(data_dir, 'results', f'{name}_ga_checkpoint.json')

        # Step 1: Mine
        if not os.path.exists(raw_csv):
            build_dataset_from_repo(repo_path, raw_csv)
        else:
            print(f"  [skip] Raw data exists: {raw_csv}")

        # Step 2: Clean
        if not os.path.exists(clean_csv):
            preprocess_dataset(raw_csv, clean_csv)
        else:
            print(f"  [skip] Clean data exists: {clean_csv}")

        df = pd.read_csv(clean_csv)
        n_files    = len(df)
        n_features = len(df.columns) - 2

        # Step 3: GA
        if os.path.exists(ga_out):
            print(f"  [skip] GA results exist: {ga_out}")
            with open(ga_out) as f:
                ga_results = json.load(f)
        else:
            ga = FeatureSelectionGA(
                csv_file        = clean_csv,
                checkpoint_path = ckpt_path,
                **ga_defaults,
            )
            ga_results = ga.evolve()
            ga_results['chromosome'] = list(ga_results['chromosome'])
            with open(ga_out, 'w') as f:
                json.dump(ga_results, f, indent=2)

        # Step 4: Baseline (optional, time-consuming)
        baseline_summary = {}
        if run_baselines:
            from src.baseline import run_baselines as _run_baselines
            bl_out = os.path.join(data_dir, 'results', f'{name}_baseline_results.json')
            if os.path.exists(bl_out):
                with open(bl_out) as f:
                    bl = json.load(f)
            else:
                bl = _run_baselines(
                    csv_file      = clean_csv,
                    ga_chromosome = tuple(ga_results['chromosome']),
                    n_trials      = 15,
                    output_path   = bl_out,
                )
            baseline_summary = {
                'all_features_mse': bl['all_features']['mean'],
                'ga_ann_mse':       bl['ga_selected']['mean'],
                'xgb_ga_mse':       bl.get('xgb_ga', {}).get('mean', float('nan')),
                'improvement_pct':  (
                    (bl['all_features']['mean'] - bl['ga_selected']['mean'])
                    / (bl['all_features']['mean'] + 1e-9) * 100
                ),
            }

        elapsed = time.time() - t0
        all_results[name] = {
            'n_files':         n_files,
            'n_features_total':n_features,
            'n_selected':      ga_results['n_selected'],
            'reduction_pct':   round((1 - ga_results['n_selected'] / n_features) * 100, 1),
            'ga_best_mse':     ga_results['best_mse'],
            'selected_features': ga_results['feature_names'],
            'elapsed_s':       round(elapsed, 1),
            **baseline_summary,
        }

    # Save aggregate
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    _print_comparison_table(all_results)
    print(f"\n  Saved to: {output_path}")
    return all_results


def _print_comparison_table(results: dict) -> None:
    print("\n" + "=" * 75)
    print("  MULTI-REPOSITORY COMPARISON TABLE")
    print("=" * 75)
    header = f"  {'Repo':<12} {'Files':>6} {'Feats':>6} {'Sel':>4} {'Reduc%':>7} "
    header += f"{'GA MSE':>8} {'All MSE':>8} {'Improv%':>8}"
    print(header)
    print("-" * 75)
    for name, r in results.items():
        improv = r.get('improvement_pct', float('nan'))
        all_mse = r.get('all_features_mse', float('nan'))
        ga_mse  = r.get('ga_ann_mse', r['ga_best_mse'])
        print(f"  {name:<12} {r['n_files']:>6} {r['n_features_total']:>6} "
              f"{r['n_selected']:>4} {r['reduction_pct']:>6.1f}% "
              f"{ga_mse:>8.4f} {all_mse:>8.4f} {improv:>7.1f}%")
    print("=" * 75)


if __name__ == '__main__':
    run_multi_repo(repo_names=['flask', 'requests'])