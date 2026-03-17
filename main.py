"""
main.py — CLI orchestrator for the Hybrid Neuro-Genetic Maintainability Framework.

Production version — local runs without cloud constraints.
Changes from cloud version:
  - GA defaults: pop=20, gen=30, stagnation=8, n-trials=30
  - Added --timeframe-months (passes through to data_collector)
  - Added --skip-ga flag to mine only, enabling combine_datasets.py workflow
  - Added --n-epochs and --no-kfold for GA fitness evaluation control
  - Expanded REPO_URLS with more open-source Python repos
  - Removed 300s clone timeout (cloud workaround)
  - Combined dataset workflow documented in help strings

Pipeline stages:
  0   Auto-clone target repo if missing
  1   Mine Git + AST metrics                    (data_collector)
  2   Clean and clip outliers                   (preprocess)
  2.5 Optional Optuna ANN hyperparameter tuning (tune)        [--run-tuning]
  3   GA feature selection                      (genetic_algorithm)
  4   Baseline comparison                       (baseline)    [--run-baselines]
  5   Ablation study                            (ablation)    [--run-ablation]
  6   Statistical significance tests            (stats)       [--run-stats]
  7   Multi-repo generalisation                 (multi_repo)  [--multi-repo]
  8   Hyperparameter sensitivity sweep          (sensitivity) [--run-sensitivity]
  9   Standalone HTML report                    (report)      [--run-report]
"""

import os
import sys
import json
import time
import argparse
import subprocess

from src.data_collector    import build_dataset_from_repo
from src.preprocess        import preprocess_dataset
from src.genetic_algorithm import FeatureSelectionGA


# Repos that can be auto-cloned by name
REPO_URLS = {
    "flask":    "https://github.com/pallets/flask.git",
    "requests": "https://github.com/psf/requests.git",
    "django":   "https://github.com/django/django.git",
    "click":    "https://github.com/pallets/click.git",
    "httpx":    "https://github.com/encode/httpx.git",
    "fastapi":  "https://github.com/tiangolo/fastapi.git",
    "rich":     "https://github.com/Textualize/rich.git",
    "black":    "https://github.com/psf/black.git",
    "pytest":   "https://github.com/pytest-dev/pytest.git",
    "pydantic": "https://github.com/pydantic/pydantic.git",
}


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Maintainability Neuro-Genetic Framework — Production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mine one repo
  python main.py --repo test_repos/flask --raw-file data/flask_dataset.csv --skip-ga

  # Mine multiple repos, then combine, then run GA on combined dataset
  python main.py --repo test_repos/flask    --raw-file data/flask_dataset.csv    --skip-ga
  python main.py --repo test_repos/requests --raw-file data/requests_dataset.csv --skip-ga
  python combine_datasets.py
  python main.py --raw-file data/combined_dataset.csv \\
                 --processed-file data/combined_dataset_clean.csv \\
                 --pop-size 20 --generations 30 --run-all

  # Full pipeline on one repo
  python main.py --repo test_repos/flask --run-all
        """
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    parser.add_argument("--repo",            type=str,
                        default="test_repos/flask",
                        help="Path to the Git repository to analyse")
    parser.add_argument("--raw-file",        type=str,
                        default="data/flask_dataset.csv",
                        help="Output path for mined raw CSV")
    parser.add_argument("--processed-file",  type=str,
                        default="data/flask_dataset_clean.csv",
                        help="Output path for cleaned CSV")

    # ── Mining ────────────────────────────────────────────────────────────────
    parser.add_argument("--timeframe-months", type=int, default=12,
                        help="Snapshot lookback window in months (default 12). "
                             "Cloud version used 3.")
    parser.add_argument("--skip-ga",          action="store_true",
                        help="Mine and clean only — skip GA and all research "
                             "modules. Use when mining multiple repos before "
                             "combining with combine_datasets.py.")

    # ── GA hyperparameters ────────────────────────────────────────────────────
    parser.add_argument("--pop-size",         type=int,   default=20,
                        help="Chromosomes per generation (default 20)")
    parser.add_argument("--generations",      type=int,   default=30,
                        help="Maximum GA generations (default 30)")
    parser.add_argument("--mutation-rate",    type=float, default=0.20,
                        help="Initial per-bit mutation probability")
    parser.add_argument("--min-mutation",     type=float, default=0.03,
                        help="Mutation rate floor (adaptive decay target)")
    parser.add_argument("--stagnation",       type=int,   default=8,
                        help="Stop if no improvement for N generations (default 8)")
    parser.add_argument("--alpha",            type=float, default=1.0,
                        help="Fitness accuracy weight")
    parser.add_argument("--beta",             type=float, default=0.5,
                        help="Fitness parsimony weight")
    parser.add_argument("--n-epochs",         type=int,   default=100,
                        help="ANN epochs per fitness evaluation (default 100)")
    parser.add_argument("--no-kfold",         action="store_true",
                        help="Use single train/val split instead of k-fold "
                             "during GA (faster but noisier fitness signal)")
    parser.add_argument("--log-transform",    dest="log_transform",
                        action="store_true",  default=True)
    parser.add_argument("--no-log-transform", dest="log_transform",
                        action="store_false",
                        help="Disable log1p target transform")

    # ── Research modules ──────────────────────────────────────────────────────
    parser.add_argument("--run-baselines",    action="store_true")
    parser.add_argument("--run-ablation",     action="store_true")
    parser.add_argument("--run-stats",        action="store_true")
    parser.add_argument("--run-tuning",       action="store_true",
                        help="Run Optuna ANN hyperparameter search before GA")
    parser.add_argument("--tune-trials",      type=int,   default=50)
    parser.add_argument("--multi-repo",       action="store_true")
    parser.add_argument("--repos",            nargs='+',
                        default=['flask', 'requests', 'click', 'httpx', 'fastapi'],
                        help="Repo names for --multi-repo")
    parser.add_argument("--run-sensitivity",  action="store_true")
    parser.add_argument("--run-report",       action="store_true")
    parser.add_argument("--n-trials",         type=int,   default=30,
                        help="Trials per method for baselines/ablation/stats "
                             "(default 30)")

    # ── Pipeline toggles ──────────────────────────────────────────────────────
    parser.add_argument("--force-collect",    action="store_true",
                        help="Re-mine even if raw CSV exists")
    parser.add_argument("--force-process",    action="store_true",
                        help="Re-clean even if processed CSV exists")
    parser.add_argument("--run-all",          action="store_true",
                        help="Enable all research modules")
    parser.add_argument("--force-all",        action="store_true",
                        help="Re-run every stage regardless of existing outputs")

    args = parser.parse_args()

    if args.run_all:
        args.run_tuning = args.run_baselines = args.run_ablation = \
            args.run_stats = args.multi_repo = args.run_sensitivity = \
            args.run_report = True

    # Ensure output directories exist
    os.makedirs("data/results", exist_ok=True)
    for path in [args.raw_file, args.processed_file]:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    t0 = time.time()

    print("=" * 60)
    print("   AUTO-MAINTAINABILITY NEURO-GENETIC FRAMEWORK")
    print(f"   Timeframe: {args.timeframe_months} months | "
          f"Pop: {args.pop_size} | Gen: {args.generations} | "
          f"Epochs: {args.n_epochs}")
    print("=" * 60)

    # Sensitivity-only shortcut
    GA_RESULTS_PATH = "data/results/ga_results.json"
    _ga_exists = (os.path.exists(GA_RESULTS_PATH)
                  and os.path.getsize(GA_RESULTS_PATH) > 0)
    _only_sensitivity = (
        args.run_sensitivity
        and not args.run_baselines
        and not args.run_ablation
        and not args.run_stats
        and not args.multi_repo
        and not args.run_tuning
        and not args.force_all
    )

    # ── Step 0: Clone repo if missing ─────────────────────────────────────────
    if _only_sensitivity and _ga_exists and os.path.exists(args.processed_file):
        print("\n[STEP 0] Skipping clone — sensitivity-only run.")
    elif not os.path.isdir(args.repo):
        repo_name = os.path.basename(args.repo.rstrip("/"))
        clone_url = REPO_URLS.get(repo_name)
        if clone_url:
            print(f"\n[STEP 0] '{args.repo}' not found — cloning from {clone_url}")
            os.makedirs(os.path.dirname(args.repo) or ".", exist_ok=True)
            try:
                result = subprocess.run(
                    ["git", "clone", clone_url, args.repo],
                    capture_output=True, text=True
                )
            except Exception as e:
                print(f"[ERROR] git clone failed: {e}")
                sys.exit(1)
            if result.returncode != 0:
                print(f"[ERROR] Clone failed:\n{result.stderr}")
                sys.exit(1)
            print("  Cloned successfully.")
        else:
            print(
                f"[ERROR] '{args.repo}' does not exist and has no entry in "
                f"REPO_URLS.\nAvailable names: {list(REPO_URLS.keys())}"
            )
            sys.exit(1)

    # ── Step 1: Mine ──────────────────────────────────────────────────────────
    if args.force_collect or args.force_all or not os.path.exists(args.raw_file):
        print(f"\n[STEP 1] Mining: {args.repo} "
              f"(timeframe={args.timeframe_months} months)")
        try:
            build_dataset_from_repo(
                args.repo,
                args.raw_file,
                timeframe_months=args.timeframe_months,
            )
        except ValueError as e:
            print(f"\n[ERROR] Mining failed: {e}")
            print(
                "  Possible causes:\n"
                "  • Wrong repo path or not enough commit history\n"
                "  • Increase --timeframe-months or use --force-collect after\n"
                "    cloning with full history (remove --depth from git clone)"
            )
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected mining error: {e}")
            sys.exit(1)
    else:
        print(f"\n[STEP 1] Skipping — using '{args.raw_file}'")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    _raw_empty = (not os.path.exists(args.raw_file)
                  or os.path.getsize(args.raw_file) == 0)
    if _raw_empty:
        print(f"\n[ERROR] Raw file missing or empty: {args.raw_file}")
        sys.exit(1)

    if args.force_process or args.force_all or not os.path.exists(args.processed_file):
        print("\n[STEP 2] Cleaning dataset...")
        try:
            preprocess_dataset(args.raw_file, args.processed_file)
        except Exception as e:
            print(f"\n[ERROR] Preprocessing failed: {e}")
            sys.exit(1)
    else:
        print(f"\n[STEP 2] Skipping — using '{args.processed_file}'")

    # ── SKIP-GA early exit ────────────────────────────────────────────────────
    if args.skip_ga:
        print(f"\n[DONE] Mine + clean complete. "
              f"Total time: {time.time() - t0:.1f}s")
        print("  Next: python combine_datasets.py")
        return

    # ── Step 2.5: Hyperparameter tuning ───────────────────────────────────────
    HP_PATH = "data/results/best_hyperparams.json"
    if args.run_tuning:
        if (os.path.exists(HP_PATH)
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 2.5] Skipping — best_hyperparams.json exists.")
        else:
            from src.tune import run_tuning
            print("\n[STEP 2.5] Optuna ANN hyperparameter tuning...")
            run_tuning(
                csv_file      = args.processed_file,
                n_trials      = args.tune_trials,
                log_transform = args.log_transform,
            )

    # ── Step 3: GA ────────────────────────────────────────────────────────────
    if _only_sensitivity and _ga_exists:
        print("\n[STEP 3] Skipping GA — using existing ga_results.json.")
        try:
            with open(GA_RESULTS_PATH) as _f:
                ga_results = json.load(_f)
            if 'chromosome' not in ga_results:
                raise KeyError("'chromosome' key missing")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] ga_results.json corrupt or old schema: {e}")
            print("  Delete data/results/ga_results.json and re-run.")
            sys.exit(1)
    else:
        print("\n[STEP 3] Starting Neuro-Genetic Optimisation...")
        ga = FeatureSelectionGA(
            csv_file          = args.processed_file,
            population_size   = args.pop_size,
            generations       = args.generations,
            mutation_rate     = args.mutation_rate,
            min_mutation_rate = args.min_mutation,
            alpha             = args.alpha,
            beta              = args.beta,
            stagnation_limit  = args.stagnation,
            log_transform     = args.log_transform,
            n_epochs          = args.n_epochs,
            use_kfold         = not args.no_kfold,
        )
        ga_results = ga.evolve()
        ga_results['chromosome'] = list(ga_results['chromosome'])
        with open(GA_RESULTS_PATH, 'w') as f:
            json.dump(ga_results, f, indent=2)
        print(f"\n  GA results saved to {GA_RESULTS_PATH}")
        print(f"  GA time so far: {time.time() - t0:.1f}s")

    ga_chrom = tuple(ga_results['chromosome'])

    # ── Step 4: Baselines ─────────────────────────────────────────────────────
    if args.run_baselines:
        BASELINE_PATH = "data/results/baseline_results.json"
        if (os.path.exists(BASELINE_PATH)
                and os.path.getsize(BASELINE_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 4] Skipping — baseline_results.json exists.")
        else:
            from src.baseline import run_baselines
            run_baselines(
                csv_file      = args.processed_file,
                ga_chromosome = ga_chrom,
                n_trials      = args.n_trials,
                log_transform = args.log_transform,
            )

    # ── Step 5: Ablation ──────────────────────────────────────────────────────
    if args.run_ablation:
        ABLATION_PATH = "data/results/ablation_results.json"
        if (os.path.exists(ABLATION_PATH)
                and os.path.getsize(ABLATION_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 5] Skipping — ablation_results.json exists.")
        else:
            from src.ablation import run_ablation
            run_ablation(
                csv_file      = args.processed_file,
                n_trials      = args.n_trials,
                log_transform = args.log_transform,
            )

    # ── Step 6: Stats ─────────────────────────────────────────────────────────
    if args.run_stats:
        STATS_PATH = "data/results/stats_results.json"
        if (os.path.exists(STATS_PATH)
                and os.path.getsize(STATS_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 6] Skipping — stats_results.json exists.")
        else:
            from src.stats import run_significance_tests
            run_significance_tests(
                csv_file      = args.processed_file,
                ga_chromosome = ga_chrom,
                n_trials      = args.n_trials,
                log_transform = args.log_transform,
            )

    # ── Step 7: Multi-repo ────────────────────────────────────────────────────
    if args.multi_repo:
        MULTI_PATH = "data/results/multi_repo_results.json"
        if (os.path.exists(MULTI_PATH)
                and os.path.getsize(MULTI_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 7] Skipping — multi_repo_results.json exists.")
        else:
            from src.multi_repo import run_multi_repo
            print("\n[STEP 7] Running multi-repository comparison...")
            run_multi_repo(
                repo_names    = args.repos,
                ga_kwargs     = dict(
                    population_size   = args.pop_size,
                    generations       = args.generations,
                    mutation_rate     = args.mutation_rate,
                    min_mutation_rate = args.min_mutation,
                    alpha             = args.alpha,
                    beta              = args.beta,
                    stagnation_limit  = args.stagnation,
                    n_epochs          = args.n_epochs,
                    use_kfold         = not args.no_kfold,
                ),
                run_baselines = args.run_baselines,
            )

    # ── Step 8: Sensitivity ───────────────────────────────────────────────────
    if args.run_sensitivity:
        SENS_PATH = "data/results/sensitivity_results.json"
        _sens_valid = (os.path.exists(SENS_PATH)
                       and os.path.getsize(SENS_PATH) > 50)
        if _sens_valid and not (args.force_collect or args.force_all):
            print("\n[STEP 8] Skipping — sensitivity_results.json exists.")
        else:
            from src.sensitivity import run_sensitivity
            print("\n[STEP 8] Running hyperparameter sensitivity sweep...")
            run_sensitivity(csv_file=args.processed_file)

    # ── Step 9: HTML Report ───────────────────────────────────────────────────
    if args.run_report:
        REPORT_PATH = "data/results/maintainability_report.html"
        if (os.path.exists(REPORT_PATH)
                and os.path.getsize(REPORT_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print(f"\n[STEP 9] Skipping — report exists at {REPORT_PATH}")
        else:
            from src.report import generate_report
            print("\n[STEP 9] Generating HTML report...")
            try:
                report_path = generate_report(
                    clean_csv   = args.processed_file,
                    output_path = REPORT_PATH,
                )
                print(f"  Open in browser: {os.path.abspath(report_path)}")
            except Exception as e:
                print(f"  [WARN] Report generation failed (non-fatal): {e}")

    print(f"\n[DONE] Total time: {time.time() - t0:.1f}s")
    print("  View results: streamlit run app.py")


if __name__ == "__main__":
    main()