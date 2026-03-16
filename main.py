"""
main.py — CLI orchestrator for the Hybrid Neuro-Genetic Maintainability Framework.

Pipeline stages:
  0   Auto-clone target repo if missing                     (subprocess git clone)
  1   Mine Git + AST metrics from the target repository     (data_collector)
  2   Clean and clip outliers; save unscaled CSV            (preprocess)
  2.5 Optional Optuna ANN hyperparameter tuning             (tune)        [--run-tuning]
  3   Run GA to evolve the optimal feature subset           (genetic_algorithm)
  4   Baseline comparison: GA vs All-Features vs Random     (baseline)    [--run-baselines]
  5   Ablation study: contribution of each metric category  (ablation)    [--run-ablation]
  6   Statistical significance test: Wilcoxon + Cohen's d   (stats)       [--run-stats]
  7   Multi-repo generalisation                             (multi_repo)  [--multi-repo]
  8   Hyperparameter sensitivity sweep                      (sensitivity) [--run-sensitivity]
  9   Standalone HTML report                                (report)      [--run-report]
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


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Maintainability Neuro-Genetic Framework"
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    parser.add_argument("--repo",           type=str, default="test_repos/flask")
    parser.add_argument("--raw-file",       type=str, default="data/flask_dataset.csv")
    parser.add_argument("--processed-file", type=str, default="data/flask_dataset_clean.csv")

    # ── GA hyperparameters ────────────────────────────────────────────────────
    parser.add_argument("--pop-size",       type=int,   default=8)
    parser.add_argument("--generations",    type=int,   default=5)
    parser.add_argument("--mutation-rate",  type=float, default=0.20)
    parser.add_argument("--min-mutation",   type=float, default=0.03)
    parser.add_argument("--stagnation",     type=int,   default=3)
    parser.add_argument("--alpha",          type=float, default=1.0)
    parser.add_argument("--beta",           type=float, default=0.5)
    # FIX Bug 1: action="store_true" + default=True means the flag is always
    # True and can never be disabled from CLI. Use store_true/store_false pair.
    parser.add_argument("--log-transform",    dest="log_transform",
                        action="store_true",  default=True)
    parser.add_argument("--no-log-transform", dest="log_transform",
                        action="store_false",
                        help="Disable log1p target transform")

    # ── Research modules ──────────────────────────────────────────────────────
    parser.add_argument("--run-baselines",   action="store_true",
                        help="Run baseline comparison after GA")
    parser.add_argument("--run-ablation",    action="store_true",
                        help="Run ablation study after GA")
    parser.add_argument("--run-stats",       action="store_true",
                        help="Run Wilcoxon significance test after GA")
    parser.add_argument("--run-tuning",      action="store_true",
                        help="Run Optuna ANN hyperparameter search BEFORE GA")
    parser.add_argument("--tune-trials",     type=int, default=50,
                        help="Number of Optuna trials for hyperparameter tuning")
    parser.add_argument("--multi-repo",      action="store_true",
                        help="Run pipeline across multiple repos after GA")
    parser.add_argument("--repos",           nargs='+',
                        default=['flask', 'requests', 'django'],
                        help="Repo names for --multi-repo")
    parser.add_argument("--run-sensitivity", action="store_true",
                        help="Run alpha/beta/pop_size sensitivity sweep")
    parser.add_argument("--run-report",      action="store_true",
                        help="Generate standalone HTML report from all results")
    parser.add_argument("--n-trials",        type=int, default=5,
                        help="Number of trials for baseline/ablation/stats")

    # ── Pipeline toggles ──────────────────────────────────────────────────────
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--run-all",       action="store_true",
                        help="Enable all research modules in one pass")
    parser.add_argument("--force-all",     action="store_true",
                        help="Re-run all stages even if outputs exist")

    args = parser.parse_args()

    if args.run_all:
        args.run_tuning = args.run_baselines = args.run_ablation = \
            args.run_stats = args.multi_repo = args.run_sensitivity = \
            args.run_report = True

    # FIX Bug 9: ensure output directories exist for any user-specified paths
    os.makedirs("data/results", exist_ok=True)
    raw_dir = os.path.dirname(args.raw_file)
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
    clean_dir = os.path.dirname(args.processed_file)
    if clean_dir:
        os.makedirs(clean_dir, exist_ok=True)

    t0 = time.time()

    print("=" * 60)
    print("   AUTO-MAINTAINABILITY NEURO-GENETIC FRAMEWORK")
    print("=" * 60)

    # FIX Bug 2: compute _only_sensitivity BEFORE Step 0 so we can skip the
    # clone entirely when just running a sensitivity sweep on existing data.
    GA_RESULTS_PATH = "data/results/ga_results.json"
    _ga_exists      = (os.path.exists(GA_RESULTS_PATH)
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

    # ── Step 0: Clone repo if missing ────────────────────────────────────────
    # Skip entirely if we are doing a sensitivity-only run on existing data —
    # the repo is not needed for sensitivity (it only needs the clean CSV).
    REPO_URLS = {
        "flask":    "https://github.com/pallets/flask.git",
        "requests": "https://github.com/psf/requests.git",
        "django":   "https://github.com/django/django.git",
    }

    if _only_sensitivity and _ga_exists and os.path.exists(args.processed_file):
        print("\n[STEP 0] Skipping clone — sensitivity-only run on existing data.")
    elif not os.path.isdir(args.repo):
        repo_name = os.path.basename(args.repo.rstrip("/"))
        clone_url = REPO_URLS.get(repo_name)
        if clone_url:
            print(f"\n[STEP 0] '{args.repo}' not found — cloning from {clone_url}")
            os.makedirs(os.path.dirname(args.repo) or ".", exist_ok=True)
            # FIX Bug 3: add timeout so a slow/hung clone doesn't block forever
            try:
                result = subprocess.run(
                    ["git", "clone", clone_url, args.repo],
                    capture_output=True, text=True, timeout=300
                )
            except subprocess.TimeoutExpired:
                print("[ERROR] git clone timed out after 5 minutes.")
                sys.exit(1)
            if result.returncode != 0:
                print(f"[ERROR] Clone failed:\n{result.stderr}")
                sys.exit(1)
            print("  Cloned successfully.")
        else:
            print(
                f"[ERROR] '{args.repo}' does not exist and has no entry in "
                f"REPO_URLS.\nEither push the repo, add it to REPO_URLS in "
                f"main.py, or point --repo at an existing local path."
            )
            sys.exit(1)

    # ── Step 1: Mine ─────────────────────────────────────────────────────────
    if args.force_collect or args.force_all or not os.path.exists(args.raw_file):
        print(f"\n[STEP 1] Mining: {args.repo}")
        # FIX Bug 4: wrap in try/except — ValueError raised when no Python files
        # are found (e.g. wrong repo path, too-shallow clone, empty snapshot).
        try:
            build_dataset_from_repo(args.repo, args.raw_file)
        except ValueError as e:
            print(f"\n[ERROR] Mining failed: {e}")
            print(
                "  Possible causes:\n"
                "  • The repo path is wrong or the clone is too shallow.\n"
                "  • All files returned empty features (no commits in window).\n"
                "  • Try --force-collect to re-mine, or pre-commit the CSV."
            )
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected mining error: {e}")
            sys.exit(1)
    else:
        print(f"\n[STEP 1] Skipping — using '{args.raw_file}'")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    # FIX Bug 5: also check that raw_file is non-empty before preprocessing.
    # An empty file (0 bytes) would cause pandas to raise an EmptyDataError.
    _raw_empty = (
        not os.path.exists(args.raw_file)
        or os.path.getsize(args.raw_file) == 0
    )
    if _raw_empty:
        print(f"\n[ERROR] Raw file is missing or empty: {args.raw_file}")
        print("  Run Step 1 first (mine the repository).")
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

    # ── Step 2.5: Hyperparameter tuning ──────────────────────────────────────
    HP_PATH = "data/results/best_hyperparams.json"
    if args.run_tuning:
        if os.path.exists(HP_PATH) and not (args.force_collect or args.force_all):
            print("\n[STEP 2.5] Skipping tuning — best_hyperparams.json already exists.")
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
        # Sensitivity-only run: load existing GA results to get ga_chrom,
        # then skip straight to Step 8. FIX Bug 6: args.run_tuning is always
        # defined by argparse so getattr fallback was unnecessary.
        print("\n[STEP 3] Skipping GA — using existing ga_results.json "
              "for sensitivity-only run.")
        try:
            with open(GA_RESULTS_PATH) as _f:
                ga_results = json.load(_f)
            # FIX Bug 7: validate that the loaded JSON has the required keys
            if 'chromosome' not in ga_results:
                raise KeyError("'chromosome' key missing from ga_results.json")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] ga_results.json is corrupt or uses an old schema: {e}")
            print("  Delete data/results/ga_results.json and re-run the GA.")
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
        if (os.path.exists(BASELINE_PATH) and os.path.getsize(BASELINE_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 4] Skipping — baseline_results.json already exists.")
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
        if (os.path.exists(ABLATION_PATH) and os.path.getsize(ABLATION_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 5] Skipping — ablation_results.json already exists.")
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
        if (os.path.exists(STATS_PATH) and os.path.getsize(STATS_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 6] Skipping — stats_results.json already exists.")
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
        if (os.path.exists(MULTI_PATH) and os.path.getsize(MULTI_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print("\n[STEP 7] Skipping — multi_repo_results.json already exists.")
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
                ),
                run_baselines = args.run_baselines,
            )

    # ── Step 8: Sensitivity ───────────────────────────────────────────────────
    if args.run_sensitivity:
        SENS_PATH = "data/results/sensitivity_results.json"
        _sens_valid = (os.path.exists(SENS_PATH)
                        and os.path.getsize(SENS_PATH) > 50)  # >50 bytes = not empty/corrupt
        if _sens_valid and not (args.force_collect or args.force_all):
            print("\n[STEP 8] Skipping — sensitivity_results.json already exists.")
        else:
            from src.sensitivity import run_sensitivity
            print("\n[STEP 8] Running hyperparameter sensitivity sweep...")
            run_sensitivity(csv_file=args.processed_file)

    # ── Step 9: HTML Report ───────────────────────────────────────────────────
    # FIX Bug 8: add skip guard — report is expensive and rarely needs to
    # regenerate unless results have changed. Use --force-all to force refresh.
    if args.run_report:
        REPORT_PATH = "data/results/maintainability_report.html"
        if (os.path.exists(REPORT_PATH) and os.path.getsize(REPORT_PATH) > 50
                and not (args.force_collect or args.force_all)):
            print(f"\n[STEP 9] Skipping — report already exists at {REPORT_PATH}")
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
    print("  Launch dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()