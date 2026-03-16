"""
main.py — CLI orchestrator for the Hybrid Neuro-Genetic Maintainability Framework.

Pipeline stages:
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
    parser.add_argument("--pop-size",       type=int,   default=15)
    parser.add_argument("--generations",    type=int,   default=10)
    parser.add_argument("--mutation-rate",  type=float, default=0.20)
    parser.add_argument("--min-mutation",   type=float, default=0.03)
    parser.add_argument("--stagnation",     type=int,   default=5)
    parser.add_argument("--alpha",          type=float, default=1.0)
    parser.add_argument("--beta",           type=float, default=0.5)
    parser.add_argument("--log-transform",  action="store_true", default=True)

    # ── Research modules ──────────────────────────────────────────────────────
    parser.add_argument("--run-baselines",   action="store_true",
                        help="Run baseline comparison after GA (slow: n_trials x 3 ANNs)")
    parser.add_argument("--run-ablation",    action="store_true",
                        help="Run ablation study after GA (slow: 7 combos x n_trials)")
    parser.add_argument("--run-stats",       action="store_true",
                        help="Run Wilcoxon significance test after GA")
    parser.add_argument("--run-tuning",      action="store_true",
                        help="Run Optuna ANN hyperparameter search BEFORE GA "
                             "(recommended first run)")
    parser.add_argument("--tune-trials",     type=int, default=50,
                        help="Number of Optuna trials for hyperparameter tuning")
    parser.add_argument("--multi-repo",      action="store_true",
                        help="Run pipeline across multiple repos after GA")
    parser.add_argument("--repos",           nargs='+',
                        default=['flask', 'requests', 'django'],
                        help="Repo names for --multi-repo (must be in REPO_REGISTRY)")
    parser.add_argument("--run-sensitivity", action="store_true",
                        help="Run alpha/beta/pop_size sensitivity sweep")
    parser.add_argument("--run-report",      action="store_true",
                        help="Generate standalone HTML report from all results")
    parser.add_argument("--n-trials",        type=int, default=20,
                        help="Number of trials for baseline/ablation/stats")

    # ── Pipeline toggles ──────────────────────────────────────────────────────
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--run-all",       action="store_true",
                        help="Shorthand: enable baselines + ablation + stats + "
                             "multi-repo + sensitivity + report")
    parser.add_argument("--force-all",     action="store_true",
                        help="Re-run all stages even if outputs exist")

    args = parser.parse_args()

    if args.run_all:
        args.run_tuning = args.run_baselines = args.run_ablation = \
            args.run_stats = args.multi_repo = args.run_sensitivity = \
            args.run_report = True

    os.makedirs("data/results", exist_ok=True)

    t0 = time.time()

    print("=" * 60)
    print("   AUTO-MAINTAINABILITY NEURO-GENETIC FRAMEWORK")
    print("=" * 60)

    # ── Step 0: Clone repo if missing ─────────────────────────────────────────
    REPO_URLS = {
        "flask":    "https://github.com/pallets/flask.git",
        "requests": "https://github.com/psf/requests.git",
        "django":   "https://github.com/django/django.git",
    }

    if not os.path.isdir(args.repo):
        repo_name = os.path.basename(args.repo.rstrip("/"))
        clone_url = REPO_URLS.get(repo_name)
        if clone_url:
            print(f"\n[STEP 0] '{args.repo}' not found — cloning from {clone_url}")
            os.makedirs(os.path.dirname(args.repo) or ".", exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth=200", clone_url, args.repo],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"[ERROR] Clone failed:\n{result.stderr}")
                raise RuntimeError(f"Could not clone {clone_url}")
            print("  Cloned successfully.")
        else:
            raise ValueError(
                f"'{args.repo}' does not exist and has no entry in REPO_URLS. "
                f"Either push the repo or add it to REPO_URLS."
            )

    # ── Step 1: Mine ──────────────────────────────────────────────────────────
    if args.force_collect or args.force_all or not os.path.exists(args.raw_file):
        print(f"\n[STEP 1] Mining: {args.repo}")
        build_dataset_from_repo(args.repo, args.raw_file)
    else:
        print(f"\n[STEP 1] Skipping — using '{args.raw_file}'")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    if args.force_process or args.force_all or not os.path.exists(args.processed_file):
        print("\n[STEP 2] Cleaning dataset...")
        preprocess_dataset(args.raw_file, args.processed_file)
    else:
        print(f"\n[STEP 2] Skipping — using '{args.processed_file}'")

    # ── Step 2.5: Hyperparameter tuning ───────────────────────────────────────
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
    with open("data/results/ga_results.json", 'w') as f:
        json.dump(ga_results, f, indent=2)
    print(f"\n  GA results saved to data/results/ga_results.json")
    print(f"  GA time so far: {time.time() - t0:.1f}s")

    ga_chrom = tuple(ga_results['chromosome'])

    # ── Step 4: Baselines ─────────────────────────────────────────────────────
    if args.run_baselines:
        BASELINE_PATH = "data/results/baseline_results.json"
        if os.path.exists(BASELINE_PATH) and not (args.force_collect or args.force_all):
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
        if os.path.exists(ABLATION_PATH) and not (args.force_collect or args.force_all):
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
        if os.path.exists(STATS_PATH) and not (args.force_collect or args.force_all):
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
        if os.path.exists(MULTI_PATH) and not (args.force_collect or args.force_all):
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
        if os.path.exists(SENS_PATH) and not (args.force_collect or args.force_all):
            print("\n[STEP 8] Skipping — sensitivity_results.json already exists.")
        else:
            from src.sensitivity import run_sensitivity
            print("\n[STEP 8] Running hyperparameter sensitivity sweep...")
            run_sensitivity(csv_file=args.processed_file)

    # ── Step 9: HTML Report ───────────────────────────────────────────────────
    if args.run_report:
        from src.report import generate_report
        print("\n[STEP 9] Generating HTML report...")
        report_path = generate_report(
            clean_csv   = args.processed_file,
            output_path = 'data/results/maintainability_report.html',
        )
        print(f"  Open in browser: {os.path.abspath(report_path)}")

    print(f"\n[DONE] Total time: {time.time() - t0:.1f}s")
    print("  Launch dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()