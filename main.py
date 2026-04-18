"""
main.py — CLI orchestrator for the Neuro-Genetic Maintainability pipeline.

Called by the FastAPI backend as a subprocess so its stdout streams live
to the WebSocket log console. Also usable standalone:

  python main.py --repo test_repos/flask --run-all
  python main.py --repo test_repos/flask --run-tuning --run-baselines
  python main.py --repo test_repos/flask --run-sensitivity

All stages are skipped if their output file already exists unless a
--force-* flag is passed.
"""

import os

# Prevent segmentation faults on macOS by resolving OpenMP conflicts
# between PyTorch and XGBoost. Must be set BEFORE libraries are imported.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import os
import sys
import time

# Ensure src/ is importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = "data/results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _banner(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def _skip(path: str, label: str) -> bool:
    if _exists(path):
        print(f"  [SKIP] {label} — output already exists: {path}")
        return True
    return False


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_collect(repo_path, raw_file, timeframe_months, force):
    _banner("STAGE 1 — Data Collection")
    if not force and _skip(raw_file, "Data collection"):
        return
    from src.data_collector import build_dataset_from_repo
    build_dataset_from_repo(repo_path, raw_file, timeframe_months=timeframe_months)


def stage_preprocess(raw_file, processed_file, force):
    _banner("STAGE 2 — Preprocessing")
    if not force and _skip(processed_file, "Preprocessing"):
        return
    if not _exists(raw_file):
        print(f"  [ERROR] Raw file not found: {raw_file}")
        sys.exit(1)
    from src.preprocess import preprocess_dataset
    preprocess_dataset(raw_file, processed_file)


def stage_tune(processed_file, tune_trials, force):
    _banner("STAGE 3 — Hyperparameter Tuning")
    hp_path = os.path.join(RESULTS_DIR, "best_hyperparams.json")
    if not force and _skip(hp_path, "Hyperparameter tuning"):
        return
    from src.tune import run_tuning
    run_tuning(processed_file, n_trials=tune_trials)


def stage_ga(processed_file, args, force):
    _banner("STAGE 4 — GA Feature Selection")
    ga_path = os.path.join(RESULTS_DIR, "ga_results.json")
    if not force and _skip(ga_path, "GA"):
        return
    t0 = time.time()
    from src.genetic_algorithm import FeatureSelectionGA
    ga = FeatureSelectionGA(
        csv_file          = processed_file,
        population_size   = args.pop_size,
        generations       = args.generations,
        mutation_rate     = args.mutation_rate,
        min_mutation_rate = args.min_mutation,
        alpha             = args.alpha,
        beta              = args.beta,
        stagnation_limit  = args.stagnation,
    )
    result = ga.evolve()
    # Persist final result with elapsed time
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result['elapsed_s'] = round(time.time() - t0, 1)
    result['complete']  = True
    with open(ga_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  GA complete — best MSE: {result['best_mse']:.4f} | "
          f"features: {result['n_selected']}/{result['n_total']}")
    return result


def _load_ga_chromosome():
    ga_path = os.path.join(RESULTS_DIR, "ga_results.json")
    if not _exists(ga_path):
        print("  [ERROR] ga_results.json not found. Run GA first.")
        sys.exit(1)
    with open(ga_path) as f:
        return tuple(json.load(f)['chromosome'])


def stage_baselines(processed_file, n_trials, force):
    _banner("STAGE 5 — Baseline Comparison")
    bl_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    if not force and _skip(bl_path, "Baselines"):
        return
    chromosome = _load_ga_chromosome()
    from src.baseline import run_baselines
    run_baselines(processed_file, ga_chromosome=chromosome, n_trials=n_trials)


def stage_ablation(processed_file, n_trials, force):
    _banner("STAGE 6 — Ablation Study")
    abl_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    if not force and _skip(abl_path, "Ablation"):
        return
    from src.ablation import run_ablation
    run_ablation(processed_file, n_trials=n_trials)


def stage_stats(processed_file, n_trials, force):
    _banner("STAGE 7 — Statistical Significance")
    stats_path = os.path.join(RESULTS_DIR, "stats_results.json")
    if not force and _skip(stats_path, "Stats"):
        return
    chromosome = _load_ga_chromosome()
    from src.stats import run_significance_tests
    run_significance_tests(processed_file, ga_chromosome=chromosome, n_trials=n_trials)


def stage_multi_repo(repo_names, ga_kwargs, run_baselines_flag, force):
    _banner("STAGE 8 — Multi-Repository Generalisation")
    mr_path = os.path.join(RESULTS_DIR, "multi_repo_results.json")
    if not force and _skip(mr_path, "Multi-repo"):
        return
    from src.multi_repo import run_multi_repo
    run_multi_repo(
        repo_names    = repo_names,
        ga_kwargs     = ga_kwargs,
        run_baselines = run_baselines_flag,
    )


def stage_sensitivity(processed_file, force):
    _banner("STAGE 9 — Sensitivity Sweep")
    sens_path = os.path.join(RESULTS_DIR, "sensitivity_results.json")
    if not force and _skip(sens_path, "Sensitivity"):
        return
    from src.sensitivity import run_sensitivity
    run_sensitivity(processed_file)


def stage_report(processed_file, force):
    _banner("STAGE 10 — HTML Report")
    report_path = os.path.join(RESULTS_DIR, "maintainability_report.html")
    if not force and _skip(report_path, "Report"):
        return
    from src.report import generate_report
    generate_report(processed_file)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Neuro-Genetic Maintainability Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--repo",            default="test_repos/flask")
    p.add_argument("--raw-file",        default="data/flask_dataset.csv")
    p.add_argument("--processed-file",  default="data/flask_dataset_clean.csv")
    p.add_argument("--timeframe-months",type=int, default=12)

    # GA
    p.add_argument("--pop-size",        type=int,   default=15)
    p.add_argument("--generations",     type=int,   default=10)
    p.add_argument("--mutation-rate",   type=float, default=0.20)
    p.add_argument("--min-mutation",    type=float, default=0.03)
    p.add_argument("--stagnation",      type=int,   default=5)
    p.add_argument("--alpha",           type=float, default=1.0)
    p.add_argument("--beta",            type=float, default=0.5)

    # Research modules
    p.add_argument("--n-trials",        type=int,   default=20)
    p.add_argument("--tune-trials",     type=int,   default=50)
    p.add_argument("--run-tuning",      action="store_true")
    p.add_argument("--run-baselines",   action="store_true")
    p.add_argument("--run-ablation",    action="store_true")
    p.add_argument("--run-stats",       action="store_true")
    p.add_argument("--multi-repo",      action="store_true")
    p.add_argument("--repos",           nargs="+",  default=["flask", "requests", "django"])
    p.add_argument("--run-sensitivity", action="store_true")
    p.add_argument("--run-report",      action="store_true")
    p.add_argument("--run-all",         action="store_true")

    # Force flags
    p.add_argument("--force-collect",   action="store_true")
    p.add_argument("--force-process",   action="store_true")
    p.add_argument("--force-ga",        action="store_true")
    p.add_argument("--force-all",       action="store_true")

    # Skip flags (for partial re-runs)
    p.add_argument("--skip-collect",    action="store_true")
    p.add_argument("--skip-ga",         action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    force_all = args.force_all

    # Resolve --run-all
    if args.run_all:
        args.run_tuning     = True
        args.run_baselines  = True
        args.run_ablation   = True
        args.run_stats      = True
        args.multi_repo     = True
        args.run_sensitivity = True
        args.run_report     = True

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.raw_file) or '.', exist_ok=True)

    print(f"\n🧬 Neuro-Genetic Maintainability Pipeline")
    print(f"   Repo     : {args.repo}")
    print(f"   Raw CSV  : {args.raw_file}")
    print(f"   Clean CSV: {args.processed_file}")
    print(f"   GA       : pop={args.pop_size}  gens={args.generations}  "
          f"α={args.alpha}  β={args.beta}")

    # Stage 1: collect
    if not args.skip_collect:
        stage_collect(args.repo, args.raw_file,
                      args.timeframe_months,
                      force=force_all or args.force_collect)

    # Stage 2: preprocess
    stage_preprocess(args.raw_file, args.processed_file,
                     force=force_all or args.force_process)

    if not _exists(args.processed_file):
        print(f"\n[ERROR] Processed file missing after preprocess: {args.processed_file}")
        sys.exit(1)

    # Stage 3: tune (optional)
    if args.run_tuning:
        stage_tune(args.processed_file, args.tune_trials, force=force_all)

    # Stage 4: GA
    if not args.skip_ga:
        stage_ga(args.processed_file, args, force=force_all or args.force_ga)

    # Stage 5: baselines
    if args.run_baselines:
        stage_baselines(args.processed_file, args.n_trials, force=force_all)

    # Stage 6: ablation
    if args.run_ablation:
        stage_ablation(args.processed_file, args.n_trials, force=force_all)

    # Stage 7: stats
    if args.run_stats:
        stage_stats(args.processed_file, args.n_trials, force=force_all)

    # Stage 8: multi-repo
    if args.multi_repo:
        ga_kwargs = dict(
            population_size   = args.pop_size,
            generations       = args.generations,
            mutation_rate     = args.mutation_rate,
            min_mutation_rate = args.min_mutation,
            alpha             = args.alpha,
            beta              = args.beta,
            stagnation_limit  = args.stagnation,
        )
        stage_multi_repo(args.repos, ga_kwargs,
                         run_baselines_flag=args.run_baselines,
                         force=force_all)

    # Stage 9: sensitivity
    if args.run_sensitivity:
        stage_sensitivity(args.processed_file, force=force_all)

    # Stage 10: report
    if args.run_report:
        stage_report(args.processed_file, force=force_all)

    _banner("PIPELINE COMPLETE ✓")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Dashboard: http://localhost:5173")


if __name__ == "__main__":
    main()
