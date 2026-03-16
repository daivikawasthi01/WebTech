"""
stats.py — Statistical significance testing for research-grade claims.

Uses the Wilcoxon signed-rank test (non-parametric, appropriate for small n)
to determine whether GA-selected features produce statistically significantly
lower MSE than the all-features baseline.

Also computes:
  - Cohen's d (effect size) to indicate practical significance
  - Percentage improvement
  - p-value with significance threshold annotation

Outputs saved to data/results/stats_results.json.
"""

import json
import os
import numpy as np
# Removed unused `import pandas as pd` — it was only referenced by the
# dead `n_total = len(pd.read_csv(...))` line (removed below).
from scipy.stats import wilcoxon, ttest_rel

from src.ann_model import train_and_evaluate_ann


def _cohens_d(a: list, b: list) -> float:
    """
    Compute Cohen's d effect size between two paired samples.
    d = mean(a - b) / std(a - b)
    Interpretation: 0.2 small, 0.5 medium, 0.8 large.
    """
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff) / (np.std(diff) + 1e-9))


def run_significance_tests(
    csv_file: str,
    ga_chromosome: tuple,
    n_trials: int       = 30,
    log_transform: bool = True,
    alpha_level: float  = 0.05,
    output_path: str    = "data/results/stats_results.json",
) -> dict:
    """
    Runs n_trials paired evaluations of GA vs All-Features, then tests whether
    the MSE difference is statistically significant.

    Why paired? Each trial uses the same random seed for both methods, so any
    difference in MSE is attributable to feature selection, not data-split luck.

    Returns dict with p-values, effect sizes, and interpretation strings.
    """
    # Removed dead assignment: `n_total = len(pd.read_csv(csv_file))`
    # — this value was assigned and never used anywhere in the function.
    ga_mask = list(ga_chromosome)

    print(f"\n[STATS] Running {n_trials} paired trials (GA vs All Features)")

    ga_mses  = []
    all_mses = []

    for trial in range(n_trials):
        seed = trial * 13 + 3   # paired: same seed for both methods

        mse_ga = train_and_evaluate_ann(
            csv_file,
            feature_mask  = ga_mask,
            use_kfold     = False,   # FIX: was `use_kfold = ,` — SyntaxError
            split_seed    = seed,
            log_transform = log_transform,
        )
        mse_all = train_and_evaluate_ann(
            csv_file,
            feature_mask  = None,    # all features
            use_kfold     = False,
            split_seed    = seed,
            log_transform = log_transform,
        )
        ga_mses.append(mse_ga)
        all_mses.append(mse_all)

        print(f"  Trial {trial+1:02d}/{n_trials} — GA: {mse_ga:.4f}  All: {mse_all:.4f}")

    # --- Wilcoxon signed-rank test ---
    try:
        stat_wilcoxon, p_wilcoxon = wilcoxon(ga_mses, all_mses, alternative='less')
    except ValueError as e:
        stat_wilcoxon, p_wilcoxon = 0.0, 1.0
        print(f"  [WARN] Wilcoxon test issue: {e}")

    # --- Paired t-test ---
    stat_ttest, p_ttest = ttest_rel(ga_mses, all_mses)

    # --- Effect size ---
    d = _cohens_d(all_mses, ga_mses)

    if abs(d) < 0.2:
        effect_label = "negligible"
    elif abs(d) < 0.5:
        effect_label = "small"
    elif abs(d) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    mean_ga  = float(np.mean(ga_mses))
    mean_all = float(np.mean(all_mses))
    pct_improvement = float((mean_all - mean_ga) / (mean_all + 1e-9) * 100)
    significant     = bool(p_wilcoxon < alpha_level)

    results = {
        'n_trials':           n_trials,
        'ga_mses':            ga_mses,
        'all_mses':           all_mses,
        'mean_ga_mse':        mean_ga,
        'mean_all_mse':       mean_all,
        'std_ga_mse':         float(np.std(ga_mses)),
        'std_all_mse':        float(np.std(all_mses)),
        'pct_improvement':    pct_improvement,
        'wilcoxon_statistic': float(stat_wilcoxon),
        'wilcoxon_p_value':   float(p_wilcoxon),
        'ttest_statistic':    float(stat_ttest),
        'ttest_p_value':      float(p_ttest),
        'cohens_d':           float(d),
        'effect_size':        effect_label,
        'significant':        significant,
        'alpha_level':        alpha_level,
        'interpretation':     _interpret(significant, p_wilcoxon, d,
                                         pct_improvement, effect_label),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    print(f"\n  Saved to: {output_path}")
    return results


def _interpret(significant, p_val, d, pct_improvement, effect_label) -> str:
    direction = "lower" if d > 0 else "higher"
    sig_str   = (f"statistically significant (p={p_val:.4f})"
                 if significant
                 else f"not statistically significant (p={p_val:.4f})")
    return (
        f"GA-selected features achieved {abs(pct_improvement):.1f}% {direction} MSE "
        f"than the all-features baseline. The difference is {sig_str} "
        f"(Wilcoxon signed-rank, alpha=0.05) with a {effect_label} effect size "
        f"(Cohen's d={d:.3f})."
    )


def _print_summary(r: dict):
    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │  STATISTICAL SIGNIFICANCE TEST RESULTS          │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  GA Mean MSE       : {r['mean_ga_mse']:.4f} ± {r['std_ga_mse']:.4f}         │")
    print(f"  │  All-Feat Mean MSE : {r['mean_all_mse']:.4f} ± {r['std_all_mse']:.4f}         │")
    print(f"  │  Improvement       : {r['pct_improvement']:+.1f}%                        │")
    print(f"  │  Wilcoxon p-value  : {r['wilcoxon_p_value']:.4f}                        │")
    print(f"  │  Cohen's d         : {r['cohens_d']:.3f} ({r['effect_size']})               │")
    sig = "YES ✓" if r['significant'] else "NO  ✗"
    print(f"  │  Significant?      : {sig}                          │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  {r['interpretation'][:48]}  │")
    print("  └──────────────────────────────────────────────────┘")


if __name__ == "__main__":
    import json
    with open("data/results/ga_results.json") as f:
        ga_results = json.load(f)
    run_significance_tests(
        csv_file      = "data/flask_dataset.csv",
        ga_chromosome = tuple(ga_results['chromosome']),
        n_trials      = 20,
    )