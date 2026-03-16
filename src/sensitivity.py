"""
sensitivity.py — Hyperparameter sensitivity analysis for the GA fitness function.

Answers: "How sensitive are the results to alpha, beta, and population size?"

Sweeps:
  alpha in [0.5, 1.0, 1.5, 2.0]   — accuracy weight
  beta  in [0.1, 0.5, 1.0, 2.0]   — parsimony weight
  pop   in [5, 10, 15, 20]         — population size

Output: data/results/sensitivity_results.json
"""

import json
import os
import time


_SENSITIVITY_GA_DEFAULTS = dict(
    generations       = 2,  # 2 gens: 4runs x 2gens x 8pop = ~64 ANN trains
    mutation_rate     = 0.20,
    min_mutation_rate = 0.03,
    stagnation_limit  = 3,
    log_transform     = True,
)


def run_sensitivity(
    csv_file: str,
    alphas: list     = None,
    betas: list      = None,
    pop_sizes: list  = None,
    output_path: str = 'data/results/sensitivity_results.json',
) -> dict:
    """
    Grid-sweeps alpha x beta x pop_size and records GA outcomes.

    Returns nested dict:
      results[alpha][beta][pop_size] = {
          'best_mse':   float,
          'n_selected': int,
          'fitness':    float,
          'elapsed_s':  float,
      }
    """
    from src.genetic_algorithm import FeatureSelectionGA

    alphas    = alphas    or [0.5, 1.5]   # 2 alpha values
    betas     = betas     or [0.1, 1.0]   # 2 beta values
    pop_sizes = pop_sizes or [8]          # 1 pop size — keeps total runs to 4 (2x2x1)

    total = len(alphas) * len(betas) * len(pop_sizes)
    print(f"\n[SENSITIVITY] Grid sweep: {total} combinations "
          f"({len(alphas)}a x {len(betas)}b x {len(pop_sizes)} pop sizes)\n")

    results = {}
    done    = 0

    for alpha in alphas:
        results[str(alpha)] = {}
        for beta in betas:
            results[str(alpha)][str(beta)] = {}
            for pop in pop_sizes:
                done += 1
                tag = f"a={alpha} b={beta} pop={pop}"
                print(f"  [{done:02d}/{total}] {tag}", end='  ', flush=True)
                t0 = time.time()

                ckpt = f"data/results/sensitivity_ckpt_{alpha}_{beta}_{pop}.json"
                ga   = FeatureSelectionGA(
                    csv_file        = csv_file,
                    population_size = pop,
                    alpha           = alpha,
                    beta            = beta,
                    checkpoint_path = ckpt,
                    **_SENSITIVITY_GA_DEFAULTS,
                )
                res     = ga.evolve()
                elapsed = time.time() - t0

                entry = {
                    'best_mse':   res['best_mse'],
                    'n_selected': res['n_selected'],
                    'fitness':    res['best_fitness'],
                    'elapsed_s':  round(elapsed, 1),
                }
                results[str(alpha)][str(beta)][str(pop)] = entry
                print(f"MSE={res['best_mse']:.4f}  "
                      f"feats={res['n_selected']}  ({elapsed:.0f}s)")

                if os.path.exists(ckpt):
                    os.remove(ckpt)

    # FIX 1: os.path.dirname() returns '' when output_path has no directory
    # component (e.g. 'results.json'), which causes makedirs('') to raise
    # FileNotFoundError on some platforms. Guard with `or '.'`.
    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        'alphas':    alphas,
        'betas':     betas,
        'pop_sizes': pop_sizes,
        'results':   results,
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    _print_sensitivity_summary(results, alphas, betas, pop_sizes)
    print(f"\n  Saved to: {output_path}")
    return payload


def _print_sensitivity_summary(results, alphas, betas, pop_sizes):
    mid_pop = str(pop_sizes[len(pop_sizes) // 2])
    print(f"\n  MSE grid (pop_size={mid_pop}) — lower is better")

    # FIX 2: Python 3.11 forbids backslashes inside f-string {} expressions.
    # Extract the column header string into a plain variable first.
    col_header = "a \\ b"
    header = f"  {col_header:>8} " + "".join(f"  b={b:<5}" for b in betas)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for alpha in alphas:
        row = f"  a={str(alpha):<6} "
        for beta in betas:
            mse = results[str(alpha)][str(beta)][mid_pop]['best_mse']
            row += f"  {mse:.4f} "
        print(row)


if __name__ == '__main__':
    run_sensitivity('data/flask_dataset_clean.csv')