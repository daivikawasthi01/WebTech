"""
src/report.py — Standalone HTML report generator.

Reads all JSON results from data/results/ and produces a single self-contained
HTML file that can be shared without running the Streamlit dashboard.
"""

import json
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def _badge(value: float, thresholds: list[float], labels: list[str],
           colors: list[str]) -> str:
    """
    Return a coloured <span> badge based on *value* vs *thresholds*.

    Example:
        _badge(0.03, [0.05, 0.10], ['Good', 'Warn', 'Poor'],
               ['#34d399', '#fbbf24', '#f87171'])
    """
    for threshold, label, color in zip(thresholds, labels, colors):
        if value <= threshold:
            return (
                f'<span style="background:{color};color:#fff;'
                f'padding:2px 8px;border-radius:4px;font-size:12px;">'
                f'{label}</span>'
            )
    # Fallback: last label / color
    return (
        f'<span style="background:{colors[-1]};color:#fff;'
        f'padding:2px 8px;border-radius:4px;font-size:12px;">'
        f'{labels[-1]}</span>'
    )


def _section(title: str, body: str) -> str:
    return (
        f'<section style="margin:32px 0;">'
        f'<h2 style="color:#7c3aed;border-bottom:1px solid #3b1f6e;'
        f'padding-bottom:6px;">{title}</h2>'
        f'{body}'
        f'</section>'
    )


def _kv_table(rows: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<tr><td style="padding:6px 12px;color:#94a3b8;">{k}</td>'
        f'<td style="padding:6px 12px;color:#e2e8f0;">{v}</td></tr>'
        for k, v in rows
    )
    return (
        '<table style="border-collapse:collapse;width:100%;'
        'background:#1e1e2e;border-radius:8px;">'
        f'{cells}</table>'
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def generate_report(
    clean_csv: str,
    output_path: str = "data/results/maintainability_report.html",
) -> str:
    """
    Generate a self-contained HTML report from all available result JSONs.
    Returns the path to the written file.
    """
    RESULTS_DIR = "data/results"
    ga_data      = _load(os.path.join(RESULTS_DIR, "ga_results.json"))
    bl_data      = _load(os.path.join(RESULTS_DIR, "baseline_results.json"))
    abl_data     = _load(os.path.join(RESULTS_DIR, "ablation_results.json"))
    stats_data   = _load(os.path.join(RESULTS_DIR, "stats_results.json"))
    multi_data   = _load(os.path.join(RESULTS_DIR, "multi_repo_results.json"))
    sens_data    = _load(os.path.join(RESULTS_DIR, "sensitivity_results.json"))

    sections: list[str] = []

    # ── GA summary ────────────────────────────────────────────────────────────
    if ga_data:
        mse_badge = _badge(
            ga_data["best_mse"],
            thresholds=[0.5, 1.0],
            labels=["Excellent", "Good", "Poor"],
            colors=["#34d399", "#fbbf24", "#f87171"],
        )
        rows = [
            ("Best MSE",           f"{ga_data['best_mse']:.4f} &nbsp;{mse_badge}"),
            ("Best Fitness",        f"{ga_data['best_fitness']:.4f}"),
            ("Features selected",   f"{ga_data['n_selected']} / {ga_data['n_total']}"),
            ("Feature reduction",   f"{(1 - ga_data['n_selected']/ga_data['n_total'])*100:.0f}%"),
            ("Elapsed",             f"{ga_data.get('elapsed_s', '—')} s"),
        ]
        sections.append(_section("GA Optimisation Summary", _kv_table(rows)))

    # ── Baseline comparison ───────────────────────────────────────────────────
    if bl_data:
        rows = []
        for method_key, label in [
            ("ga_selected",  "GA Selected"),
            ("all_features", "All Features"),
            ("random_subset","Random Subset"),
        ]:
            d = bl_data.get(method_key, {})
            mean = d.get("mean", float("nan"))
            std  = d.get("std",  float("nan"))
            badge = _badge(
                mean,
                thresholds=[0.5, 1.0],
                labels=["Good", "Moderate", "Poor"],
                colors=["#34d399", "#fbbf24", "#f87171"],
            )
            rows.append((label, f"{mean:.4f} ± {std:.4f} &nbsp;{badge}"))
        sections.append(_section("Baseline Comparison", _kv_table(rows)))

    # ── Statistical significance ──────────────────────────────────────────────
    if stats_data:
        sig_badge = _badge(
            stats_data["wilcoxon_p_value"],
            thresholds=[0.01, 0.05],
            labels=["p<0.01 ✓", "p<0.05 ✓", "Not significant"],
            colors=["#34d399", "#a3e635", "#f87171"],
        )
        rows = [
            ("Wilcoxon p-value",  f"{stats_data['wilcoxon_p_value']:.4f} &nbsp;{sig_badge}"),
            ("Cohen's d",          f"{stats_data['cohens_d']:.3f} ({stats_data['effect_size']})"),
            ("Improvement",        f"{stats_data['pct_improvement']:+.1f}%"),
        ]
        sections.append(_section("Statistical Significance", _kv_table(rows)))

    # ── Ablation study ────────────────────────────────────────────────────────
    if abl_data:
        sorted_combos = sorted(abl_data.items(), key=lambda x: x[1].get("mean", 999))
        rows = [
            (combo, f"{data['mean']:.4f} ± {data['std']:.4f}")
            for combo, data in sorted_combos
        ]
        sections.append(_section("Ablation Study (sorted by MSE)", _kv_table(rows)))

    # ── Multi-repo ────────────────────────────────────────────────────────────
    if multi_data:
        rows = [
            (repo,
             f"MSE {r.get('ga_ann_mse', r['ga_best_mse']):.4f} | "
             f"{r['n_selected']}/{r['n_features_total']} feats | "
             f"{r['reduction_pct']}% reduction")
            for repo, r in multi_data.items()
        ]
        sections.append(_section("Multi-Repository Generalisation", _kv_table(rows)))

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    body = "\n".join(sections) if sections else (
        "<p style='color:#94a3b8;'>No results found — run the pipeline first.</p>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Neuro-Genetic Maintainability Report</title>
  <style>
    body  {{ font-family: 'Segoe UI', sans-serif; background:#0f0f1a;
             color:#e2e8f0; margin:0; padding:32px 48px; }}
    h1    {{ color:#a78bfa; }}
    small {{ color:#64748b; }}
  </style>
</head>
<body>
  <h1>🧬 Neuro-Genetic Maintainability Report</h1>
  <small>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
  {body}
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[report] HTML report saved → {output_path}")
    return output_path