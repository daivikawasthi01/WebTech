"""
report.py — Generates a standalone HTML report from all pipeline results.

Produces a single self-contained HTML file (no external dependencies) that
includes:
  - Executive summary metrics
  - Selected feature list with category labels
  - GA convergence table
  - Baseline comparison table
  - Ablation study table
  - Statistical significance result
  - Multi-repo comparison table (if available)
  - Per-file risk score table (top 30 riskiest files)

The HTML is fully offline-capable — suitable for sharing with supervisors,
attaching to a submission, or displaying in a corporate dashboard.

Usage:
  python -m src.report
  python main.py --run-report
"""

import json
import os
from datetime import datetime


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _badge(value, thresholds, labels, colors):
    """Return a coloured badge span based on value vs thresholds."""
    for thresh, label, color in zip(thresholds, labels, colors):
        if value <= thresh:
            return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{label}</span>'
    t, l, c = thresholds[-1], labels[-1], colors[-1]
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{l}</span>'


def generate_report(
    results_dir: str = 'data/results',
    clean_csv: str   = 'data/flask_dataset_clean.csv',
    output_path: str = 'data/results/maintainability_report.html',
) -> str:
    """
    Reads all available result JSONs and writes a standalone HTML report.
    Returns the output path.
    """
    ga       = _load(os.path.join(results_dir, 'ga_results.json'))
    bl       = _load(os.path.join(results_dir, 'baseline_results.json'))
    abl      = _load(os.path.join(results_dir, 'ablation_results.json'))
    stats    = _load(os.path.join(results_dir, 'stats_results.json'))
    multi    = _load(os.path.join(results_dir, 'multi_repo_results.json'))
    hp       = _load(os.path.join(results_dir, 'best_hyperparams.json'))
    sens     = _load(os.path.join(results_dir, 'sensitivity_results.json'))

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── Shared CSS ────────────────────────────────────────────────────────
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f0f1a; color: #e2e8f0; line-height: 1.6; }
    .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
    h1 { font-size: 28px; font-weight: 700; color: #a78bfa; margin-bottom: 4px; }
    h2 { font-size: 18px; font-weight: 600; color: #c4b5fd; margin: 32px 0 12px;
         border-left: 4px solid #7c3aed; padding-left: 12px; }
    h3 { font-size: 14px; font-weight: 600; color: #94a3b8;
         text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
    .subtitle { color: #64748b; font-size: 14px; margin-bottom: 32px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px; margin-bottom: 24px; }
    .card { background: #1e1e2e; border-radius: 10px; padding: 20px;
            border-left: 4px solid #7c3aed; }
    .card .val { font-size: 28px; font-weight: 700; color: #a78bfa; }
    .card .lbl { font-size: 12px; color: #64748b; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 8px; }
    th { background: #1e1e2e; color: #94a3b8; padding: 10px 14px;
         text-align: left; font-weight: 600; }
    td { padding: 9px 14px; border-bottom: 1px solid #1e293b; }
    tr:hover td { background: #1e1e2e; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 11px; font-weight: 600; }
    .tag-s { background: #3b1f6e; color: #c4b5fd; }
    .tag-t { background: #0c3352; color: #7dd3fc; }
    .tag-e { background: #0a3d2e; color: #6ee7b7; }
    .sig-yes { color: #34d399; font-weight: 700; }
    .sig-no  { color: #f87171; font-weight: 700; }
    .section { background: #13131f; border-radius: 10px;
               padding: 24px; margin-bottom: 24px; }
    footer { color: #334155; font-size: 12px; margin-top: 40px; text-align: center; }
    """

    # ── Build sections ─────────────────────────────────────────────────────

    sections = []

    # Header
    sections.append(f"""
    <h1>🧬 Software Maintainability Report</h1>
    <p class="subtitle">Hybrid Neuro-Genetic Framework &nbsp;·&nbsp; Generated {generated_at}</p>
    """)

    # Summary cards
    if ga:
        improv = ''
        if bl:
            improv_val = (bl['all_features']['mean'] - bl['ga_selected']['mean']) / \
                         (bl['all_features']['mean'] + 1e-9) * 100
            improv = f'<div class="card"><div class="val">{improv_val:+.1f}%</div><div class="lbl">MSE Improvement over All-Features</div></div>'
        sig_card = ''
        if stats:
            sig = '✓ Significant' if stats['significant'] else '✗ Not Significant'
            sig_cls = 'sig-yes' if stats['significant'] else 'sig-no'
            sig_card = f'<div class="card"><div class="val {sig_cls}">{sig}</div><div class="lbl">Wilcoxon p={stats["wilcoxon_p_value"]:.4f}</div></div>'

        sections.append(f"""
        <div class="grid">
          <div class="card"><div class="val">{ga['best_mse']:.4f}</div>
            <div class="lbl">GA Best Validation MSE</div></div>
          <div class="card"><div class="val">{ga['n_selected']}/{ga['n_total']}</div>
            <div class="lbl">Features Selected</div></div>
          <div class="card"><div class="val">{(1-ga['n_selected']/ga['n_total'])*100:.0f}%</div>
            <div class="lbl">Feature Reduction</div></div>
          {improv}
          {sig_card}
        </div>
        """)

    # Selected features
    if ga:
        from src.constants import CATEGORY_A_STRUCTURAL, CATEGORY_B_TEXTUAL, CATEGORY_C_EVOLUTIONARY
        fcat = ({f: ('Structural', 'tag-s')   for f in CATEGORY_A_STRUCTURAL} |
                {f: ('Textual',    'tag-t')    for f in CATEGORY_B_TEXTUAL}    |
                {f: ('Evolutionary','tag-e')   for f in CATEGORY_C_EVOLUTIONARY})
        rows = ''
        for feat in ga['feature_names']:
            cat_name, css_cls = fcat.get(feat, ('Unknown', ''))
            rows += f'<tr><td>{feat}</td><td><span class="tag {css_cls}">{cat_name}</span></td></tr>'
        sections.append(f"""
        <div class="section">
          <h2>Selected Features</h2>
          <table><thead><tr><th>Feature</th><th>Category</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>""")

    # Tuned hyperparameters
    if hp:
        rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>'
                       for k, v in hp.items() if k != 'best_cv_mse')
        cv_mse = hp.get('best_cv_mse', '')
        sections.append(f"""
        <div class="section">
          <h2>Tuned Hyperparameters (Optuna)</h2>
          <p style="color:#64748b;font-size:13px;margin-bottom:12px">
            Best 5-fold CV MSE: <strong style="color:#a78bfa">{cv_mse:.4f}</strong></p>
          <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>""")

    # GA convergence
    if ga and ga.get('history'):
        rows = ''
        for h in ga['history']:
            rows += (f'<tr><td>{h["generation"]}</td><td>{h["global_best_mse"]:.4f}</td>'
                     f'<td>{h["best_mse"]:.4f}</td><td>{h["n_features"]}</td>'
                     f'<td>{h["mutation_rate"]:.3f}</td></tr>')
        sections.append(f"""
        <div class="section">
          <h2>GA Convergence</h2>
          <table><thead><tr>
            <th>Gen</th><th>Global Best MSE</th><th>Gen Best MSE</th>
            <th>Features</th><th>Mutation Rate</th>
          </tr></thead><tbody>{rows}</tbody></table>
        </div>""")

    # Baseline comparison
    if bl:
        method_map = [
            ('all_features',  'All Features (base paper)'),
            ('random_subset', 'Random Same-Size Subset'),
            ('ga_selected',   'GA + ANN (ours)'),
            ('xgb_ga',        'GA + XGBoost'),
        ]
        rows = ''
        best_mse = min(bl[k]['mean'] for k, _ in method_map if k in bl)
        for key, label in method_map:
            if key not in bl:
                continue
            m, s = bl[key]['mean'], bl[key]['std']
            bold_open  = '<strong style="color:#a78bfa">' if abs(m - best_mse) < 1e-6 else ''
            bold_close = '</strong>' if abs(m - best_mse) < 1e-6 else ''
            rows += f'<tr><td>{label}</td><td>{bold_open}{m:.4f}{bold_close}</td><td>{s:.4f}</td></tr>'
        sections.append(f"""
        <div class="section">
          <h2>Baseline Comparison ({bl.get('n_trials','?')} trials each)</h2>
          <table><thead><tr><th>Method</th><th>Mean MSE ↓</th><th>Std Dev</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>""")

    # Ablation
    if abl:
        sorted_abl = sorted(abl.items(), key=lambda x: x[1]['mean'])
        rows = ''
        for combo, data in sorted_abl:
            rows += (f'<tr><td>{combo}</td><td>{data["n_features"]}</td>'
                     f'<td>{data["mean"]:.4f}</td><td>{data["std"]:.4f}</td></tr>')
        sections.append(f"""
        <div class="section">
          <h2>Ablation Study — Feature Category Contribution</h2>
          <table><thead><tr>
            <th>Combination</th><th>Features</th><th>Mean MSE ↓</th><th>Std Dev</th>
          </tr></thead><tbody>{rows}</tbody></table>
        </div>""")

    # Statistical significance
    if stats:
        sig_class = 'sig-yes' if stats['significant'] else 'sig-no'
        sig_label = 'Statistically Significant ✓' if stats['significant'] \
                    else 'Not Statistically Significant ✗'
        sections.append(f"""
        <div class="section">
          <h2>Statistical Significance</h2>
          <div class="grid">
            <div class="card"><div class="val">{stats['wilcoxon_p_value']:.4f}</div>
              <div class="lbl">Wilcoxon p-value (one-sided)</div></div>
            <div class="card"><div class="val">{stats['cohens_d']:.3f}</div>
              <div class="lbl">Cohen's d ({stats['effect_size']} effect)</div></div>
            <div class="card"><div class="val {sig_class}">{sig_label}</div>
              <div class="lbl">alpha = {stats['alpha_level']}</div></div>
            <div class="card"><div class="val">{stats['pct_improvement']:+.1f}%</div>
              <div class="lbl">MSE Improvement over All-Features</div></div>
          </div>
          <p style="color:#94a3b8;font-size:13px;margin-top:12px">{stats['interpretation']}</p>
        </div>""")

    # Multi-repo
    if multi:
        rows = ''
        for repo, r in multi.items():
            all_mse = r.get('all_features_mse', float('nan'))
            ga_mse  = r.get('ga_ann_mse', r['ga_best_mse'])
            improv  = r.get('improvement_pct', float('nan'))
            rows += (f'<tr><td><strong>{repo}</strong></td>'
                     f'<td>{r["n_files"]}</td><td>{r["n_features_total"]}</td>'
                     f'<td>{r["n_selected"]} ({r["reduction_pct"]}%)</td>'
                     f'<td>{ga_mse:.4f}</td><td>{all_mse:.4f}</td>'
                     f'<td>{improv:+.1f}%</td></tr>')
        sections.append(f"""
        <div class="section">
          <h2>Multi-Repository Comparison</h2>
          <table><thead><tr>
            <th>Repository</th><th>Files</th><th>Total Feats</th>
            <th>Selected</th><th>GA MSE</th><th>All-Feat MSE</th><th>Improvement</th>
          </tr></thead><tbody>{rows}</tbody></table>
        </div>""")

    # Sensitivity summary
    if sens:
        alphas    = sens['alphas']
        betas     = sens['betas']
        pop_sizes = sens['pop_sizes']
        mid_pop   = str(pop_sizes[len(pop_sizes)//2])
        rows = ''
        for a in alphas:
            for b in betas:
                entry = sens['results'][str(a)][str(b)].get(mid_pop, {})
                mse   = entry.get('best_mse', float('nan'))
                nsel  = entry.get('n_selected', '?')
                rows += f'<tr><td>{a}</td><td>{b}</td><td>{mid_pop}</td><td>{mse:.4f}</td><td>{nsel}</td></tr>'
        sections.append(f"""
        <div class="section">
          <h2>Sensitivity Analysis (pop_size={mid_pop})</h2>
          <table><thead><tr>
            <th>Alpha</th><th>Beta</th><th>Pop</th><th>Best MSE</th><th>Features Sel.</th>
          </tr></thead><tbody>{rows}</tbody></table>
        </div>""")

    sections.append('<footer>Generated by Neuro-Genetic Maintainability Framework</footer>')

    # ── Assemble HTML ──────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Maintainability Report</title>
  <style>{css}</style>
</head>
<body>
  <div class="container">
    {''.join(sections)}
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  [REPORT] Written to: {output_path}")
    return output_path


if __name__ == '__main__':
    generate_report()