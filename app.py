import json
import os
import sys
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

from src.constants import (
    CATEGORY_A_STRUCTURAL,
    CATEGORY_B_TEXTUAL,
    CATEGORY_C_EVOLUTIONARY,
)

st.set_page_config(
    page_title="Neuro-Genetic Maintainability",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
    border-left: 4px solid #7c3aed;
}
.metric-value { font-size: 28px; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 13px; color: #94a3b8; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

RESULTS_DIR    = "data/results"
GA_PATH        = os.path.join(RESULTS_DIR, "ga_results.json")
BASELINE_PATH  = os.path.join(RESULTS_DIR, "baseline_results.json")
ABLATION_PATH  = os.path.join(RESULTS_DIR, "ablation_results.json")
STATS_PATH     = os.path.join(RESULTS_DIR, "stats_results.json")
MULTI_PATH     = os.path.join(RESULTS_DIR, "multi_repo_results.json")
SENS_PATH      = os.path.join(RESULTS_DIR, "sensitivity_results.json")

CATEGORY_COLORS = {
    "Structural":  "#7c3aed",
    "Textual":     "#0ea5e9",
    "Evolutionary":"#10b981",
    "Unknown":     "#94a3b8",
}

FEATURE_CATEGORY = (
    {f: "Structural"   for f in CATEGORY_A_STRUCTURAL}
    | {f: "Textual"    for f in CATEGORY_B_TEXTUAL}
    | {f: "Evolutionary" for f in CATEGORY_C_EVOLUTIONARY}
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    margin=dict(l=10, r=10, t=40, b=10),
)

GRID_COLOR = "rgba(255,255,255,0.1)"


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def status(path, label):
    mark = "✓" if exists(path) else "○"
    cls  = "status-ok" if exists(path) else "status-warn"
    return f'<span class="{cls}">{mark} {label}</span>'


# Sidebar
with st.sidebar:
    st.title("NeuroGA Config")
    st.markdown("---")

    repo_path  = st.text_input("Repository path", "test_repos/flask")
    raw_file   = st.text_input("Raw data file",   "data/flask_dataset.csv")
    clean_file = st.text_input("Clean data file",  "data/flask_dataset_clean.csv")

    st.markdown("**GA Settings**")
    pop_size    = st.slider("Population size",   5,  30, 15)
    generations = st.slider("Max generations",   3,  30, 10)
    alpha       = st.slider("Alpha (accuracy)",  0.1, 3.0, 1.0, 0.1)
    beta        = st.slider("Beta (parsimony)",  0.0, 2.0, 0.5, 0.1)
    mut_rate    = st.slider("Initial mutation",  0.05, 0.40, 0.20, 0.01)
    min_mut     = st.slider("Min mutation",      0.01, 0.10, 0.03, 0.01)
    stagnation  = st.slider("Stagnation limit",  2, 10, 5)

    st.markdown("**Research Modules**")
    n_trials    = st.slider("Trials per method", 5, 30, 20)
    tune_trials = st.slider("Optuna trials",    10, 100, 50)

    do_tuning   = st.checkbox("Hyperparameter tuning")
    do_baselines= st.checkbox("Baselines",   value=True)
    do_ablation = st.checkbox("Ablation",    value=True)
    do_stats    = st.checkbox("Stat tests",  value=True)
    do_multi    = st.checkbox("Multi-repo")
    do_sens     = st.checkbox("Sensitivity")
    do_report   = st.checkbox("HTML report", value=True)

    repos_input = st.text_input("Repos (space-separated)", "flask requests django")

    st.markdown("---")
    st.markdown("**Data Status**", unsafe_allow_html=True)
    for path, label in [
        (raw_file,      "Raw data"),
        (clean_file,    "Clean data"),
        (GA_PATH,       "GA results"),
        (BASELINE_PATH, "Baselines"),
        (ABLATION_PATH, "Ablation"),
        (STATS_PATH,    "Stats"),
        (MULTI_PATH,    "Multi-repo"),
        (SENS_PATH,     "Sensitivity"),
    ]:
        st.markdown(status(path, label), unsafe_allow_html=True)


st.title("Neuro-Genetic Software Maintainability")
st.caption("Automated feature selection via hybrid neuro-genetic optimisation across structural, textual, and evolutionary metrics.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Pipeline", "GA Results", "Baselines",
    "Ablation", "File Risk", "Multi-Repo", "Sensitivity",
])


# Pipeline tab
with tab1:
    st.header("Pipeline Runner")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        stages = [
            ("Mine repository",       exists(raw_file)),
            ("Clean data",            exists(clean_file)),
            ("Hyperparameter tuning", exists(os.path.join(RESULTS_DIR, "best_hyperparams.json"))),
            ("GA optimisation",       exists(GA_PATH)),
            ("Baseline comparison",   exists(BASELINE_PATH)),
            ("Ablation study",        exists(ABLATION_PATH)),
            ("Significance tests",    exists(STATS_PATH)),
            ("Multi-repo comparison", exists(MULTI_PATH)),
            ("Sensitivity sweep",     exists(SENS_PATH)),
            ("HTML report",           exists(os.path.join(RESULTS_DIR, "maintainability_report.html"))),
        ]
        for label, done in stages:
            box = (
                '<input type="checkbox" checked disabled style="margin-right:8px;">'
                if done else
                '<input type="checkbox" disabled style="margin-right:8px;">'
            )
            st.markdown(f"{box}{label}", unsafe_allow_html=True)

    with col_b:
        run_full = st.button("▶ Run Full Pipeline", type="primary", use_container_width=True)
        run_ga   = st.button("▶ GA Only",           use_container_width=True)
        run_res  = st.button("▶ Research Modules",  use_container_width=True)

    if run_full or run_ga or run_res:
        cmd = [
            sys.executable, "main.py",
            "--repo", repo_path,
            "--raw-file", raw_file,
            "--processed-file", clean_file,
            "--pop-size", str(pop_size),
            "--generations", str(generations),
            "--alpha", str(alpha),
            "--beta", str(beta),
            "--mutation-rate", str(mut_rate),
            "--min-mutation", str(min_mut),
            "--stagnation", str(stagnation),
            "--n-trials", str(n_trials),
        ]
        if run_full or run_res:
            if do_tuning:    cmd += ["--run-tuning", "--tune-trials", str(tune_trials)]
            if do_baselines: cmd.append("--run-baselines")
            if do_ablation:  cmd.append("--run-ablation")
            if do_stats:     cmd.append("--run-stats")
            if do_multi:     cmd += ["--multi-repo", "--repos"] + repos_input.split()
            if do_sens:      cmd.append("--run-sensitivity")
            if do_report:    cmd.append("--run-report")

        with st.spinner("Running… check terminal for live output."):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("Done!")
            st.rerun()
        else:
            st.error("Pipeline failed.")
            with st.expander("Error output"):
                st.code(result.stderr)

    if exists(clean_file):
        st.divider()
        df = pd.read_csv(clean_file)
        c1, c2, c3 = st.columns(3)
        c1.metric("Files",              len(df))
        c2.metric("Features",           len(df.columns) - 2)
        c3.metric("Bug-prone (>0 bugs)", int((df.iloc[:, -1] > 0).sum()))
        st.dataframe(df.head(10), use_container_width=True)


# GA Results tab
with tab2:
    st.header("GA Optimisation Results")
    ga = load(GA_PATH)

    if not ga:
        st.info("Run the GA first (Pipeline tab).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best MSE",        f"{ga['best_mse']:.4f}")
        c2.metric("Best Fitness",    f"{ga['best_fitness']:.4f}")
        c3.metric("Features",        f"{ga['n_selected']}/{ga['n_total']}")
        c4.metric("Reduction",       f"{(1 - ga['n_selected']/ga['n_total'])*100:.0f}%")

        st.divider()
        left, right = st.columns([3, 2])

        with left:
            history = ga.get("history", [])
            if history:
                df_h = pd.DataFrame(history)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(
                    x=df_h["generation"], y=df_h["global_best_mse"],
                    name="Global best MSE", line=dict(color="#7c3aed", width=2.5),
                    mode="lines+markers",
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=df_h["generation"], y=df_h["best_mse"],
                    name="Gen best MSE", line=dict(color="#a78bfa", width=1.5, dash="dot"),
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=df_h["generation"], y=df_h["mutation_rate"],
                    name="Mutation rate", line=dict(color="#f59e0b", width=1, dash="dash"),
                ), secondary_y=True)
                fig.update_layout(height=320, **PLOT_LAYOUT,
                    legend=dict(orientation="h", y=-0.2))
                fig.update_yaxes(title_text="MSE",           gridcolor=GRID_COLOR, secondary_y=False)
                fig.update_yaxes(title_text="Mutation rate",                        secondary_y=True)
                fig.update_xaxes(title_text="Generation",    gridcolor=GRID_COLOR)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(df_h, x="generation", y="n_features",
                              color_discrete_sequence=["#10b981"],
                              title="Features selected per generation")
                fig2.update_layout(height=200, **PLOT_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

        with right:
            st.markdown("**Selected features**")
            chrom = ga.get("chromosome", [])
            all_names = (
                pd.read_csv(clean_file).columns[1:-1].tolist()
                if exists(clean_file)
                else [f"feature_{i}" for i in range(len(chrom))]
            )
            rows = [
                {"Feature": name, "Selected": "✓" if bit else "—",
                 "Category": FEATURE_CATEGORY.get(name, "Unknown")}
                for name, bit in zip(all_names, chrom)
            ]
            df_feat = pd.DataFrame(rows)

            CAT_BG = {"Structural": "#3b1f6e", "Textual": "#0c3352", "Evolutionary": "#0a3d2e"}
            def highlight(row):
                bg = CAT_BG.get(row["Category"], "#2a2a2a") if row["Selected"] == "✓" else ""
                return [f"background-color:{bg}"] * len(row)

            st.dataframe(df_feat.style.apply(highlight, axis=1),
                         use_container_width=True, height=420, hide_index=True)

            selected_cats = df_feat[df_feat["Selected"] == "✓"]["Category"].value_counts()
            if not selected_cats.empty:
                fig3 = px.pie(values=selected_cats.values, names=selected_cats.index,
                              color=selected_cats.index,
                              color_discrete_map=CATEGORY_COLORS, hole=0.5)
                fig3.update_layout(height=200, **PLOT_LAYOUT, showlegend=True,
                                   legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig3, use_container_width=True)


# Baselines tab
with tab3:
    st.header("Baseline Comparison")
    bl = load(BASELINE_PATH)

    if not bl:
        st.info("Enable Baselines in the Pipeline tab and run.")
    else:
        methods = {
            "All Features":  bl.get("all_features",  {}),
            "Random Subset": bl.get("random_subset", {}),
            "GA Selected":   bl.get("ga_selected",   {}),
        }
        colors = ["#94a3b8", "#f59e0b", "#7c3aed"]

        cols = st.columns(3)
        for col, (name, data) in zip(cols, methods.items()):
            col.metric(name, f"{data.get('mean', 0):.4f}",
                       delta=f"±{data.get('std', 0):.4f}", delta_color="off")

        left, right = st.columns([3, 2])

        with left:
            fig = go.Figure()
            for (name, data), color in zip(methods.items(), colors):
                fig.add_trace(go.Bar(
                    x=[name], y=[data.get("mean", 0)],
                    error_y=dict(type="data", array=[data.get("std", 0)], visible=True),
                    marker_color=color, name=name, width=0.4,
                ))
            fig.update_layout(title="Mean MSE by method (lower is better)",
                              height=360, **PLOT_LAYOUT, showlegend=False,
                              yaxis_title="Mean MSE",
                              yaxis=dict(gridcolor=GRID_COLOR), bargap=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig2 = go.Figure()
            for (name, data), color in zip(methods.items(), colors):
                if data.get("mses"):
                    fig2.add_trace(go.Box(y=data["mses"], name=name, marker_color=color,
                                          boxpoints="all", jitter=0.3, pointpos=-1.8))
            fig2.update_layout(title="MSE distribution across trials",
                               height=360, **PLOT_LAYOUT, showlegend=False,
                               yaxis=dict(gridcolor=GRID_COLOR))
            st.plotly_chart(fig2, use_container_width=True)

        rows = [
            {"Method": name,
             "Mean MSE": f"{d.get('mean',0):.4f}",
             "Std":      f"{d.get('std',0):.4f}",
             "Min":      f"{d.get('min',0):.4f}",
             "Max":      f"{d.get('max',0):.4f}",
             "Trials":   bl.get("n_trials", "—")}
            for name, d in methods.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"GA used {bl.get('ga_n_features','?')}/{bl.get('total_features','?')} features.")

        stats = load(STATS_PATH)
        if stats:
            sig = "Significant" if stats["significant"] else "Not significant"
            st.info(f"**{sig}** — Wilcoxon p={stats['wilcoxon_p_value']:.4f} | "
                    f"Cohen's d={stats['cohens_d']:.3f} ({stats['effect_size']}) | "
                    f"Improvement: {stats['pct_improvement']:+.1f}%")


# Ablation tab
with tab4:
    st.header("Ablation Study")
    abl = load(ABLATION_PATH)

    if not abl:
        st.info("Enable Ablation in the Pipeline tab and run.")
    else:
        df_abl = pd.DataFrame([
            {"Combination": k, "Categories": " + ".join(v["categories"]),
             "Features": v["n_features"], "Mean MSE": v["mean"], "Std": v["std"]}
            for k, v in abl.items()
        ]).sort_values("Mean MSE")

        bar_color_map = {
            "A + B + C (Full)": "#7c3aed",
            "C only (Evolutionary)": "#10b981",
            "B only (Textual)": "#0ea5e9",
        }
        colors = [bar_color_map.get(c, "#94a3b8") for c in df_abl["Combination"]]

        fig = go.Figure(go.Bar(
            x=df_abl["Mean MSE"], y=df_abl["Combination"],
            orientation="h", marker_color=colors,
            error_x=dict(type="data", array=df_abl["Std"].tolist(), visible=True),
        ))
        fig.update_layout(title="Mean MSE per category combination (lower is better)",
                          height=380, **PLOT_LAYOUT,
                          xaxis_title="Mean MSE",
                          xaxis=dict(gridcolor=GRID_COLOR),
                          yaxis=dict(automargin=True))
        st.plotly_chart(fig, use_container_width=True)

        display = df_abl.copy()
        display["Mean MSE"] = display["Mean MSE"].map(lambda x: f"{x:.4f}")
        display["Std"]      = display["Std"].map(lambda x: f"{x:.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        best  = df_abl.iloc[0]
        worst = df_abl.iloc[-1]
        st.success(f"Best:  {best['Combination']} (MSE = {best['Mean MSE']:.4f})")
        st.warning(f"Worst: {worst['Combination']} (MSE = {worst['Mean MSE']:.4f})")

        a_only = abl.get("A only (Structural)", {}).get("mean")
        abc    = abl.get("A + B + C (Full)",    {}).get("mean")
        if a_only and abc:
            improvement = (a_only - abc) / (a_only + 1e-9) * 100
            st.info(f"Adding textual + evolutionary metrics to structural-only "
                    f"reduces MSE by **{improvement:.1f}%**.")


# File Risk tab
with tab5:
    st.header("Per-File Bug Risk Scores")
    ga = load(GA_PATH)

    if not ga or not exists(clean_file):
        st.info("Run the GA pipeline first.")
    else:
        if st.button("Generate Predictions", type="primary"):
            with st.spinner("Training final model and predicting…"):
                try:
                    from src.ann_model import get_predictions
                    files, y_true, y_pred, mse = get_predictions(
                        clean_file, tuple(ga["chromosome"])
                    )
                    df_risk = pd.DataFrame({
                        "File":           files,
                        "True Bugs":      y_true.astype(int),
                        "Predicted Score":np.round(y_pred, 2),
                        "Risk Level":     pd.cut(
                            y_pred, bins=[-0.1, 0.5, 2.0, 1e9],
                            labels=["Low", "Medium", "High"],
                        ).astype(str),
                    }).sort_values("Predicted Score", ascending=False)

                    st.session_state["df_risk"] = df_risk
                    st.session_state["risk_mse"] = mse
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if "df_risk" in st.session_state:
            df_risk = st.session_state["df_risk"]
            mse     = st.session_state["risk_mse"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Validation MSE", f"{mse:.4f}")
            c2.metric("High risk",   int((df_risk["Risk Level"] == "High").sum()))
            c3.metric("Medium risk", int((df_risk["Risk Level"] == "Medium").sum()))
            c4.metric("Low risk",    int((df_risk["Risk Level"] == "Low").sum()))

            left, right = st.columns([2, 1])
            with left:
                max_val = max(df_risk["True Bugs"].max(), df_risk["Predicted Score"].max()) + 1
                fig = px.scatter(df_risk, x="True Bugs", y="Predicted Score",
                                 color="Risk Level",
                                 color_discrete_map={"High":   "#f87171",
                                                      "Medium": "#fbbf24",
                                                      "Low":    "#34d399"},
                                 hover_data=["File"],
                                 title="True vs Predicted Bug Commits")
                fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                         mode="lines",
                                         line=dict(dash="dash", color="#64748b"),
                                         name="Perfect prediction"))
                fig.update_layout(height=360, **PLOT_LAYOUT,
                                  xaxis=dict(gridcolor=GRID_COLOR),
                                  yaxis=dict(gridcolor=GRID_COLOR))
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown("**Top 10 riskiest files**")
                st.dataframe(df_risk[["File","Predicted Score","Risk Level"]].head(10),
                             use_container_width=True, hide_index=True, height=340)

            st.divider()
            filter_levels = st.multiselect("Filter by risk level",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"])
            st.dataframe(df_risk[df_risk["Risk Level"].isin(filter_levels)],
                         use_container_width=True, hide_index=True, height=380)


# Multi-repo tab
with tab6:
    st.header("Multi-Repository Generalisation")
    mr = load(MULTI_PATH)

    if not mr:
        st.info("Enable Multi-repo in the Pipeline tab and run.")
    else:
        repos  = list(mr.keys())
        colors = px.colors.qualitative.Vivid[:len(repos)]

        cols = st.columns(len(repos))
        for col, (repo, data) in zip(cols, mr.items()):
            col.metric(repo.capitalize(),
                       f"MSE {data.get('ga_ann_mse', data['ga_best_mse']):.4f}",
                       delta=f"{data['n_selected']}/{data['n_features_total']} feats",
                       delta_color="off")

        left, right = st.columns([3, 2])

        with left:
            ga_mses  = [r.get("ga_ann_mse", r["ga_best_mse"]) for r in mr.values()]
            all_mses = [r.get("all_features_mse", float("nan"))  for r in mr.values()]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="All Features", x=repos, y=all_mses,
                                  marker_color="#94a3b8", width=0.35))
            fig.add_trace(go.Bar(name="GA Selected",  x=repos, y=ga_mses,
                                  marker_color="#7c3aed", width=0.35))
            fig.update_layout(barmode="group",
                              title="GA vs All-Features MSE per repository",
                              height=340, **PLOT_LAYOUT,
                              yaxis_title="Mean MSE",
                              yaxis=dict(gridcolor=GRID_COLOR),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            reductions = [r["reduction_pct"] for r in mr.values()]
            fig2 = go.Figure(go.Bar(
                x=repos, y=reductions, marker_color=colors, width=0.5,
                text=[f"{v:.0f}%" for v in reductions], textposition="outside",
            ))
            fig2.update_layout(title="Feature reduction % per repo", height=340,
                               **PLOT_LAYOUT, yaxis_title="Reduction %",
                               yaxis=dict(gridcolor=GRID_COLOR, range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)

        rows = [
            {"Repository": repo,
             "Files":       r["n_files"],
             "Total Feats": r["n_features_total"],
             "Selected":    r["n_selected"],
             "Reduction":   f"{r['reduction_pct']}%",
             "GA MSE":      f"{r.get('ga_ann_mse', r['ga_best_mse']):.4f}",
             "All-Feat MSE":f"{r.get('all_features_mse', float('nan')):.4f}",
             "Improvement": f"{r.get('improvement_pct', float('nan')):+.1f}%",
             "Time (s)":    r.get("elapsed_s", "—")}
            for repo, r in mr.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        all_selected = [set(r["selected_features"]) for r in mr.values()]
        if all_selected:
            common = set.intersection(*all_selected)
            union  = set.union(*all_selected)
            if common:
                st.success(f"{len(common)} features selected across all repos: "
                           f"{', '.join(sorted(common))}")
            st.info(f"{len(union)} unique features selected across all repos combined.")


# Sensitivity tab
with tab7:
    st.header("Hyperparameter Sensitivity")
    sens = load(SENS_PATH)

    if not sens:
        st.info("Enable Sensitivity in the Pipeline tab and run.")
    else:
        alphas    = sens["alphas"]
        betas     = sens["betas"]
        pop_sizes = sens["pop_sizes"]
        results   = sens["results"]

        pop_choice    = st.select_slider("Population size",
                            options=[str(p) for p in pop_sizes],
                            value=str(pop_sizes[len(pop_sizes) // 2]))
        metric_choice = st.radio("Metric", ["best_mse", "n_selected"], horizontal=True,
                                  format_func=lambda x: "MSE" if x == "best_mse"
                                                         else "Features selected")

        left, right = st.columns([3, 2])

        with left:
            z = [[results[str(a)][str(b)].get(pop_choice, {}).get(metric_choice, float("nan"))
                  for b in betas]
                 for a in alphas]

            fig = go.Figure(go.Heatmap(
                z=z,
                x=[f"β={b}" for b in betas],
                y=[f"α={a}" for a in alphas],
                colorscale="Viridis",
                reversescale=(metric_choice == "best_mse"),
                text=[[f"{v:.4f}" if metric_choice == "best_mse" else str(int(v))
                       for v in row] for row in z],
                texttemplate="%{text}",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                title=f"{'MSE' if metric_choice == 'best_mse' else 'Features'} "
                      f"heatmap (pop={pop_choice})",
                height=360, **PLOT_LAYOUT,
                xaxis_title="Beta (parsimony weight)",
                yaxis_title="Alpha (accuracy weight)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            best_combo, best_mean = None, float("inf")
            for a in alphas:
                for b in betas:
                    mses = [results[str(a)][str(b)].get(str(p), {}).get("best_mse", float("inf"))
                            for p in pop_sizes]
                    m = sum(mses) / len(mses)
                    if m < best_mean:
                        best_mean, best_combo = m, (a, b)

            if best_combo:
                a_b, b_b = best_combo
                pop_mses = [results[str(a_b)][str(b_b)].get(str(p), {}).get("best_mse", float("nan"))
                            for p in pop_sizes]
                pop_nsel = [results[str(a_b)][str(b_b)].get(str(p), {}).get("n_selected", 0)
                            for p in pop_sizes]

                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=pop_sizes, y=pop_mses, name="MSE",
                                           line=dict(color="#7c3aed", width=2.5),
                                           mode="lines+markers"), secondary_y=False)
                fig2.add_trace(go.Scatter(x=pop_sizes, y=pop_nsel, name="Features",
                                           line=dict(color="#10b981", width=2, dash="dot"),
                                           mode="lines+markers"), secondary_y=True)
                fig2.update_layout(title=f"Pop size effect (α={a_b}, β={b_b})",
                                   height=360, **PLOT_LAYOUT,
                                   legend=dict(orientation="h", y=-0.2))
                fig2.update_yaxes(title_text="MSE",      gridcolor=GRID_COLOR, secondary_y=False)
                fig2.update_yaxes(title_text="Features",                        secondary_y=True)
                fig2.update_xaxes(title_text="Population size")
                st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Raw data"):
            flat = [
                {"Alpha": a, "Beta": b, "Pop": p,
                 "MSE":      round(results[str(a)][str(b)].get(str(p), {}).get("best_mse", float("nan")), 4),
                 "Features": results[str(a)][str(b)].get(str(p), {}).get("n_selected", "?"),
                 "Time (s)": results[str(a)][str(b)].get(str(p), {}).get("elapsed_s", "?")}
                for a in alphas for b in betas for p in pop_sizes
            ]
            st.dataframe(pd.DataFrame(flat), use_container_width=True, hide_index=True)

        # Check if default settings are near-optimal
        default = results.get("1.0", {}).get("0.5", {}).get(
            str(pop_sizes[len(pop_sizes) // 2]), {}
        ).get("best_mse", float("nan"))
        min_mse = min(
            results[str(a)][str(b)].get(str(p), {}).get("best_mse", float("inf"))
            for a in alphas for b in betas for p in pop_sizes
        )
        if not (min_mse == float("inf") or default != default):
            pct = (default - min_mse) / (min_mse + 1e-9) * 100
            if pct < 5:
                st.success(f"Default settings (alpha=1.0, beta=0.5) are within "
                           f"{pct:.1f}% of the global best — results are robust.")
            else:
                st.warning(f"Default settings are {pct:.1f}% above the global optimum. "
                           f"Consider tuning alpha and beta for this repository.")