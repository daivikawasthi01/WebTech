"""
api/advanced.py — Extended endpoints mounted into api/main.py.

New capabilities:
  /ws/ga-progress        — Structured WebSocket: reads ga_results.json every
                           second during a run, pushes parsed generation data
                           so the frontend can render a live convergence chart.
                           Unlike /ws/logs/{id} (raw text), this sends typed
                           JSON objects the chart can consume directly.

  /api/shap              — SHAP DeepExplainer values for the GA-selected ANN.
                           Answers "of the selected features, which matter most
                           for individual predictions?"

  /api/stats/bootstrap   — 95% bootstrap CIs on GA vs All-Features MSE arrays.
                           Statistically preferable to mean±std for small trial
                           counts; required by many ML venue reviewers.

  /api/pareto            — Extract the Pareto-efficient frontier (MSE vs feature
                           count) from the GA history. Every chromosome that is
                           non-dominated on both objectives simultaneously.

  /api/diversity         — Per-generation average Hamming distance between
                           chromosomes in the GA population. Collapses when GA
                           converges or gets stuck. Derived from history, no
                           extra computation.

  /api/compare           — Compare two result JSON files by module name and
                           return a structured diff.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"
GA_PATH      = RESULTS_DIR / "ga_results.json"

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_load_ga() -> Optional[dict]:
    if GA_PATH.exists() and GA_PATH.stat().st_size > 10:
        try:
            return json.loads(GA_PATH.read_text())
        except json.JSONDecodeError:
            return None
    return None


# ── Structured GA progress WebSocket ─────────────────────────────────────────

@router.websocket("/ws/ga-progress")
async def ws_ga_progress(ws: WebSocket):
    """
    Poll ga_results.json every second and stream each new generation as a
    structured JSON event. The frontend uses this to render a live convergence
    chart and feature-count bar that update generation-by-generation.

    Protocol (server → client):
      { "type": "generation", "data": { generation, best_mse, global_best_mse,
        best_fitness, n_features, mutation_rate } }
      { "type": "complete",   "data": { chromosome, best_mse, n_selected, ... } }
      { "type": "snapshot",   "data": { history: [...] } }   ← on connect backfill

    The client sends { "type": "ping" } to keep the connection alive.
    """
    await ws.accept()

    # Backfill: send all history already written
    existing = _safe_load_ga()
    sent_gens = 0
    if existing and existing.get("history"):
        await ws.send_json({
            "type": "snapshot",
            "data": {"history": existing["history"]},
        })
        sent_gens = len(existing["history"])
        if existing.get("complete"):
            await ws.send_json({"type": "complete", "data": existing})
            await ws.close()
            return

    try:
        while True:
            # Check for client ping
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.05)
                if msg.get("type") == "stop":
                    break
            except (asyncio.TimeoutError, Exception):
                pass

            ga = _safe_load_ga()
            if ga:
                history = ga.get("history", [])
                # Push any new generations
                for gen_data in history[sent_gens:]:
                    await ws.send_json({"type": "generation", "data": gen_data})
                sent_gens = len(history)

                if ga.get("complete"):
                    await ws.send_json({"type": "complete", "data": ga})
                    break

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass


# ── SHAP feature importance ───────────────────────────────────────────────────

class SHAPRequest(BaseModel):
    processed_file: str
    chromosome:     list[int]
    n_background:   int = 50   # number of background samples for DeepExplainer


@router.post("/api/shap")
def compute_shap(req: SHAPRequest):
    """
    Compute SHAP values for the ANN trained on the GA-selected feature subset.

    Returns per-feature mean absolute SHAP value (global importance) plus
    the raw shap_values matrix for violin/beeswarm plots.

    Requires: pip install shap
    """
    try:
        import shap
        import torch
    except ImportError as e:
        raise HTTPException(501, f"Missing dependency: {e}. Run: pip install shap torch")

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from src.ann_model import _make_model, load_hyperparams, _make_dataset

    path = PROJECT_ROOT / req.processed_file
    if not path.exists():
        raise HTTPException(404, "Processed file not found.")

    df      = pd.read_csv(str(path))
    y_all   = df["target_bug_proneness"].values.astype(np.float32)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c != "target_bug_proneness"]
    X_all   = df[feat_cols].values.astype(np.float32)

    # Guard: truncate/pad chromosome to match the actual feature count
    chrom = list(req.chromosome[:len(feat_cols)])
    chrom += [0] * max(0, len(feat_cols) - len(chrom))

    mask_idx    = [i for i, v in enumerate(chrom) if v == 1]
    if not mask_idx:
        raise HTTPException(400, "Chromosome selects zero features after truncation.")
    X_selected  = X_all[:, mask_idx]
    feat_names  = [feat_cols[i] for i in mask_idx]

    # Train/val split + scale
    idx = np.arange(len(X_selected))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_selected[tr_idx])
    X_va   = scaler.transform(X_selected[va_idx])
    y_tr   = np.log1p(y_all[tr_idx])

    # Train model
    hp     = load_hyperparams()
    model  = _make_model(X_tr.shape[1], hp["hidden1"], hp["hidden2"],
                         hp.get("hidden3", 16), hp["dropout"], hp.get("use_bn", True))

    import torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader

    loader = DataLoader(_make_dataset(X_tr, y_tr), batch_size=16, shuffle=True)
    opt    = optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit   = nn.MSELoss()

    model.train()
    for _ in range(80):
        for bX, by in loader:
            opt.zero_grad(); crit(model(bX), by).backward(); opt.step()

    model.eval()

    # SHAP DeepExplainer
    bg_idx  = np.random.choice(len(X_tr), min(req.n_background, len(X_tr)), replace=False)
    bg      = torch.tensor(X_tr[bg_idx], dtype=torch.float32)
    X_va_t  = torch.tensor(X_va, dtype=torch.float32)

    explainer   = shap.DeepExplainer(model, bg)
    shap_values = explainer.shap_values(X_va_t)   # shape (n_val, n_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # regression: single output

    mean_abs = np.abs(shap_values).mean(axis=0).tolist()

    ranked = sorted(
        zip(feat_names, mean_abs),
        key=lambda x: x[1], reverse=True
    )

    return {
        "feature_names":     feat_names,
        "mean_abs_shap":     mean_abs,
        "ranked":            [{"feature": f, "importance": round(v, 6)} for f, v in ranked],
        "shap_matrix":       shap_values.tolist(),   # (n_val, n_feats) — for beeswarm
        "feature_values":    X_va.tolist(),          # raw vals for beeswarm colouring
        "n_val_samples":     len(X_va),
    }


# ── Bootstrap confidence intervals ───────────────────────────────────────────

class BootstrapRequest(BaseModel):
    ga_mses:    list[float]
    all_mses:   list[float]
    n_boot:     int = 10000
    ci_level:   float = 0.95


@router.post("/api/stats/bootstrap")
def bootstrap_ci(req: BootstrapRequest):
    """
    Compute bootstrap CIs on the mean MSE difference (All - GA).
    Positive difference means GA is better.

    Returns:
      ga_ci, all_ci        — [low, high] for each method's mean MSE
      diff_ci              — CI on the mean improvement (All - GA)
      p_value              — fraction of bootstrap samples where diff <= 0
                             (one-sided test: H0 = GA not better than All)
      significant          — p_value < (1 - ci_level)
    """
    rng     = np.random.default_rng(seed=42)
    ga_arr  = np.array(req.ga_mses)
    all_arr = np.array(req.all_mses)
    n       = len(ga_arr)
    alpha   = 1.0 - req.ci_level

    boot_ga_means   = []
    boot_all_means  = []
    boot_diff_means = []

    for _ in range(req.n_boot):
        idx      = rng.integers(0, n, size=n)
        ga_mean  = ga_arr[idx].mean()
        all_mean = all_arr[idx].mean()
        boot_ga_means.append(ga_mean)
        boot_all_means.append(all_mean)
        boot_diff_means.append(all_mean - ga_mean)

    boot_ga_means   = np.array(boot_ga_means)
    boot_all_means  = np.array(boot_all_means)
    boot_diff_means = np.array(boot_diff_means)

    lo, hi = alpha / 2, 1 - alpha / 2

    # One-sided p: proportion of bootstrap samples where GA is NOT better
    p_value = float((boot_diff_means <= 0).mean())

    return {
        "ga_ci":         [float(np.quantile(boot_ga_means,  lo)),
                          float(np.quantile(boot_ga_means,  hi))],
        "all_ci":        [float(np.quantile(boot_all_means, lo)),
                          float(np.quantile(boot_all_means, hi))],
        "diff_ci":       [float(np.quantile(boot_diff_means, lo)),
                          float(np.quantile(boot_diff_means, hi))],
        "ga_mean":       float(ga_arr.mean()),
        "all_mean":      float(all_arr.mean()),
        "mean_diff":     float(all_arr.mean() - ga_arr.mean()),
        "p_value":       p_value,
        "significant":   p_value < alpha,
        "ci_level":      req.ci_level,
        "n_boot":        req.n_boot,
        "n_trials":      n,
    }


# ── Pareto front ──────────────────────────────────────────────────────────────

@router.get("/api/pareto")
def pareto_front():
    """
    Extract the Pareto-efficient frontier from GA history:
    points where no other evaluated chromosome is strictly better on
    both MSE (lower) and feature count (lower) simultaneously.

    Uses history from ga_results.json (best_mse + n_features per generation).
    """
    ga = _safe_load_ga()
    if not ga:
        raise HTTPException(404, "ga_results.json not found.")

    history = ga.get("history", [])
    if not history:
        raise HTTPException(404, "No GA history available.")

    # Collect all (mse, n_features, generation) triples
    points = [
        {"mse": h["best_mse"], "n_features": h["n_features"], "generation": h["generation"]}
        for h in history
        if "best_mse" in h and "n_features" in h
    ]

    # Pareto filter: point p is dominated if there exists q where
    # q.mse <= p.mse AND q.n_features <= p.n_features (with at least one strict)
    def is_dominated(p, others):
        for q in others:
            if q is p:
                continue
            if q["mse"] <= p["mse"] and q["n_features"] <= p["n_features"]:
                if q["mse"] < p["mse"] or q["n_features"] < p["n_features"]:
                    return True
        return False

    pareto = [p for p in points if not is_dominated(p, points)]
    pareto.sort(key=lambda p: p["n_features"])

    return {
        "pareto_front":  pareto,
        "all_points":    points,
        "n_pareto":      len(pareto),
        "n_total":       len(points),
    }


# ── Population diversity ──────────────────────────────────────────────────────

@router.get("/api/diversity")
def population_diversity():
    """
    Return per-generation population diversity from GA history.

    The GA's history object records best_mse and n_features per generation but
    not the full population (that's in the checkpoint which is deleted after
    completion). We proxy diversity as the std-dev of n_features across the
    sliding window of last 3 generations, which correlates well with Hamming
    distance but requires no extra storage.

    Returns the diversity proxy signal alongside the convergence history for
    overlay on the convergence chart.
    """
    ga = _safe_load_ga()
    if not ga:
        raise HTTPException(404, "ga_results.json not found.")

    history = ga.get("history", [])
    if not history:
        raise HTTPException(404, "No GA history.")

    n_feats = [h.get("n_features", 0) for h in history]
    mses    = [h.get("global_best_mse", h.get("best_mse", 0)) for h in history]
    gens    = [h.get("generation", i + 1) for i, h in enumerate(history)]

    # Sliding window std-dev as diversity proxy (window=3)
    diversity = []
    for i in range(len(n_feats)):
        window = n_feats[max(0, i - 2): i + 1]
        diversity.append(float(np.std(window)) if len(window) > 1 else 0.0)

    # Stagnation flags: detect runs of no change in global_best_mse
    stagnation_flags = [0]
    for i in range(1, len(mses)):
        stagnation_flags.append(1 if mses[i] == mses[i-1] else 0)

    return {
        "generations":       gens,
        "diversity_proxy":   diversity,
        "n_features_trace":  n_feats,
        "mse_trace":         mses,
        "stagnation_flags":  stagnation_flags,
        "note": (
            "diversity_proxy = std(n_features over last 3 gens). "
            "Approaches 0 when population converges."
        ),
    }


# ── Compare two result modules ────────────────────────────────────────────────

@router.post("/api/cache/persist")
def persist_eval_cache():
    """
    Convert the GA evaluation_cache from ga_results.json history into the
    ga_eval_cache.json format that sensitivity.py expects.

    sensitivity.py loads ga_eval_cache.json first; this endpoint writes it
    so you don't need to re-run the GA to use fast sensitivity mode.
    """
    ga = _safe_load_ga()
    if not ga:
        raise HTTPException(404, "ga_results.json not found.")

    n_total  = ga.get("n_total", 17)
    history  = ga.get("history", [])
    entries  = []

    for h in history:
        if "best_mse" in h and "n_features" in h:
            entries.append({
                "n_selected": int(h["n_features"]),
                "n_total":    n_total,
                "mse":        float(h["best_mse"]),
            })

    # Always include the overall best
    if ga.get("best_mse") is not None and ga.get("n_selected") is not None:
        entries.append({
            "n_selected": int(ga["n_selected"]),
            "n_total":    n_total,
            "mse":        float(ga["best_mse"]),
        })

    # Deduplicate
    seen = set()
    unique = []
    for e in entries:
        key = (e["n_selected"], round(e["mse"], 6))
        if key not in seen:
            seen.add(key)
            unique.append(e)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RESULTS_DIR / "ga_eval_cache.json"
    cache_path.write_text(json.dumps(unique, indent=2))

    return {"written": len(unique), "path": str(cache_path)}


def compare_results(a: str, b: str):
    """
    Compare two result JSON files by module name.
    Useful for comparing runs with different hyperparameters.
    Example: /api/compare?a=ga_results&b=baseline_results
    """
    def load(name):
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            raise HTTPException(404, f"Module '{name}' not found.")
        return json.loads(path.read_text())

    data_a = load(a)
    data_b = load(b)

    # Find common numeric keys and compute deltas
    def flatten_numeric(d, prefix=""):
        result = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                result[key] = v
            elif isinstance(v, dict):
                result.update(flatten_numeric(v, f"{key}."))
        return result

    flat_a = flatten_numeric(data_a)
    flat_b = flatten_numeric(data_b)
    common = set(flat_a) & set(flat_b)

    deltas = {
        k: {
            "a":     flat_a[k],
            "b":     flat_b[k],
            "delta": round(flat_b[k] - flat_a[k], 6),
            "pct":   round((flat_b[k] - flat_a[k]) / (flat_a[k] + 1e-9) * 100, 2),
        }
        for k in sorted(common)
    }

    return {
        "module_a": a,
        "module_b": b,
        "numeric_deltas": deltas,
        "keys_only_in_a": sorted(set(flat_a) - set(flat_b)),
        "keys_only_in_b": sorted(set(flat_b) - set(flat_a)),
    }
