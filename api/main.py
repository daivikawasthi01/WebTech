"""
api/main.py — FastAPI backend for Neuro-Genetic Maintainability Framework.

Key upgrades over Streamlit:
  - Real-time WebSocket log streaming: GA/pipeline stdout streamed line-by-line
    as it runs. No more blind spinners.
  - Non-blocking: pipeline runs in a background thread; UI stays responsive.
  - Job management: track multiple jobs, cancel via DELETE /api/jobs/{id}
  - Full REST API: every result module has a clean endpoint
  - Prediction endpoint: per-file risk scoring on demand
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Path setup (api/ sits next to src/) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

app = FastAPI(title="Neuro-GA Maintainability API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job registry ────────────────────────────────────────────────────
# { job_id: { status, command, process, log_buffer, started_at, finished_at } }
_jobs: dict = {}
_jobs_lock = threading.Lock()

# Each job has a ring-buffer of recent log lines for late WebSocket subscribers
_LOG_BUFFER_SIZE = 2000


# ── Pydantic models ───────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    repo_path:       str   = "test_repos/flask"
    raw_file:        str   = "data/flask_dataset.csv"
    processed_file:  str   = "data/flask_dataset_clean.csv"
    pop_size:        int   = 15
    generations:     int   = 10
    alpha:           float = 1.0
    beta:            float = 0.5
    mutation_rate:   float = 0.20
    min_mutation:    float = 0.03
    stagnation:      int   = 5
    n_trials:        int   = 20
    tune_trials:     int   = 50
    repos:           list[str] = ["flask", "requests", "django"]
    # Module flags
    run_tuning:      bool  = False
    run_baselines:   bool  = False
    run_ablation:    bool  = False
    run_stats:       bool  = False
    run_multi:       bool  = False
    run_sensitivity: bool  = False
    run_report:      bool  = True
    run_all:         bool  = False
    force_collect:   bool  = False
    force_process:   bool  = False


class PredictRequest(BaseModel):
    processed_file: str
    chromosome: list[int]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(name: str) -> Optional[dict]:
    path = RESULTS_DIR / f"{name}.json"
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _file_status() -> dict:
    checks = {
        "raw_data":       "data/flask_dataset.csv",
        "clean_data":     "data/flask_dataset_clean.csv",
        "hyperparams":    "data/results/best_hyperparams.json",
        "ga_results":     "data/results/ga_results.json",
        "baselines":      "data/results/baseline_results.json",
        "ablation":       "data/results/ablation_results.json",
        "stats":          "data/results/stats_results.json",
        "multi_repo":     "data/results/multi_repo_results.json",
        "sensitivity":    "data/results/sensitivity_results.json",
        "report":         "data/results/maintainability_report.html",
    }
    return {
        key: (PROJECT_ROOT / path).exists()
        for key, path in checks.items()
    }


def _build_command(cfg: PipelineConfig) -> list[str]:
    """Translate PipelineConfig into a main.py CLI command."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "main.py"),
        "--repo",           cfg.repo_path,
        "--raw-file",       cfg.raw_file,
        "--processed-file", cfg.processed_file,
        "--pop-size",       str(cfg.pop_size),
        "--generations",    str(cfg.generations),
        "--alpha",          str(cfg.alpha),
        "--beta",           str(cfg.beta),
        "--mutation-rate",  str(cfg.mutation_rate),
        "--min-mutation",   str(cfg.min_mutation),
        "--stagnation",     str(cfg.stagnation),
        "--n-trials",       str(cfg.n_trials),
    ]
    if cfg.run_all:
        cmd.append("--run-all")
        return cmd
    if cfg.run_tuning:
        cmd += ["--run-tuning", "--tune-trials", str(cfg.tune_trials)]
    if cfg.run_baselines:
        cmd.append("--run-baselines")
    if cfg.run_ablation:
        cmd.append("--run-ablation")
    if cfg.run_stats:
        cmd.append("--run-stats")
    if cfg.run_multi and cfg.repos:
        cmd += ["--multi-repo", "--repos"] + cfg.repos
    if cfg.run_sensitivity:
        cmd.append("--run-sensitivity")
    if cfg.run_report:
        cmd.append("--run-report")
    if cfg.force_collect:
        cmd.append("--force-collect")
    if cfg.force_process:
        cmd.append("--force-process")
    return cmd


def _run_job_thread(job_id: str, cmd: list[str]):
    """Runs the pipeline subprocess in a background thread, captures output."""
    with _jobs_lock:
        job = _jobs[job_id]
        job["status"] = "running"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,               # line-buffered
        cwd=str(PROJECT_ROOT),
    )

    with _jobs_lock:
        _jobs[job_id]["process"] = process

    for line in process.stdout:
        line = line.rstrip("\n")
        with _jobs_lock:
            _jobs[job_id]["log_buffer"].append(line)

    process.wait()

    with _jobs_lock:
        _jobs[job_id]["status"]      = "done" if process.returncode == 0 else "failed"
        _jobs[job_id]["returncode"]  = process.returncode
        _jobs[job_id]["finished_at"] = datetime.now().isoformat()


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Health check + file existence map."""
    return {"ok": True, "files": _file_status()}


@app.get("/api/results/{module}")
def get_results(module: str):
    """
    Load a JSON result by module name.
    module: ga_results | baseline_results | ablation_results |
            stats_results | multi_repo_results | sensitivity_results |
            best_hyperparams
    """
    data = _load_json(module)
    if data is None:
        raise HTTPException(404, f"Result '{module}' not found or empty.")
    return data


@app.get("/api/data/summary")
def data_summary(processed_file: str = "data/flask_dataset_clean.csv"):
    """Return dataset summary stats for the Pipeline tab data preview."""
    path = PROJECT_ROOT / processed_file
    if not path.exists():
        raise HTTPException(404, "Processed file not found.")
    df = pd.read_csv(path)
    target_col = "target_bug_proneness"
    bug_prone  = int((df[target_col] > 0).sum()) if target_col in df.columns else 0
    n_features = len(df.select_dtypes(include="number").columns) - (1 if target_col in df.columns else 0)
    return {
        "n_files":    len(df),
        "n_features": n_features,
        "bug_prone":  bug_prone,
        "bug_prone_pct": round(bug_prone / len(df) * 100, 1) if len(df) else 0,
        "columns":    df.columns.tolist(),
        "preview":    df.head(10).replace({float("nan"): None}).to_dict(orient="records"),
    }


@app.get("/api/feature-names")
def feature_names(processed_file: str = "data/flask_dataset_clean.csv"):
    """
    Return the exact ordered list of feature names as ann_model.py sees them:
    numeric columns from the processed CSV, excluding target_bug_proneness.

    This is the canonical source of truth for ChromosomeEditor bit alignment —
    index i in this list corresponds to bit i in the GA chromosome.
    """
    path = PROJECT_ROOT / processed_file
    if not path.exists():
        raise HTTPException(404, "Processed file not found.")
    df = pd.read_csv(path)
    target_col = "target_bug_proneness"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    features = [c for c in numeric_cols if c != target_col]
    return {"features": features, "n_features": len(features)}


@app.post("/api/pipeline/run")
def run_pipeline(cfg: PipelineConfig):
    """
    Launch the pipeline as a background job.
    Returns job_id immediately — poll /api/jobs/{id} or subscribe to
    WS /ws/logs/{id} for live output.
    """
    job_id = str(uuid.uuid4())[:8]
    cmd    = _build_command(cfg)

    with _jobs_lock:
        _jobs[job_id] = {
            "status":      "queued",
            "command":     " ".join(cmd),
            "process":     None,
            "log_buffer":  deque(maxlen=_LOG_BUFFER_SIZE),
            "started_at":  datetime.now().isoformat(),
            "finished_at": None,
            "returncode":  None,
        }

    thread = threading.Thread(target=_run_job_thread, args=(job_id, cmd), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def list_jobs():
    with _jobs_lock:
        return [
            {
                "job_id":      jid,
                "status":      j["status"],
                "started_at":  j["started_at"],
                "finished_at": j["finished_at"],
            }
            for jid, j in _jobs.items()
        ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return {
        "job_id":      job_id,
        "status":      job["status"],
        "returncode":  job["returncode"],
        "started_at":  job["started_at"],
        "finished_at": job["finished_at"],
        "log_tail":    list(job["log_buffer"])[-100:],  # last 100 lines for REST
    }


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    """Kill a running job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    proc = job.get("process")
    if proc and proc.poll() is None:
        proc.terminate()
        with _jobs_lock:
            _jobs[job_id]["status"] = "cancelled"
        return {"cancelled": True}
    return {"cancelled": False, "reason": "Job not running"}


@app.post("/api/predict")
def predict_risk(req: PredictRequest):
    """
    Run get_predictions on the processed CSV with the given chromosome.
    Returns per-file risk scores.
    """
    path = PROJECT_ROOT / req.processed_file
    if not path.exists():
        raise HTTPException(404, "Processed file not found.")
    try:
        from src.ann_model import get_predictions

        # Guard: chromosome may be longer than the actual feature count if the
        # GA was run on a different version of the dataset. Truncate or pad
        # to match the real feature count to prevent numpy IndexError.
        df_tmp = pd.read_csv(str(path))
        num_cols = df_tmp.select_dtypes(include="number").columns.tolist()
        real_n = len([c for c in num_cols if c != "target_bug_proneness"])
        chrom = list(req.chromosome[:real_n])  # truncate if too long
        chrom += [0] * max(0, real_n - len(chrom))  # pad if too short

        files, y_true, y_pred, mse = get_predictions(str(path), tuple(chrom))
        # Assign risk levels
        def risk_level(score: float) -> str:
            if score >= 2.0: return "High"
            if score >= 0.5: return "Medium"
            return "Low"

        rows = [
            {
                "file":            f,
                "true_bugs":       int(t),
                "predicted_score": round(float(p), 3),
                "risk_level":      risk_level(float(p)),
            }
            for f, t, p in zip(files, y_true, y_pred)
        ]
        rows.sort(key=lambda r: r["predicted_score"], reverse=True)
        return {
            "mse":   round(float(mse), 4),
            "files": rows,
            "summary": {
                "high":   sum(1 for r in rows if r["risk_level"] == "High"),
                "medium": sum(1 for r in rows if r["risk_level"] == "Medium"),
                "low":    sum(1 for r in rows if r["risk_level"] == "Low"),
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/report")
def download_report():
    """Serve the generated HTML report for download."""
    path = RESULTS_DIR / "maintainability_report.html"
    if not path.exists():
        raise HTTPException(404, "Report not generated yet.")
    return FileResponse(str(path), media_type="text/html",
                        filename="maintainability_report.html")


# ── WebSocket: live log streaming ─────────────────────────────────────────────

@app.websocket("/ws/logs/{job_id}")
async def ws_logs(websocket: WebSocket, job_id: str):
    """
    Stream pipeline logs to the browser in real time.

    Protocol:
      Server → client:
        { "type": "log",    "line": "..." }
        { "type": "status", "status": "done"|"failed"|"cancelled" }
        { "type": "history","lines": [...] }   ← sent on connect (backfill)
    """
    await websocket.accept()

    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    # Backfill: send lines already captured before this client connected
    with _jobs_lock:
        backfill = list(job["log_buffer"])
    if backfill:
        await websocket.send_json({"type": "history", "lines": backfill})

    # Stream new lines as they arrive
    sent_up_to = len(backfill)

    try:
        while True:
            with _jobs_lock:
                buf    = job["log_buffer"]
                status = job["status"]
                total  = len(buf)

            # Send any new lines
            if total > sent_up_to:
                new_lines = list(buf)[sent_up_to:total]
                for line in new_lines:
                    await websocket.send_json({"type": "log", "line": line})
                sent_up_to = total

            # If job is finished, notify and close
            if status in ("done", "failed", "cancelled"):
                await websocket.send_json({"type": "status", "status": status})
                break

            await asyncio.sleep(0.15)  # poll interval

    except WebSocketDisconnect:
        pass


# ── Mount advanced endpoints ──────────────────────────────────────────────────
from api.advanced import router as advanced_router
app.include_router(advanced_router)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
