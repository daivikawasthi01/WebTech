# Automated Software Maintainability Assessment
## Hybrid Neuro-Genetic Framework

> Extends: Vescan & Barac-Antonescu (2025). *Software maintainability prediction based on change metric using neural network models.* Engineering Applications of Artificial Intelligence, 142, 110032. (Elsevier, Q1 SCIE)

---

## What This Project Does

This framework mines a Python Git repository, extracts 18 cross-dimensional code metrics, and uses a **Genetic Algorithm** to automatically select the optimal subset of those features for an **ANN-based bug-proneness predictor**.

The base paper trains a single ANN on a manually-chosen structural feature set. This project replaces that with a fully automated, leakage-free, three-dimensional pipeline:

| Limitation in Base Paper | This Project's Contribution |
|---|---|
| Manual correlation-based feature selection | Automated GA binary-chromosome selection |
| Structural metrics only | Three categories: structural + textual + evolutionary (Git) |
| No temporal data integrity | Snapshot-based mining — future commits never seen at train time |
| Abstract "change metric" target | Concrete bug-fix commit count from post-snapshot Git history |
| All features used, no parsimony | α-β fitness function balancing accuracy vs. feature compactness |

---

## Project Structure

```
.
├── main.py                    # CLI entry point — orchestrates all pipeline stages
├── app.py                     # Streamlit dashboard (visual results + pipeline runner)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build for reproducible deployment
├── src/
│   ├── constants.py           # Feature category definitions (A/B/C)
│   ├── data_collector.py      # Git + AST metric mining
│   ├── preprocess.py          # Outlier clipping, correlation filtering (no leakage)
│   ├── ann_model.py           # PyTorch ANN — training, k-fold CV, predictions
│   ├── genetic_algorithm.py   # GA with memoization, adaptive mutation, checkpointing
│   ├── baseline.py            # ANN-All / ANN-Random / ANN-GA / XGB-GA comparison
│   ├── ablation.py            # Per-category contribution study (7 combinations)
│   ├── stats.py               # Wilcoxon signed-rank test + Cohen's d
│   ├── tune.py                # Optuna ANN hyperparameter search
│   ├── multi_repo.py          # Cross-repository generalisation study
│   ├── sensitivity.py         # α / β / pop_size sensitivity sweep
│   └── report.py              # Standalone HTML report generator
└── data/
    └── results/               # All JSON outputs written here
```

---

## Installation

**Requirements:** Python 3.9+, Git

```bash
git clone https://github.com/yourusername/neuro-genetic-maintainability.git
cd neuro-genetic-maintainability
pip install -r requirements.txt
```

Clone a target repository to test against:

```bash
mkdir -p test_repos
git clone https://github.com/pallets/flask.git test_repos/flask
```

### Docker (optional)

```bash
docker build -t neuro-ga-maint .

# Run the Streamlit dashboard
docker run -p 8501:8501 -v $(pwd)/data:/app/data neuro-ga-maint

# Run the full CLI pipeline
docker run -v $(pwd)/data:/app/data -v $(pwd)/test_repos:/app/test_repos \
  neuro-ga-maint python main.py --repo test_repos/flask --run-all
```

---

## How to Run

### Step 1 — Full pipeline in one command

```bash
python main.py --repo test_repos/flask --run-all
```

This runs every stage in order: mine → clean → tune → GA → baselines → ablation → stats → multi-repo → sensitivity → HTML report.

### Step 2 — Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501. Use the sidebar to configure hyperparameters and tick which research modules to run. Click **▶ Run Full Pipeline**.

---

## CLI Reference

### Core arguments

| Argument | Default | Description |
|---|---|---|
| `--repo` | `test_repos/flask` | Path to the Git repository to analyse |
| `--raw-file` | `data/flask_dataset.csv` | Where to write the mined raw CSV |
| `--processed-file` | `data/flask_dataset_clean.csv` | Where to write the cleaned CSV |

### GA hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--pop-size` | `15` | Number of chromosomes per generation |
| `--generations` | `10` | Maximum generations |
| `--mutation-rate` | `0.20` | Initial per-gene mutation probability |
| `--min-mutation` | `0.03` | Mutation rate floor (exponential decay target) |
| `--stagnation` | `5` | Early-stop if no improvement for N generations |
| `--alpha` | `1.0` | Accuracy weight in fitness function |
| `--beta` | `0.5` | Parsimony weight in fitness function |

### Research module flags

| Flag | Description |
|---|---|
| `--run-tuning` | Run Optuna ANN hyperparameter search before GA (recommended on first run) |
| `--tune-trials` | Number of Optuna trials (default 50) |
| `--run-baselines` | Compare GA vs All-Features vs Random vs XGBoost |
| `--run-ablation` | Test all 7 category combinations (A, B, C, A+B, A+C, B+C, A+B+C) |
| `--run-stats` | Wilcoxon signed-rank test + Cohen's d vs all-features baseline |
| `--multi-repo` | Run GA across multiple repos (see `--repos`) |
| `--repos` | Space-separated repo names for multi-repo (default: `flask requests django`) |
| `--run-sensitivity` | Sweep α, β, and population size to validate robustness |
| `--run-report` | Generate standalone HTML report |
| `--run-all` | Enable all of the above in one pass |
| `--n-trials` | Trials per method for baselines/ablation/stats (default 20) |

### Force / skip flags

| Flag | Description |
|---|---|
| `--force-collect` | Re-mine even if raw CSV already exists |
| `--force-process` | Re-clean even if processed CSV already exists |
| `--force-all` | Re-run every stage regardless of existing outputs |

---

## Typical Workflows

### First run on a new repository

```bash
# Mine + clean + tune ANN + run GA (recommended order)
python main.py --repo test_repos/flask --run-tuning

# Then run research modules using the tuned hyperparameters
python main.py --repo test_repos/flask \
  --run-baselines --run-ablation --run-stats --run-report
```

### Replicate the base paper's no-parsimony condition

```bash
# alpha >> beta: prioritise accuracy, ignore feature count
python main.py --alpha 2.0 --beta 0.1 --generations 15
```

### Extreme parsimony experiment

```bash
# beta >> alpha: aggressively prune features
python main.py --alpha 0.5 --beta 2.0 --generations 15
```

### Multi-repository generalisation

```bash
# Clone additional repos first
git clone https://github.com/psf/requests.git test_repos/requests
git clone https://github.com/django/django.git test_repos/django

python main.py --multi-repo --repos flask requests django \
  --run-baselines --run-report
```

### Force a complete re-run

```bash
python main.py --run-all --force-all
```

---

## Output Files

All results are written to `data/results/`:

| File | Contents |
|---|---|
| `ga_results.json` | Best chromosome, MSE, fitness, convergence history |
| `best_hyperparams.json` | Tuned ANN hyperparameters from Optuna |
| `baseline_results.json` | MSE distributions for All-Features / Random / GA-ANN / GA-XGB |
| `ablation_results.json` | Mean MSE per feature category combination |
| `stats_results.json` | Wilcoxon p-value, Cohen's d, improvement % |
| `multi_repo_results.json` | Per-repo GA results and feature selection consistency |
| `sensitivity_results.json` | α × β × pop_size grid MSE values |
| `maintainability_report.html` | Self-contained HTML report — open in any browser |

---

## Feature Categories

All 18 features are split across three orthogonal dimensions. Names match the CSV columns exactly.

**Category A — Structural** (8 features)
`loc`, `num_functions`, `num_classes`, `cyclomatic_complexity`, `nesting_depth`, `class_coupling`, `halstead_volume`, `maintainability_index`

**Category B — Textual** (5 features)
`avg_identifier_length`, `comment_ratio`, `blank_line_ratio`, `avg_line_length`, `code_duplication_pct`

**Category C — Evolutionary** (5 features)
`commit_frequency`, `num_authors`, `code_churn`, `added_deleted_ratio`, `days_since_last_change`

**Target variable:** `bug_fix_commits` — number of commits containing fix/bug/patch/issue/defect keywords in the post-snapshot Git window.

---

## Experimental Results (Flask)

Tested against the Flask repository (Python, 1000+ commits).

| Metric | Value |
|---|---|
| Validation MSE | 0.0530 |
| Features selected | 6 / 18 |
| Feature reduction | ~67% fewer than all-features baseline |

**GA-selected subset:**
- `cyclomatic_complexity` (structural)
- `halstead_volume` (structural)
- `class_coupling` (structural)
- `comment_ratio` (textual — absent from base paper)
- `commit_frequency` (evolutionary — absent from base paper)
- `num_authors` (evolutionary — absent from base paper)

**Key finding:** The GA consistently selects a cross-dimensional subset, rejecting purely structural features in isolation. This validates the core hypothesis that structural complexity must be paired with developer churn and readability signals — a finding not observable in the base paper, which does not mine evolutionary or textual dimensions.

---

## Architecture

```
Repository (Git + Python source)
        │
        ▼
┌─────────────────────┐
│   data_collector.py │  AST walk + radon + git log → raw CSV (18 features + target)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    preprocess.py    │  Drop NaNs → clip 99th-pct outliers → remove |r|>0.95 cols
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│            genetic_algorithm.py          │
│                                          │
│  Population of 19-bit chromosomes        │
│  ┌─────────────────────────────────┐     │
│  │         ann_model.py            │     │
│  │  PyTorch 3-layer ANN            │     │
│  │  Input → 32 → 16 → 1 (MSE)     │◄────┤  fitness = α×(1/MSE) + β×(1−k/n)
│  └─────────────────────────────────┘     │
│  Tournament Selection                    │
│  Single-Point Crossover                  │
│  Adaptive Exponential Mutation Decay     │
│  Memoization cache + Checkpointing       │
└──────────────────────────────────────────┘
        │
        ▼
  Best chromosome → baseline.py / ablation.py / stats.py → JSON results
        │
        ▼
  app.py (Streamlit) + report.py (HTML)
```

---

## References

1. Vescan, A., & Barac-Antonescu, D. (2025). Software maintainability prediction based on change metric using neural network models. *Engineering Applications of Artificial Intelligence*, 142, 110032. https://doi.org/10.1016/j.engappai.2025.110032
2. Radon — Python code metrics tool. https://radon.readthedocs.io
3. GitPython — Git repository interaction library. https://gitpython.readthedocs.io
4. Optuna — Hyperparameter optimization framework. https://optuna.org