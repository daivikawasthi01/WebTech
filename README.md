# Automated Software Maintainability Assessment
### Hybrid Neuro-Genetic Framework

> **Course**: Soft Computing &nbsp;|&nbsp; **Authors**: Daivik Awasthi
>
> *Department of Information Technology, Netaji Subhas University of Technology, India*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.3%2B-blueviolet)
![Status](https://img.shields.io/badge/Status-Final%20Submission-brightgreen)

---

## What This Project Does

Software maintenance accounts for 60–80% of a system's total lifecycle cost. This framework automatically predicts which files in a codebase are most likely to need future bug fixes — before problems arise.

It does this by:

1. **Mining** a project's Git history and source code to extract 19 metrics across three dimensions: structural complexity, code readability, and developer activity
2. **Evolving** the best subset of those metrics using a Genetic Algorithm (GA) with a dual accuracy–parsimony fitness function
3. **Training** a Neural Network (ANN), tuned via Bayesian optimisation (Optuna), on the GA-selected subset to predict future bug-fix commit activity

This extends Vescan & Barac-Antonescu (2025) by replacing manual correlation-based feature selection with a fully automated, leakage-free, three-dimensional pipeline.

---

## Quick Start (3 Steps)

> **Prerequisites**: Python 3.9+ and Git must be installed.

```bash
# Step 1 — Install everything (run once)
bash setup.sh

# Step 2 — Run the full analysis
bash run.sh

# Step 3 — View the results
open data/results/maintainability_report.html
```

Results are also available at **http://localhost:5173** if the dashboard is running.

---

## Table of Contents

- [Results at a Glance](#results-at-a-glance)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Features Extracted](#features-extracted)
- [Output Files](#output-files)
- [Advanced Usage](#advanced-usage)
- [Dependencies](#dependencies)
- [References](#references)

---

## Results at a Glance

Evaluated across three mature open-source Python repositories:

| Repository | Files | Features Selected (*k*) | Feature Reduction | Validation MSE |
|---|---|---|---|---|
| Flask | 80 | 10 | 44.4% | **0.139** |
| Requests | 35 | 4 | 71.4% | **0.220** |
| FastAPI | 935 | 8 | 55.6% | **0.254** |

The GA-ANN achieves **17.4% lower MSE** than the all-features baseline on FastAPI (Cohen's *d* = 1.615, large effect) while using **55.6% fewer features**.

**Feature selection matrix** — which features the GA chose per repository:

| Feature | Category | Flask | Requests | FastAPI |
|---|---|:---:|:---:|:---:|
| `avg_cyclomatic_complexity` | Structural | ✓ | ✓ | ✓ |
| `halstead_volume` | Structural | ✓ | — | — |
| `nesting_depth` | Structural | ✓ | — | — |
| `class_coupling` | Structural | ✓ | — | — |
| `maintainability_index` | Structural | — | ✓ | — |
| `number_of_methods_per_class` | Structural | — | — | ✓ |
| `loc` | Structural | — | — | ✓ |
| `comment_ratio` | Textual | ✓ | ✓ | ✓ |
| `blank_line_ratio` | Textual | ✓ | — | — |
| `avg_line_length` | Textual | — | — | ✓ |
| `commit_frequency` | Evolutionary | ✓ | — | ✓ |
| `author_count` | Evolutionary | ✓ | — | ✓ |
| `code_churn` | Evolutionary | ✓ | — | — |
| `bug_fix_ratio` | Evolutionary | ✓ | — | — |
| `days_since_last_change` | Evolutionary | — | ✓ | ✓ |

> **Key finding**: The GA consistently selects cross-dimensional subsets in every repository — it never settles on structural metrics alone. `avg_cyclomatic_complexity` and `comment_ratio` appear in all three repositories, establishing them as the most universally relevant predictors. This validates the core hypothesis that structural complexity must be paired with developer-activity and readability signals to accurately characterise maintainability risk — a finding the base paper could not make, as it used structural metrics only.

---

## How It Works

```
Git Repository (source code + commit history)
        │
        ▼
┌──────────────────────┐
│   Data Collection    │  AST analysis (radon) + Git log (GitPython)
│                      │  → 19 raw features per file + bug-fix label
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│    Preprocessing     │  Drop NaNs → clip 99th-pct outliers
│                      │  → remove |r| > 0.95 correlated features
│                      │  → Min-Max scale to [0,1]
└──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│          Genetic Algorithm (GA)                  │
│                                                  │
│  Binary chromosome c ∈ {0,1}^n encodes which    │
│  features to include                             │
│                                                  │
│  Fitness: F(c) = α·(1/MSE) + β·(1 − k/n)       │
│                  ↑ accuracy    ↑ parsimony        │
│                                                  │
│  Operators: tournament selection (k=3),          │
│  single-point crossover, adaptive exponential    │
│  mutation decay (μ₀=0.20 → μ_min=0.03)          │
│  Elitism (top-2 preserved), memoisation cache   │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────┐
│    ANN Evaluator     │  3-layer MLP (PyTorch)
│                      │  Architecture tuned by 50 Optuna trials
│                      │  before GA begins — frozen for all experiments
│                      │  5-fold cross-validation MSE as fitness signal
└──────────────────────┘
        │
        ▼
  Baseline comparison · Ablation study · Statistical tests · HTML Report
```

**Snapshot-based temporal isolation**: the feature observation window and the bug-fix label window are strictly non-overlapping intervals in Git history. No future-commit information can contaminate training data.

**What makes this different from the base paper (Vescan & Barac-Antonescu, 2025):**

| Dimension | Base Paper (NNCP) | This Project |
|---|---|---|
| Feature selection | Manual Pearson thresholds | Automated GA binary search |
| Metric dimensions | CK structural only | Structural + Textual + Evolutionary |
| Optimisation goal | Accuracy (MSE) only | Accuracy + Parsimony (dual objective) |
| Temporal integrity | Not specified | Snapshot isolation — no leakage |
| Target variable | Lines changed per class | Bug-fix commit count from Git |
| ANN tuning | Manual, fixed layers | Optuna Bayesian search (50 trials) |
| Validation | 2 projects (UIMS, QUES) | 3 open-source repos (Flask, Requests, FastAPI) |

---

## Installation

### Automatic (Recommended)

```bash
bash setup.sh
```

This one script will:
- Create a Python virtual environment
- Install all required libraries
- Download the test repositories (Flask, Requests, FastAPI)
- Verify everything is working

### Manual

```bash
git clone https://github.com/yourusername/neuro-genetic-maintainability.git
cd neuro-genetic-maintainability

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

mkdir -p test_repos
git clone https://github.com/pallets/flask.git         test_repos/flask
git clone https://github.com/psf/requests.git          test_repos/requests
git clone https://github.com/tiangolo/fastapi.git      test_repos/fastapi
```

---

## Running the Project

### Option 1 — One Command (Recommended)

```bash
bash run.sh
```

Runs the complete pipeline and generates an HTML report. Takes **15–30 minutes**.

### Option 2 — Interactive Dashboard

```bash
bash start.sh
```

Then open **http://localhost:5173** to:
- Select a repository
- Tune GA hyperparameters interactively
- Watch pipeline logs in real time
- Download results

### Option 3 — Command Line (Full Control)

```bash
python main.py --repo test_repos/flask --run-all
```

This runs all stages in sequence: mine → clean → tune → GA → baselines → ablation → stats → multi-repo → sensitivity → report.

### Option 4 — Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend Dashboard | http://localhost:5173 |
| Backend API | http://localhost:8000 |

---

## Project Structure

```
.
├── main.py                  # Entry point — runs the full pipeline
├── setup.sh                 # One-time setup script
├── run.sh                   # Run analysis
├── start.sh                 # Launch interactive dashboard
├── requirements.txt         # Python dependencies (pinned versions)
├── Dockerfile               # For reproducible containerised execution
│
├── src/
│   ├── constants.py         # Feature category definitions (A/B/C)
│   ├── data_collector.py    # Git log + AST metric extraction
│   ├── preprocess.py        # Outlier removal, correlation filtering, scaling
│   ├── ann_model.py         # PyTorch 3-layer MLP (training, k-fold CV)
│   ├── genetic_algorithm.py # GA with adaptive mutation, elitism, memoisation
│   ├── baseline.py          # ANN-All / ANN-Random / GA-ANN / XGB-GA comparison
│   ├── ablation.py          # 7-combination category contribution study (A/B/C)
│   ├── stats.py             # Wilcoxon signed-rank test + Cohen's d
│   ├── tune.py              # Optuna Bayesian hyperparameter search (50 trials)
│   ├── multi_repo.py        # Cross-repository generalisation study
│   ├── sensitivity.py       # α / β / pop_size robustness sweep
│   └── report.py            # HTML report generator
│
├── api/                     # FastAPI backend (WebSocket logs + REST)
├── frontend/                # Vite + React dashboard
└── data/
    └── results/             # All JSON and HTML outputs written here
```

---

## Features Extracted

All features are split across three independent categories. A pairwise Pearson correlation filter (|r| > 0.95) is applied before genetic search, typically reducing the active space from 19 to 18 dimensions.

**Category A — Structural** (8 features) — internal complexity at a fixed point in time

`avg_cyclomatic_complexity`, `halstead_volume`, `number_of_methods_per_class`, `nesting_depth`, `class_coupling`, `maintainability_index`, `loc`, `num_classes`

**Category B — Textual** (5 features) — readability and documentation quality

`avg_identifier_length`, `comment_ratio`, `blank_line_ratio`, `avg_line_length`, `code_duplication_pct`

**Category C — Evolutionary** (6 features) — developer activity from Git history

`commit_frequency`, `author_count`, `code_churn`, `added_deleted_ratio`, `days_since_last_change`, `bug_fix_ratio`

**Target variable**: `bug_fix_commits` — count of commits in the post-snapshot window containing keywords: *fix*, *bug*, *patch*, *issue*, *defect*.

---

## Output Files

All outputs are written to `data/results/`:

| File | Contents |
|---|---|
| `maintainability_report.html` | **Interactive HTML report — open this in any browser** |
| `ga_results.json` | Best chromosome, MSE, fitness score, convergence curve |
| `best_hyperparams.json` | Tuned ANN architecture from Optuna (layer sizes, dropout, LR, etc.) |
| `baseline_results.json` | MSE distributions: All-Features vs Random vs GA-ANN vs XGB-GA |
| `ablation_results.json` | Mean MSE for all 7 category combinations (A, B, C, A+B, A+C, B+C, A+B+C) |
| `stats_results.json` | Wilcoxon p-value, Cohen's d, improvement percentage |
| `multi_repo_results.json` | Per-repository GA results and feature selection consistency |
| `sensitivity_results.json` | α × β × pop_size grid of MSE values |

**CSV files** (in `data/`):
- `flask_dataset.csv` — Raw 19 features + target per file
- `flask_dataset_clean.csv` — After outlier removal and preprocessing

---

## Advanced Usage

<details>
<summary><strong>CLI Reference — all arguments</strong></summary>

### Core arguments

| Argument | Default | Description |
|---|---|---|
| `--repo` | `test_repos/flask` | Path to the Git repository to analyse |
| `--raw-file` | `data/flask_dataset.csv` | Where to write the raw mined CSV |
| `--processed-file` | `data/flask_dataset_clean.csv` | Where to write the cleaned CSV |

### GA hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--pop-size` | `15` | Chromosomes per generation |
| `--generations` | `10` | Maximum generations |
| `--mutation-rate` | `0.20` | Initial per-gene mutation probability (μ₀) |
| `--min-mutation` | `0.03` | Mutation rate floor after exponential decay |
| `--stagnation` | `5` | Early-stop if no improvement for N generations |
| `--alpha` | `1.0` | Accuracy weight in fitness function |
| `--beta` | `0.5` | Parsimony weight in fitness function |

### Research module flags

| Flag | Description |
|---|---|
| `--run-tuning` | Run Optuna ANN hyperparameter search (50 trials) before GA |
| `--run-baselines` | Compare GA-ANN vs All-Features vs Random vs XGB-GA |
| `--run-ablation` | Test all 7 feature category combinations (A, B, C, A+B, A+C, B+C, A+B+C) |
| `--run-stats` | Wilcoxon signed-rank test + Cohen's d over 5-trial MSE distributions |
| `--multi-repo` | Run GA across multiple repos (see `--repos`) |
| `--repos` | Space-separated repo names (default: `flask requests fastapi`) |
| `--run-sensitivity` | Sweep α ∈ {0.5,1.0,1.5,2.0}, β ∈ {0.1,0.5,1.0,2.0}, P ∈ {5,8,10,15} |
| `--run-report` | Generate standalone HTML report |
| `--run-all` | Enable everything above in one pass |
| `--n-trials` | Trials per method for baselines/ablation/stats (default: 20) |

### Force / skip flags

| Flag | Description |
|---|---|
| `--force-collect` | Re-mine even if raw CSV already exists |
| `--force-process` | Re-clean even if processed CSV already exists |
| `--force-all` | Re-run every stage regardless of existing outputs |

</details>

<details>
<summary><strong>Experiment recipes</strong></summary>

**Replicate base paper (accuracy-only, no parsimony):**
```bash
# α >> β: prioritise MSE, ignore feature count
python main.py --alpha 2.0 --beta 0.1 --generations 15
```

**Extreme parsimony (aggressively prune features):**
```bash
# β >> α: maximum feature reduction at marginal MSE cost
python main.py --alpha 0.5 --beta 2.0 --generations 15
```

**Multi-repository generalisation (all three repos):**
```bash
python main.py --multi-repo --repos flask requests fastapi \
  --run-baselines --run-report
```

**Force a complete re-run from scratch:**
```bash
python main.py --run-all --force-all
```

</details>

<details>
<summary><strong>ANN architecture details (Optuna search space)</strong></summary>

The ANN is a three-hidden-layer MLP in PyTorch. Optuna runs 50 trials on the full feature set to find the optimal configuration before GA begins. The tuned hyperparameters are then frozen for all GA, baseline, and ablation experiments to ensure fair comparison.

| Hyperparameter | Search Range | Purpose |
|---|---|---|
| Learning rate | Log-uniform [5×10⁻⁵, 5×10⁻³] | Gradient step size |
| Hidden layer 1 | {32, 64, 128} | First-stage capacity |
| Hidden layer 2 | {16, 32, 64} (≤ H1) | Second-stage abstraction |
| Hidden layer 3 | {0, 8, 16, 32} (≤ H2) | Optional depth |
| Dropout rate | Uniform [0.1, 0.5] | Stochastic regularisation |
| Weight decay | Log-uniform [10⁻⁵, 10⁻²] | L2 coefficient |
| Batch size | {8, 16, 32} | Mini-batch stability |
| Batch normalisation | Boolean | Covariate shift reduction |

The funnel constraint (H₁ ≥ H₂ ≥ H₃) prevents the network from memorising small-dataset targets. All features are Min-Max scaled to [0,1] before training. Activation function: ReLU throughout.

</details>

---

## Dependencies

All versions are pinned in `requirements.txt` for full reproducibility.

| Library | Version | Purpose |
|---|---|---|
| `torch` | 2.0+ | Neural network implementation (PyTorch MLP) |
| `optuna` | 3.3+ | Bayesian hyperparameter tuning (50 trials) |
| `scikit-learn` | 1.3+ | K-fold CV, Min-Max scaling, utilities |
| `xgboost` | 2.0+ | XGB-GA baseline model |
| `pandas` | 2.0+ | Data handling and CSV I/O |
| `numpy` | 1.24+ | Numerical computing |
| `scipy` | 1.11+ | Wilcoxon signed-rank test, Cohen's d |
| `GitPython` | 3.1+ | Git log mining and commit history |
| `radon` | 6.0+ | AST-based code metrics (cyclomatic complexity, Halstead, MI) |
| `plotly` | 5.18+ | Interactive charts in HTML report |
| `tqdm` | 4.66+ | Progress bars |
| `python-dateutil` | 2.8+ | Date handling for snapshot windows |

**Optional (for the interactive dashboard):**
- Node.js 18+ (Vite + React UI)
- Docker 20+ (containerised execution)

**System requirements:**
- Python 3.9+, Git (latest)
- OS: Linux, macOS, or Windows (WSL2 recommended)
- Disk: ~2 GB (includes cloned test repositories)
- FastAPI is the largest subject (~935 files); its GA run takes approximately 8 minutes

---

## References

1. Vescan, A., & Barac-Antonescu, D. (2025). Software maintainability prediction based on change metric using neural network models. *Engineering Applications of Artificial Intelligence*, 144, 110032. https://doi.org/10.1016/j.engappai.2025.110032
2. Nagappan, N., & Ball, T. (2005). Use of relative code churn measures to predict system defect density. *ICSE 2005*, pp. 284–292.
3. Rahman, F., & Devanbu, P. (2013). How, and why, process metrics are better. *ICSE 2013*, pp. 432–441.
4. Radon — Python code metrics. https://radon.readthedocs.io
5. GitPython — Git repository interaction. https://gitpython.readthedocs.io
6. Optuna — Hyperparameter optimisation framework. https://optuna.org

---

<div align="center">

**Daivik Awasthi** (`2023UIT3079`)

Department of Information Technology, NSUT India &nbsp;|&nbsp; April 2026

</div>
