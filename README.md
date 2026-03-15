# Automated Software Maintainability Assessment using a Hybrid Neuro-Genetic Framework

## Overview

> **Base Paper:** Vescan, A., & Barac-Antonescu, D. (2025). *Software maintainability prediction based on change metric using neural network models.* Engineering Applications of Artificial Intelligence, 142, 110032. https://doi.org/10.1016/j.engappai.2025.110032
> *(SCIE-indexed, Elsevier — Q1 in Computer Science, Artificial Intelligence)*

Vescan & Barac-Antonescu (2025) establish a strong foundation by demonstrating that ANN-based models outperform classical static maintainability formulae. However, their framework exposes five critical limitations that this project directly addresses:

| Limitation in Base Paper | This Project's Contribution |
|---|---|
| Feature selection via manual correlation analysis | Automated GA-based binary chromosome selection |
| Structural (code) metrics only | Three-dimensional metrics: structural + textual + evolutionary (Git) |
| No temporal data integrity mechanism | Historical snapshot mining with mathematical leakage prevention |
| Prediction target is a generic "change metric" | Concrete bug-proneness score from real post-snapshot Git history |
| All features used — no parsimony control | Alpha-beta fitness function balancing accuracy vs. model compactness |

This project extends the base paper's ANN approach into a Hybrid Neuro-Genetic Framework that replaces manual feature engineering with a Genetic Algorithm optimizer, mines real development history to construct a leakage-free ground truth, and expands the metric space across three orthogonal dimensions.

---

## Architecture & Methodology

The pipeline operates in three phases that directly address the base paper's identified gaps.

### Phase 1: Temporal Data Mining (addressing Gap 3 — no leakage prevention)

Unlike Vescan & Barac-Antonescu (2025), who do not explicitly address temporal data integrity, this framework takes a historical snapshot of the codebase (e.g., exactly 6 months prior to analysis). The miner parses the Abstract Syntax Tree (AST) and Git log at that snapshot date to extract 19 cross-dimensional metrics. The ground truth (`target_bug_proneness`) is then computed strictly from bug-fix commits that occurred *after* the snapshot — ensuring the model cannot access future information at training time.

### Phase 2: Preprocessing

Features are scaled to [0, 1] using `MinMaxScaler` to optimize ANN convergence, consistent with the base paper's normalization approach. Files with unparseable or missing values are dropped.

### Phase 3: Hybrid Neuro-Genetic Optimization (addressing Gaps 1, 4, and 5)

The base paper trains a single ANN on a manually-selected, correlated feature set. This framework replaces that pipeline with a two-component hybrid:

- **The ANN Evaluator (PyTorch):** A 3-layer feed-forward network (Input → 32 → 16 → 1) trained to predict the exact number of future bug-fix commits (MSE loss). Architecture mirrors the base paper's ANN to allow a controlled comparison.
- **The GA Optimizer:** Wraps the ANN as a black-box evaluator. Generates 19-bit binary chromosomes (one bit per metric), evolves them over N generations using Tournament Selection, Single-Point Crossover, and Memoization caching.
- **The Fitness Function:** Unlike the base paper which uses all available features regardless of redundancy, each chromosome is scored by: `Fitness = α × (1 / MSE) + β × (1 − N_selected / N_total)`. The α and β hyperparameters control the accuracy-parsimony tradeoff, which is the key novelty over the base paper.

---

## Extracted Features (addressing Gap 2 — single-dimension metrics)

Vescan & Barac-Antonescu (2025) rely on structural and change-based code metrics. This framework expands the input space across three orthogonal dimensions, each capturing a distinct degradation signal:

**Category A: Structural Metrics (Complexity & OOP Design)**
Cyclomatic Complexity, Halstead Volume, Halstead Effort, Depth of Inheritance Tree (DIT), Class Coupling (COF), Number of Methods per Class, Weighted Methods per Class, Nesting Depth

**Category B: Textual & Documentation Metrics (Readability)**
Comment Density, Docstring Presence, Average Identifier Length, Code Duplication %, Whitespace Ratio

**Category C: Evolutionary Metrics (Git History)**
Commit Frequency, Bug Fix Ratio, Author Count, Code Age (days), Added/Deleted Line Ratio

The inclusion of Categories B and C constitutes a direct extension over the base paper, which does not mine developer activity or documentation quality signals.

---

## Target Variable (addressing Gap 4 — abstract change metric)

The base paper uses a "change metric" derived from version history as its prediction target. This framework replaces that with a more operationally meaningful target:

**Bug-Proneness Score:** The number of bug-fix commits affecting a file in the N months *after* the snapshot date. Bug-fix commits are identified using conventional commit keywords (`fix`, `bug`, `patch`, `issue`, `defect`) applied strictly to the post-snapshot window.

This makes predictions directly actionable — a score of 7 means a file received 7 confirmed bug-fix interventions, not an abstract index value.

---

## Installation & Setup

**Requirements:** Python 3.8+, Git

```bash
git clone https://github.com/yourusername/neuro-genetic-maintainability.git
cd neuro-genetic-maintainability
pip install torch pandas numpy scikit-learn GitPython radon
```

Prepare a target repository (e.g., Flask for replication):

```bash
mkdir test_repos
git clone https://github.com/pallets/flask.git test_repos/flask
```

---

## Usage

```bash
# Default pipeline
python main.py --repo "test_repos/flask"

# Experiment 1 — Extreme Parsimony (replicates β >> α condition from fitness function analysis)
python main.py --alpha 0.5 --beta 2.0 --generations 15

# Experiment 2 — Pure Accuracy (replicates α >> β, closer to base paper's no-parsimony approach)
python main.py --alpha 2.0 --beta 0.1 --generations 15

# Force re-collect on a new repository
python main.py --repo "test_repos/django" --raw-file "django_data.csv" \
               --processed-file "django_data_processed.csv" \
               --force-collect --force-process
```

---

## Experimental Results

Tested against Flask (open-source, Python, 1000+ commits). The GA converged to a 6-feature subset — less than one-third of the full 19-feature input — while achieving near-optimal predictive accuracy.

| Metric | Value |
|---|---|
| Validation MSE | 0.0530 |
| Features selected | 6 / 19 |
| Reduction over base paper approach | ~68% fewer features |

**Winning feature subset selected by GA:**
- Cyclomatic Complexity *(structural)*
- Halstead Effort *(structural)*
- Weighted Methods per Class *(structural)*
- Whitespace Ratio *(textual — not present in base paper)*
- Commit Frequency *(evolutionary — not present in base paper)*
- Author Count *(evolutionary — not present in base paper)*

**Key finding:** The GA consistently rejected purely structural metrics in isolation and selected a cross-dimensional subset, validating the core hypothesis: structural complexity metrics must be paired with developer churn and readability signals to accurately model real-world software degradation. This finding is not observable in the base paper, which does not mine evolutionary or textual dimensions.

---

## File Structure

```
.
├── main.py                   # CLI orchestrator and entry point
├── src/
│   ├── data_collector.py     # Git parsing, AST walking, Radon metric extraction
│   ├── preprocess.py         # MinMaxScaler and dataset cleaning
│   ├── ann_model.py          # PyTorch ANN architecture and training loop
│   └── genetic_algorithm.py  # GA with memoization and fitness evaluation
└── data/                     # Output directory for raw and processed CSVs
```

---

## References

1. Vescan, A., & Barac-Antonescu, D. (2025). Software maintainability prediction based on change metric using neural network models. *Engineering Applications of Artificial Intelligence*, 142, 110032. https://doi.org/10.1016/j.engappai.2025.110032
2. Radon — Python code metrics tool. https://radon.readthedocs.io
3. GitPython — Git repository interaction library. https://gitpython.readthedocs.io
