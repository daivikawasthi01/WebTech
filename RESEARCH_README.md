# Research Technical Documentation: Hybrid Neuro-Genetic Framework for Software Maintainability

This document provides the formal technical specifications, mathematical foundations, and experimental protocols for the **Neuro-Genetic Maintainability Framework**. It is designed to serve as the primary resource for drafting a research paper based on this repository.

---

## 1. Abstract
We propose a hybrid framework that integrates **Artificial Neural Networks (ANN)** and **Genetic Algorithms (GA)** to automate software maintainability assessment. Unlike traditional models that rely purely on structural complexity (e.g., LOC, Cyclomatic Complexity), this framework incorporates **Textual** and **Evolutionary** dimensions mined from Git history. The GA optimizes feature selection to maximize predictive accuracy while minimizing feature redundancy (parsimony), addressing the limitations of manual feature selection found in previous literature.

---

## 2. Theoretical Framework & Contribution
This work extends the methodology of **Vescan & Barac-Antonescu (2025)** by introducing three critical advancements:
1.  **Automated Feature Selection**: Replaces manual correlation-based selection with a GA-driven binary optimization.
2.  **Multidimensional Feature Space**: Expands from 1D (Structural) to 3D (Structural + Textual + Evolutionary).
### 2.1 Comparative Analysis: Baseline vs. Proposed Framework

| Feature | Vescan & Barac-Antonescu (2025) | Proposed Framework (This Project) |
|---|---|---|
| **Feature Selection** | Manual (Correlation-based) | **Automated (Genetic Algorithm)** |
| **Metric Dimensions** | 1D: Structural only | **3D: Structural + Textual + Evolutionary** |
| **Optimization Goal** | Accuracy only | **Accuracy + Parsimony (Feature Reduction)** |
| **Data Integrity** | Not specified (Potential leakage) | **Snapshot-based temporal isolation (No leakage)** |
| **Target Variable** | Abstract "Change Metric" | **Concrete Bug-Fix Commit Count (Post-Snapshot)** |
| **Optimization Method** | Fixed NN Architecture | **Optuna-driven Bayesian Hyperparameter Tuning** |

---

## 3. Methodology

### 3.1 Taxonomy of Software Metrics
We define a 19-dimensional feature space categorized into three distinct domains:

| Category | Domain | Features | Count |
| :--- | :--- | :--- | :--- |
| **Category A** | **Structural** | Cyclomatic complexity, Halstead volume, depth of inheritance, class coupling, weighted methods, etc. | 8 |
| **Category B** | **Textual** | Comment density, whitespace ratio, docstring presence, identifier length, duplication. | 5 |
| **Category C** | **Evolutionary** | Commit frequency, author count, bug-fix ratio, code age, churn, added/deleted ratio. | 6 |

#### Category A: Structural (8 Features)
*Metrics representing code complexity and Object-Oriented design.*
- **Cyclomatic Complexity**: Measures logical control flow paths.
- **Halstead Volume & Effort**: Measures information content and mental effort required.
- **DIT (Depth of Inheritance Tree)**: Measure of OOP complexity.
- **WMC (Weighted Methods per Class)**: Total complexity of methods in a class.
- **Nesting Depth**: Maximum depth of control structures.
- **Class Coupling**: Count of classes a specific file depends on.

#### Category B: Textual (5 Features)
*Metrics representing readability, documentation, and code quality.*
- **Comment Density**: Ratio of comment lines to code.
- **Docstring Presence**: Boolean/count indicating formal documentation.
- **Average Identifier Length**: Signal for naming clarity.
- **Code Duplication %**: Measure of technical debt (DRY principle).
- **Whitespace Ratio**: Visual layout quality signal.

#### Category C: Evolutionary (6 Features)
*Metrics derived from Git history to capture developer churn and volatility.*
- **Commit Frequency**: Update rate of the file.
- **Author Count**: Measuring "ownership" and potential coordination overhead (Brook's Law).
- **Bug-Fix Ratio**: Historical prevalence of defects.
- **Code Age**: Time since creation or last major refactor.
- **Code Churn**: Cumulative lines added and deleted.
- **Added/Deleted Ratio**: Signal of expansion vs. refactoring.

---

### 3.2 Preprocessing & Dilation
- **Correlation Filtering**: Highly collinear features ($|r| > 0.95$) are automatically pruned. In practice, this typically reduces the 19-dimensional space to **18 features** for GA selection (e.g., dropping `weighted_methods_per_class`).
- **Log Transformation**: Bug counts are log-normalized to stabilize training on zero-inflated distributions.
- **Min-Max Scaling**: Features are scaled to [0, 1] to ensure weight symmetry in the ANN.

---

### 3.3 Neuro-Predictor (ANN Architecture)
The predictive core is a **Flexible 3-layer Multi-Layer Perceptron (MLP)** optimized for small, imbalanced repository datasets.

- **Architecture**: Input layer → Hidden 1 → Hidden 2 → Optional Hidden 3 → 1 (Regressor).
- **Regularization**: Batch Normalization, Dropout (0.1–0.5), and Adam with Weight Decay ($10^{-4}$).
- **Optimization**: Hyperparameters (layer sizes, learning rate, dropout) are determined via 50 trials of **Bayesian Optimization (Optuna)** to minimize 5-fold CV MSE.
- **Optimizer**: Adam with an adaptive learning rate (ReduceLROnPlateau).

---

### 3.4 Genetic Optimization (GA)
The GA performs binary feature selection where a chromosome $\mathbf{c} \in \{0, 1\}^{19}$ represents the presence/absence of a feature.

#### Fitness Function
The multi-objective fitness $F$ is defined as:
$$F(\mathbf{c}) = \alpha \cdot \left(\frac{1}{MSE + \epsilon}\right) + \beta \cdot \left(1 - \frac{k}{n}\right)$$
Where:
- $MSE$: Mean Squared Error calculated via stratified 5-fold cross-validation.
- $k$: Number of features selected.
- $n$: Total available features (19).
- $\alpha, \beta$: Weights balancing accuracy vs. parsimony.

#### GA Operators
- **Selection**: Tournament Selection (size $k=3$).
- **Crossover**: Single-Point Crossover.
- **Mutation**: Bit-flip mutation with **Adaptive Exponential Decay**, starting high to encourage exploration and cooling to refine the solution.
- **Elitism**: Top 2 chromosomes are preserved across generations.
- **Memoization**: Fitness scores are cached to avoid redundant ANN training.

---

## 4. Experimental Design

### 4.1 Hyperparameter Tuning (Optuna Search Space)
Before the GA or Baseline runs, **Optuna** (a Bayesian optimization framework) is used to find the optimal ANN configuration. The search space is defined to balance model capacity with regularization:

| Hyperparameter | Search Space / Distribution | Purpose |
|---|---|---|
| **Learning Rate** | Log-uniform $[5 \times 10^{-5}, 5 \times 10^{-3}]$ | Capped at $0.003$ to ensure stability during GA |
| **Hidden Layer 1** | Categorical $\{32, 64, 128\}$ | Initial feature transformation capacity |
| **Hidden Layer 2** | Categorical $\{16, 32, 64\}$ | Second-stage abstraction ($H_2 \le H_1$ constrained) |
| **Hidden Layer 3** | Categorical $\{0, 8, 16, 32\}$ | Optional deepening ($0$ disables layer; $H_3 \le H_2$) |
| **Dropout Rate** | Uniform $[0.1, 0.5]$ | Stochastic regularization |
| **Weight Decay** | Log-uniform $[1 \times 10^{-5}, 10^{-2}]$ | L2 Regularization coefficient |
| **Batch Size** | Categorical $\{8, 16, 32\}$ | Mini-batch optimization stability |
| **BatchNorm** | Boolean $\{$True, False$\}$ | Covariate shift reduction |

### 4.2 Research Questions (RQs)
1.  **RQ1 (Performance)**: Does the GA-selected feature subset outperform the "All-Features" baseline?
2.  **RQ2 (Significance of Dimensions)**: What is the relative contribution of Evolutionary and Textual features compared to purely Structural metrics? (Analyzed via **Ablation Studies**).
3.  **RQ3 (Generalization)**: How consistent is the feature selection across different open-source projects? (Analyzed via **Multi-Repo Study**).
4.  **RQ4 (Sensitivity)**: How does the choice of $\alpha$ and $\beta$ impact the model's complexity and error?

### 4.3 Statistical Validation
To ensure results are not due to random chance (ANN initialization or data splits):
- **Wilcoxon Signed-Rank Test**: A non-parametric paired test used to compare MSE distributions of GA-selected vs. All-Features models.
- **Cohen's d**: Measures the effect size (magnitude of improvement).

---

## 5. Implementation Workflow & Monitoring

The framework provides a dual interface: a **CLI-based pipeline** for large-scale experiments and an **Interactive Dashboard** for qualitative analysis.

### 5.1 Interactive Dashboard (FastAPI + React)
Built to facilitate researcher exploration, the new decoupled dashboard provides:
- **Real-time Log Streaming**: WebSocket-based output streaming from the backend pipeline.
- **Async Execution**: The pipeline runs in a background thread; the UI remains responsive throughout.
- **REST API Results**: Clean JSON endpoints for per-module research data.
- **File-Risk Assessment**: Heatmaps and tables ranking repository files by their predicted bug-proneness.
- **What-If Analysis**: Tools to toggle individual features and observe the impact on predictive MSE instantly via the Chromosome Editor.

### 5.2 CLI Workflow for Researchers
Run `data_collector.py` to mine the target repository. 
- *Input*: Local Git Repo path.
- *Output*: `raw_dataset.csv`.

### Phase 2: Preprocessing
Run `preprocess.py` to clip outliers and handle multicollinearity (removes features with $|r| > 0.95$ to ensure GA does not select redundant signals).

### Phase 3: Optimization
Run `main.py --run-tuning --run-ga` to evolve the optimal feature subset.

## 6. How to Run for Research Validation

### Option 1: Full Pipeline (CLI)
Recommended for automated data collection for multiple repositories.
```bash
python3 main.py --repo test_repos/flask --run-all
```

### Option 2: Full Stack (Interactive)
Starts the FastAPI backend and React frontend for visual data exploration.
```bash
chmod +x start.sh
./start.sh
```
The dashboard will be available at `http://localhost:5173`. Use the **Pipeline** tab to configure and launch trials.

---

## 7. Interpreting Results
The framework outputs JSON files in `data/results/` which can be directly converted into paper figures:
- `ga_results.json`: Convergence plots and best chromosome.
- `ablation_results.json`: Bar charts showing the value of Textual/Evolutionary data.
- `stats_results.json`: p-values and effect sizes for the Significance section of the paper.

---

## References
1.  Vescan, A., & Barac-Antonescu, D. (2025). Software maintainability prediction based on change metric using neural network models. *Engineering Applications of Artificial Intelligence*.
2.  Radon, GitPython, and PyTorch documentation.
