# Neuro-Genetic Maintainability Project Deep Dive

This file is the code-grounded deep-dive README for the project in this repository.
Use it when someone wants more than the normal README:

- exact formulas
- exact feature names used by the code
- why each module exists
- what the pipeline does stage by stage
- what terms like parsimony, temporal leakage, Wilcoxon, SHAP, and Pareto mean in this project

If there is ever a mismatch between older documentation and the current implementation, treat the code as ground truth:

- `src/constants.py`
- `src/data_collector.py`
- `src/preprocess.py`
- `src/ann_model.py`
- `src/genetic_algorithm.py`
- `main.py`

## 1. Project In One Minute

This project predicts file-level software maintainability risk using a hybrid pipeline:

1. Mine source-code and Git-history metrics from a repository.
2. Build a supervised dataset where each row is a file.
3. Use a Genetic Algorithm to choose the most useful feature subset.
4. Use a PyTorch ANN to predict future bug-proneness from that subset.
5. Validate the result with baselines, ablation, significance tests, multi-repo checks, sensitivity analysis, SHAP, and a dashboard.

The central idea is:

- accuracy matters
- fewer features also matter
- future information must not leak into the past

So the project is not just "train a neural network." It is a research-style pipeline that combines:

- software metrics mining
- temporal data isolation
- feature selection
- neural regression
- statistical validation
- explainability

## 2. What Problem The Project Solves

The project tries to answer:

"Given the state of a file at a historical snapshot, can we predict how bug-prone that file will be in future commits?"

That is why the target is not a readability score or a static complexity score. The target is a future-looking bug signal derived from commit history.

This makes the problem:

- supervised
- regression-based
- temporally constrained
- suitable for maintainability-risk ranking

## 3. Main Contributions

Compared with a simpler paper-style baseline, this codebase adds:

1. Automated feature selection with a Genetic Algorithm instead of manual selection.
2. Three metric families instead of only structural metrics.
3. Temporal isolation between feature extraction time and target time.
4. Optuna-based hyperparameter tuning for the ANN.
5. Research validation modules: baselines, ablation, stats, multi-repo, sensitivity.
6. FastAPI + React dashboard with live logs and advanced analytics.

## 4. Very Important Ground-Truth Note

Older docs in this repo mention feature names such as:

- `loc`
- `maintainability_index`
- `comment_ratio`
- `num_authors`
- `bug_fix_commits`

Those are not the exact current code-level names used in the main implementation.

The actual current implementation uses names such as:

- `avg_cyclomatic_complexity`
- `halstead_volume`
- `comment_density`
- `author_count`
- `target_bug_proneness`

If your professor asks "what does the code actually use?", answer from `src/constants.py`, `src/data_collector.py`, and the processed CSV, not from older simplified text.

## 5. Repository Map

Core pipeline files:

- `main.py`: CLI orchestrator for the full pipeline
- `src/data_collector.py`: mines structural, textual, and evolutionary metrics
- `src/preprocess.py`: cleans dataset, clips outliers, removes highly correlated features
- `src/ann_model.py`: PyTorch ANN training, evaluation, prediction
- `src/tune.py`: Optuna search for ANN hyperparameters
- `src/genetic_algorithm.py`: GA-based feature selection
- `src/baseline.py`: All-features, random-subset, GA-selected, and XGBoost baselines
- `src/ablation.py`: category-combination experiments
- `src/stats.py`: Wilcoxon signed-rank test and Cohen's d
- `src/multi_repo.py`: cross-repository generalization
- `src/sensitivity.py`: rescoring-based alpha/beta/population sensitivity
- `src/report.py`: standalone HTML summary

Backend and UI:

- `api/main.py`: FastAPI backend, REST endpoints, job launching, log WebSocket
- `api/advanced.py`: SHAP, bootstrap CI, Pareto, diversity, cache persistence
- `frontend/src/App.jsx`: main React shell
- `frontend/src/components/PipelineTab.jsx`: pipeline control panel and live logs
- `frontend/src/utils/api.js`: frontend API/WebSocket client

## 6. End-To-End Pipeline

The pipeline in `main.py` runs these stages:

1. Stage 1: data collection
2. Stage 2: preprocessing
3. Stage 3: hyperparameter tuning
4. Stage 4: GA feature selection
5. Stage 5: baseline comparison
6. Stage 6: ablation study
7. Stage 7: statistical significance
8. Stage 8: multi-repository generalization
9. Stage 9: sensitivity sweep
10. Stage 10: HTML report generation

Important orchestration behavior:

- Existing outputs are skipped unless force flags are used.
- `--run-all` enables all research stages.
- All results are written to `data/results/`.

## 7. Exact Implemented Feature Space

### 7.1 Raw Feature Registry

The raw feature registry in `src/constants.py` contains 19 features:

Structural:

- `avg_cyclomatic_complexity`
- `halstead_volume`
- `halstead_effort`
- `depth_of_inheritance_tree`
- `number_of_methods_per_class`
- `weighted_methods_per_class`
- `nesting_depth`
- `class_coupling`

Textual:

- `comment_density`
- `whitespace_ratio`
- `docstring_presence`
- `avg_identifier_length`
- `code_duplication_pct`

Evolutionary:

- `commit_frequency`
- `author_count`
- `bug_fix_ratio`
- `code_age_days`
- `code_churn`
- `added_deleted_ratio`

### 7.2 Why The Final CSV Usually Has Fewer Features

`src/preprocess.py` removes highly correlated numeric features using:

- absolute Pearson correlation threshold `|r| > 0.95`

So the final processed dataset often has 18 features instead of 19.

For the current checked Flask cleaned CSV in this repo:

- rows: `80`
- numeric features after preprocessing: `18`

Current feature list:

- `avg_cyclomatic_complexity`
- `halstead_volume`
- `halstead_effort`
- `depth_of_inheritance_tree`
- `number_of_methods_per_class`
- `nesting_depth`
- `class_coupling`
- `comment_density`
- `whitespace_ratio`
- `docstring_presence`
- `avg_identifier_length`
- `code_duplication_pct`
- `commit_frequency`
- `author_count`
- `bug_fix_ratio`
- `code_age_days`
- `code_churn`
- `added_deleted_ratio`

In practice, `weighted_methods_per_class` is a natural candidate to be dropped because it is derived from other structural signals.

## 8. Temporal Snapshot Logic And Why Leakage Is Avoided

This is one of the most important research ideas in the codebase.

For each file:

1. The code chooses a snapshot date:

```python
snapshot_date = now - relativedelta(months=timeframe_months)
```

2. Commits touching that file are split into:

- `past_commits`: commits on or before the snapshot
- `future_commits`: commits after the snapshot

3. Features are extracted only from:

- the file contents at the last past snapshot state
- Git history up to the snapshot

4. The target is computed only from future commits:

```python
target_bug_fixes = sum(
    1 for c in future_commits
    if any(kw in c.message.lower() for kw in bug_keywords)
)
```

That target is stored as:

- `target_bug_proneness`

Bug keywords used in the code:

- `fix`
- `bug`
- `patch`
- `issue`
- `resolve`
- `error`

Why this matters:

- If we mined future commits into features, the model would cheat.
- This design keeps the prediction task realistic.

This is what people mean by:

- temporal leakage prevention
- no future information leakage
- snapshot-based supervision

## 9. Exact Feature Calculations

## 9.1 Structural Metrics

These are computed from the historical code snapshot.

`avg_cyclomatic_complexity`

- computed with `radon.complexity.cc_visit`
- code formula:

```python
sum(b.complexity for b in blocks) / max(1, len(blocks))
```

`halstead_volume` and `halstead_effort`

- computed with `radon.metrics.h_visit`
- the code uses Radon's final values directly
- conceptually:
  - Volume measures information content
  - Effort estimates mental effort required to understand/implement

`depth_of_inheritance_tree`

- code approximation:
- maximum number of base classes among class definitions

`number_of_methods_per_class`

- average number of methods across classes:

```python
total_methods / max(1, len(classes))
```

`weighted_methods_per_class`

- explicitly derived in code as:

```python
number_of_methods_per_class * avg_cyclomatic_complexity
```

`nesting_depth`

- recursively computes the deepest control-structure nesting
- tracked across `if`, `for`, `while`, `with`, `try`, `except`

`class_coupling`

- counts referenced names not defined inside the file:

```python
len(referenced_names - defined_names)
```

## 9.2 Textual Metrics

`comment_density`

```python
comment_lines / total_lines
```

`whitespace_ratio`

```python
empty_lines / total_lines
```

`docstring_presence`

- binary feature
- `1` if triple-quoted docstring markers are present, else `0`

`avg_identifier_length`

```python
sum(len(name) for name in identifiers) / max(1, len(identifiers))
```

`code_duplication_pct`

- duplicates are repeated non-empty lines

```python
duplicated / len(non_empty)
```

## 9.3 Evolutionary Metrics

These are computed from Git history up to the snapshot date.

`commit_frequency`

```python
len(past_commits)
```

`author_count`

```python
len(past_authors)
```

`bug_fix_ratio`

```python
past_bug_fixes / max(1, len(past_commits))
```

`code_age_days`

- days from the snapshot date back to the oldest available past commit for the file

`code_churn`

```python
insertions + deletions
```

`added_deleted_ratio`

```python
insertions / max(1, deletions)
```

Implementation detail:

- churn is extracted efficiently with a single `git log --numstat` call per file
- this replaced a much slower diff-per-commit strategy

## 10. Preprocessing Logic

`src/preprocess.py` does four main things:

1. Drop rows with missing values.
2. Clip extreme outliers at the 99th percentile for each numeric feature.
3. Remove highly correlated numeric features using `|r| > 0.95`.
4. Save the cleaned CSV without scaling.

Outlier clipping:

```python
cap = clipped[col].quantile(0.99)
clipped[col] = clipped[col].clip(upper=cap)
```

Correlation filtering:

```python
corr_matrix = clipped.corr().abs()
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
```

Very important design choice:

- scaling is not done in preprocessing
- scaling is deferred to `ann_model.py`
- this avoids train/validation leakage

If you are asked why this matters:

- fitting a scaler before the split lets the validation distribution leak into training
- the current code fits the scaler only on `X_train`

Also important:

- non-numeric columns like `repo` are preserved but excluded from clipping and correlation math

## 11. ANN Model

The ANN is defined in `src/ann_model.py`.

Architecture:

- input layer
- hidden layer 1
- hidden layer 2
- optional hidden layer 3
- single output neuron for regression

Per hidden layer the code may use:

- `Linear`
- optional `BatchNorm1d`
- `ReLU`
- `Dropout`

Important implementation details:

- `hidden3 = 0` disables the third hidden layer
- dropout on the optional third layer is lighter: `dropout / 2`
- optimizer is Adam with weight decay
- scheduler is `ReduceLROnPlateau`
- gradients are clipped to `1.0`

Key training choices:

- loss: MSE
- optimizer: Adam
- regularization: dropout + weight decay + optional batch norm
- early stopping based on validation MSE patience

## 11.1 Target Transform

The target is usually trained in log space:

```python
y_model = np.log1p(y_all)
```

Predictions are mapped back with:

```python
np.expm1(...)
```

Why `log1p`:

- bug counts are zero-inflated and skewed
- log transform compresses large values
- `log1p(0)` is valid, which is useful when many files have zero future bug-fix commits

## 11.2 Stratified K-Fold

The code does not stratify on the raw count directly.
It bins the target into:

- `0`
- `1`
- `2+`

Code:

```python
y_bins = np.clip(y_all.astype(int), 0, 2)
```

Why:

- small repository datasets can otherwise produce folds with no bug-prone files
- this makes cross-validation more stable

The code also clamps fold count to the minimum class count so `StratifiedKFold` does not crash on tiny datasets.

## 11.3 Divergence Guard

Inside `_train_fold`, the ANN uses an adaptive divergence cap:

```python
divergence_cap = max(4.0 * naive_mse, 2.0)
```

If the model diverges or becomes numerically unstable:

- return the cap instead of a nonsense value

This prevents a bad training run from destroying higher-level evaluation modules.

## 12. Optuna Hyperparameter Tuning

`src/tune.py` tunes the ANN before later stages reuse the best settings.

Search space:

- `lr`: log-uniform `[5e-5, 5e-3]`
- `hidden1`: `[32, 64, 128]`
- `hidden2`: `[16, 32, 64]`
- `hidden3`: `[0, 8, 16, 32]`
- `dropout`: `[0.1, 0.5]`
- `weight_decay`: log-uniform `[1e-5, 1e-2]`
- `batch_size`: `[8, 16, 32]`
- `use_bn`: `[True, False]`

Search constraints:

- `hidden2 <= hidden1`
- if `hidden3 > 0`, then `hidden3 <= hidden2`

Study settings:

- sampler: `TPESampler(seed=42)`
- pruner: `MedianPruner(n_startup_trials=10)`
- objective: minimize 5-fold stratified CV MSE on all features

One subtle implementation detail:

- when tuned hyperparameters are loaded later, learning rate is capped to `0.003`
- this prevents unstable overly aggressive tuned rates from ruining later experiments

## 13. Genetic Algorithm

The GA in `src/genetic_algorithm.py` is the core feature selector.

Each chromosome is a binary vector:

- `1` means keep the feature
- `0` means drop the feature

The chromosome length equals the number of numeric features in the processed dataset.

## 13.1 Fitness Function

Exact code formula:

```python
fitness = (
    alpha * (1.0 / (mse + 1e-6))
    + beta * (1.0 - n_selected / self.num_features)
)
```

Interpretation:

- first term rewards low prediction error
- second term rewards small feature subsets

Meaning of hyperparameters:

- `alpha`: weight for accuracy
- `beta`: weight for parsimony

If `alpha` increases:

- the GA cares more about predictive accuracy

If `beta` increases:

- the GA cares more about using fewer features

## 13.2 Why MSE Is Inverted

The GA is maximizing fitness, not minimizing error.

So it uses:

- `1 / (mse + epsilon)`

This means:

- lower MSE gives higher fitness

The `1e-6` prevents division by zero.

## 13.3 Selection, Crossover, Mutation, Elitism

Selection:

- tournament selection with `k = 3`

Crossover:

- single-point crossover

Mutation:

- bit-flip mutation
- mutation rate decays linearly from the initial rate to the minimum rate

Exact mutation-rate logic:

```python
rate = (
    min_mutation_rate
    + (mutation_rate - min_mutation_rate)
    * (1.0 - generation / generations)
)
```

Elitism:

- top `2` chromosomes are copied into the next generation unchanged

Validity constraint:

- a chromosome is never allowed to have zero selected features

## 13.4 Why The Split Seed Changes Per Generation

This is a very good professor question.

The code evaluates fitness with:

```python
fitness, mse = self.calculate_fitness(chrom, split_seed=gen)
```

Why:

- if every generation used the same split seed, the GA could overfit to one particular validation pattern
- varying the seed across generations makes the search less biased toward one lucky fold layout

The evaluation cache key includes both:

- chromosome
- split seed

so comparisons remain fair within the same generation.

## 13.5 Other GA Engineering Features

Implemented features:

- checkpoint save/resume
- memoization of evaluated chromosomes
- interim `ga_results.json` writes after each generation
- stagnation-based early stopping

Stored GA history includes:

- generation number
- best MSE
- global best MSE
- best fitness
- number of selected features
- mutation rate

## 14. Baseline Comparison

`src/baseline.py` runs four methods:

1. ANN on all features
2. ANN on a random subset with the same size as the GA subset
3. ANN on GA-selected features
4. XGBoost on GA-selected features

Why these baselines matter:

- all-features tests whether feature selection is useful
- random-subset controls for subset size
- XGBoost checks whether the ANN is actually the right model family

XGBoost settings used in code:

- `n_estimators=200`
- `learning_rate=0.05`
- `max_depth=4`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `early_stopping_rounds=20`

## 15. Ablation Study

`src/ablation.py` answers:

"Which feature families matter most?"

Tested combinations:

- A only
- B only
- C only
- A + B
- A + C
- B + C
- A + B + C

Where:

- A = Structural
- B = Textual
- C = Evolutionary

This is one of the strongest modules for defending the inclusion of Git-history and textual signals.

## 16. Statistical Significance

`src/stats.py` compares GA-selected ANN vs all-features ANN over repeated paired trials.

Metrics:

- mean MSE
- standard deviation
- percentage improvement
- Wilcoxon signed-rank statistic and p-value
- Cohen's d

Improvement formula:

```python
improvement_pct = (all_mean - ga_mean) / (all_mean + 1e-9) * 100
```

Wilcoxon is used because:

- it is non-parametric
- it works well for paired samples
- it does not assume normality as strongly as a t-test

Cohen's d in the code:

```python
differences = all_arr - ga_arr
cohens_d = diff_mean / (diff_std + 1e-9)
```

Interpretation bands in code:

- `< 0.2`: negligible
- `< 0.5`: small
- `< 0.8`: medium
- otherwise: large

Significance threshold:

- `p < 0.05`

## 17. Multi-Repository Generalization

`src/multi_repo.py` runs the pipeline independently across repositories and compares outcomes.

What it measures:

- number of files
- number of total features
- number of selected features
- feature reduction percentage
- GA best MSE
- all-feature baseline MSE
- improvement percentage
- selected feature names
- elapsed time

Why it matters:

- it tests whether the chosen features are repository-specific or generalizable

Current registry includes:

- `flask`
- `requests`
- `django`
- `fastapi`
- `numpy`

## 18. Sensitivity Analysis

`src/sensitivity.py` is fast because it usually does not retrain the ANN.

Instead it:

1. loads cached or historical GA evaluations
2. rescales them under different `alpha`, `beta`, and population-size settings
3. writes a grid of new scores

Key formula reused:

```python
alpha * (1.0 / (mse + 1e-6)) + beta * (1.0 - n_selected / max(n_total, 1))
```

Why this is clever:

- it answers "what if alpha/beta were different?" without rerunning the full pipeline

Population-size simulation:

- random samples are drawn from cached evaluations
- best rescored candidate in that sampled pool is used as the proxy result

## 19. Reporting

`src/report.py` builds a standalone HTML report from saved JSON outputs.

It summarizes:

- GA outcome
- baselines
- significance
- ablation
- multi-repo

This report is useful for demo and submission because it does not require the dashboard to be running.

## 20. FastAPI Backend

`api/main.py` converts the CLI/research pipeline into a web backend.

Main REST endpoints:

- `GET /api/status`
- `GET /api/results/{module}`
- `GET /api/data/summary`
- `GET /api/feature-names`
- `POST /api/pipeline/run`
- `GET /api/jobs`
- `GET /api/jobs/{id}`
- `DELETE /api/jobs/{id}`
- `POST /api/predict`
- `GET /api/report`

Main WebSocket:

- `WS /ws/logs/{job_id}`

This is how the frontend gets live streaming logs while the pipeline is running.

### 20.1 Prediction Risk Levels

The backend turns predicted scores into simple labels:

- `High` if score `>= 2.0`
- `Medium` if score `>= 0.5`
- `Low` otherwise

These thresholds are implemented in `api/main.py`.

## 21. Advanced Analytics Endpoints

`api/advanced.py` provides extra research endpoints.

Important ones:

- `WS /ws/ga-progress`: structured generation-by-generation GA telemetry
- `POST /api/shap`: SHAP explanations for the ANN
- `POST /api/stats/bootstrap`: bootstrap confidence intervals
- `GET /api/pareto`: Pareto-efficient MSE vs feature-count frontier
- `GET /api/diversity`: diversity proxy over generations
- `POST /api/cache/persist`: writes `ga_eval_cache.json` for fast sensitivity runs

Important advanced concepts:

SHAP:

- feature attribution method
- tells which selected features contributed most to predictions

Bootstrap CI:

- repeated resampling with replacement
- used here to compute confidence intervals on mean MSE differences

Pareto front:

- a point is Pareto-efficient if no other point is better on both objectives at once
- here the objectives are:
  - lower MSE
  - fewer features

Diversity proxy:

- code uses standard deviation of selected-feature counts over a sliding 3-generation window
- not exact Hamming distance
- but a useful convergence proxy without storing full populations

## 22. React Frontend

The frontend is a Vite + React dashboard.

Main roles:

- show pipeline state
- launch experiments
- stream logs
- display results
- provide extra tools like file risk, SHAP, chromosome editor, and advanced charts

Important implementation notes:

- `frontend/src/App.jsx` now lazy-loads heavy tabs
- `frontend/src/components/PipelineTab.jsx` lazy-loads `LiveGAProgress` only when needed
- `frontend/src/utils/api.js` supports direct API origin configuration through `VITE_API_ORIGIN`

Why this matters:

- Plotly-heavy views are expensive
- lazy loading improves first render performance
- direct API origin support avoids proxy-related local dev issues

## 23. Important Engineering Fixes Implemented In This Repository

These are good talking points because they show engineering maturity beyond the raw model idea.

### 23.1 Data Leakage Fix

`src/preprocess.py` used to scale too early.
Now scaling is done only after the train/validation split in `src/ann_model.py`.

### 23.2 Non-Numeric Metadata Safety

`repo` or other string columns are preserved but excluded from numeric preprocessing.

### 23.3 Lazy Torch Imports

Dashboard tabs that only display saved results can still work even if `torch` is not installed, because torch is imported lazily inside model-related functions.

### 23.4 Safer Stratified CV

The code bins targets into `[0, 1, 2+]` and clamps fold count to avoid small-dataset crashes.

### 23.5 Learning-Rate Stability Guard

Loaded tuned learning rates are capped at `0.003` to avoid unstable later runs.

### 23.6 GA Generalization Fix

The GA varies split seeds by generation instead of evaluating every chromosome on one identical fold partition forever.

### 23.7 Multi-Repo Compatibility Fix

`src/multi_repo.py` was refactored to match the real call signature and output format expected by the dashboard and report modules.

### 23.8 Fast Sensitivity Mode

`src/sensitivity.py` avoids ANN retraining when cache/history is available.

### 23.9 Frontend Stability And Performance Fixes

Recent frontend/backend improvements in this repo include:

- lazy loading heavy analytics tabs
- lazy loading live Plotly telemetry only when a job is active
- configurable direct API origin in `frontend/src/utils/api.js`
- extra local CORS origins in `api/main.py`

## 24. Current Saved Result Snapshot In This Repo

These are the saved results that were present when this deep-dive README was written.
They can change if you rerun the pipeline.

Processed Flask dataset:

- rows: `80`
- numeric features after preprocessing: `18`

Saved tuned ANN hyperparameters:

- `lr = 0.0028531725003682526`
- `hidden1 = 64`
- `hidden2 = 16`
- `hidden3 = 0`
- `dropout = 0.12489118508423548`
- `weight_decay = 0.0002948274130474986`
- `batch_size = 16`
- `use_bn = False`
- best CV MSE: `0.1830`

Saved baseline means:

- all features: about `0.2242`
- random subset: about `0.2495`
- GA-selected ANN: about `0.1748`
- GA-selected XGBoost: about `0.3856`

Saved significance result:

- GA mean MSE: about `0.1810`
- all-features mean MSE: about `0.2191`
- improvement: about `17.4%`
- Wilcoxon p-value: about `1.9e-06`
- Cohen's d: about `1.636`
- effect size: `large`
- significant: `True`

Saved multi-repo snapshot:

- Flask: about `21.57%` improvement, `50%` feature reduction
- Requests: about `14.53%` improvement, `68.8%` feature reduction

One caveat:

- the current `ga_results.json` in the repo may reflect an interim or older run
- so if exact GA totals differ from the cleaned CSV or other result files, trust the latest rerun and the processed dataset

## 25. Common Viva Questions And Good Answers

### Q1. Why is this a regression problem and not classification?

Because the target is a future bug-fix count, not just a binary defect label.
The code predicts `target_bug_proneness` as a numeric value.

### Q2. Why use `log1p` on the target?

Because bug counts are skewed and zero-inflated. `log1p` compresses large values and still works when the count is zero.

### Q3. Why use a Genetic Algorithm at all?

Because the goal is feature subset search over a combinatorial binary space.
For `n` features there are `2^n` possible subsets, so exhaustive search is impractical.

### Q4. What does parsimony mean here?

Parsimony means preferring a smaller feature subset when predictive power is similar.
In the fitness function this is the `beta * (1 - k/n)` part.

### Q5. Why use Wilcoxon instead of a paired t-test?

Wilcoxon is safer when the paired MSE differences may not be normally distributed, especially with relatively small trial counts.

### Q6. Why compare against a random subset baseline?

To prove that GA is not winning just because it uses fewer features.
The random subset baseline controls for subset size.

### Q7. Why compare against XGBoost?

Because feature selection and model family are different questions.
This tells us whether the selected features help only the ANN or are useful more generally.

### Q8. Why is temporal leakage a serious issue?

If future commits influence features, the model sees information that would not exist at prediction time.
That creates unrealistically optimistic performance.

### Q9. Why use SHAP?

To explain which selected features are driving model predictions.
That improves interpretability beyond just reporting the final chromosome.

### Q10. What is the Pareto frontier here?

It is the set of non-dominated solutions in the trade-off between:

- lower MSE
- fewer selected features

No Pareto point can be improved on both objectives simultaneously by another point.

## 26. Short Glossary

ANN:

- Artificial Neural Network

GA:

- Genetic Algorithm

MSE:

- Mean Squared Error

Parsimony:

- preference for simpler or smaller feature sets

Temporal leakage:

- contamination of training/evaluation by future information

Stratified K-Fold:

- cross-validation where class-like distributions are preserved across folds

BatchNorm:

- batch normalization, used to stabilize hidden-layer activations

Dropout:

- random hidden-unit masking during training for regularization

Weight decay:

- L2-like regularization applied by the optimizer

Early stopping:

- stop training when validation performance stops improving

Tournament selection:

- GA parent-selection method where a few candidates compete and the best is chosen

Elitism:

- copying top-performing chromosomes directly into the next generation

SHAP:

- feature-attribution method based on Shapley values

Bootstrap CI:

- confidence interval estimated by resampling with replacement

Cohen's d:

- standardized effect size

Wilcoxon signed-rank:

- non-parametric paired significance test

Pareto-efficient:

- not dominated on all objectives by any other point

## 27. How To Run The Project Quickly

Full CLI pipeline:

```bash
python main.py --repo test_repos/flask --run-all
```

Tune first, then run research modules:

```bash
python main.py --repo test_repos/flask --run-tuning
python main.py --repo test_repos/flask --run-baselines --run-ablation --run-stats --run-report
```

Frontend + backend:

```bash
./start.sh
```

Or run the backend directly:

```bash
./venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## 28. Best Files To Cite During A Demo Or Viva

If someone asks "show me where that is in code", these are the highest-value files:

- `src/data_collector.py`: snapshot logic, target definition, feature engineering
- `src/preprocess.py`: leakage-safe preprocessing
- `src/ann_model.py`: model, scaling, log-transform, CV
- `src/tune.py`: Optuna search space
- `src/genetic_algorithm.py`: fitness function and GA operators
- `src/stats.py`: Wilcoxon and Cohen's d
- `src/sensitivity.py`: fast cache-based rescoring
- `api/main.py`: job system and REST/WebSocket API
- `api/advanced.py`: SHAP, bootstrap, Pareto, diversity

## 29. Final One-Sentence Summary

This project is a leakage-aware, feature-selecting, research-grade software-maintainability prediction system that combines Git-aware metric mining, ANN regression, Genetic Algorithm optimization, statistical validation, and an interactive analytics dashboard.
