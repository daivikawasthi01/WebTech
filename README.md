# Automated Software Maintainability Assessment using a Hybrid Neuro-Genetic Framework

## Project Architecture & Methodology

### 1. The Core Problem & Research Gap
Traditional software maintainability models heavily rely on static, linear metrics or the classic Maintainability Index (MI). However, these approaches have significant blind spots:
* They primarily evaluate objective, structural code measurements while ignoring subjective factors and the actual development history of the systems.
* Many traditional metrics provide redundant information because they are highly correlated with one another.
* Rigid, threshold-based tools fail to capture the complex, non-linear degradation of modern software systems.

This project proposes a paradigm shift: rather than calculating a theoretical score, the framework will use an Artificial Neural Network (ANN) to predict tangible, real-world maintenance pain, optimized by a Genetic Algorithm (GA) to eliminate metric redundancy.

### 2. The Input Vector (The Independent Variables)
The framework analyzes 19 orthogonal parameters across three dimensions, ensuring a holistic view of the software's state.
* **Category A: Structural Metrics (Complexity & OOP Design)**: Cyclomatic Complexity, Halstead Volume/Effort, Depth of Inheritance Tree (DIT), Class Coupling (COF), Number of Methods per Class, Weighted Methods per Class, and Nesting Depth.
* **Category B: Textual & Documentation Metrics (Readability)**: Comment Density, Docstring Presence, Average Identifier Length, Code Duplication %, and Whitespace Ratio.
* **Category C: Evolutionary Metrics (Git History)**: Commit Frequency, Bug Fix Ratio, Author Count, Code Age, and Added/Deleted Line Ratio.

### 3. The Target Variable (The Ground Truth)
To prevent the ANN from simply memorizing an outdated formula, the classic Maintainability Index is removed from the input vector. Instead, the framework uses historical Git data as the supervised target variable (the "Ground Truth").

**Target Metric: The "Bug-Proneness" Score (or Composite Pain Index)**
The model is trained to predict the future instability of a file. By taking a historical snapshot of the 19 input parameters from *X* months ago, the ANN learns to predict how many bug-fix commits that specific file required over the subsequent *X* months.

### 4. The Hybrid Neuro-Genetic Framework
Because feeding 19 highly correlated parameters into an ANN can cause overfitting and increase computational overhead, a Genetic Algorithm is used as a wrapper for optimal feature selection.
* **Chromosome Representation:** A 19-bit binary array where `1` includes the feature in the ANN training, and `0` drops it.
* **The Evaluator:** For each generation, a lightweight ANN is trained on the selected features to predict the ground truth.
* **The Fitness Function:** The GA evaluates each chromosome by balancing predictive accuracy against model parsimony.

### 5. Expected Contribution
This framework will demonstrate that an evolutionarily optimized subset of structural, textual, and historical metrics can predict actual software defect rates and maintenance bottlenecks far more accurately than traditional, isolated static analysis tools.
