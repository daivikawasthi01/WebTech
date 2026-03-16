"""
constants.py — Shared feature category definitions.

Centralising these here means ablation.py, baseline.py, and the Streamlit app
all refer to the same ground truth about which features belong to which category.

All names exactly match the column names written by data_collector.py.
"""

# Category A: Structural — code complexity and OOP design
# Matches: collect_structural_metrics() in data_collector.py
CATEGORY_A_STRUCTURAL = [
    'loc',
    'num_functions',
    'num_classes',
    'cyclomatic_complexity',
    'nesting_depth',
    'class_coupling',
    'halstead_volume',
    'maintainability_index',
]

# Category B: Textual — readability and documentation quality
# Matches: collect_textual_metrics() in data_collector.py
CATEGORY_B_TEXTUAL = [
    'avg_identifier_length',
    'comment_ratio',
    'blank_line_ratio',
    'avg_line_length',
    'code_duplication_pct',
]

# Category C: Evolutionary — Git history signals
# Matches: collect_evolutionary_metrics_and_target() in data_collector.py
# ('bug_fix_commits' is the regression target, not a feature — excluded here)
CATEGORY_C_EVOLUTIONARY = [
    'commit_frequency',
    'num_authors',
    'code_churn',
    'added_deleted_ratio',
    'days_since_last_change',
]

ALL_FEATURES = CATEGORY_A_STRUCTURAL + CATEGORY_B_TEXTUAL + CATEGORY_C_EVOLUTIONARY

CATEGORY_LABELS = {
    'A': 'Structural',
    'B': 'Textual',
    'C': 'Evolutionary',
}