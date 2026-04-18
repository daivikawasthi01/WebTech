"""
constants.py — Shared feature category definitions.

Category C now includes code_churn and added_deleted_ratio, computed via
git log --numstat (single subprocess call per file, ~10-50ms). This replaces
the old commit.stats.total approach which ran git diff once per commit and
was prohibitively slow. The new approach is safe for FastAPI backends.
"""

# Category A: Structural — code complexity and OOP design
CATEGORY_A_STRUCTURAL = [
    'avg_cyclomatic_complexity',
    'halstead_volume',
    'halstead_effort',
    'depth_of_inheritance_tree',
    'number_of_methods_per_class',
    'weighted_methods_per_class',
    'nesting_depth',
    'class_coupling',
]

# Category B: Textual — readability and documentation quality
CATEGORY_B_TEXTUAL = [
    'comment_density',
    'whitespace_ratio',
    'docstring_presence',
    'avg_identifier_length',
    'code_duplication_pct',
]

# Category C: Evolutionary — Git history signals
CATEGORY_C_EVOLUTIONARY = [
    'commit_frequency',
    'author_count',
    'bug_fix_ratio',
    'code_age_days',
    'code_churn',           # total lines added + deleted (git log --numstat)
    'added_deleted_ratio',  # lines added / lines deleted (git log --numstat)
]

ALL_FEATURES = CATEGORY_A_STRUCTURAL + CATEGORY_B_TEXTUAL + CATEGORY_C_EVOLUTIONARY

CATEGORY_LABELS = {
    'A': 'Structural',
    'B': 'Textual',
    'C': 'Evolutionary',
}