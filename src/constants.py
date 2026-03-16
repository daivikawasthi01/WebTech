"""
constants.py — Shared feature category definitions.

added_deleted_ratio removed from CATEGORY_C_EVOLUTIONARY — it required
commit.stats.total which diffs every commit and hangs on cloud deployments
with large repos. Removing it keeps the GA chromosome honest and fast.
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
    # 'added_deleted_ratio' removed — required commit.stats.total, too slow on cloud
]

ALL_FEATURES = CATEGORY_A_STRUCTURAL + CATEGORY_B_TEXTUAL + CATEGORY_C_EVOLUTIONARY

CATEGORY_LABELS = {
    'A': 'Structural',
    'B': 'Textual',
    'C': 'Evolutionary',
}