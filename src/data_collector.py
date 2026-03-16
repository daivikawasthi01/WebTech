"""
src/data_collector.py — Mine Git history + AST metrics for every Python file
in a repository and write a raw CSV dataset.

Metrics collected
─────────────────
Category A – Structural (8):
  loc, num_functions, num_classes, cyclomatic_complexity,
  nesting_depth, class_coupling,          ← previously missing
  halstead_volume, maintainability_index

Category B – Textual (5):
  avg_identifier_length, comment_ratio, blank_line_ratio,
  avg_line_length, code_duplication_pct   ← previously missing

Category C – Evolutionary (5) + target:
  commit_frequency, code_churn, num_authors,
  added_deleted_ratio,                    ← previously missing
  days_since_last_change,
  bug_fix_commits  (target)
"""

import ast
import csv
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import git
from radon.metrics import h_visit, mi_visit
from radon.visitors import ComplexityVisitor


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Return the maximum nesting depth in the AST subtree rooted at *node*."""
    NESTING_NODES = (
        ast.If, ast.For, ast.While, ast.With, ast.Try,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    )
    child_depths = [
        _max_nesting(child, depth + (1 if isinstance(node, NESTING_NODES) else 0))
        for child in ast.iter_child_nodes(node)
    ]
    return max(child_depths) if child_depths else depth


# ---------------------------------------------------------------------------
# Per-file metric collectors
# ---------------------------------------------------------------------------

def collect_structural_metrics(source: str) -> dict:
    metrics: dict = {}
    lines = source.splitlines()

    # Basic counts
    metrics["loc"] = len(lines)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        metrics.update({
            "num_functions": 0, "num_classes": 0,
            "cyclomatic_complexity": 1,
            "nesting_depth": 0, "class_coupling": 0,
            "halstead_volume": 0.0, "maintainability_index": 0.0,
        })
        return metrics

    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,
                                                               ast.AsyncFunctionDef))]
    classes   = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    metrics["num_functions"] = len(functions)
    metrics["num_classes"]   = len(classes)

    # Cyclomatic complexity (average across all functions, min 1)
    try:
        cv = ComplexityVisitor.from_code(source)
        complexities = [b.complexity for b in cv.functions + cv.classes]
        metrics["cyclomatic_complexity"] = (
            sum(complexities) / len(complexities) if complexities else 1
        )
    except Exception:
        metrics["cyclomatic_complexity"] = 1

    # Nesting depth — BUG FIX: was missing entirely
    try:
        metrics["nesting_depth"] = _max_nesting(tree)
    except Exception:
        metrics["nesting_depth"] = 0

    # Class coupling — BUG FIX: was missing entirely
    # Count unique external attribute names referenced inside class bodies.
    try:
        names_in_classes: set[str] = set()
        for cls in classes:
            for node in ast.walk(cls):
                if isinstance(node, ast.Attribute):
                    names_in_classes.add(node.attr)
        metrics["class_coupling"] = len(names_in_classes)
    except Exception:
        metrics["class_coupling"] = 0

    # Halstead volume
    try:
        hal = h_visit(source)
        metrics["halstead_volume"] = hal.total.volume if hal else 0.0
    except Exception:
        metrics["halstead_volume"] = 0.0

    # Maintainability index
    try:
        metrics["maintainability_index"] = mi_visit(source, multi=True)
    except Exception:
        metrics["maintainability_index"] = 0.0

    return metrics


def collect_textual_metrics(source: str) -> dict:
    metrics: dict = {}
    lines = source.splitlines()
    non_empty = [l for l in lines if l.strip()]
    total     = max(1, len(lines))

    # Average identifier length
    identifiers = re.findall(r'\b[A-Za-z_]\w+\b', source)
    metrics["avg_identifier_length"] = (
        sum(len(i) for i in identifiers) / len(identifiers) if identifiers else 0.0
    )

    # Comment ratio
    comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
    metrics["comment_ratio"] = comment_lines / total

    # Blank line ratio
    blank_lines = sum(1 for l in lines if not l.strip())
    metrics["blank_line_ratio"] = blank_lines / total

    # Average line length (non-blank)
    metrics["avg_line_length"] = (
        sum(len(l) for l in non_empty) / len(non_empty) if non_empty else 0.0
    )

    # Code duplication percentage — BUG FIX: was missing entirely
    # Fraction of non-blank lines that are exact duplicates of another line.
    stripped = [l.strip() for l in lines if l.strip()]
    if stripped:
        duplicate_count = len(stripped) - len(set(stripped))
        metrics["code_duplication_pct"] = duplicate_count / len(stripped)
    else:
        metrics["code_duplication_pct"] = 0.0

    return metrics


def collect_evolutionary_metrics_and_target(
    repo: git.Repo,
    file_rel_path: str,
    lookback_commits: int = 100,
) -> dict:
    metrics: dict = {}

    try:
        past_commits = list(repo.iter_commits(paths=file_rel_path,
                                              max_count=lookback_commits))
    except Exception:
        past_commits = []

    metrics["commit_frequency"]      = len(past_commits)
    metrics["num_authors"]           = len({c.author.email for c in past_commits})

    # Code churn (total insertions + deletions)
    total_churn = 0
    total_added   = 0
    total_deleted = 0
    for commit in past_commits:
        try:
            file_stats = commit.stats.files.get(file_rel_path, {})
            ins = file_stats.get("insertions", 0)
            dele= file_stats.get("deletions",  0)
            total_churn   += ins + dele
            total_added   += ins
            total_deleted += dele
        except Exception:
            pass
    metrics["code_churn"] = total_churn

    # Added/deleted ratio — BUG FIX: was missing entirely
    metrics["added_deleted_ratio"] = total_added / max(1, total_deleted)

    # Days since last change
    if past_commits:
        last_ts = past_commits[0].committed_date
        now_ts  = datetime.now(timezone.utc).timestamp()
        metrics["days_since_last_change"] = max(0, (now_ts - last_ts) / 86_400)
    else:
        metrics["days_since_last_change"] = 0

    # Target: number of commits with bug-fix keywords
    BUG_PATTERN = re.compile(
        r'\b(fix|bug|error|issue|defect|patch|hotfix|crash|regression)\b',
        re.IGNORECASE,
    )
    metrics["bug_fix_commits"] = sum(
        1 for c in past_commits
        if c.message and BUG_PATTERN.search(c.message)
    )

    return metrics


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset_from_repo(
    repo_path: str,
    output_csv: str,
    lookback_commits: int = 100,
) -> None:
    """
    Walk *repo_path*, collect all three metric categories for every .py file,
    and write a CSV to *output_csv*.
    """
    print(f"[data_collector] Opening repo: {repo_path}")
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        raise ValueError(f"'{repo_path}' is not a valid Git repository.")

    repo_root = repo.working_tree_dir
    py_files  = [
        os.path.join(dirpath, fname)
        for dirpath, _, fnames in os.walk(repo_path)
        for fname in fnames
        if fname.endswith(".py")
        and ".git" not in dirpath
    ]

    if not py_files:
        raise ValueError(f"No Python files found under '{repo_path}'.")

    print(f"[data_collector] Found {len(py_files)} Python files.")

    rows: list[dict] = []
    for abs_path in py_files:
        rel_path = os.path.relpath(abs_path, repo_root)
        try:
            with open(abs_path, encoding="utf-8", errors="ignore") as fh:
                source = fh.read()
        except OSError:
            continue

        row: dict = {"file": rel_path}
        row.update(collect_structural_metrics(source))
        row.update(collect_textual_metrics(source))
        row.update(collect_evolutionary_metrics_and_target(
            repo, rel_path, lookback_commits
        ))
        rows.append(row)

    if not rows:
        raise ValueError("No metrics collected — check repository path.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[data_collector] Dataset saved → {output_csv}  ({len(rows)} rows)")