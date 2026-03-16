import os
import ast
from collections import Counter
from radon.complexity import cc_visit
from radon.metrics import h_visit
import git
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta


def collect_structural_metrics(code):
    """
    Extract Category A metrics: Cyclomatic Complexity, Halstead, DIT, WMC,
    nesting_depth, class_coupling.
    """
    if not code:
        return {}

    metrics = {}

    try:
        blocks = cc_visit(code)
        metrics['avg_cyclomatic_complexity'] = (
            sum(b.complexity for b in blocks) / max(1, len(blocks))
        )
    except Exception:
        metrics['avg_cyclomatic_complexity'] = 0

    try:
        halstead = h_visit(code)
        metrics['halstead_volume'] = halstead.total.volume
        metrics['halstead_effort'] = halstead.total.effort
    except Exception:
        metrics['halstead_volume'] = 0
        metrics['halstead_effort'] = 0

    try:
        tree    = ast.parse(code)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        metrics['depth_of_inheritance_tree'] = max(
            [len(c.bases) for c in classes] + [0]
        )

        total_methods = sum(
            len([m for m in c.body if isinstance(m, ast.FunctionDef)])
            for c in classes
        )
        metrics['number_of_methods_per_class'] = total_methods / max(1, len(classes))
        metrics['weighted_methods_per_class']  = (
            metrics['number_of_methods_per_class']
            * metrics.get('avg_cyclomatic_complexity', 1)
        )

        _NESTING_TYPES = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)

        def _max_nesting(node, depth=0):
            child_depths = [
                _max_nesting(child, depth + (1 if isinstance(child, _NESTING_TYPES) else 0))
                for child in ast.iter_child_nodes(node)
            ]
            return max(child_depths) if child_depths else depth

        metrics['nesting_depth'] = _max_nesting(tree)

        defined_names = {c.name for c in classes} | {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        metrics['class_coupling'] = len(referenced_names - defined_names)

    except Exception:
        metrics['depth_of_inheritance_tree'] = 0
        metrics['number_of_methods_per_class'] = 0
        metrics['weighted_methods_per_class']  = 0
        metrics['nesting_depth']               = 0
        metrics['class_coupling']              = 0

    return metrics


def collect_textual_metrics(code):
    """
    Extract Category B metrics: Readability, Comments, Docstrings, Whitespace,
    code_duplication_pct.
    """
    if not code:
        return {}

    lines       = code.split('\n')
    total_lines = len(lines)
    if total_lines == 0:
        return {}

    empty_lines   = sum(1 for l in lines if l.strip() == '')
    comment_lines = sum(1 for l in lines if l.strip().startswith('#'))

    metrics = {
        'comment_density':    comment_lines / total_lines,
        'whitespace_ratio':   empty_lines   / total_lines,
        'docstring_presence': 1 if ('"""' in code or "'''" in code) else 0,
    }

    try:
        tree        = ast.parse(code)
        identifiers = set(
            [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
            + [
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
            ]
        )
        metrics['avg_identifier_length'] = (
            sum(len(name) for name in identifiers) / max(1, len(identifiers))
        )
    except Exception:
        metrics['avg_identifier_length'] = 0

    non_empty = [l.strip() for l in lines if l.strip()]
    if non_empty:
        counts     = Counter(non_empty)
        duplicated = sum(v - 1 for v in counts.values() if v > 1)
        metrics['code_duplication_pct'] = duplicated / len(non_empty)
    else:
        metrics['code_duplication_pct'] = 0.0

    return metrics


def collect_evolutionary_metrics_and_target(repo_path, file_rel_path, timeframe_months=3):
    """
    Extract Category C metrics and Target Ground Truth (Bug-Proneness Score).

    timeframe_months reduced from 6 -> 3 so shallow/recent clones have enough
    history to produce past_commits.

    Fallback: if no commits fall before the snapshot date (shallow clone or very
    new file), all available commits are used as past_commits so the file is not
    silently dropped from the dataset.

    Tree path lookup: tries direct key access first, then walks the tree segment
    by segment to handle nested paths that GitPython sometimes fails to resolve
    with a single bracket lookup.
    """
    try:
        repo = git.Repo(repo_path)
    except git.exc.InvalidGitRepositoryError:
        print(f"Not a valid git repository: {repo_path}")
        return {}, None

    commits_touching_file = list(repo.iter_commits(paths=file_rel_path))

    if not commits_touching_file:
        return {}, None

    now           = datetime.now(timezone.utc)
    snapshot_date = now - relativedelta(months=timeframe_months)

    past_commits   = []
    future_commits = []

    for c in commits_touching_file:
        commit_date = c.committed_datetime.astimezone(timezone.utc)
        if commit_date <= snapshot_date:
            past_commits.append(c)
        else:
            future_commits.append(c)

    # FIX: if no commits pre-date the snapshot (shallow clone or very new file),
    # fall back to treating all available commits as past so the file is kept.
    if not past_commits and commits_touching_file:
        past_commits   = commits_touching_file
        future_commits = []

    if not past_commits:
        return {}, None

    past_authors   = set()
    past_bug_fixes = 0
    total_added    = 0
    total_deleted  = 0
    bug_keywords   = ['fix', 'bug', 'patch', 'issue', 'resolve', 'error']

    for commit in past_commits:
        past_authors.add(commit.author.email)
        message = commit.message.lower()
        if any(keyword in message for keyword in bug_keywords):
            past_bug_fixes += 1
        try:
            stats          = commit.stats.total
            total_added   += stats.get('insertions', 0)
            total_deleted += stats.get('deletions',  0)
        except Exception:
            pass

    target_bug_fixes = 0
    for commit in future_commits:
        if any(kw in commit.message.lower() for kw in bug_keywords):
            target_bug_fixes += 1

    try:
        code_age_days = (
            snapshot_date
            - past_commits[-1].committed_datetime.astimezone(timezone.utc)
        ).days
    except IndexError:
        code_age_days = 0

    metrics = {
        'commit_frequency':    len(past_commits),
        'author_count':        len(past_authors),
        'bug_fix_ratio':       past_bug_fixes / max(1, len(past_commits)),
        'code_age_days':       code_age_days,
        'added_deleted_ratio': total_added / max(1, total_deleted),
        'target_bug_proneness': target_bug_fixes,
    }

    # Retrieve the file's code as it existed at the snapshot date.
    # FIX: try direct key lookup first, then walk path segments — GitPython
    # sometimes fails to resolve nested paths with a single bracket access.
    historical_code  = None
    last_past_commit = past_commits[0]

    try:
        blob            = last_past_commit.tree[file_rel_path]
        historical_code = blob.data_stream.read().decode('utf-8', errors='replace')
    except (KeyError, AttributeError):
        try:
            obj = last_past_commit.tree
            for part in file_rel_path.split('/'):
                obj = obj[part]
            historical_code = obj.data_stream.read().decode('utf-8', errors='replace')
        except Exception:
            historical_code = None

    return metrics, historical_code


def extract_all_metrics_for_file(repo_path, file_rel_path):
    """
    Combines all metrics for a single file into a single feature vector.
    """
    evolutionary_metrics, historical_code = collect_evolutionary_metrics_and_target(
        repo_path, file_rel_path
    )

    if historical_code is None:
        return {}

    structural = collect_structural_metrics(historical_code)
    textual    = collect_textual_metrics(historical_code)

    return {**structural, **textual, **evolutionary_metrics}


def build_dataset_from_repo(repo_path, output_csv_path):
    print(f"Scanning repository at {repo_path}...")
    dataset = []

    for root, _, files in os.walk(repo_path):
        if '.git' in root or '__pycache__' in root or 'venv' in root:
            continue

        for file in files:
            if file.endswith('.py'):
                abs_path      = os.path.join(root, file)
                file_rel_path = os.path.relpath(abs_path, repo_path).replace('\\', '/')

                features = extract_all_metrics_for_file(repo_path, file_rel_path)

                if features:
                    features['file_name'] = file_rel_path
                    dataset.append(features)

    df = pd.DataFrame(dataset)

    if df.empty or 'file_name' not in df.columns:
        # Count raw .py files so the error message is more informative
        py_count = sum(
            1 for root, _, files in os.walk(repo_path)
            if '.git' not in root and '__pycache__' not in root
            for f in files if f.endswith('.py')
        )
        print(
            f"[WARN] No valid Python files extracted from '{repo_path}'. "
            f"({py_count} .py files found by os.walk — all returned empty features.) "
            "Check that the repo has enough git history for the snapshot window."
        )
        raise ValueError(f"No Python files found under '{repo_path}'.")

    cols = ['file_name'] + [c for c in df.columns if c != 'file_name']
    df   = df[cols]

    df.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! Dataset generated with {len(df)} valid Python files.")
    print(f"Saved to: {output_csv_path}")
    return df


if __name__ == "__main__":
    repo_directory = "test_repos/flask"
    output_file    = "flask_dataset.csv"
    build_dataset_from_repo(repo_directory, output_file)