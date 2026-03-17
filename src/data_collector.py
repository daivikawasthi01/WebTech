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

        _NESTING_TYPES = (ast.If, ast.For, ast.While, ast.With, ast.Try,
                          ast.ExceptHandler)

        def _max_nesting(node, depth=0):
            child_depths = [
                _max_nesting(child,
                             depth + (1 if isinstance(child, _NESTING_TYPES) else 0))
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
                if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                     ast.AsyncFunctionDef))
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


def collect_evolutionary_metrics_and_target(
        repo_path, file_rel_path, timeframe_months=12):
    """
    Extract Category C metrics and target ground truth (bug-proneness score).

    timeframe_months=12 (production default):
        12 months gives far more bug-fix commits in the target window and
        richer evolutionary metric history. The cloud deployment used 3 months
        to avoid timeouts on fresh shallow clones.

    No max_count cap on iter_commits:
        The max_count=100 limit was a cloud workaround. Locally we walk the
        full commit history for accurate commit_frequency, bug_fix_ratio, and
        author_count values.

    commit.stats.total intentionally absent:
        Runs git diff on every commit — prohibitively slow. added_deleted_ratio
        is not computed and is absent from constants.py.

    Fallback for empty past_commits:
        If no commits pre-date the snapshot, all available commits are used
        so the file is not silently dropped from the dataset.
    """
    try:
        repo = git.Repo(repo_path)
    except git.exc.InvalidGitRepositoryError:
        print(f"Not a valid git repository: {repo_path}")
        return {}, None

    # Full commit history — no cap (max_count=100 was a cloud workaround)
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

    # Fallback: if no commits pre-date the snapshot use all available commits
    if not past_commits and commits_touching_file:
        past_commits   = commits_touching_file
        future_commits = []

    if not past_commits:
        return {}, None

    past_authors   = set()
    past_bug_fixes = 0
    bug_keywords   = ['fix', 'bug', 'patch', 'issue', 'resolve', 'error']

    for commit in past_commits:
        past_authors.add(commit.author.email)
        if any(kw in commit.message.lower() for kw in bug_keywords):
            past_bug_fixes += 1

    target_bug_fixes = sum(
        1 for c in future_commits
        if any(kw in c.message.lower() for kw in bug_keywords)
    )

    try:
        code_age_days = (
            snapshot_date
            - past_commits[-1].committed_datetime.astimezone(timezone.utc)
        ).days
    except IndexError:
        code_age_days = 0

    metrics = {
        'commit_frequency':     len(past_commits),
        'author_count':         len(past_authors),
        'bug_fix_ratio':        past_bug_fixes / max(1, len(past_commits)),
        'code_age_days':        max(0, code_age_days),
        'target_bug_proneness': target_bug_fixes,
    }

    # Retrieve file content at snapshot date
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


def extract_all_metrics_for_file(repo_path, file_rel_path,
                                 timeframe_months=12):
    evolutionary_metrics, historical_code = collect_evolutionary_metrics_and_target(
        repo_path, file_rel_path, timeframe_months=timeframe_months
    )

    if historical_code is None:
        return {}

    structural = collect_structural_metrics(historical_code)
    textual    = collect_textual_metrics(historical_code)

    return {**structural, **textual, **evolutionary_metrics}


def build_dataset_from_repo(repo_path, output_csv_path,
                            timeframe_months=12):
    """
    Mine all Python files in repo_path and write a CSV to output_csv_path.

    timeframe_months (default 12):
        Production default. Controls snapshot lookback window.
        Cloud deployment used 3 months to avoid timeouts.
    """
    print(f"\nScanning repository: {repo_path}")
    print(f"Snapshot window: {timeframe_months} months back")
    dataset = []

    py_files = []
    for root, _, files in os.walk(repo_path):
        if '.git' in root or '__pycache__' in root or 'venv' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))

    print(f"Found {len(py_files)} Python files. Mining (this takes a while "
          f"on large repos)...")

    for i, abs_path in enumerate(py_files, 1):
        file_rel_path = os.path.relpath(abs_path, repo_path).replace('\\', '/')

        if i % 50 == 0 or i == len(py_files):
            print(f"  [{i:4d}/{len(py_files)}] {file_rel_path}")

        features = extract_all_metrics_for_file(
            repo_path, file_rel_path,
            timeframe_months=timeframe_months
        )

        if features:
            features['file_name'] = file_rel_path
            dataset.append(features)

    df = pd.DataFrame(dataset)

    if df.empty or 'file_name' not in df.columns:
        raise ValueError(
            f"No valid Python files extracted from '{repo_path}'. "
            f"({len(py_files)} .py files found — all returned empty features.) "
            "Check that the repo has enough commit history."
        )

    cols = ['file_name'] + [c for c in df.columns if c != 'file_name']
    df   = df[cols]

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    bug_prone = (df['target_bug_proneness'] > 0).sum()
    print(f"\nDataset saved: {len(df)} files | "
          f"{bug_prone} bug-prone ({bug_prone/len(df)*100:.1f}%) | "
          f"→ {output_csv_path}")
    return df


if __name__ == "__main__":
    build_dataset_from_repo(
        "test_repos/flask",
        "data/flask_dataset.csv",
        timeframe_months=12
    )