import os
import ast
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
import git
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

def collect_structural_metrics(code):
    """
    Extract Category A metrics: Cyclomatic Complexity, Halstead, DIT, WMC, etc.
    """
    if not code:
        return {}

    metrics = {}
    
    # Cyclomatic Complexity
    try:
        blocks = cc_visit(code)
        metrics['avg_cyclomatic_complexity'] = sum([b.complexity for b in blocks]) / max(1, len(blocks))
    except:
        metrics['avg_cyclomatic_complexity'] = 0

    # Halstead Metrics
    try:
        halstead = h_visit(code)
        metrics['halstead_volume'] = halstead.total.volume
        metrics['halstead_effort'] = halstead.total.effort
    except:
        metrics['halstead_volume'] = 0
        metrics['halstead_effort'] = 0

    # AST-based OOP metrics
    try:
        tree = ast.parse(code)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Depth of Inheritance Tree (shallow estimation via max bases)
        metrics['depth_of_inheritance_tree'] = max([len(c.bases) for c in classes] + [0])
        
        # Number of Methods and Weighted Methods per Class
        total_methods = sum([len([m for m in c.body if isinstance(m, ast.FunctionDef)]) for c in classes])
        metrics['number_of_methods_per_class'] = total_methods / max(1, len(classes))
        
        # Approximate Weighted Methods by using average complexity of the block
        metrics['weighted_methods_per_class'] = metrics['number_of_methods_per_class'] * metrics.get('avg_cyclomatic_complexity', 1)
        
    except Exception:
        metrics['depth_of_inheritance_tree'] = 0
        metrics['number_of_methods_per_class'] = 0
        metrics['weighted_methods_per_class'] = 0

    return metrics


def collect_textual_metrics(code):
    """
    Extract Category B metrics: Readability, Comments, Docstrings, Whitespace.
    """
    if not code:
        return {}
        
    lines = code.split('\n')
    total_lines = len(lines)
    if total_lines == 0:
        return {}

    empty_lines = len([l for l in lines if l.strip() == ''])
    comment_lines = len([l for l in lines if l.strip().startswith('#')])
    
    metrics = {
        'comment_density': comment_lines / total_lines,
        'whitespace_ratio': empty_lines / total_lines,
        'docstring_presence': 1 if '"""' in code or "'''" in code else 0,
    }
    
    # Calculate average identifier length using AST
    try:
        tree = ast.parse(code)
        identifiers = set([node.id for node in ast.walk(tree) if isinstance(node, ast.Name)] + \
                      [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))])
        metrics['avg_identifier_length'] = sum(len(name) for name in identifiers) / max(1, len(identifiers))
    except Exception:
        metrics['avg_identifier_length'] = 0
        
    return metrics


def collect_evolutionary_metrics_and_target(repo_path, file_rel_path, timeframe_months=6):
    """
    Extract Category C metrics and Target Ground Truth (Bug-Proneness Score)
    """
    try:
        repo = git.Repo(repo_path)
    except git.exc.InvalidGitRepositoryError:
        print(f"Not a valid git repository: {repo_path}")
        return {}, None

    commits_touching_file = list(repo.iter_commits(paths=file_rel_path))
    
    if not commits_touching_file:
        return {}, None

    now = datetime.now(timezone.utc)
    snapshot_date = now - relativedelta(months=timeframe_months)
    
    past_commits = []
    future_commits = []
    
    for c in commits_touching_file:
        commit_date = c.committed_datetime.astimezone(timezone.utc)
        if commit_date <= snapshot_date:
            past_commits.append(c)
        else:
            future_commits.append(c)

    past_authors = set()
    past_bug_fixes = 0
    bug_keywords = ['fix', 'bug', 'patch', 'issue', 'resolve', 'error']

    for commit in past_commits:
        past_authors.add(commit.author.email)
        message = commit.message.lower()
        if any(keyword in message for keyword in bug_keywords):
            past_bug_fixes += 1

    target_bug_fixes = 0
    for commit in future_commits:
        message = commit.message.lower()
        if any(keyword in message for keyword in bug_keywords):
            target_bug_fixes += 1

    try:
        code_age_days = (snapshot_date - past_commits[-1].committed_datetime.astimezone(timezone.utc)).days
    except IndexError:
        code_age_days = 0

    # Evolution Metrics evaluated strictly in the past timeframe
    metrics = {
        'commit_frequency': len(past_commits),
        'author_count': len(past_authors),
        'bug_fix_ratio': past_bug_fixes / max(1, len(past_commits)),
        'code_age_days': code_age_days
    }
    
    # Ground Truth: Bug Fix Count (Pain Score) that happened AFTER the snapshot
    metrics['target_bug_proneness'] = target_bug_fixes 
    
    # Retrieve the file's code exactly as it existed at the snapshot date
    historical_code = None
    if past_commits:
        last_past_commit = past_commits[0] # The most recent commit before the snapshot
        tree = last_past_commit.tree
        try:
            # Read the file content as it existed X months ago
            blob = tree[file_rel_path]
            historical_code = blob.data_stream.read().decode('utf-8')
        except KeyError:
            # The file may have been created or moved after the snapshot, or the path is incorrect
            pass
            
    return metrics, historical_code


def extract_all_metrics_for_file(repo_path, file_rel_path):
    """
    Combines all metrics for a single file into a single feature vector, mapped to its ground truth.
    """
    evolutionary_metrics, historical_code = collect_evolutionary_metrics_and_target(repo_path, file_rel_path)
    
    if historical_code is None:
        return {} # File didn't exist or couldn't be parsed at snapshot
        
    structural = collect_structural_metrics(historical_code)
    textual = collect_textual_metrics(historical_code)
    
    return {**structural, **textual, **evolutionary_metrics}
    
def build_dataset_from_repo(repo_path, output_csv_path):
    print(f"Scanning repository at {repo_path}...")
    dataset = []
    
    # Walk through all files in the repository
    for root, _, files in os.walk(repo_path):
        # Skip hidden directories like .git or environment folders
        if '.git' in root or '__pycache__' in root or 'venv' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                abs_path = os.path.join(root, file)
                # Git expects forward slashes for paths
                file_rel_path = os.path.relpath(abs_path, repo_path).replace('\\', '/')
                
                features = extract_all_metrics_for_file(repo_path, file_rel_path)
                
                # If features were extracted (meaning the file existed 6 months ago)
                if features: 
                    features['file_name'] = file_rel_path
                    dataset.append(features)
                    
    # Convert list of dictionaries to a pandas DataFrame and save
    df = pd.DataFrame(dataset)
    # Reorder columns to put file_name first
    cols = ['file_name'] + [c for c in df.columns if c != 'file_name']
    df = df[cols]
    
    df.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! Dataset generated with {len(df)} valid Python files.")
    print(f"Saved to: {output_csv_path}")
    return df

if __name__ == "__main__":
    # Target the entire Flask repository
    repo_directory = "test_repos/flask"
    output_file = "flask_dataset.csv"
    
    build_dataset_from_repo(repo_directory, output_file)
