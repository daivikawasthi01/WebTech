"""
combine_datasets.py — Merge per-repo raw CSVs into one combined dataset.

Usage:
    python combine_datasets.py

    # Override input directory and output path:
    python combine_datasets.py --input-dir data --output data/combined_dataset.csv

By default, scans data/ for any CSV files matching *_dataset.csv (but not
*_dataset_clean.csv or combined_dataset.csv itself), adds a 'repo' column,
concatenates them, and writes the result.

Run this AFTER mining all target repos:
    python main.py --repo test_repos/flask    --raw-file data/flask_dataset.csv    --skip-ga
    python main.py --repo test_repos/requests --raw-file data/requests_dataset.csv --skip-ga
    python main.py --repo test_repos/click    --raw-file data/click_dataset.csv    --skip-ga
    python combine_datasets.py
    python main.py --raw-file data/combined_dataset.csv \\
                   --processed-file data/combined_dataset_clean.csv \\
                   --pop-size 20 --generations 30 --stagnation 8 --run-all
"""

import argparse
import glob
import os
import pandas as pd


def combine(input_dir: str = "data", output_path: str = "data/combined_dataset.csv"):
    pattern = os.path.join(input_dir, "*_dataset.csv")
    paths   = [
        p for p in glob.glob(pattern)
        if "clean" not in p and "combined" not in p
    ]

    if not paths:
        print(f"[ERROR] No *_dataset.csv files found in '{input_dir}'.")
        print("  Mine at least one repo first:")
        print("  python main.py --repo test_repos/flask --raw-file data/flask_dataset.csv")
        return None

    print(f"Found {len(paths)} dataset(s):")
    dfs = []
    for path in sorted(paths):
        df       = pd.read_csv(path)
        # derive repo name from filename: flask_dataset.csv → flask
        repo_tag = os.path.basename(path).replace("_dataset.csv", "")
        df.insert(1, "repo", repo_tag)
        print(f"  {repo_tag:15s} — {len(df):4d} files | "
              f"{(df['target_bug_proneness'] > 0).sum():3d} bug-prone "
              f"({(df['target_bug_proneness'] > 0).mean()*100:.1f}%)")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Summary stats
    total      = len(combined)
    bug_prone  = (combined['target_bug_proneness'] > 0).sum()
    pct        = bug_prone / total * 100
    mean_bugs  = combined['target_bug_proneness'].mean()

    print(f"\nCombined dataset:")
    print(f"  Total files  : {total}")
    print(f"  Bug-prone    : {bug_prone} ({pct:.1f}%)")
    print(f"  Mean bug count: {mean_bugs:.2f}")
    print(f"  Repos        : {combined['repo'].nunique()}")
    print(f"  Features     : {len(combined.columns) - 3}  "
          f"(excl. file_name, repo, target)")

    if pct < 15:
        print("\n  [WARN] Bug-prone rate < 15%. Consider increasing --timeframe-months "
              "when mining, or add more repositories.")
    if total < 300:
        print(f"\n  [WARN] Only {total} files. Aim for 400+ for reliable GA results. "
              "Add more repos.")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-repo datasets")
    parser.add_argument("--input-dir", default="data",
                        help="Directory containing *_dataset.csv files")
    parser.add_argument("--output",    default="data/combined_dataset.csv",
                        help="Output path for combined CSV")
    args = parser.parse_args()
    combine(args.input_dir, args.output)