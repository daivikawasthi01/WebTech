"""
preprocess.py

Fixes applied vs original:
  - MinMaxScaler removed from this module entirely.
    Why: scaling before train/val split causes data leakage.
    Fix: ann_model.py fits the scaler exclusively on X_train after splitting.

  - Hardcoded column names removed.
    The original code used df['file_name'] and df['target_bug_proneness'],
    but data_collector.py writes 'file' and 'bug_fix_commits'.
    This caused a KeyError on every run.
    Fix: detect identifier and target columns by position (first / last),
    which is robust to any column naming convention.
"""

import pandas as pd
import numpy as np


def preprocess_dataset(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Clean the raw mined dataset and save it unscaled.
    Scaling is deferred to ann_model.train_and_evaluate_ann() so it is
    applied correctly after the train/val split.

    Steps:
      1. Drop rows with any NaN values.
      2. Clip extreme outliers (values beyond 99th percentile per feature).
      3. Remove highly correlated features (|r| > 0.95).
      4. Save the cleaned, unscaled CSV.
    """
    print(f"Loading raw dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Original shape : {df.shape}")

    # Detect identifier and target by position — not by hardcoded name.
    id_col     = df.columns[0]   # 'file' as written by data_collector.py
    target_col = df.columns[-1]  # 'bug_fix_commits' as written by data_collector.py

    identifiers = df[id_col].reset_index(drop=True)
    target      = df[target_col].reset_index(drop=True)
    features    = df.drop(columns=[id_col, target_col])

    # Step 1: drop NaNs
    full = pd.concat([identifiers, features, target], axis=1)
    full = full.dropna().reset_index(drop=True)
    print(f"  After NaN drop : {full.shape}")

    identifiers = full[id_col]
    target      = full[target_col]
    features    = full.drop(columns=[id_col, target_col])

    # Step 2: clip outliers per feature at the 99th percentile
    clipped = features.copy()
    for col in clipped.columns:
        cap = clipped[col].quantile(0.99)
        clipped[col] = clipped[col].clip(upper=cap)

    outliers_clipped = (features != clipped).sum().sum()
    print(f"  Outlier values clipped : {outliers_clipped}")

    # Step 3: remove highly correlated features (|r| > 0.95)
    corr_matrix = clipped.corr().abs()
    upper_tri   = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    if to_drop:
        print(f"  Dropping {len(to_drop)} highly correlated features (|r|>0.95): {to_drop}")
        clipped = clipped.drop(columns=to_drop)
    else:
        print("  No features dropped for correlation (all |r| <= 0.95)")

    # Step 4: reassemble and save — NO scaling
    processed_df = pd.concat([identifiers, clipped, target], axis=1).reset_index(drop=True)
    processed_df.to_csv(output_csv, index=False)
    print(f"  Saved cleaned (unscaled) dataset to: {output_csv}")
    print(f"  Final shape: {processed_df.shape}")

    return processed_df


if __name__ == "__main__":
    result = preprocess_dataset("data/flask_dataset.csv", "data/flask_dataset_clean.csv")
    print("\nSample (first 3 rows):")
    print(result.head(3))