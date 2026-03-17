"""
preprocess.py

Fixes applied vs original:
- MinMaxScaler removed from this module entirely.
  Why: Scaling the full dataset before any train/val split causes data leakage
  — the scaler learns the min/max of the validation set and encodes that
  information into the training features. ann_model.py now fits the scaler
  exclusively on X_train after the split.

- Non-numeric column handling added.
  Why: combine_datasets.py adds a 'repo' string column. The original code only
  dropped 'file_name' and 'target_bug_proneness', leaving any other string
  columns in the feature matrix. When outlier clipping called .quantile() on a
  string column it returned a string, and clip(upper=string) raised
  "unsupported operand type(s) for -: 'str' and 'str'".
  Fix: after dropping known metadata columns, any remaining non-numeric columns
  are detected and set aside. They are re-attached at the end but never clipped
  or used in the correlation filter.
"""

import pandas as pd
import numpy as np


# Columns that are metadata / identifiers — never treated as features
_META_COLS = {'file_name', 'repo'}


def preprocess_dataset(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Clean the raw mined dataset and save it unscaled.

    Steps:
    1. Drop rows with any NaN values.
    2. Clip extreme outliers (99th percentile per numeric feature).
    3. Remove highly correlated features (|r| > 0.95).
    4. Save cleaned, unscaled CSV.

    Scaling is deferred to ann_model.train_and_evaluate_ann() so it is
    applied correctly after the train/val split.
    """
    print(f"Loading raw dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Original shape         : {df.shape}")

    # ── Separate known metadata, target, and feature columns ─────────────────
    target_col  = 'target_bug_proneness'
    id_col      = 'file_name'

    # Collect all metadata columns that exist in this CSV
    meta_cols_present = [c for c in _META_COLS if c in df.columns]

    identifiers = df[id_col].reset_index(drop=True)
    target      = df[target_col].reset_index(drop=True)

    # Drop ALL metadata + target — whatever remains are candidate features
    drop_cols = meta_cols_present + [target_col]
    features  = df.drop(columns=drop_cols)

    # Separate numeric vs non-numeric feature columns
    # Non-numeric columns (e.g. a stray string column) are preserved but
    # never clipped or filtered.
    numeric_cols     = features.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric_cols:
        print(f"  Non-numeric columns (excluded from processing): {non_numeric_cols}")

    numeric_features     = features[numeric_cols].copy()
    non_numeric_features = features[non_numeric_cols].copy() if non_numeric_cols else None

    # ── Step 1: Drop NaN rows ─────────────────────────────────────────────────
    full = pd.concat([identifiers, numeric_features, target], axis=1)
    full = full.dropna().reset_index(drop=True)
    print(f"  After NaN drop         : {full.shape}")

    identifiers      = full[id_col]
    target           = full[target_col]
    numeric_features = full.drop(columns=[id_col, target_col])

    if non_numeric_features is not None:
        non_numeric_features = non_numeric_features.loc[full.index].reset_index(drop=True)

    # ── Step 2: Clip outliers per numeric feature at 99th percentile ──────────
    clipped = numeric_features.copy()
    for col in clipped.columns:
        cap         = clipped[col].quantile(0.99)
        clipped[col] = clipped[col].clip(upper=cap)

    outliers_clipped = (numeric_features != clipped).sum().sum()
    print(f"  Outlier values clipped : {outliers_clipped}")

    # ── Step 3: Remove highly correlated numeric features (|r| > 0.95) ────────
    corr_matrix = clipped.corr().abs()
    upper_tri   = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

    if to_drop:
        print(f"  Dropping {len(to_drop)} highly correlated features (|r|>0.95):")
        for col in to_drop:
            # Show which column it correlates with
            correlated_with = upper_tri[col][upper_tri[col] > 0.95].index.tolist()
            print(f"    - {col}  (r>{0.95:.2f} with {correlated_with})")
        clipped = clipped.drop(columns=to_drop)
    else:
        print("  No features dropped for correlation (all |r| ≤ 0.95)")

    print(f"  Features remaining     : {len(clipped.columns)}")

    # ── Step 4: Reassemble and save ───────────────────────────────────────────
    parts = [identifiers.reset_index(drop=True)]
    # Re-attach non-numeric metadata (e.g. repo) after file_name
    if non_numeric_features is not None:
        parts.append(non_numeric_features.reset_index(drop=True))
    parts.append(clipped.reset_index(drop=True))
    parts.append(target.reset_index(drop=True))

    processed_df = pd.concat(parts, axis=1)
    processed_df.to_csv(output_csv, index=False)

    bug_prone = (processed_df[target_col] > 0).sum()
    pct       = bug_prone / len(processed_df) * 100
    print(f"  Bug-prone files        : {bug_prone} / {len(processed_df)} ({pct:.1f}%)")
    print(f"  Final shape            : {processed_df.shape}")
    print(f"  Saved to               : {output_csv}")
    return processed_df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = preprocess_dataset(
        "data/combined_dataset.csv",
        "data/combined_dataset_clean.csv"
    )
    print("\nSample (first 3 rows):")
    print(result.head(3))