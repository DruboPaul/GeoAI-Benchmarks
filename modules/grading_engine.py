"""
Grading Engine Module — Drubo Live Benchmark Platform
======================================================
Self-contained scoring logic ported from grade.py with additional
convenience wrappers for the Streamlit frontend.

Scoring formula:
    score_t = clamp(1 − RMSE_agent / RMSE_baseline, 0, 1)
    final   = mean(score_t  for t in targets)

Authors: Drubo / Antigravity
"""

import numpy as np
import pandas as pd
from pathlib import Path

REQUIRED_TARGETS = [
    "stem_count",
    "basal_area",
    "carbon_storage",
    "leaf_area",
    "percent_tree_cover",
]

TARGET_UNITS = {
    "stem_count": "trees/plot",
    "basal_area": "m²/plot",
    "carbon_storage": "kg C/plot",
    "leaf_area": "m²/plot",
    "percent_tree_cover": "%",
}

TARGET_DESCRIPTIONS = {
    "stem_count": "Number of live tree stems (DBH ≥ 2.54 cm)",
    "basal_area": "Total basal area of trees",
    "carbon_storage": "Above-ground carbon stored",
    "leaf_area": "Total leaf area",
    "percent_tree_cover": "Canopy cover percentage",
}


# ===========================================================================
# Core functions
# ===========================================================================

def compute_rmse(pred, truth):
    """Root Mean Squared Error."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def compute_mae(pred, truth):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) -
                                np.asarray(truth, dtype=float))))


def compute_r2(pred, truth):
    """R² (coefficient of determination)."""
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_baseline_means(df_train, group_cols=None):
    """
    Compute stratum × land_use grouped means as the naïve baseline.
    Falls back to global mean if groups have no variance.
    """
    if group_cols is None:
        group_cols = ["stratum", "land_use"]

    # Ensure group columns exist
    available = [c for c in group_cols if c in df_train.columns]
    if not available:
        # Global mean fallback
        means = {}
        for t in REQUIRED_TARGETS:
            if t in df_train.columns:
                means[t] = df_train[t].mean()
        return pd.DataFrame([means])

    grouped = df_train.groupby(available)[REQUIRED_TARGETS].mean().reset_index()
    return grouped


def validate_submission_csv(df, n_expected_rows=None):
    """
    Validate a submission DataFrame.  Returns (is_valid, reason_str).
    """
    required_cols = ["plot_id"] + REQUIRED_TARGETS
    
    # Check columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"

    extra_cols = [c for c in df.columns if c not in required_cols]
    # Extra columns are allowed but noted

    # Check row count
    if n_expected_rows is not None and len(df) != n_expected_rows:
        return False, f"Row count mismatch: expected {n_expected_rows}, got {len(df)}"

    # Check for NaN, Inf, negative
    for t in REQUIRED_TARGETS:
        if df[t].isnull().any():
            return False, f"Column '{t}' contains NaN values."
        if np.isinf(df[t]).any():
            return False, f"Column '{t}' contains infinite values."
        if (df[t] < 0).any():
            return False, f"Column '{t}' contains negative values."
        if t == "percent_tree_cover" and (df[t] > 100).any():
            return False, f"Column '{t}' has values > 100%."

    return True, "Validation passed."


def validate_ground_truth_csv(df):
    """Validate a ground truth CSV."""
    required_cols = ["plot_id", "stratum", "land_use"] + REQUIRED_TARGETS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Ground truth missing columns: {missing}"
    if len(df) == 0:
        return False, "Ground truth is empty."
    return True, "Ground truth is valid."


# ===========================================================================
# Main grading function
# ===========================================================================

def grade_submission(df_pred, df_truth, df_baseline=None):
    """
    Grade a prediction DataFrame against ground truth.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Must have columns: plot_id + 5 targets
    df_truth : pd.DataFrame
        Must have columns: plot_id, stratum, land_use + 5 targets
    df_baseline : pd.DataFrame or None
        Stratum × land_use means.  Computed from truth if None.

    Returns
    -------
    dict  with keys:
        valid : bool
        mean_score : float
        target_breakdown : dict
        overall_rmse : float
        overall_mae : float
    """
    # Validate prediction
    is_valid, reason = validate_submission_csv(df_pred, n_expected_rows=len(df_truth))
    if not is_valid:
        return {
            "valid": False,
            "reason": reason,
            "mean_score": 0.0,
            "target_breakdown": {},
        }

    # Merge on plot_id
    merged = pd.merge(df_truth, df_pred, on="plot_id", suffixes=("_truth", "_pred"))
    if len(merged) != len(df_truth):
        return {
            "valid": False,
            "reason": "Plot IDs in submission do not match ground truth.",
            "mean_score": 0.0,
            "target_breakdown": {},
        }

    # Compute baseline predictions
    if df_baseline is not None and len(df_baseline) > 0:
        baseline_cols = [c for c in ["stratum", "land_use"] if c in df_baseline.columns]
        if baseline_cols:
            merged = pd.merge(merged, df_baseline, on=baseline_cols,
                              suffixes=("", "_base"), how="left")
        else:
            for t in REQUIRED_TARGETS:
                merged[f"{t}_base"] = df_truth[t].mean()
    else:
        # Compute from truth
        baseline_df = compute_baseline_means(df_truth)
        baseline_cols = [c for c in ["stratum", "land_use"] if c in baseline_df.columns]
        if baseline_cols:
            merged = pd.merge(merged, baseline_df, on=baseline_cols,
                              suffixes=("", "_base"), how="left")
        else:
            for t in REQUIRED_TARGETS:
                merged[f"{t}_base"] = df_truth[t].mean()

    target_scores = {}
    breakdown = {}

    for t in REQUIRED_TARGETS:
        y_true = merged[f"{t}_truth"].to_numpy(dtype=float)
        y_pred = merged[f"{t}_pred"].to_numpy(dtype=float)

        # Baseline column resolution
        base_col = f"{t}_base" if f"{t}_base" in merged.columns else t
        if base_col in merged.columns:
            y_base = merged[base_col].to_numpy(dtype=float)
        else:
            y_base = np.full_like(y_true, np.mean(y_true))

        # Fill NaN in baseline with global mean
        nan_mask = np.isnan(y_base)
        if nan_mask.any():
            y_base[nan_mask] = np.mean(y_true)

        rmse_agent = compute_rmse(y_pred, y_true)
        rmse_base = compute_rmse(y_base, y_true)
        mae_agent = compute_mae(y_pred, y_true)
        r2 = compute_r2(y_pred, y_true)

        if rmse_base <= 1e-8:
            score_t = 0.0
        else:
            score_t = float(np.clip(1.0 - rmse_agent / rmse_base, 0.0, 1.0))

        target_scores[t] = score_t
        breakdown[t] = {
            "rmse_agent": round(rmse_agent, 4),
            "rmse_baseline": round(rmse_base, 4),
            "mae_agent": round(mae_agent, 4),
            "r2": round(r2, 4),
            "score": round(score_t, 4),
            "unit": TARGET_UNITS.get(t, ""),
            "description": TARGET_DESCRIPTIONS.get(t, ""),
        }

    final_score = float(np.mean(list(target_scores.values())))

    return {
        "valid": True,
        "mean_score": round(final_score, 6),
        "target_breakdown": breakdown,
        "per_target_scores": {k: round(v, 4) for k, v in target_scores.items()},
    }


def grade_from_files(pred_path, truth_path, baseline_path=None):
    """
    Convenience wrapper to grade from file paths.
    """
    df_pred = pd.read_csv(pred_path)
    df_truth = pd.read_csv(truth_path)
    df_baseline = pd.read_csv(baseline_path) if baseline_path and Path(baseline_path).exists() else None
    return grade_submission(df_pred, df_truth, df_baseline)
