"""Metric computation for supervised ranking evaluation.

Computes per-category and aggregated ranking metrics from a DataFrame
of OOF predictions with columns: product_code, category, true_score, predicted_score.

The target column may arrive as ``true_score`` (panel-mean rating, what a
rebuild from the gated bundle writes) or ``true_rank`` (within-category rank,
what the public release ships). Every entry point here normalises it — see
shared/oof_io.py — so both forms give bit-identical metrics.

Metrics:
    - Spearman rho (category-weighted)
    - Pairwise accuracy (category-weighted)
    - Recall@1 (macro-averaged across categories)
    - Recall@3 (macro-averaged across categories)
"""

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# Callers reach this module through several sys.path roots (food_similarity/,
# molecular/scripts/, a bare `python evaluation/bootstrap_fast.py`), so shared/
# cannot be assumed to be importable already.
_SHARED = Path(__file__).resolve().parents[2] / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from oof_io import TARGET, normalize_target  # noqa: E402


# The only columns any metric below reads. product_code is in the OOF schema and
# in these docstrings, but is never read numerically here or by the bootstrap.
#
# `true_score` is the CANONICAL spelling, not the only one on disk: the public
# release ships the same target as `true_rank` (see shared/oof_io.py). Every
# entry point below runs normalize_target() first, so these keys are always
# present by the time they are used. Sorting on either form gives the same
# order — the keys sort by category first and rank is monotone in score within a
# category — so the canonical order is identical in both trees, which is the
# whole point of this function.
CANONICAL_SORT_KEYS = ["category", TARGET, "predicted_score"]


def canonical_order(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pin row order so index-resampling bootstraps are reproducible.

    WHY THIS EXISTS. compute_bca_cis resamples row INDICES, so permuting the
    input rows moves the interval even though every point estimate is invariant.
    The release build re-keys product_code and re-sorts rows at a step AFTER the
    tables are rendered (the build script's product-code reminting step),
    so shipped CI caches were computed under a row order the shipped data no
    longer has. Sorting into one canonical order makes both trees agree.

    WHY NOT product_code. Sorting by the code would still be re-keyable: the
    build draws a fresh cryptographically random per-category permutation, so
    (category, product_code) orders the source tree differently from the released
    tree and the intervals would go on disagreeing — the defect would look fixed
    while staying live.

    WHY TIES ARE SAFE. Rows tied on all three keys are indistinguishable to
    every consumer: these are the only values read, so residual ordering among
    tied rows cannot move a metric. Verified empirically in
    tests/test_bootstrap_order_invariance.py.
    """
    return (
        normalize_target(results_df)
        .sort_values(CANONICAL_SORT_KEYS, kind="mergesort")
        .reset_index(drop=True)
    )


def _spearman_per_category(group: pd.DataFrame) -> float:
    """Spearman correlation for one category."""
    if len(group) < 2:
        return np.nan
    mask = ~(group["true_score"].isna() | group["predicted_score"].isna())
    g = group[mask]
    if len(g) < 2:
        return np.nan
    rho, _ = spearmanr(g["true_score"], g["predicted_score"])
    return rho


def _kendall_per_category(group: pd.DataFrame) -> float:
    """Kendall's tau-b correlation for one category (handles ties)."""
    if len(group) < 2:
        return np.nan
    mask = ~(group["true_score"].isna() | group["predicted_score"].isna())
    g = group[mask]
    if len(g) < 2:
        return np.nan
    tau, _ = kendalltau(g["true_score"], g["predicted_score"], variant="b")
    return tau


def _pairwise_accuracy_per_category(group: pd.DataFrame) -> tuple:
    """Pairwise accuracy for one category. Returns (correct, total)."""
    true = group["true_score"].values
    pred = group["predicted_score"].values
    n = len(true)
    correct = 0.0
    total = 0
    for i, j in combinations(range(n), 2):
        true_diff = true[i] - true[j]
        pred_diff = pred[i] - pred[j]
        if abs(true_diff) < 1e-10:
            correct += 0.5  # Tie in ground truth: half credit
            total += 1
        elif abs(pred_diff) < 1e-10:
            correct += 0.5  # Tie in prediction: half credit
            total += 1
        else:
            if true_diff * pred_diff > 0:
                correct += 1.0
            total += 1
    return correct, total


def _recall_at_k_per_category(group: pd.DataFrame, k: int) -> float:
    """Recall@k for one category: is the best product (highest true score)
    in the top-k predicted products?

    Handles ties in predictions using expected value under random tie-breaking.
    """
    if len(group) == 0:
        return np.nan

    true = group["true_score"].values
    pred = group["predicted_score"].values

    best_idx = np.argmax(true)
    best_pred = pred[best_idx]

    n_strictly_better = np.sum(pred > best_pred + 1e-10)
    if n_strictly_better >= k:
        return 0.0

    n_tied = np.sum(np.abs(pred - best_pred) < 1e-10)
    slots_available = k - n_strictly_better
    return min(slots_available, n_tied) / n_tied


def compute_per_category_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all metrics per category.

    Args:
        results_df: DataFrame with columns: product_code, category, true_score, predicted_score

    Returns:
        DataFrame with one row per category and columns for each metric.
    """
    # Accepts a frame carrying either target spelling; see shared/oof_io.py.
    results_df = normalize_target(results_df)
    rows = []
    for cat, group in results_df.groupby("category"):
        correct, total = _pairwise_accuracy_per_category(group)
        rows.append({
            "category": cat,
            "n_products": len(group),
            "spearman": _spearman_per_category(group),
            "kendall_tau": _kendall_per_category(group),
            "pairwise_accuracy": correct / total if total > 0 else np.nan,
            "correct_pairs": correct,
            "total_pairs": total,
            "recall_at_1": _recall_at_k_per_category(group, 1),
            "recall_at_2": _recall_at_k_per_category(group, 2),
            "recall_at_3": _recall_at_k_per_category(group, 3),
        })
    return pd.DataFrame(rows)


def compute_all_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregated metrics across all categories.

    Spearman and pairwise accuracy are category-weighted (by number of products/pairs).
    Recall@1 and recall@3 are macro-averaged (equal weight per category).

    Args:
        results_df: DataFrame with columns: product_code, category, true_score, predicted_score

    Returns:
        dict of metric_name -> value
    """
    per_cat = compute_per_category_metrics(results_df)

    # Category-weighted Spearman
    valid_sp = per_cat.dropna(subset=["spearman"])
    weighted_spearman = (
        (valid_sp["spearman"] * valid_sp["n_products"]).sum()
        / valid_sp["n_products"].sum()
        if len(valid_sp) > 0 else np.nan
    )

    # Category-weighted Kendall tau-b
    valid_kt = per_cat.dropna(subset=["kendall_tau"])
    weighted_kendall = (
        (valid_kt["kendall_tau"] * valid_kt["n_products"]).sum()
        / valid_kt["n_products"].sum()
        if len(valid_kt) > 0 else np.nan
    )

    # Category-weighted pairwise accuracy
    total_correct = per_cat["correct_pairs"].sum()
    total_pairs = per_cat["total_pairs"].sum()
    weighted_pairwise_acc = total_correct / total_pairs if total_pairs > 0 else np.nan

    # Macro-averaged recall
    recall_1 = per_cat["recall_at_1"].mean()
    recall_2 = per_cat["recall_at_2"].mean()
    recall_3 = per_cat["recall_at_3"].mean()

    return {
        "spearman": weighted_spearman,
        "kendall_tau": weighted_kendall,
        "pairwise_accuracy": weighted_pairwise_acc,
        "recall_at_1": recall_1,
        "recall_at_2": recall_2,
        "recall_at_3": recall_3,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a readable string."""
    lines = ["Metrics:"]
    for name, val in metrics.items():
        lines.append(f"  {name}: {val:.4f}")
    return "\n".join(lines)
