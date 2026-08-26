#!/usr/bin/env python3
"""Hierarchical (partial-pooling) estimation of per-category pairwise accuracy.

The companion to run_category_weighted.py: macro weighting treats every category
equally but leaves each category's estimate unshrunk, so a 5-product category
contributes its full sampling noise. This shrinks instead.

Method: empirical-Bayes / DerSimonian-Laird random-effects shrinkage. Each
category's observed pairwise accuracy p_c (sampling variance v_c) is shrunk toward
the grand mean by B_c = tau^2/(tau^2 + v_c), where tau^2 (between-category
variance) is estimated per model. Small, noisy categories shrink more.

Two choices here are deliberate and were changed after review; both move the number,
so do not "simplify" them back:

  1. The shrink target is the PRECISION-WEIGHTED random-effects grand mean, not the
     pair-count-weighted micro mean. Shrinking toward micro would reintroduce exactly
     the large-category dominance that this analysis exists to remove.
  2. The sampling variance uses n_products, NOT the C(n,2) pairs. Pairs that share a
     product are dependent, so a pair count would understate v_c and under-shrink.

The earlier micro-target / pair-count version gave .666 for the best model instead
of .682 — same qualitative conclusion, different headline, hence this note.

Reads ONLY results/oof_predictions/. No API, no gated data, no training. Runs in
seconds: there is no bootstrap here, the shrinkage is closed-form.

Output: results/category_balanced/hierarchical_estimation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FS_DIR))
sys.path.insert(0, str(FS_DIR.parent / "shared"))

from evaluation.metrics import compute_per_category_metrics  # noqa: E402
from oof_io import read_oof  # noqa: E402

OOF = FS_DIR / "results" / "oof_predictions"
OUT = FS_DIR / "results" / "category_balanced"

MODELS = [
    ("llm_gemini_3_1_pro_preview_ingredients_image.csv", "Gemini 3.1 Pro"),
    ("llm_qwen3_5_397b_a17b_ingredients_image.csv", "Qwen 3.5 397B"),
    ("dist_pred_cosine_NCI.csv", "MMRF (cosine)"),
    ("dist_pred_l2_NCI.csv", "MMRF (L2)"),
    ("ridge_SNCTI_bench.csv", "Ridge"),
    ("bradley_terry_SNCTI_bench.csv", "Bradley-Terry"),
    ("hierarchical_bt_SNCTI_bench.csv", "Hierarchical BT"),
    ("kernel_ranksvm_SNCTI_bench.csv", "Kernel RankSVM"),
    ("lightgbm_reg_SNCTI_bench.csv", "LightGBM"),
    ("nested_bt_gemini_nnls.csv", "NNLS (BT+Gemini)"),
    ("nested_bt_gemini_rank.csv", "Rank avg."),
    ("nested_bt_gemini_mean.csv", "Mean"),
]


def eb_shrink(pc):
    """DerSimonian-Laird random-effects shrinkage of per-category accuracy."""
    p = pc["pairwise_accuracy"].values.astype(float)
    n = pc["n_products"].values.astype(float)
    # Clip only for the variance term: a category at exactly 0 or 1 would get v_c = 0
    # and therefore infinite weight.
    p_g = np.clip(p, 0.5 / np.maximum(n, 1), 1 - 0.5 / np.maximum(n, 1))
    v = p_g * (1 - p_g) / n
    w = 1.0 / v
    mu_w = np.sum(w * p) / np.sum(w)
    Q = np.sum(w * (p - mu_w) ** 2)
    k = len(p)
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / denom) if denom > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    mu_re = np.sum(w_re * p) / np.sum(w_re)
    B = tau2 / (tau2 + v)
    p_shrunk = mu_re + B * (p - mu_re)
    return mu_re, float(np.mean(p_shrunk)), pd.Series(p_shrunk, index=pc.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, name in MODELS:
        pc = compute_per_category_metrics(read_oof(OOF / f)).set_index("category")
        micro = pc["correct_pairs"].sum() / pc["total_pairs"].sum()
        mu_re, hier, sh = eb_shrink(pc)
        naive_macro = pc["pairwise_accuracy"].mean()
        rows.append(dict(model=name, micro=round(micro, 3), naive_macro=round(naive_macro, 3),
                         hierarchical=round(hier, 3)))
    df = pd.DataFrame(rows).sort_values("hierarchical", ascending=False)
    df.to_csv(OUT / "hierarchical_estimation.csv", index=False)
    print(df.to_string(index=False))

    base = df.set_index("model")["micro"]
    order_micro = base.sort_values(ascending=False)
    rho = spearmanr(order_micro.values,
                    df.set_index("model")["hierarchical"].reindex(order_micro.index).values)[0]
    print(f"\nBest model (micro): {order_micro.index[0]}")
    print(f"Best model (hierarchical): {df.iloc[0]['model']}")
    print(f"Spearman(hierarchical ranking vs micro ranking): {rho:.3f}")
    print(f"\nwrote {(OUT / 'hierarchical_estimation.csv').relative_to(FS_DIR.parent)}")


if __name__ == "__main__":
    main()
