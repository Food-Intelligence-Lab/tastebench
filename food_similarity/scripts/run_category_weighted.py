#!/usr/bin/env python3
"""Category-balanced (macro) aggregation, small-category exclusion, and ranking stability.

Generates the appendix table for the category-weighted evaluation: every model's
micro (the paper's headline "Pw. Acc.") beside its macro (equal-weight) score, a
sweep that drops the smallest categories, and the Spearman correlation between
each scheme's model ranking and the paper's.

  * MICRO pairwise accuracy = sum(correct pairs) / sum(total pairs) -> the paper's
    headline; dominated by the large categories (Milk contributes 153 of 935 pairs).
  * MACRO pairwise accuracy = mean over categories of per-category pairwise accuracy
    -> every category weighted equally.

Reads ONLY results/oof_predictions/. No API, no gated data, no training.

Pairwise accuracy uses the SAME tie handling as the frozen metric
(evaluation/metrics.py): true tie -> 0.5, pred tie -> 0.5, else concordant -> 1.
main() asserts the vectorized implementation reproduces the frozen per-category
values exactly before it reports anything, so a drift in either implementation
fails here rather than silently changing the appendix.

Bootstrap resamples products WITHIN categories (seed 42), matching the frozen
resampling scheme; intervals are PERCENTILE, not BCa, so they are slightly
narrower than Table 1's BCa intervals. Point estimates are unaffected.

Every OOF frame is read through _read_oof, which pins row order first. The
bootstrap draws row POSITIONS, so permuting the input moves the interval even
though every point estimate is invariant — and the release build does permute it:
remint_product_codes.py re-keys product_code and re-sorts every OOF CSV by the new
public code, so the source tree and the released tree hand these scripts the same
rows in a different order. Sorting on (category, true_score, predicted_score) makes
both agree. Not on product_code: that column is re-keyed by a fresh random
per-category permutation, so ordering by it would keep the two trees disagreeing
while looking fixed.

Usage:
    python3 scripts/run_category_weighted.py            # with bootstrap CIs (~minutes)
    python3 scripts/run_category_weighted.py --no-ci    # point estimates only (seconds)

Outputs (results/category_balanced/):
    category_product_counts.csv   products per category, ascending
    micro_vs_macro.csv            per-model micro/macro + percentile CIs
    exclusion_sweep.csv           micro/macro under min-products thresholds 5..10
    ranking_by_scheme.csv         model rank (1=best) under each scheme
    ranking_correlation.csv       Spearman of each scheme's ranking vs micro_all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FS_DIR))
sys.path.insert(0, str(FS_DIR.parent / "shared"))

from evaluation.metrics import CANONICAL_SORT_KEYS, compute_per_category_metrics  # noqa: E402
from oof_io import read_oof  # noqa: E402

OOF_DIR = FS_DIR / "results" / "oof_predictions"
OUT = FS_DIR / "results" / "category_balanced"
N_BOOT = 5000
SEED = 42

MODELS = [
    ("llm_gemini_3_1_pro_preview_ingredients_image.csv", "Gemini 3.1 Pro", "Unsupervised"),
    ("llm_qwen3_5_397b_a17b_ingredients_image.csv", "Qwen 3.5 397B", "Unsupervised"),
    ("dist_pred_cosine_NCI.csv", "MMRF (cosine)", "Unsupervised"),
    ("dist_pred_l2_NCI.csv", "MMRF (L2)", "Unsupervised"),
    ("ridge_SNCTI_bench.csv", "Ridge", "Supervised-linear"),
    ("bradley_terry_SNCTI_bench.csv", "Bradley-Terry", "Supervised-pairwise"),
    ("hierarchical_bt_SNCTI_bench.csv", "Hierarchical BT", "Supervised-pairwise"),
    ("kernel_ranksvm_SNCTI_bench.csv", "Kernel RankSVM", "Supervised-pairwise"),
    ("lightgbm_reg_SNCTI_bench.csv", "LightGBM", "Supervised-nonlinear"),
    ("nested_bt_gemini_nnls.csv", "NNLS (BT+Gemini)", "Ensemble"),
    ("nested_bt_gemini_rank.csv", "Rank avg.", "Ensemble"),
    ("nested_bt_gemini_mean.csv", "Mean", "Ensemble"),
]


def _read_oof(name):
    """Read one OOF frame in a row order both the source and released trees share.

    Rows tied on all three keys are interchangeable here: those are the only values
    any metric below reads, so residual ordering among ties cannot move a number.
    """
    return (read_oof(OOF_DIR / name)
            .sort_values(CANONICAL_SORT_KEYS, kind="mergesort")
            .reset_index(drop=True))


def cat_correct_total(true, pred):
    """Vectorized (correct, total) for one category, matching metrics.py tie rules."""
    n = len(true)
    if n < 2:
        return 0.0, 0
    dt = true[:, None] - true[None, :]
    dp = pred[:, None] - pred[None, :]
    iu = np.triu_indices(n, k=1)
    dt, dp = dt[iu], dp[iu]
    true_tie = np.abs(dt) < 1e-10
    pred_tie = (~true_tie) & (np.abs(dp) < 1e-10)
    concordant = (~true_tie) & (~pred_tie) & (dt * dp > 0)
    correct = 0.5 * true_tie.sum() + 0.5 * pred_tie.sum() + 1.0 * concordant.sum()
    return float(correct), int(len(dt))


def micro_macro(df):
    """(micro, macro, per-category dict) from an OOF frame."""
    cc, tt, per = 0.0, 0, {}
    for cat, g in df.groupby("category"):
        c, t = cat_correct_total(g["true_score"].values, g["predicted_score"].values)
        per[cat] = c / t if t else np.nan
        cc += c
        tt += t
    micro = cc / tt
    macro = float(np.nanmean(list(per.values())))
    return micro, macro, per


def boot(df, min_products=None):
    if min_products is not None:
        keep = df.groupby("category")["product_code"].transform("size") >= min_products
        df = df[keep]
    groups = [g[["true_score", "predicted_score"]].values for _, g in df.groupby("category")]
    rng = np.random.default_rng(SEED)
    micros, macros = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        cc, tt, accs = 0.0, 0, []
        for arr in groups:
            n = len(arr)
            idx = rng.integers(0, n, n)
            s = arr[idx]
            c, t = cat_correct_total(s[:, 0], s[:, 1])
            cc += c
            tt += t
            accs.append(c / t if t else np.nan)
        micros[b] = cc / tt
        macros[b] = np.nanmean(accs)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return ci(micros), ci(macros)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ci", action="store_true",
                    help="skip the 5,000-replicate percentile bootstrap; point estimates "
                         "and the ranking correlations are unaffected by it")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ref = _read_oof(MODELS[0][0])
    counts = ref.groupby("category").size().sort_values()
    counts.to_csv(OUT / "category_product_counts.csv", header=["n_products"])
    print("Product counts min..max:", counts.min(), "..", counts.max(),
          "| <5:", int((counts < 5).sum()), "| <8:", int((counts < 8).sum()))

    # The vectorized pairwise accuracy must agree with the frozen metric exactly,
    # or the appendix and the main tables would be reporting different quantities.
    g = _read_oof(MODELS[0][0])
    frozen = compute_per_category_metrics(g)
    frozen_micro = frozen["correct_pairs"].sum() / frozen["total_pairs"].sum()
    my_micro, _, _ = micro_macro(g)
    assert abs(frozen_micro - my_micro) < 1e-9, (frozen_micro, my_micro)
    print(f"[verify] vectorized micro == frozen metric ({my_micro:.6f}) OK")

    rows = []
    for oof, name, group in MODELS:
        df = _read_oof(oof)
        micro, macro, _ = micro_macro(df)
        if args.no_ci:
            (mi_lo, mi_hi), (ma_lo, ma_hi) = (np.nan, np.nan), (np.nan, np.nan)
        else:
            (mi_lo, mi_hi), (ma_lo, ma_hi) = boot(df)
        rows.append(dict(model=name, group=group, micro=micro, micro_lo=mi_lo,
                         micro_hi=mi_hi, macro=macro, macro_lo=ma_lo, macro_hi=ma_hi,
                         delta=macro - micro))
        print(f"{name:22s} micro={micro:.3f}[{mi_lo:.3f},{mi_hi:.3f}] "
              f"macro={macro:.3f}[{ma_lo:.3f},{ma_hi:.3f}] delta={macro-micro:+.3f}")
    main_df = pd.DataFrame(rows)
    if args.no_ci:
        # Never overwrite shipped intervals with NaN: --no-ci is a fast check, not a
        # regeneration. The manifest hash in results/input_cache_sha256.txt covers
        # this file, so a NaN write would show up as cache drift much later.
        print("  --no-ci: not writing micro_vs_macro.csv (its CI columns would be NaN)")
    else:
        main_df.to_csv(OUT / "micro_vs_macro.csv", index=False)

    sweep = []
    for thr in [5, 6, 7, 8, 9, 10]:
        kept = int((counts >= thr).sum())
        for oof, name, group in MODELS:
            df = _read_oof(oof)
            keep = df.groupby("category")["product_code"].transform("size") >= thr
            micro, macro, _ = micro_macro(df[keep])
            sweep.append(dict(min_products=thr, kept_categories=kept, model=name,
                              micro=micro, macro=macro))
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(OUT / "exclusion_sweep.csv", index=False)

    schemes = {"micro_all": main_df.set_index("model")["micro"],
               "macro_all": main_df.set_index("model")["macro"]}
    for thr in [8, 10]:
        s = sweep_df[sweep_df.min_products == thr].set_index("model")
        schemes[f"micro_ge{thr}"] = s["micro"]
        schemes[f"macro_ge{thr}"] = s["macro"]
    order = main_df.set_index("model")["micro"].sort_values(ascending=False).index
    rank_tbl = pd.DataFrame({k: v.rank(ascending=False, method="min")
                             for k, v in schemes.items()}).loc[order]
    rank_tbl.to_csv(OUT / "ranking_by_scheme.csv")

    base = schemes["micro_all"]
    corr = {k: spearmanr(base.values, v.reindex(base.index).values)[0]
            for k, v in schemes.items()}
    pd.Series(corr, name="spearman_vs_micro_all").to_csv(OUT / "ranking_correlation.csv")

    print("\nRanking by scheme (1=best):")
    print(rank_tbl.to_string())
    print("\nSpearman of ranking vs paper micro_all:")
    for k, v in corr.items():
        print(f"  {k:12s} {v:.3f}")
    print("\nTop model per scheme:", {k: v.idxmax() for k, v in schemes.items()})
    print("Best (macro_all):", schemes["macro_all"].idxmax(),
          "| Best (macro_ge8):", schemes["macro_ge8"].idxmax())
    print(f"\nwrote {OUT.relative_to(FS_DIR.parent)}/")


if __name__ == "__main__":
    main()
