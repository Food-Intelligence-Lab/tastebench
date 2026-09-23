"""Guard the category-balanced appendix numbers against their generators.

The appendix quotes four figures for the best model (NNLS BT+Gemini): micro .683,
macro .641, ranking Spearman .881 vs the paper's micro ordering, and empirical-Bayes
.682. Those are prose in a .tex file, so nothing stops them drifting away from
results/category_balanced/ once someone edits a generator. This asserts them.

Run the two generators first — this reads their output, it does not recompute:

    python3 scripts/run_category_weighted.py --no-ci
    python3 scripts/hierarchical_estimation.py
    python3 scripts/check_category_balanced.py

Exit 0 = every published number matches. Exit 1 = a number moved, with the
expected and observed values printed. Exit 2 = the outputs are not there at all.
"""
from pathlib import Path

import pandas as pd

FS_DIR = Path(__file__).resolve().parent.parent
CB = FS_DIR / "results" / "category_balanced"
BEST = "NNLS (BT+Gemini)"

# (label, expected, decimals). Decimals match the precision the appendix quotes at,
# so this fails on a real change and not on a last-bit floating-point difference.
EXPECT = [
    ("best-model micro (paper headline)", 0.6829, 4),
    ("best-model macro (category-balanced)", 0.6409, 4),
    ("Spearman(macro_all ranking, micro_all ranking)", 0.8811, 4),
    ("empirical-Bayes aggregate, best model", 0.682, 3),
]


def main() -> int:
    need = ["exclusion_sweep.csv", "ranking_correlation.csv", "ranking_by_scheme.csv",
            "hierarchical_estimation.csv", "category_product_counts.csv"]
    missing = [f for f in need if not (CB / f).exists()]
    if missing:
        print(f"  ERROR: {CB} is missing {', '.join(missing)}")
        print("  Run scripts/run_category_weighted.py and scripts/hierarchical_estimation.py first.")
        return 2

    sweep = pd.read_csv(CB / "exclusion_sweep.csv")
    # min_products=5 keeps every category (the smallest has exactly 5 products), so this
    # row IS the all-category micro/macro pair. Reading it here instead of
    # micro_vs_macro.csv lets the fast --no-ci path verify the same numbers.
    all_cat = sweep[(sweep.min_products == 5) & (sweep.model == BEST)]
    if len(all_cat) != 1:
        print(f"  ERROR: expected exactly one min_products=5 row for {BEST}, got {len(all_cat)}")
        return 2
    all_cat = all_cat.iloc[0]

    corr = pd.read_csv(CB / "ranking_correlation.csv", index_col=0)["spearman_vs_micro_all"]
    hier = pd.read_csv(CB / "hierarchical_estimation.csv").set_index("model")["hierarchical"]

    got = [float(all_cat.micro), float(all_cat.macro),
           float(corr.loc["macro_all"]), float(hier.loc[BEST])]

    failures = []
    for (label, want, dp), obs in zip(EXPECT, got):
        ok = round(obs, dp) == round(want, dp)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {obs:.6f} (expected {want})")
        if not ok:
            failures.append((label, want, obs))

    counts = pd.read_csv(CB / "category_product_counts.csv")
    if len(counts) != 24 or int(counts.n_products.min()) != 5:
        failures.append(("24 categories, smallest n=5",
                         "24 / 5", f"{len(counts)} / {counts.n_products.min()}"))
        print(f"  FAIL category shape: {len(counts)} categories, min n={counts.n_products.min()}")
    else:
        print("  ok   24 categories, smallest has 5 products "
              "(so R1's <5 threshold excludes none)")

    ranks = pd.read_csv(CB / "ranking_by_scheme.csv").set_index("model")
    off = [c for c in ranks.columns if ranks.loc[BEST, c] != 1.0]
    if off:
        failures.append(("best model is #1 under every scheme", "rank 1 everywhere",
                         f"not #1 under {off}"))
        print(f"  FAIL best model is not #1 under: {off}")
    else:
        print(f"  ok   {BEST} ranks #1 under all {len(ranks.columns)} schemes")

    if failures:
        print(f"\n  ERROR: {len(failures)} published number(s) no longer match the generators.")
        print("  Either the generators changed or the appendix is stale — reconcile before shipping.")
        return 1
    print("\n  category-balanced appendix numbers verified against "
          "results/category_balanced/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
