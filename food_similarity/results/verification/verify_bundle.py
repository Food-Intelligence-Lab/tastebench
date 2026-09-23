"""Recompute every published number from the bundle alone.

This script must run with NO gated NECTAR data, NO API key, and no access to
anything outside this directory (plus the paper's own metric helper, which is
committed and deterministic). If it passes on a clean checkout that lacks
data/consolidated_datasets/, the bundle is sufficient.

Each check prints the recomputed value, the value published in the rebuttal
responses, and PASS/FAIL against a stated tolerance.

Usage:
    python verify_bundle.py            # exits non-zero on any FAIL
"""

import argparse
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# This script runs in two layouts: inside the rebuttal bundle (data/<group>/) and
# inside the release tree, where the derived files sit beside it and the OOF
# variants live with the other OOFs. Resolve by filename across both.
def _find_tree():
    """Locate the code tree, if we are running inside one."""
    for d in [HERE, *HERE.parents]:
        if (d / "food_similarity").is_dir() and (d / "human_baseline").is_dir():
            return d
    return None


_TREE = _find_tree()

SEARCH_DIRS = [
    HERE / "data" / "human_agreement",                  # bundle layout
    HERE / "data" / "recognition",
    HERE / "data" / "oof",
    HERE,                                               # release: results/verification/
    HERE.parent / "oof_predictions" / "ablations",       # release: OOF variants
]
if _TREE is not None:                                    # release: cross-area files
    SEARCH_DIRS += [
        _TREE / "human_baseline" / "results" / "verification",
        _TREE / "food_similarity" / "results" / "verification",
        _TREE / "food_similarity" / "results" / "oof_predictions" / "ablations",
    ]


class _Resolver:
    """DATA / "group/name.csv" -> the first existing candidate, by basename."""

    def __truediv__(self, rel):
        name = Path(rel).name
        for d in SEARCH_DIRS:
            p = d / name
            if p.exists():
                return p
        return Path(name)          # non-existent; callers check .exists()


DATA = _Resolver()

# The OOF target column is `true_score` in this bundle and `true_rank` in the
# public release (the panel means are withheld; only their within-category order
# ships). shared/oof_io.py is the one place that knows both spellings, and it is
# importable only when this file sits inside a code tree — running from the
# standalone bundle there is nothing but `true_score` to read, so plain read_csv
# is exactly right there.
if _TREE is not None and (_TREE / "shared" / "oof_io.py").exists():
    sys.path.insert(0, str(_TREE / "shared"))
    from oof_io import read_oof
else:
    read_oof = pd.read_csv

RESULTS = []


def check(name, got, want, tol, note=""):
    ok = got is not None and abs(got - want) <= tol
    RESULTS.append((name, got, want, ok))
    g = "  n/a " if got is None else f"{got:.4f}"
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<46s} got {g}   published {want:.4f}"
          + (f"   ({note})" if note else ""))
    return ok


# ---------------------------------------------------------------------------
# r2: Krippendorff's alpha = .077, from the coincidence matrices
# ---------------------------------------------------------------------------

def alpha_from_coincidence(df_block):
    vals = sorted(set(df_block.value_a) | set(df_block.value_b))
    o = {(a, b): 0.0 for a in vals for b in vals}
    for r in df_block.itertuples():
        o[(r.value_a, r.value_b)] = r.coincidence
    n_c = {c: sum(o[(c, k)] for k in vals) for c in vals}
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")

    def delta2(c, k):
        lo, hi = (c, k) if c <= k else (k, c)
        s = sum(n_c[g] for g in vals if lo <= g <= hi)
        return (s - (n_c[c] + n_c[k]) / 2.0) ** 2

    D_o = sum(o[(c, k)] * delta2(c, k) for c in vals for k in vals) / n
    D_e = sum(n_c[c] * n_c[k] * delta2(c, k) for c in vals for k in vals) / (n * (n - 1))
    return 1.0 - D_o / D_e if D_e else float("nan")


def verify_alpha():
    print("\nr2 — inter-rater agreement (Krippendorff's alpha)")
    f = DATA / "alpha_coincidence.csv"
    if not f.exists():
        return check("alpha (mean over blocks)", None, 0.077, 0.002)
    d = pd.read_csv(f)
    alphas = [alpha_from_coincidence(g) for _, g in d.groupby(["category", "block_id"])]
    alphas = [a for a in alphas if np.isfinite(a)]
    print(f"         {len(alphas)} blocks reconstructed from the coincidence matrices")
    return check("alpha (mean over blocks)", float(np.mean(alphas)), 0.077, 0.002)


# ---------------------------------------------------------------------------
# r2: rater-vs-rater pairwise agreement = .547, from the sign counts
# ---------------------------------------------------------------------------

def verify_rater_vs_rater():
    print("\nr2 — one panelist vs another panelist")
    f = DATA / "rater_pair_signs.csv"
    if not f.exists():
        return check("rater-vs-rater pairwise agreement", None, 0.547, 0.002)
    d = pd.read_csv(f)
    credit = total = 0.0
    for r in d.itertuples():
        n = r.n_pos + r.n_neg + r.n_tie
        tot = comb(n, 2)
        if tot == 0:
            continue
        same = comb(r.n_pos, 2) + comb(r.n_neg, 2)
        opp = r.n_pos * r.n_neg
        tie = tot - same - opp
        credit += same + 0.5 * tie
        total += tot
    return check("rater-vs-rater pairwise agreement", credit / total, 0.547, 0.002,
                 f"{int(total):,} rater-pair comparisons")


# ---------------------------------------------------------------------------
# r1: block allocation — 25 units, 1 significant after BH-FDR, median range 0.24
# ---------------------------------------------------------------------------

def verify_block_variance():
    print("\nr1 — block allocation does not bias the targets")
    f = DATA / "block_group_stats.csv"
    if not f.exists():
        check("multi-block units", None, 25, 0)
        return False
    d = pd.read_csv(f)
    units = d.unit.nunique()
    check("multi-block units", float(units), 25.0, 0.0)

    # one-way ANOVA from group (n, mean, sd) — closed form, exact
    pvals, ranges = [], []
    from scipy import stats
    for _, g in d.groupby("unit"):
        k = len(g)
        n = g.n.values.astype(float)
        # means are centered on each unit's grand mean; F and the range are
        # shift-invariant, so the centered values give identical results
        m = g["mean_centered"].values
        s = g["sd"].values
        N = n.sum()
        grand = (n * m).sum() / N
        ss_between = (n * (m - grand) ** 2).sum()
        ss_within = ((n - 1) * s ** 2).sum()
        df_b, df_w = k - 1, N - k
        if df_w <= 0 or ss_within <= 0:
            continue
        F = (ss_between / df_b) / (ss_within / df_w)
        pvals.append(float(stats.f.sf(F, df_b, df_w)))
        ranges.append(float(m.max() - m.min()))

    p = np.array(sorted(pvals))
    mm = len(p)
    bh = np.array([pv <= (i + 1) / mm * 0.05 for i, pv in enumerate(p)])
    nsig = (np.where(bh)[0].max() + 1) if bh.any() else 0
    ok1 = check("raw p<0.05 count", float((p < 0.05).sum()), 5.0, 0.0)
    ok2 = check("significant after BH-FDR", float(nsig), 1.0, 0.0)
    ok3 = check("median between-block mean difference", float(np.median(ranges)), 0.24, 0.005)
    return ok1 and ok2 and ok3


# ---------------------------------------------------------------------------
# r2: split-half reliability .825 — ATTESTATION ONLY
# ---------------------------------------------------------------------------

def verify_split_half():
    print("\nr2 — split-half reliability of the panel mean  [attestation, not reproduction]")
    f = DATA / "split_half_iterations.csv"
    if not f.exists():
        return check("split-half mean over iterations", None, 0.825, 0.01)
    d = pd.read_csv(f)
    v = d.split_half_pairwise_agreement.dropna()
    print(f"         {len(v)} iterations, sd {v.std():.4f}, "
          f"2.5-97.5% [{v.quantile(.025):.3f},{v.quantile(.975):.3f}]")
    return check("split-half mean over iterations", float(v.mean()), 0.825, 0.01)


# ---------------------------------------------------------------------------
# r1: recognition audit — rates, accuracy-by-flag, margin decomposition
# ---------------------------------------------------------------------------

def verify_recognition():
    print("\nr1 — recognition audit")
    f = DATA / "recognition_flags.csv"
    if not f.exists():
        check("identification rate", None, 0.125, 0.002)
        return False
    d = pd.read_csv(f)
    r = d[d.has_reasoning]
    ok = []
    ok.append(check("identification rate", float(r.identification.mean()), 0.125, 0.002))
    ok.append(check("brand-mention rate", float(r.brand.mean()), 0.102, 0.002))
    ok.append(check("external-recall rate", float(r.external_recall.mean()), 0.024, 0.002))

    # accuracy-by-flag: identification
    fl, cl = r[r.identification], r[~r.identification]
    ok.append(check("accuracy delta, identified vs not",
                    float(fl.gemini_correct.mean() - cl.gemini_correct.mean()), 0.042, 0.005))

    # margin decomposition vs Bradley-Terry
    ok.append(check("Gemini overall accuracy", float(r.gemini_correct.mean()), 0.662, 0.003))
    ok.append(check("Bradley-Terry overall accuracy", float(r.bt_correct.mean()), 0.613, 0.003))
    rec, non = r[r.identification], r[~r.identification]
    ok.append(check("margin, recognized pairs",
                    float((rec.gemini_correct - rec.bt_correct).mean()), 0.026, 0.005))
    ok.append(check("margin, not-recognized pairs",
                    float((non.gemini_correct - non.bt_correct).mean()), 0.052, 0.005))
    return all(ok)


# ---------------------------------------------------------------------------
# r1: potent additives change nothing (micro +.000, macro +.0005)
# ---------------------------------------------------------------------------

def _pairwise_acc(df, macro=False):
    """Within-category pairwise ranking accuracy; ties get half credit."""
    per_cat = {}
    for cat, g in df.groupby("category"):
        t = g.true_score.values
        p = g.predicted_score.values
        num = den = 0.0
        for i, j in combinations(range(len(g)), 2):
            if t[i] == t[j]:
                continue
            den += 1
            if p[i] == p[j]:
                num += 0.5
            elif (p[i] > p[j]) == (t[i] > t[j]):
                num += 1
        if den:
            per_cat[cat] = (num, den)
    if macro:
        return float(np.mean([n / d for n, d in per_cat.values()]))
    tn = sum(n for n, _ in per_cat.values())
    td = sum(d for _, d in per_cat.values())
    return tn / td


def verify_additives():
    print("\nr1 — potent additives (MSG, yeast extract)")
    need = ["c2_oof_cached.csv", "c2_oof_reextracted_top3.csv", "c2_oof_top3_plus_additives.csv"]
    if not all((DATA / n).exists() for n in need):
        check("additive effect (micro)", None, 0.0, 0.0005)
        return False
    base = read_oof(DATA / "c2_oof_reextracted_top3.csv")
    add = read_oof(DATA / "c2_oof_top3_plus_additives.csv")
    cached = read_oof(DATA / "c2_oof_cached.csv")
    ok = []
    ok.append(check("control reproduces the paper bench (micro)",
                    _pairwise_acc(cached), 0.6102, 0.0005))
    ok.append(check("additive effect (micro)",
                    _pairwise_acc(add) - _pairwise_acc(base), 0.0000, 0.0005))
    ok.append(check("additive effect (macro)",
                    _pairwise_acc(add, macro=True) - _pairwise_acc(base, macro=True),
                    0.0005, 0.0005))
    return all(ok)


def main():
    argparse.ArgumentParser().parse_args()

    print("=" * 78)
    print("Verifying published numbers from the bundle only")
    print("  no gated NECTAR data, no API key, no model downloads")
    print("=" * 78)

    verify_alpha()
    verify_rater_vs_rater()
    verify_block_variance()
    verify_split_half()
    verify_recognition()
    verify_additives()

    n = len(RESULTS)
    npass = sum(1 for *_, ok in RESULTS if ok)
    print("\n" + "=" * 78)
    print(f"{npass}/{n} checks passed")
    failed = [r[0] for r in RESULTS if not r[3]]
    if failed:
        print("FAILED: " + "; ".join(failed))
    print("=" * 78)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
