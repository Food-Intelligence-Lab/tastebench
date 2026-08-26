"""BCa intervals must not depend on input row order or on product_code values.

Each test asserts BIT-IDENTICAL bounds, not approximate agreement: the fix is a
canonical sort, so any drift at all would mean an order dependence survived.

Run: python food_similarity/evaluation/tests/test_bootstrap_order_invariance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(FS))
sys.path.insert(0, str(FS.parent / "shared"))

from evaluation.bootstrap import compute_bca_cis  # noqa: E402
from evaluation.bootstrap_fast import (  # noqa: E402
    compute_bca_pw_acc,
    compute_bca_recall_at_k,
)
from evaluation.metrics import CANONICAL_SORT_KEYS, compute_all_metrics  # noqa: E402
from oof_io import read_oof  # noqa: E402

OOF = FS / "results" / "oof_predictions" / "bradley_terry_SNCTI_bench.csv"
# The one OOF in the paper's own row set that actually has a tie on all three
# canonical keys (rank-averaging maps two products to the same rank). 6 of the
# 216 committed OOFs do; this is the tie test's real-data case.
TIE_OOF = FS / "results" / "oof_predictions" / "nested_bt_gemini_rank.csv"
NB = 1000          # enough that an order effect shows; keeps the suite ~minutes
PERMS = (0, 1, 7, 12345)


def _load():
    return read_oof(OOF).dropna(subset=["predicted_score", "true_score"])


def _all_cis(df):
    """Every interval a render script can ask for, as one flat dict."""
    out = {k: v for k, v in compute_bca_cis(df, n_bootstrap=NB).items()}
    out["fast_pw_acc"] = compute_bca_pw_acc(df, n_bootstrap=NB)[1:]
    for k in (1, 2, 3):
        out[f"fast_recall_at_{k}"] = compute_bca_recall_at_k(df, k, n_bootstrap=NB)[1:]
    return out


def _diff(a, b):
    """Names of intervals that are not bit-identical between two runs."""
    return [k for k in a if a[k][0] != b[k][0] or a[k][1] != b[k][1]]


def _report(name, base, other, label):
    bad = _diff(base, other)
    if bad:
        print(f"  FAIL {name}: {label} moved {len(bad)} interval(s)")
        for k in bad:
            print(f"    {k:22s} {base[k]} -> {other[k]}")
    else:
        print(f"  ok   {name}: {label} — all {len(base)} intervals bit-identical")
    return not bad


def check_row_order_invariance(df, base):
    """Test 1: several row permutations, all must give identical bounds."""
    print("TEST 1  row-order invariance")
    ok = True
    for s in PERMS:
        shuffled = df.sample(frac=1.0, random_state=s).reset_index(drop=True)
        assert not shuffled["product_code"].equals(
            df.reset_index(drop=True)["product_code"]), f"seed={s} was a no-op shuffle"
        ok &= _report("row-order", base, _all_cis(shuffled), f"permutation seed={s}")
    return ok


def check_rekey_invariance(df, base):
    """Test 2: re-mint product_code the way the release build does.

    Sorting by (category, product_code) would fail this — the build's map is a
    fresh random per-category permutation, so it reorders rows arbitrarily.
    """
    print("TEST 2  re-key invariance (simulates remint_product_codes.py)")
    ok = True
    for s in PERMS:
        rng = np.random.default_rng(s)
        d = df.copy()
        for cat, g in d.groupby("category"):
            codes = sorted(g["product_code"].unique())
            public = list(range(1, len(codes) + 1))
            rng.shuffle(public)
            m = dict(zip(codes, public))
            d.loc[g.index, "product_code"] = [m[c] for c in g["product_code"]]
        # The build re-keys AND re-sorts by the new public code.
        d = d.sort_values(["category", "product_code"], kind="mergesort").reset_index(drop=True)
        ok &= _report("re-key", base, _all_cis(d), f"remint seed={s}")
    return ok


def check_ties_are_inert(base_df, label):
    """Test 3: permuting rows tied on all sort keys must change nothing."""
    print(f"TEST 3  tie handling is inert  [{label}]")
    ties = base_df.groupby(CANONICAL_SORT_KEYS).size()
    ties = ties[ties > 1]
    if len(ties):
        print(f"  {label} has {len(ties)} REAL tied group(s) on {CANONICAL_SORT_KEYS}")
        df = base_df.copy()
    else:
        print(f"  {label} has NO ties on {CANONICAL_SORT_KEYS}; injecting a synthetic one")
        df = base_df.copy()
        # Duplicate two rows from different categories so each tie group has 2
        # members that differ ONLY in product_code.
        dup = df.groupby("category").head(1).head(2).copy()
        dup["product_code"] = dup["product_code"].max() + 1000 + np.arange(len(dup))
        df = pd.concat([df, dup], ignore_index=True)
        ties = df.groupby(CANONICAL_SORT_KEYS).size()
        ties = ties[ties > 1]
        print(f"  injected {len(ties)} tied group(s)")

    base = _all_cis(df)
    ok = True
    for s in PERMS:
        rng = np.random.default_rng(s)
        # Permute ONLY within tie groups; every other row keeps its position.
        order = np.arange(len(df))
        for keys in ties.index:
            mask = np.ones(len(df), dtype=bool)
            for col, val in zip(CANONICAL_SORT_KEYS, keys):
                mask &= (df[col] == val).values
            pos = np.where(mask)[0]
            order[pos] = rng.permutation(pos)
        d = df.iloc[order].reset_index(drop=True)
        ok &= _report("ties", base, _all_cis(d), f"within-tie permutation seed={s}")
    return ok


def check_points_unchanged(df):
    """Test 5: point estimates are order-invariant already — prove it still holds."""
    print("TEST 5  point estimates unchanged by the sort")
    from evaluation.metrics import canonical_order
    a = compute_all_metrics(df)
    b = compute_all_metrics(canonical_order(df))
    bad = [k for k in a if not (a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])))]
    if bad:
        print("  FAIL: sorting moved point estimate(s)")
        for k in bad:
            print(f"    {k:22s} {a[k]!r} -> {b[k]!r}")
    else:
        print(f"  ok   all {len(a)} point estimates identical (max delta 0.0)")
    return not bad


# pytest entry points. The functions above take arguments because the script
# driver below parameterises them, and pytest reads any argument as a fixture
# request — so collecting them directly failed with "fixture 'df' not found"
# and the suite errored out in the released tree. These no-arg wrappers let the
# same checks run under `pytest` and as `python3 test_bootstrap_order_invariance.py`.

_BASE_CIS = None


def _fixtures():
    """Fresh frame, cached baseline CIs.

    The baseline is a 10,000-resample BCa over every metric, so recomputing it
    per test took the suite past six minutes. It depends only on the committed
    OOF, so compute it once. The DataFrame is re-read each call because the
    checks permute their copy and must not see another test's ordering.
    """
    global _BASE_CIS
    df = _load()
    if _BASE_CIS is None:
        _BASE_CIS = _all_cis(df)
    return df, _BASE_CIS


def test_points_unchanged():
    df, _ = _fixtures()
    assert check_points_unchanged(df)


def test_row_order_invariance():
    df, base = _fixtures()
    assert check_row_order_invariance(df, base)


def test_rekey_invariance():
    df, base = _fixtures()
    assert check_rekey_invariance(df, base)


def test_ties_are_inert_real():
    tie_df = read_oof(TIE_OOF).dropna(subset=["predicted_score", "true_score"])
    assert check_ties_are_inert(tie_df, TIE_OOF.name)


def test_ties_are_inert_synthetic():
    df, _ = _fixtures()
    assert check_ties_are_inert(df, OOF.name)


def main() -> int:
    df = _load()
    print(f"OOF: {OOF.name}  n={len(df)} rows, "
          f"{df['category'].nunique()} categories, n_bootstrap={NB}\n")
    base = _all_cis(df)
    tie_df = read_oof(TIE_OOF).dropna(subset=["predicted_score", "true_score"])
    results = [
        check_points_unchanged(df),
        check_row_order_invariance(df, base),
        check_rekey_invariance(df, base),
        check_ties_are_inert(tie_df, TIE_OOF.name),   # real ties
        check_ties_are_inert(df, OOF.name),           # synthetic ties
    ]
    print()
    if all(results):
        print("ALL PASS")
        return 0
    print(f"FAILED {results.count(False)} of {len(results)} test group(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
