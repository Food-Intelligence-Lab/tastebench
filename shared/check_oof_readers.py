"""Gate: every out-of-fold reader tolerates both target forms.

WHY THIS EXISTS. An OOF file's target column is ``true_score`` when it was built
from the gated NECTAR ratings and ``true_rank`` when it came out of the public
release (see shared/oof_io.py). A script that hard-codes one form crashes on the
other, and a Tier 1 reproduction sees BOTH — it starts on released files and
overwrites them with retrained ones partway through.

That already shipped once. ``food_similarity/reproduce.sh`` Phase 4 died with
``KeyError: ['true_score']`` in ``train/run_taste_gnn_nectar.py``, which reads an
archived OOF before retraining. Nothing caught it: the release build re-renders
every table, but no build step and no verification bundle runs a Tier 1 script,
and the sweep that was supposed to find unconverted readers exempted
``food_similarity/train/`` wholesale on the assumption that everything there
BUILDS its target rather than reading one. Most of it does. Four files did not.

So this checks two separate things, neither of which needs the gated bundle:

BEHAVIOUR
    Builds the same tiny OOF twice, once per target spelling, and asserts the
    shared loader and every metric entry point accept both and return identical
    numbers. This is the claim the release rests on — "ranking the target moves
    no published number" — reduced to an assertion.

COVERAGE
    Finds every line that reads a CSV out of an OOF directory and fails unless it
    goes through ``read_oof``. Paths are resolved through local assignments, so
    ``archive = OOF / "..."`` followed by ``pd.read_csv(archive)`` is caught —
    naming, not directory, is what decides. There are no directory-wide
    exemptions, by design.

Usage:
    python3 shared/check_oof_readers.py               # this tree
    python3 shared/check_oof_readers.py --tree DIR    # a built release tree
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oof_io import RANK_TARGET, TARGET, normalize_target, read_oof  # noqa: E402

# A path expression is "an OOF path" if it names one. Deliberately loose: a false
# positive costs one line of exemption, a false negative costs a release.
OOF_TOKEN = re.compile(r"oof", re.IGNORECASE)

READ_CALL = re.compile(r"(?<![\w.])(?:pd\.)?read_csv\s*\(([^)]*)")
ASSIGN = re.compile(r"^\s*([A-Za-z_][\w]*)\s*=\s*(.+)$")

# Reads that name an OOF path but are not reading an OOF file. Each entry is
# (path suffix, substring of the offending line, reason). Currently EMPTY, and
# that is the target state: if it grows, the detector is wrong, not the code.
EXEMPT_LINES: list[tuple[str, str, str]] = []

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "wandb"}


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------

def _fixture() -> pd.DataFrame:
    """A four-category OOF with ties, NaNs and a non-monotone prediction."""
    rng = np.random.default_rng(0)
    rows = []
    for cat in ["Bacon", "Cheddar", "Mozzarella", "Yogurt"]:
        for code in range(6):
            rows.append({"category": cat, "product_code": code,
                         TARGET: float(rng.integers(0, 4)),      # ties on purpose
                         "predicted_score": float(rng.normal())})
    df = pd.DataFrame(rows)
    df.loc[3, "predicted_score"] = np.nan
    df.loc[7, TARGET] = np.nan
    return df


def check_behaviour(tmp: Path) -> list[str]:
    """Both spellings must give bit-identical metrics through every entry point."""
    sys.path.insert(0, str(HERE.parent / "food_similarity"))
    from evaluation.metrics import (canonical_order, compute_all_metrics,
                                    compute_per_category_metrics)

    failures = []
    scored = _fixture()

    # The rank form as the release writes it: average rank WITHIN category.
    ranked = scored.rename(columns={TARGET: RANK_TARGET}).copy()
    ranked[RANK_TARGET] = (scored.groupby("category")[TARGET]
                           .rank(method="average").to_numpy())

    p_score, p_rank = tmp / "oof_true_score.csv", tmp / "oof_true_rank.csv"
    scored.to_csv(p_score, index=False)
    ranked.to_csv(p_rank, index=False)

    a, b = read_oof(p_score), read_oof(p_rank)
    for name, frame in [("true_score fixture", a), ("true_rank fixture", b)]:
        if TARGET not in frame.columns:
            failures.append(f"read_oof did not present {TARGET} for the {name}")
        if RANK_TARGET in frame.columns:
            failures.append(f"read_oof left {RANK_TARGET} behind for the {name}")

    # The six reported metrics, plus the two order-sensitive helpers.
    ma, mb = compute_all_metrics(a), compute_all_metrics(b)
    worst = 0.0
    for k in ma:
        if isinstance(ma[k], float) and np.isfinite(ma[k]) and np.isfinite(mb[k]):
            worst = max(worst, abs(ma[k] - mb[k]))
        elif ma[k] != mb[k] and not (pd.isna(ma[k]) and pd.isna(mb[k])):
            failures.append(f"metric {k}: {ma[k]} vs {mb[k]}")
    if worst > 0.0:
        failures.append(f"metrics moved between target forms: max |delta| = {worst:.3e}")

    pa = compute_per_category_metrics(a).drop(columns=["category"])
    pb = compute_per_category_metrics(b).drop(columns=["category"])
    if not np.allclose(pa.to_numpy(dtype=float), pb.to_numpy(dtype=float),
                       rtol=0, atol=0, equal_nan=True):
        failures.append("compute_per_category_metrics differs between target forms")

    # canonical_order pins the bootstrap's row order; both forms must agree on it,
    # or a CI computed in one tree would not reproduce in the other.
    # .equals(), not ==: a NaN prediction is a legitimate row here and NaN != NaN.
    ca = canonical_order(a)[["category", "product_code", "predicted_score"]]
    cb = canonical_order(b)[["category", "product_code", "predicted_score"]]
    if not ca.equals(cb):
        failures.append("canonical_order() disagrees between target forms")

    # A frame that never had a target must still pass through untouched: several
    # consumers only want predicted_score.
    bare = a.drop(columns=[TARGET])
    if list(normalize_target(bare).columns) != list(bare.columns):
        failures.append("normalize_target() altered a target-free frame")

    print(f"  behaviour: {'FAIL' if failures else 'ok'} — "
          f"6 metrics, per-category table and canonical_order identical across "
          f"true_score/true_rank (max |delta| = {worst:.3e})")
    return failures


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def _oof_names(text: str) -> set[str]:
    """Local names bound to something that names an OOF path.

    Iterated to a fixpoint so a chain (``OOF = ...oof_predictions``;
    ``archive = OOF / "x.csv"``) propagates. That chain is the exact shape of the
    read that shipped broken.
    """
    names: set[str] = set()
    for _ in range(6):
        grew = False
        for line in text.splitlines():
            m = ASSIGN.match(line)
            if not m:
                continue
            lhs, rhs = m.group(1), m.group(2)
            if lhs in names:
                continue
            if OOF_TOKEN.search(rhs) or any(
                    re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", rhs) for n in names):
                names.add(lhs)
                grew = True
        if not grew:
            break
    return names


def scan_file(path: Path, rel: str) -> list[str]:
    text = path.read_text(errors="replace")
    if "read_csv" not in text:
        return []
    names = _oof_names(text)
    problems = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for m in READ_CALL.finditer(line):
            arg = m.group(1)
            reads_oof = OOF_TOKEN.search(arg) or any(
                re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", arg) for n in names)
            if not reads_oof:
                continue
            if any(rel.endswith(suffix) and frag in line for suffix, frag, _ in EXEMPT_LINES):
                continue
            problems.append(f"{rel}:{line_no}: reads an OOF with plain read_csv "
                            f"-- use read_oof (shared/oof_io.py):\n      {line.strip()}")
    return problems


def check_coverage(tree: Path) -> list[str]:
    problems, scanned = [], 0
    for p in sorted(tree.rglob("*")):
        if p.suffix not in {".py", ".sh"} or not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.resolve() == Path(__file__).resolve():
            continue
        scanned += 1
        problems += scan_file(p, p.relative_to(tree).as_posix())
    print(f"  coverage:  {'FAIL' if problems else 'ok'} — scanned {scanned} "
          f"script(s) under {tree}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=str(HERE.parent),
                    help="tree to scan (default: the tree this file lives in)")
    ap.add_argument("--coverage-only", action="store_true")
    a = ap.parse_args()
    tree = Path(a.tree).resolve()

    print("checking that every out-of-fold reader accepts true_score AND true_rank")
    failures = []
    if not a.coverage_only:
        import tempfile
        import traceback
        with tempfile.TemporaryDirectory() as td:
            try:
                failures += check_behaviour(Path(td))
            except Exception as exc:
                # A reader that handles only one form raises rather than
                # returning a wrong number, so the exception IS the finding.
                # Report it as one instead of dumping a traceback at a reader.
                print("  behaviour: FAIL — raised while reading the fixtures")
                traceback.print_exc(limit=3)
                failures.append(f"behaviour check raised {type(exc).__name__}: {exc}")
    failures += check_coverage(tree)

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nOK: both target forms are handled everywhere they can arrive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
