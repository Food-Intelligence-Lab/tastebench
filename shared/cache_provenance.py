"""Bind each BCa CI cache to the sha256 of the files it was bootstrapped from.

WHY THIS EXISTS. The CI caches were trusted, never validated, and that is why a
real defect shipped. verify_paper.sh re-renders every table; the render scripts
read a cached CI file and skip the bootstrap when the cache covers every row, so
the check confirmed a cache had been *copied faithfully* and never once tested
whether it *follows from the data*. When the release build re-keyed product_code
and re-sorted the OOF rows after the tables were rendered, 68 of 72 intervals in
cis_table_results.csv stopped being reproducible from the shipped inputs, and
nothing went red.

A cache is now accompanied by a sidecar ``<cache>.sha256`` naming its inputs. A
loader that finds a mismatch must not use the cache. Sidecar rather than an extra
CSV column, because every render script parses these CSVs with a fixed schema
(``row,metric,point,ci_lo,ci_hi`` and friends) and a new column would break them.

The format is deliberately ``sha256sum -c`` compatible, matching
food_similarity/results/input_cache_sha256.txt, which reproduce.sh Phase 1
already verifies that way:

    <64 hex>  food_similarity/results/oof_predictions/bradley_terry_SC.csv

Set TASTEBENCH_CACHE_STRICT=1 to turn a stale cache into a hard error instead of
a recompute. CI should set it: a silent recompute is correct but slow, and a
build that quietly spends 25 minutes re-bootstrapping is a signal worth failing
on.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# Everything is hashed relative to the repository root so the sidecars are
# identical in the source tree and the released tree.
ROOT = Path(__file__).resolve().parent.parent

STRICT_ENV = "TASTEBENCH_CACHE_STRICT"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sidecar_path(cache_path: Path | str) -> Path:
    """``results/cis_x.csv`` -> ``results/cis_x.csv.sha256``.

    Suffixed rather than replaced so the sidecar sorts next to its cache and
    cannot collide with a future ``cis_x.sha256`` artifact.
    """
    return Path(str(cache_path) + ".sha256")


def manifest_lines(input_paths: Iterable[Path | str]) -> list[str]:
    """``sha256sum``-format lines for the given inputs, sorted by path.

    Sorted because callers discover inputs by globbing, and glob order is not
    guaranteed stable across filesystems — an unsorted manifest would appear to
    change when nothing had.
    """
    seen: dict[str, Path] = {}
    for p in input_paths:
        p = Path(p).resolve()
        seen[_rel(p)] = p
    return [f"{_sha256(seen[rel])}  {rel}" for rel in sorted(seen)]


def write_sidecar(cache_path: Path | str, input_paths: Sequence[Path | str],
                  header: str | None = None) -> Path:
    """Record what a freshly computed cache was derived from.

    `header` overrides the leading comment. The default describes a BCa cache,
    which is what all but one caller is; the feature matrix supplies its own so
    the file does not tell a reader to "regenerate the cache" for something that
    is not one.
    """
    missing = [str(p) for p in input_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"cannot record provenance for {cache_path}: missing input(s) {missing}")
    out = sidecar_path(cache_path)
    body = "\n".join(manifest_lines(input_paths))
    if header is None:
        header = (f"# BCa inputs for {Path(cache_path).name}. Regenerate the cache if these\n"
                  f"# no longer match; see shared/cache_provenance.py.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{header}\n{body}\n")
    return out


def check_sidecar(cache_path: Path | str, input_paths: Sequence[Path | str],
                  kind: str = "CI CACHE",
                  remedy: str = "ignoring the cache and re-running the bootstrap") -> bool:
    """True iff the cache's recorded inputs still hash the same.

    Returns False (and says why) for a missing sidecar too: an unprovenanced
    cache is exactly the state that let the defect ship, so it does not get the
    benefit of the doubt.

    `kind`/`remedy` only shape the message. They exist because this is also used
    to bind the feature matrix, where "re-running the bootstrap" is not a thing
    anyone can do — see `check_matrix_sidecar`.
    """
    cache_path = Path(cache_path)
    side = sidecar_path(cache_path)
    tag = f"{cache_path.name}"

    if not side.exists():
        return _stale(tag, f"no provenance sidecar at {_rel(side)}", kind, remedy)

    recorded = {}
    for line in side.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        recorded[rel.strip()] = digest.strip()

    actual = {}
    for line in manifest_lines(input_paths):
        digest, _, rel = line.partition("  ")
        actual[rel] = digest

    if set(recorded) != set(actual):
        added = sorted(set(actual) - set(recorded))
        gone = sorted(set(recorded) - set(actual))
        return _stale(tag, f"input set changed (+{len(added)} -{len(gone)}): "
                           f"{(added + gone)[:3]}", kind, remedy)

    changed = [rel for rel in actual if actual[rel] != recorded[rel]]
    if changed:
        return _stale(tag, f"{len(changed)} input(s) changed content, e.g. "
                           f"{sorted(changed)[:3]}", kind, remedy)
    return True


def _stale(tag: str, reason: str, kind: str = "CI CACHE",
           remedy: str = "ignoring the cache and re-running the bootstrap") -> bool:
    msg = f"STALE {kind}: {tag}: {reason}"
    if os.environ.get(STRICT_ENV) == "1":
        raise SystemExit(f"ERROR: {msg}\n"
                         f"       {STRICT_ENV}=1 refuses to proceed from an artifact "
                         f"that does not follow from its inputs.\n"
                         f"       {remedy}.")
    print(f"{msg}\n  -> {remedy}.", file=sys.stderr)
    return False


# --- feature-matrix provenance ----------------------------------------------
#
# WHY THIS EXISTS, SEPARATELY FROM THE CI CACHES. Every published supervised and
# distance number is a function of one file — food_similarity/data/product_features.pkl
# — and nothing bound the two together. The matrix is gated, so it is untracked;
# `git status` cannot notice it changing, and the out-of-fold files it produced
# carry no record of which matrix that was.
#
# That gap cost a full day. A stale, orphaned copy of the matrix
# (zero_shot_baselines/tests/regression_baseline/supervised_product_features.pkl,
# since deleted) differed from the shipped one in two feature blocks, and with
# nothing authoritative to compare against, the shipped matrix was diagnosed as
# the drifted one. It was not — retraining from it reproduces all 110 out-of-fold
# files exactly. The sidecar makes that answerable in a second rather than a day.
#
# It is a TRIPWIRE, not a cache: a CI cache that goes stale can be recomputed on
# the spot, but a feature matrix that no longer matches the published numbers
# cannot be repaired by re-running anything. So the remedy text says to
# regenerate and re-verify rather than promising a silent recompute.
MATRIX_REL = "food_similarity/data/product_features.pkl"

_MATRIX_REMEDY = ("the published out-of-fold files may not follow from this matrix; "
                  "regenerate it with food_similarity/prepare_data.py and re-verify, "
                  "or restore the recorded one")


def matrix_path() -> Path:
    """The feature matrix every published supervised/distance number comes from."""
    return ROOT / MATRIX_REL


def write_matrix_sidecar(matrix: Path | str | None = None) -> Path | None:
    """Bind the matrix on disk to its own sha256. None when there is no matrix.

    The matrix is its own recorded input. Its true upstream inputs are gated
    NECTAR CSVs and ~400 MB of caches that a release tree never has, so hashing
    those would make the sidecar uncheckable exactly where it is needed.
    """
    p = Path(matrix) if matrix is not None else matrix_path()
    if not p.exists():
        return None
    return write_sidecar(p, [p], header=(
        f"# sha256 of the {p.name} that the published out-of-fold files were\n"
        f"# trained from. This is a TRIPWIRE, not a cache: if it no longer matches,\n"
        f"# the shipped numbers may not follow from the matrix on disk. Rewrite it\n"
        f"# only after re-verifying. See shared/cache_provenance.py."))


def check_matrix_sidecar(matrix: Path | str | None = None) -> bool:
    """True when the matrix matches what was recorded — or when there is none.

    NO MATRIX = TRUE, unconditionally and before anything else is read. A public
    tree ships no `product_features.pkl` (it is gated), and Tier 0 must not be
    told that a file it is not supposed to have is in a bad state. Absence is the
    normal case there, not a failure; the check only has an opinion once a matrix
    is actually present.
    """
    p = Path(matrix) if matrix is not None else matrix_path()
    if not p.exists():
        return True
    return check_sidecar(p, [p], kind="FEATURE MATRIX", remedy=_MATRIX_REMEDY)


# --- cache CSV I/O ----------------------------------------------------------
#
# A cache exists so that re-rendering skips the bootstrap. That is only sound if
# rendering from the cache and rendering from a fresh bootstrap print the SAME
# digits, and with the obvious defaults they do not.
#
# The caches used to be written with float_format="%.6f" and read with a bare
# read_csv. Both truncate: the writer drops everything past the sixth decimal,
# and pandas' default C parser does not round-trip the last bit even when the
# text carries it. A bound landing on an exact rounding boundary — recall@2 for
# MMRF (L2) sat on 0.2875 — then printed .288 from a fresh bootstrap and .287
# from the cache, so `verify_paper.sh --check-diff` failed on a table nobody had
# touched. Only writing the shortest round-tripping repr AND reading with
# float_precision="round_trip" reproduces the in-memory double exactly.
#
# Route every cache through these two helpers rather than passing the keywords
# at each call site; the policy has to hold at all ~20 of them, and one writer
# added later with "%.6f" silently reintroduces the bug.


def read_cache_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """Read a CI cache, preserving the exact float64 values that were written."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _main(argv: Sequence[str] | None = None) -> int:
    """`python3 shared/cache_provenance.py --write-matrix | --check-matrix`.

    The CI caches are provenanced by whichever render script computes them. The
    feature matrix has no such owner — it is produced by prepare_data.py and then
    simply read by everything — so the record has to be writable by hand.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-matrix", action="store_true",
                   help="record the feature matrix now on disk as the authoritative one")
    g.add_argument("--check-matrix", action="store_true",
                   help="verify it still matches; exit 0 when it does, or when absent")
    a = ap.parse_args(argv)

    if a.write_matrix:
        side = write_matrix_sidecar()
        if side is None:
            print(f"no matrix at {matrix_path()}; nothing to record")
            return 0
        print(f"recorded {_rel(matrix_path())} -> {_rel(side)}")
        return 0

    if check_matrix_sidecar():
        p = matrix_path()
        print(f"feature matrix provenance: OK"
              if p.exists() else
              f"feature matrix provenance: no matrix present (public tree) — nothing to check")
        return 0
    return 1


def write_cache_csv(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    """Write a CI cache so every float round-trips bit-exactly.

    No float_format: pandas then emits the shortest string that reads back as
    the same double, which is both exact and more readable than "%.17g".
    """
    kwargs.pop("float_format", None)
    df.to_csv(path, index=False, **kwargs)


if __name__ == "__main__":
    raise SystemExit(_main())
