"""Confirm the shipped feature matrix is usable before Tier 1 runs on it.

Two callers, two correct answers, so the CALLER says which it wants:

  standalone (``python3 scripts/check_product_features.py``)
      A bare public tree legitimately has no feature matrix — it ships in the
      gated NECTAR bundle. Reporting failure there made a clean checkout look
      broken to anyone running the checks in order, so absence SKIPs and exits 0.

  ``--required`` (reproduce.sh Phase 3)
      reproduce.sh is the Tier 1 entry point and cannot proceed without the
      matrix. An unconditional skip made Phase 3 print success on a tree that has
      no matrix and let Phase 4 die instead — one phase later, with an error
      naming neither the missing file nor how to obtain it.

The flag rather than caller-detection: the same file shape as
``scripts/check_input_caches.py --require``, which Phase 1 already uses for
exactly this reason, and the requirement is a property of the CALL, not something
the script can infer about its environment.

Usage:
    python3 scripts/check_product_features.py             # absent -> SKIP, exit 0
    python3 scripts/check_product_features.py --required  # absent -> exit 1
"""
import argparse
import pickle
import sys
from pathlib import Path

PKL = Path(__file__).resolve().parent.parent / "data" / "product_features.pkl"

BLOCKS = ["category_subset", "nutrition", "compound", "text", "image"]


def _how_to_get_it() -> None:
    print(f"  missing  {PKL.name}  (food_similarity/data/)")
    print("  It ships in the gated NECTAR bundle (see data/GATED.md); Tier 1")
    print("  needs NECTAR access regardless. To rebuild it from raw NECTAR:")
    print("      REBUILD_FEATURES=1 bash food_similarity/reproduce.sh")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--required", action="store_true",
                    help="a missing matrix is a failure (Tier 1 cannot run without one)")
    a = ap.parse_args()

    if not PKL.exists():
        if a.required:
            print("ERROR: Tier 1 needs the feature matrix and it is not here.")
            _how_to_get_it()
            return 1
        # Absent is the NORMAL state in the public release. A precondition, not a
        # fault — a matrix that IS present but unusable still fails below,
        # whether or not --required was passed.
        print("SKIP: the feature matrix is not here, which is expected in the "
              "public release.")
        _how_to_get_it()
        return 0

    try:
        with open(PKL, "rb") as f:
            pf = pickle.load(f)
    except Exception as exc:
        # A truncated or non-pickle file used to end the run in a traceback from
        # whichever consumer touched it first. Name the file here instead.
        print(f"ERROR: {PKL.name} is present but could not be loaded: "
              f"{type(exc).__name__}: {exc}")
        return 1

    if not isinstance(pf, dict) or not pf:
        print(f"ERROR: {PKL.name} is not the expected non-empty "
              f"{{(category, product_code): features}} mapping "
              f"(got {type(pf).__name__}).")
        return 1
    if not all(isinstance(v, dict) for v in pf.values()):
        print(f"ERROR: {PKL.name} values are not feature dicts; this is not a "
              f"product-features matrix.")
        return 1

    # The matrix is keyed on the original NECTAR codes; the release is keyed on
    # the re-keyed public ones. Ask for the map HERE — Phase 3, first seconds —
    # rather than letting Phase 4 find out minutes later with an empty pair array.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import product_code_map as pcm
    try:
        pf = pcm.to_public_features(pf, what=PKL.name)
    except SystemExit as exc:
        print(str(exc).lstrip("\n"))
        return 1

    analogs = sum(1 for v in pf.values() if v.get("is_analog"))
    missing = {b: sum(1 for v in pf.values() if v.get(b) is None) for b in BLOCKS}
    print(f"  product_features.pkl: {len(pf)} products ({analogs} analogs)")
    for b in BLOCKS:
        n = missing[b]
        print(f"    {b:16s} present for {len(pf)-n}/{len(pf)}")
    absent = [b for b in BLOCKS if missing[b] == len(pf)]
    if absent:
        print(f"ERROR: no product carries {', '.join(absent)}; this matrix cannot "
              f"drive the supervised models.")
        return 1
    if analogs < 200:
        print(f"ERROR: expected ~215 analogs, found {analogs}")
        return 1

    # Provenance. Everything above asks whether the matrix is STRUCTURALLY usable;
    # none of it asks whether it is the SAME matrix the published out-of-fold
    # files were trained from. A matrix can be perfectly well-formed, pass every
    # check here, and still not be the one behind the paper — which is exactly
    # what happened, and it cost a day of misdiagnosis before the shipped matrix
    # was cleared. Unreachable when the matrix is absent: that returns above.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    try:
        from cache_provenance import check_matrix_sidecar, sidecar_path
    except ImportError:
        # This file arrives by task overlay (read from disk) while
        # shared/cache_provenance.py arrives from the committed tree, so a build
        # made before the matrix-provenance helpers were committed has one but
        # not the other. Say which half is missing instead of dying in a
        # traceback; a tree built from a commit that has both never gets here.
        print("  NOTE: shared/cache_provenance.py in this tree predates the "
              "feature-matrix\n        provenance helpers, so the matrix could "
              "not be checked against a sidecar.")
        return 0

    side = sidecar_path(PKL)
    if not side.exists():
        # A warning, not a failure. An unprovenanced matrix is unverified, not
        # known-wrong, and a gated bundle unpacked before the sidecar existed is
        # a legitimate state — failing it would block Tier 1 on a tree that is
        # probably fine. Record it and the next run is checkable.
        print(f"  NOTE: {PKL.name} has no provenance sidecar, so nothing can tell "
              f"you whether it\n        is the matrix the published numbers came "
              f"from. Record the current one with:\n"
              f"            python3 shared/cache_provenance.py --write-matrix")
    elif not check_matrix_sidecar(PKL):
        print(f"ERROR: {PKL.name} does not match its recorded provenance — the "
              f"published\n       out-of-fold files may not follow from this "
              f"matrix. See the sidecar at\n       {side.name}.")
        return 1
    else:
        print(f"  provenance: matches {side.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
