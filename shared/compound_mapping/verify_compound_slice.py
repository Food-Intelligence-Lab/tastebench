"""Check that the compound slice reproduces the full-source compound vectors exactly.

Extracts the compound feature twice in one process — once from the slice, once
from FoodAtlas + FooDB with DISABLE_COMPOUND_SLICE set — and compares the
resulting per-product vectors bit-for-bit.

This is the claim the slice rests on. Anything short of an exact match means the
released tree reproduces different numbers than the paper, so the comparison is
exact (bitwise, then a max-|diff| report) rather than approximate.

Usage (needs FoodAtlas + FooDB present, since it must compute the reference):
    python3 shared/compound_mapping/verify_compound_slice.py
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "food_similarity"))


def extract(use_slice: bool):
    for m in [m for m in list(sys.modules) if m == "prepare_data"]:
        del sys.modules[m]
    os.environ.pop("DISABLE_COMPOUND_SLICE", None)
    if not use_slice:
        os.environ["DISABLE_COMPOUND_SLICE"] = "1"
    import prepare_data as P
    products = P.load_nectar_products()
    return P.extract_compound(products)


def main() -> int:
    import prepare_data as P0
    if not P0.COMPOUND_SLICE.exists():
        print("ERROR: no slice — run build_compound_slice.py first")
        return 1

    # This check computes the REFERENCE from the full upstream sources, so it
    # needs the gated NECTAR product list plus FoodAtlas and FooDB. None of the
    # three is in the public release, and README/THIRD_PARTY_NOTICES point here
    # for the bit-exactness claim — so a public reader following that pointer
    # used to get a FileNotFoundError traceback. Skip cleanly instead, and say
    # what is missing. Only the preconditions are skipped: if the inputs are
    # present and the vectors disagree, this still fails below.
    def _rel(p):
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    missing = [_rel(p) for p in (P0.INGREDIENTS_CSV, P0.SENSORY_CSV,
                                 P0.FOOD_ATLAS_DIR, P0.FOODB_DIR)
               if not p.exists()]
    if missing:
        print("SKIP: cannot recompute the reference vectors here — this check "
              "needs inputs that are not in the public release:")
        for m in missing:
            print(f"    missing  {m}")
        print("  The slice itself ships and is verified by its recorded hash in\n"
              "  food_similarity/results/input_cache_sha256.txt (reproduce.sh\n"
              "  Phase 1). Re-running THIS comparison needs gated NECTAR access\n"
              "  plus the two public downloads (bash data/download_public.sh).")
        return 0

    print("computing from slice ...")
    a = extract(True)
    print("computing from FoodAtlas + FooDB ...")
    b = extract(False)

    only_a, only_b = set(a) - set(b), set(b) - set(a)
    print(f"\nproducts: slice={len(a)}  full={len(b)}")
    if only_a or only_b:
        print(f"  KEY MISMATCH: slice-only={len(only_a)} full-only={len(only_b)}")
        for k in list(only_a)[:3] + list(only_b)[:3]:
            print(f"      {k}")
        return 1

    identical = 0
    worst = 0.0
    worst_key = None
    for k in sorted(a, key=str):
        va, vb = np.asarray(a[k]), np.asarray(b[k])
        if va.shape != vb.shape:
            print(f"  SHAPE MISMATCH {k}: {va.shape} vs {vb.shape}")
            return 1
        if np.array_equal(va, vb):
            identical += 1
        d = float(np.max(np.abs(va - vb))) if va.size else 0.0
        if d > worst:
            worst, worst_key = d, k

    print(f"bit-identical vectors : {identical} of {len(a)}")
    print(f"max |difference|      : {worst:.3e}" + (f"  at {worst_key}" if worst_key else ""))
    if identical == len(a):
        print("\nPASS - the slice reproduces the compound features exactly")
        return 0
    print("\nFAIL - the slice does not reproduce the compound features")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
