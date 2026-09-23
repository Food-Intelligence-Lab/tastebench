"""Cut the 1.9 GB embedding caches down to the compounds this benchmark actually uses.

C29 committed to releasing the embedding caches so every number regenerates without an
API key. The full caches cover the 136k-compound FoodAtlas/FooDB union and total 1.9 GB,
which is too large to ship in a git repository -- so the commitment sat blocked on hosting
rather than on anything to do with the NDA. (The caches carry no NECTAR content at all:
every one is keyed by SMILES, none by product code.)

The 247 products touch **738 distinct SMILES**. Slicing to those turns 1.9 GB into ~10 MB,
a 197x reduction, and loses nothing that any published number depends on -- the vectors
for covered SMILES are copied verbatim, not recomputed or compressed.

This follows the precedent set by `compound_slice.parquet` (34 KB, replacing an 82 MB
FoodAtlas and a 953 MB FooDB download).

What a slice does NOT cover: any compound outside the 738. An ablation that widens the
ingredient set, or the molecular task's 15k-molecule FartDB work, still needs the full
cache -- regenerate it with the other scripts in this directory. `slices/MANIFEST.json`
records the covered set so a consumer can tell the difference between "not in the slice"
and "not in the data", and the loader says so by name when it falls back.

Usage:
    python3 shared/scripts/prepare_caches/build_embedding_slices.py
    python3 shared/scripts/prepare_caches/build_embedding_slices.py --check   # verify only
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = Path(__file__).resolve().parents[2]
CACHES = SHARED / "data" / "caches"
SLICES = CACHES / "slices"
SOURCE_OF_TRUTH = CACHES / "compound_slice.parquet"

# Every embedding cache keyed by SMILES. Deliberately explicit rather than a glob: a new
# cache should be added here consciously, after checking it is keyed by SMILES and carries
# no product key.
CACHE_NAMES = [
    "fart_compound_embeddings.pkl",
    "fart_classifier_dense_embeddings.pkl",
    "fart_taste_probs_embeddings.pkl",
    "taste_gnn_best_compound_embeddings.pkl",
    "_tg_seed0_compound_embeddings.pkl",
    "_tg_seed1_compound_embeddings.pkl",
    "_tg_seed2_compound_embeddings.pkl",
    "_tg_seed3_compound_embeddings.pkl",
    "_tg_seed4_compound_embeddings.pkl",
]


def needed_smiles() -> set:
    if not SOURCE_OF_TRUTH.exists():
        raise SystemExit(
            f"ERROR: {SOURCE_OF_TRUTH.name} is not here, and it defines which compounds a\n"
            f"       slice must cover. Build it first with\n"
            f"       shared/compound_mapping/build_compound_slice.py"
        )
    df = pd.read_parquet(SOURCE_OF_TRUTH)
    return set(df["smiles"].dropna().unique())


def _digest(obj: dict) -> str:
    """Order-independent digest over (smiles, bytes), so it does not depend on dict order."""
    h = hashlib.sha256()
    for k in sorted(obj):
        h.update(k.encode())
        h.update(np.asarray(obj[k]).tobytes())
    return h.hexdigest()


def build(check_only: bool) -> int:
    need = needed_smiles()
    print(f"  {len(need)} SMILES are reachable from the {SOURCE_OF_TRUTH.name} ingredient set")
    SLICES.mkdir(parents=True, exist_ok=True)

    manifest, missing_caches, bad = {}, [], []
    for name in CACHE_NAMES:
        full = CACHES / name
        out = SLICES / name.replace(".pkl", ".slice.pkl")
        if not full.exists():
            # A tree that already has only slices is the NORMAL public state, not an error.
            if out.exists():
                with open(out, "rb") as f:
                    have = pickle.load(f)
                manifest[name] = {"n": len(have), "sha256": _digest(have), "source": "existing slice"}
                print(f"  {name:<44} full cache absent; kept existing slice ({len(have)})")
            else:
                missing_caches.append(name)
            continue

        with open(full, "rb") as f:
            src = pickle.load(f)
        sub = {k: src[k] for k in need if k in src}
        uncovered = len(need) - len(sub)
        if uncovered:
            # Not fatal by itself -- say so loudly and record it, because a silently
            # short slice would look identical to a complete one at load time.
            bad.append(f"{name}: {uncovered} of {len(need)} needed SMILES absent from the full cache")

        if check_only:
            if not out.exists():
                bad.append(f"{name}: slice not built")
                continue
            with open(out, "rb") as f:
                have = pickle.load(f)
            if _digest(have) != _digest(sub):
                bad.append(f"{name}: slice does NOT match the full cache for the covered SMILES")
            else:
                print(f"  {name:<44} slice verified against full cache ({len(sub)} vectors)")
        else:
            with open(out, "wb") as f:
                pickle.dump(sub, f, protocol=4)
            print(f"  {name:<44} {len(src):>7} -> {len(sub):<5} "
                  f"({full.stat().st_size/1e6:7.1f} MB -> {out.stat().st_size/1e6:5.2f} MB)")
        manifest[name] = {"n": len(sub), "sha256": _digest(sub), "source": "sliced from full cache"}

    if missing_caches and not check_only:
        print(f"  NOTE: {len(missing_caches)} full cache(s) absent and no slice present: "
              f"{', '.join(missing_caches)}")
        print(f"        Regenerate them with the other scripts in this directory, then re-run.")

    if bad:
        print("\n  FAILED:")
        for b in bad:
            print(f"    {b}")
        return 1

    if not check_only:
        (SLICES / "MANIFEST.json").write_text(json.dumps({
            "covers": sorted(need),
            "n_smiles": len(need),
            "source_of_truth": SOURCE_OF_TRUTH.name,
            "caches": manifest,
            "note": "Slices carry the SAME vectors as the full caches for these SMILES, "
                    "copied verbatim. Anything outside this set needs the full cache.",
        }, indent=2) + "\n")
        print(f"\n  wrote {SLICES}/MANIFEST.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify existing slices against the full caches; build nothing")
    raise SystemExit(build(ap.parse_args().check))
