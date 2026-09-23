"""Produce taste_gnn_compound_embeddings.pkl matching FART's SMILES coverage.

Reads the SMILES *keys* from shared/data/caches/fart_compound_embeddings.pkl,
runs predict_embeddings, writes to shared/data/caches/<name>.pkl + .provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from molecular.src.embed.predict_embeddings import run as predict_run

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--source_cache", type=Path, required=True,
                    help="Existing FART cache whose SMILES keys we re-embed.")
    ap.add_argument("--output_pkl", type=Path, required=True)
    ap.add_argument("--extraction_point", choices=["encoder", "penultimate"], default="penultimate")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # Both inputs are things the public tree may legitimately not have yet: the
    # checkpoint ships (molecular/results/grid/best/ckpt.pt) but the FART source
    # cache is a 431 MB derived file the user builds. Report which one is absent
    # instead of letting pickle.load or torch.load raise from inside a helper.
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        print("  the released grid ships one at molecular/results/grid/best/ckpt.pt")
        return 1
    if not args.source_cache.exists():
        print(f"ERROR: source cache not found at {args.source_cache}")
        print("  this step re-embeds the SMILES keys of the FART cache; build it first:")
        print("    python3 shared/scripts/prepare_caches/prepare_fart_embeddings.py")
        return 1

    logger.info("Reading SMILES keys from %s", args.source_cache)
    with args.source_cache.open("rb") as f:
        src = pickle.load(f)
    smiles = sorted(src.keys())
    logger.info("Total SMILES: %d", len(smiles))

    # Write to temp CSV matching predict_embeddings.py's expected format
    tmp_csv = args.output_pkl.parent / (args.output_pkl.stem + ".in.csv")
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Canonicalized SMILES": smiles}).to_csv(tmp_csv, index=False)

    stats = predict_run(
        checkpoint=args.checkpoint,
        smiles_csv=tmp_csv,
        output_pkl=args.output_pkl,
        extraction_point=args.extraction_point,
        device=args.device,
        batch_size=args.batch_size,
        smiles_column="Canonicalized SMILES",
    )

    # Provenance
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    prov = {
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "checkpoint":       str(args.checkpoint),
        "checkpoint_sha":   sha256(args.checkpoint),
        "source_cache":     str(args.source_cache),
        "source_cache_sha": sha256(args.source_cache),
        "extraction_point": args.extraction_point,
        "n_source_keys":    len(smiles),
        "n_written":        stats["n_written"],
        "n_failed":         stats["n_failed"],
        "embedding_dim":    stats["dim"],
        "git_sha":          git_sha,
    }
    prov_path = args.output_pkl.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2))
    logger.info("Provenance written to %s", prov_path)
    tmp_csv.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
