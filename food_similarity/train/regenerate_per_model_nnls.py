"""Run `compute_per_model_nnls.py` for each of the 5 supervised models.

Produces `nested_{tag}_gemini_nnls.csv` for tag in {bt, hbt, ridge,
lgbm, ksvm}. These feed `table_per_model_nnls.tex` and the canonical
NNLS row in `table_results.tex`. The BT case is run by default; the
other 4 are spawned sequentially via subprocess with SUPERVISED_MODEL
set per call.

The nested inner LOOCV is the long pole (~30 min per model). Running
sequentially keeps n_jobs at the default per process; running 4 in
parallel would cause cache contention.

Usage:
    cd food_similarity
    python3 -m train.regenerate_per_model_nnls
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

SUPERVISED_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# bradley_terry is normally already done by the earlier BT phase, so it is skipped
# by default; SKIP_BT=0 forces a rerun. But "already done" is an assumption about
# an incremental run, and it is false on a from-scratch Tier 1: nothing else writes
# results/cache/nested_bt_gemini_base_v4.npz, that directory is export-ignored so
# the release never ships it, and regenerate_bt_aggregations.py needs it. A full
# reproduce.sh run therefore spent 96 minutes regenerating the other four models
# and then died on the missing file. Include BT when its cache is absent — see
# _bt_cache_missing() below.
MODELS_DEFAULT = ["hierarchical_bt", "ridge", "lightgbm_reg", "kernel_ranksvm"]

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"

_MODEL_SHORT = {
    "bradley_terry":   "bt",
    "hierarchical_bt": "hbt",
    "ridge":           "ridge",
    "lightgbm_reg":    "lgbm",
    "kernel_ranksvm":  "ksvm",
}


BT_BASE_CACHE = SUPERVISED_DIR / "results" / "cache" / "nested_bt_gemini_base_v4.npz"


def _bt_cache_missing() -> bool:
    """True when the canonical BT nested base has not been built yet."""
    return not BT_BASE_CACHE.exists()


def main():
    skip_bt = os.environ.get("SKIP_BT", "1") == "1"
    models = MODELS_DEFAULT.copy()
    if not skip_bt or _bt_cache_missing():
        models.insert(0, "bradley_terry")
        if skip_bt:
            logger.info(
                "bradley_terry added: %s is absent, so this is a from-scratch run "
                "and nothing else would produce it.", BT_BASE_CACHE.name)

    t_start = time.time()
    results = []
    failed = []

    for model in models:
        short = _MODEL_SHORT[model]
        nnls_csv = OOF_DIR / f"nested_{short}_gemini_nnls_v4.csv"
        if nnls_csv.exists() and os.environ.get("FORCE", "0") != "1":
            logger.info(f"Skipping {model}: {nnls_csv.name} already exists "
                        f"(set FORCE=1 to override)")
            continue
        logger.info("=" * 70)
        logger.info(f"Running compute_per_model_nnls with SUPERVISED_MODEL={model}")
        logger.info("=" * 70)
        env = os.environ.copy()
        env["SUPERVISED_MODEL"] = model

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "train.compute_per_model_nnls"],
            cwd=SUPERVISED_DIR,
            env=env,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            logger.error(f"  {model} failed (returncode={proc.returncode})")
            failed.append(model)
            continue

        results.append((model, elapsed))
        logger.info(f"  {model} completed in {elapsed/60:.1f} min")

    total = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    # The per-model subprocesses were logged-and-skipped, and main() then returned
    # None -- exit 0. So a Phase 6 in which EVERY model failed still printed
    # "Phase H complete" and let reproduce.sh (set -e) carry on to Phase 7, which
    # would go on to compute CIs over whichever ensembles happened to be on disk
    # already. Say which failed, and exit nonzero so the run stops here.
    if failed:
        logger.error(f"Phase H FAILED after {total/60:.1f} min: "
                     f"{len(failed)} of {len(models)} model(s) did not complete "
                     f"({', '.join(failed)}).")
        logger.error("The ensembles on disk are unchanged for those models, so "
                     "anything downstream would be scoring stale files.")
        return 1
    logger.info(f"Phase H complete in {total/60:.1f} min")
    logger.info("=" * 70)
    for model, el in results:
        logger.info(f"  {model:<25s} {el/60:>6.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
