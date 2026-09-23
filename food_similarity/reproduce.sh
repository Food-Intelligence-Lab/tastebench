#!/usr/bin/env bash
# End-to-end reproduction for the food_similarity section.
#
# Regenerates the food_similarity OOF predictions and tables from raw
# inputs (caches + NECTAR data). Assumes prerequisites are in place:
#   - shared/data/caches/*.pkl, *.csv (see README.md → Caches)
#   - results/oof_predictions/*.csv   (already committed)
#   - data/{consolidated_datasets,food_atlas,foodb_2020_04_07_csv,
#           product_images,taste_like}/  (gated NECTAR + public)
#
# With the gated NECTAR bundle this is the heavyweight train-from-scratch
# path (~2-3h on 8 cores).  Without it, training phases (3-6) are skipped
# and the pipeline verifies the paper numbers from the committed
# out-of-fold predictions — stronger than verify_paper.sh because it
# independently recomputes every BCa CI from the OOFs.
#
# Pinned versions live in environment.yml; git SHA at run-time is
# recorded by Phase 0 below.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NEURIPS_DIR="$(cd "${HERE}/.." && pwd)"

# Phases 5 and later invoke `python3 -m train.<module>`, which resolves the
# package from the CURRENT directory, so running this script from anywhere but
# food_similarity/ died ten minutes in with "No module named 'train'" — long
# after the expensive phases had already succeeded. The earlier phases use
# absolute ${HERE} paths and so worked from anywhere, which made the failure look
# like a defect in Phase 5 rather than a wrong working directory.
cd "${HERE}"
MOL="${NEURIPS_DIR}/molecular"
PY="${PY:-python3}"

echo "== Phase 0: record git SHA + environment ==============="
git -C "${NEURIPS_DIR}" rev-parse HEAD 2>/dev/null \
  | tee "${HERE}/results/repro_git_sha.txt" || true

echo ""
echo "== Phase 1: verify input caches ========================"
"${PY}" "${HERE}/scripts/check_input_caches.py" --require 0


# This is the path that sees BOTH forms of the target column: it starts on the
# released out-of-fold files (`true_rank`) and overwrites them with retrained
# ones (`true_score`) at Phase 5. A reader that handles only one form used to
# surface as a KeyError nine minutes in, at Phase 4. Fail in the first seconds
# instead. Needs no gated data.
"${PY}" "${NEURIPS_DIR}/shared/check_oof_readers.py"

# Phase 1 verifies the input caches against a manifest but says nothing about
# product_features.pkl, which every supervised and distance number is a function
# of. The matrix is gated and therefore untracked, so git cannot notice it
# changing, and a drifted matrix used to surface only as numbers that quietly
# disagreed with the paper. Exits 0 when there is no matrix — a public tree does
# not ship one, and absence is normal there rather than a fault.
"${PY}" "${NEURIPS_DIR}/shared/cache_provenance.py" --check-matrix \
  || echo "WARNING: feature-matrix provenance drift; the numbers below may not match the paper"

echo ""
echo "== Phase 2: archived BCa CI sanity check ==============="
"${PY}" "${HERE}/scripts/sanity_check_archived_cis.py"

PKL="${HERE}/data/product_features.pkl"
if [ -f "${PKL}" ] || [ "${REBUILD_FEATURES:-0}" = "1" ]; then
    # ── Training path (NECTAR present) ──────────────────────────
    echo ""
    echo "== Phase 3: product features ==========================="
    if [ "${REBUILD_FEATURES:-0}" = "1" ]; then
        echo "   rebuilding from raw NECTAR (needs FoodAtlas, FooDB and encoder downloads)"
        "${PY}" "${HERE}/prepare_data.py"
    else
        echo "   using the shipped feature matrix (no downloads, no API key)"
        "${PY}" "${HERE}/scripts/check_product_features.py" --required
    fi

    echo ""
    echo "== Phase 4: GNN compound encoder comparison ============"
    "${PY}" "${HERE}/train/run_taste_gnn_nectar.py"

    echo ""
    echo "== Phase 5: regenerate v4.0 supervised + distance OOFs ="
    "${PY}" -m train.regenerate_supervised_oofs
    "${PY}" -m train.regenerate_distance_oofs

    echo ""
    echo "== Phase 6: per-model nested NNLS ensembles ============"
    "${PY}" -m train.regenerate_per_model_nnls
    "${PY}" -m train.regenerate_bt_aggregations
else
    # ── Verification path (no NECTAR) ───────────────────────────
    echo ""
    echo "== Phases 3-6: SKIPPED (feature matrix not found) ======"
    echo "   ${PKL##*/} ships in the gated NECTAR bundle (see data/GATED.md)."
    echo "   Verifying paper numbers from committed out-of-fold predictions."
fi

echo ""
echo "== Phase 7: BCa CIs for paper-table cells (n_jobs=8) ==="
"${PY}" "${HERE}/scripts/compute_cis_parallel.py" \
  --discover --n-jobs 8 \
  --out "${HERE}/results/cis_final_tables.csv"

echo ""
echo "== Phase 8: LLM CI regeneration ========================"
"${PY}" "${HERE}/scripts/regenerate_llm_cis.py"

echo ""
echo "== Phase 9: render food_similarity paper tables ========"
bash "${HERE}/verify.sh"

echo ""
echo "== Phase 10: render molecular paper tables ============="
bash "${MOL}/verify.sh"

echo ""
echo "Done."
echo "  food_similarity tables:  ${HERE}/results/tables/"
echo "  food_similarity figures: ${HERE}/results/figures/"
echo "  molecular tables:        ${MOL}/results/tables/tex/"
echo ""
echo "Run human_baseline/human_panelist_baseline.py separately for human-baseline numbers."
