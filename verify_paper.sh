#!/usr/bin/env bash
# Re-render every paper table and figure from the artifacts shipped in
# this directory: committed OOF prediction CSVs, parquet predictions,
# GNN grid outputs, and human-baseline analysis. No external data is
# downloaded; no models are trained.
#
# Inputs:
#   food_similarity/results/oof_predictions/
#   molecular/results/grid/, molecular/results/fart_augmented_test/
#   molecular/results/encoder_comparison/
#   human_baseline/results/
#
# Runtime is dominated by BCa bootstrap (n=10,000 resamples). The
# slowest single render is render_table_ablation_features.py, which
# sweeps CIs over 105 (model, feature-subset) pairs.
#
# Each area has its own verify.sh; this top-level script chains them, then
# re-renders the paper/charts figures and checks the render against the results
# file it came from. The figures were left out of this script once, while their
# numbers were hard-coded: a corrected results file reached the .tex tables (they
# render from it) but could not reach the figures, so the shipped
# chart_gnn_per_model.pdf contradicted the table printed beside it for as long as
# nobody looked. Rendering and checking them here is what makes that a build
# failure instead of a reviewer's discovery.
#
# Usage:  bash verify_paper.sh
#         bash verify_paper.sh --check-diff   # also check every rendered table
#                                             # and figure against
#                                             # paper/RENDERED_SHA256.txt
#                                             # (no git required; can fail)

set -euo pipefail

NEURIPS_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"

# Fail fast on a pyarrow that cannot read the committed predictions, rather
# than several minutes into a render. Interpreter and version drift are
# warnings and do not stop the run (see shared/check_env.py).
"${PY}" "$(dirname "$0")/shared/check_env.py" || exit 1

CHECK_DIFF=0
[[ "${1:-}" == "--check-diff" ]] && CHECK_DIFF=1

echo "========================================================"
echo "  TasteBench paper verification"
echo "  Re-rendering all tables from committed OOFs."
echo "  No downloads, no training."
echo "========================================================"

# Cheap and first, because it is the only check here that covers scripts this
# file never runs. The render path reads the OOFs one way; a Tier 1 reproduction
# (food_similarity/reproduce.sh) reads them the other way and is not exercised by
# any build step, which is how a hard KeyError on the shipped target column
# reached a reviewer while every render still passed. Seconds, no gated data.
echo ""
echo "[$(date +%H:%M:%S)] shared/check_oof_readers.py"
"${PY}" "${NEURIPS_DIR}/shared/check_oof_readers.py" | sed 's/^/  /'

bash "${NEURIPS_DIR}/food_similarity/verify.sh"
bash "${NEURIPS_DIR}/molecular/verify.sh"
bash "${NEURIPS_DIR}/human_baseline/verify.sh"

echo ""
echo "[$(date +%H:%M:%S)] paper/charts: make_charts.py"
( cd "${NEURIPS_DIR}/paper/charts" && "${PY}" make_charts.py ) | sed 's/^/  /'
echo ""
echo "[$(date +%H:%M:%S)] paper/charts: check_charts.py"
( cd "${NEURIPS_DIR}/paper/charts" && "${PY}" check_charts.py ) | sed 's/^/  /'

echo ""
echo "[$(date +%H:%M:%S)] All renders complete."
echo ""
echo "Outputs:"
echo "  ${NEURIPS_DIR}/paper/human_baseline/{human_baseline_table.tex, group_size_curve.pdf}"
echo "  ${NEURIPS_DIR}/paper/model_results_tables/*.tex"
echo "  ${NEURIPS_DIR}/paper/molecular_prediction/*.tex"
echo "  ${NEURIPS_DIR}/paper/charts/chart_*.{pdf,png}"

if [[ "${CHECK_DIFF}" == "1" ]]; then
  echo ""
  echo "[$(date +%H:%M:%S)] Checking rendered tables against paper/RENDERED_SHA256.txt..."
  # This used to be `git diff -- paper/`, which silently passed in the released
  # tree: there is no .git there, so the diff was empty and the check could not
  # fail. Compare against the shipped hash manifest instead.
  "${PY}" "${NEURIPS_DIR}/shared/check_rendered_outputs.py" || exit 1
fi

echo ""
echo "Done."

# Every rendered table must carry a caption. The human-baseline table shipped
# without one; this stops that recurring.
missing=0
for f in $(find "$(dirname "$0")/paper" -name '*.tex'); do
    grep -q 'caption' "$f" || { echo "NO CAPTION: $f" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "ERROR: rendered table(s) without a caption" >&2; exit 1; }
