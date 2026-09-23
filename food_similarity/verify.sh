#!/usr/bin/env bash
# Re-render every food-similarity table from the committed OOF
# prediction CSVs in results/oof_predictions/. No training, no
# external data.
#
# Runtime: ~25–30 minutes total. The long pole is
# render_table_ablation_features.py (~25 min: 105 BCa bootstrap calls
# parallelised across cores; each call uses the full Python-loop
# bootstrap and takes ~2 min). The other six renders together finish
# in ~5 min.
#
# Outputs (all written to paper/model_results_tables/):
#   table_results.tex                (combined ranking + recall metrics)
#   table_per_category.tex
#   table_per_category_nnls.tex
#   table_per_model_nnls.tex
#   table_ablation_features.tex
#   table_ablation_llm.tex
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"

# Per-run log, not a fixed /tmp path. Two areas verifying at once — or a
# reviewer re-running one while another is going — used to write the same
# file, and the `tail | sed` below then read a half-rewritten log. That
# crashed BSD sed outright (Abort trap: 6) and failed a build whose renders
# had all succeeded.
LOG="$(mktemp -t verify_food_similarity)"
trap 'rm -f "${LOG}"' EXIT


run() {
  echo ""
  echo "[$(date +%H:%M:%S)] food_similarity: $*"
  ( cd "${HERE}" && "${PY}" "$@" ) > "${LOG}" 2>&1 \
    || { echo "FAILED — last 60 lines of log (resource_tracker noise filtered):";
         grep -vE 'resource_tracker\.py|Cannot register .* for automatic cleanup|unknown resource type|raise ValueError\(' "${LOG}" | tail -60;
         exit 1; }
  tail -3 "${LOG}" | sed 's/^/  /'
}

run scripts/render_table_results.py
run scripts/render_table_per_category.py
run scripts/render_table_per_category_nnls.py
run scripts/render_table_per_model_nnls.py
run scripts/render_table_ablation_features.py
run scripts/render_table_ablation_llm.py

# Category-balanced evaluation (appendix E.7, C05/C06). Not table renderers: they
# write results/category_balanced/, which the appendix and the paper prose quote
# from. ~1 min for the 5,000-replicate percentile bootstrap, then a guard that
# fails if any quoted number has moved away from its generator.
run scripts/run_category_weighted.py
run scripts/hierarchical_estimation.py
run scripts/check_category_balanced.py
# Table 11 (ingredient aggregation). Re-runs a within-block cluster bootstrap, so
# it is the slowest renderer here — but leaving it out means its caption and CIs
# can drift from the generator without anything noticing.
run scripts/render_table_ingredient_agg.py
# Two-column variants used by the camera-ready layout.
run scripts/render_table_results_two_column.py
# Appendix tables E.7–E.8 (category-balanced evaluation, recognition audit,
# brand-blind probe). Pure formatting from committed analysis CSVs; fast.
run scripts/render_appendix_tables.py

# Verification bundle: recompute 18 published rebuttal numbers from the
# committed sidecar data (no gated NECTAR data, no API, no model downloads).
run results/verification/verify_bundle.py
