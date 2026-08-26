#!/usr/bin/env bash
# Re-render every molecular-area table from the committed parquet
# predictions in results/grid/ and results/fart_augmented_test/.
# No training, no external data.
#
# Outputs (all written to paper/molecular_prediction/):
#   table_molecular_prediction.tex
#   table_gnn_per_model.tex
#   table_molecular_per_class.tex
#   table_gnn_grid.tex
#
# CSV sidecars (parallel data versions of each table) land in
# molecular/results/tables_csv/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"

# Per-run log, not a fixed /tmp path. Two areas verifying at once — or a
# reviewer re-running one while another is going — used to write the same
# file, and the `tail | sed` below then read a half-rewritten log. That
# crashed BSD sed outright (Abort trap: 6) and failed a build whose renders
# had all succeeded.
LOG="$(mktemp -t verify_molecular)"
trap 'rm -f "${LOG}"' EXIT


run() {
  echo ""
  echo "[$(date +%H:%M:%S)] molecular: $*"
  ( cd "${HERE}" && "${PY}" "$@" ) > "${LOG}" 2>&1 \
    || { echo "FAILED — last 60 lines of log (resource_tracker noise filtered):";
         grep -vE 'resource_tracker\.py|Cannot register .* for automatic cleanup|unknown resource type|raise ValueError\(' "${LOG}" | tail -60;
         exit 1; }
  tail -3 "${LOG}" | sed 's/^/  /'
}

run scripts/render_table_molecular_prediction.py
run scripts/render_table_molecular_per_class.py
run scripts/render_table_gnn_grid.py
# Table 3 is now the three-encoder matched comparison; the old two-column
# FART-vs-GNN renderer is kept in the tree for provenance but not run.
run scripts/render_table_encoder_comparison.py
run scripts/render_table_molecular_prediction_two_column.py
