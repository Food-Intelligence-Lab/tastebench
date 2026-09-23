#!/usr/bin/env bash
# Re-render the human-baseline figure from the committed analysis
# artifacts in results/ (group_size_curve.csv, split_half_reliability.json,
# summary.json). No NECTAR data required.
#
# Outputs (written to paper/human_baseline/):
#   group_size_curve.pdf
#
# The accompanying .tex table (paper/human_baseline/human_baseline_table.tex)
# is committed as a static artifact; regenerating it requires running
# human_panelist_baseline.py with the gated NECTAR data.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"

# Per-run log, not a fixed /tmp path. Two areas verifying at once — or a
# reviewer re-running one while another is going — used to write the same
# file, and the `tail | sed` below then read a half-rewritten log. That
# crashed BSD sed outright (Abort trap: 6) and failed a build whose renders
# had all succeeded.
LOG="$(mktemp -t verify_human_baseline)"
trap 'rm -f "${LOG}"' EXIT


echo ""
echo ""
echo "[$(date +%H:%M:%S)] human_baseline: render_table_human_baseline.py"
( cd "${HERE}" && "${PY}" render_table_human_baseline.py ) > "${LOG}" 2>&1 \
  || { echo "FAILED — last 30 lines of log:"; grep -vE 'resource_tracker\.py|Cannot register .* for automatic cleanup|unknown resource type|raise ValueError\(' "${LOG}" | tail -60; exit 1; }
tail -8 "${LOG}" | sed 's/^/  /'

echo "[$(date +%H:%M:%S)] human_baseline: plot_human_baseline.py"
( cd "${HERE}" && "${PY}" plot_human_baseline.py ) > "${LOG}" 2>&1 \
  || { echo "FAILED — last 30 lines of log:"; grep -vE 'resource_tracker\.py|Cannot register .* for automatic cleanup|unknown resource type|raise ValueError\(' "${LOG}" | tail -60; exit 1; }
tail -3 "${LOG}" | sed 's/^/  /'
