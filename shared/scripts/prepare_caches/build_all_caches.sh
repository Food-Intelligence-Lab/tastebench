#!/usr/bin/env bash
# Build every derived cache listed in results/input_cache_sha256.txt.
#
#   bash shared/scripts/prepare_caches/build_all_caches.sh          # build
#   bash shared/scripts/prepare_caches/build_all_caches.sh --check  # report only
#
# None of this is needed for Tier 0: `verify_paper.sh` renders every table from
# committed out-of-fold predictions. It is also not needed for Tier 1 if you use
# the feature matrix shipped in the gated bundle — see README.md.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
PY="${PY:-python3}"
CACHES="${ROOT}/shared/data/caches"
MANIFEST="${ROOT}/food_similarity/results/input_cache_sha256.txt"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

mkdir -p "${CACHES}"

have() {  # name -> 0 if present and hash-matching (when the manifest knows it)
    local f="${CACHES}/$1"
    [ -f "${f}" ] || return 1
    if [ -f "${MANIFEST}" ]; then
        local want
        want=$(grep -F " shared/data/caches/$1" "${MANIFEST}" 2>/dev/null | awk '{print $1}' | head -1)
        if [ -n "${want}" ]; then
            local got
            got=$(shasum -a 256 "${f}" | awk '{print $1}')
            [ "${got}" = "${want}" ] || { echo "    hash mismatch: $1"; return 1; }
        fi
    fi
    return 0
}

step() {  # <cache file> <description> <command...>
    local name="$1"; shift
    local desc="$1"; shift
    if have "${name}"; then
        echo "  ok      ${name}  (${desc})"
        return 0
    fi
    if [ "${CHECK}" -eq 1 ]; then
        echo "  MISSING ${name}  (${desc})"
        return 0
    fi
    echo "  build   ${name}  (${desc})"
    ( cd "${ROOT}" && "$@" ) || { echo "  FAILED  ${name}"; return 1; }
}

echo "caches under ${CACHES}"
rc=0
step smiles_cache.csv "FoodAtlas + PubChem -> SMILES" \
    "${PY}" shared/scripts/prepare_caches/prepare_smiles_cache.py || rc=1
step chebi_smiles_cache.csv "ChEBI-only compounds -> SMILES" \
    "${PY}" shared/scripts/prepare_caches/resolve_chebi_smiles.py || rc=1
step fart_compound_embeddings.pkl "FART compound encoder (431 MB; needs FART weights)" \
    "${PY}" shared/scripts/prepare_caches/prepare_fart_embeddings.py || rc=1
# molecular.src.embed.generate_cache takes three REQUIRED arguments. Invoking it
# bare — as this script and the README both used to — exits 2 on argparse before
# doing anything, so the step could never have built the cache it claims to.
# The checkpoint is the val-best D-MPNN run, shipped as a symlink in the grid.
step taste_gnn_best_compound_embeddings.pkl "D-MPNN embeddings (177 MB; uses the shipped checkpoint)" \
    "${PY}" -m molecular.src.embed.generate_cache \
        --checkpoint   molecular/results/grid/best/ckpt.pt \
        --source_cache shared/data/caches/fart_compound_embeddings.pkl \
        --output_pkl   shared/data/caches/taste_gnn_best_compound_embeddings.pkl || rc=1

# Optional: backs no paper table, and needs an LLM endpoint.
if [ "${CHECK}" -eq 1 ]; then
    have sensory_descriptions.csv \
        && echo "  ok      sensory_descriptions.csv  (optional)" \
        || echo "  MISSING sensory_descriptions.csv  (optional; needs an LLM endpoint; skipped by default)"
else
    echo "  skip    sensory_descriptions.csv  (optional; needs an LLM endpoint — run"
    echo "          shared/scripts/generate_sensory_descriptions.py explicitly if wanted)"
fi

echo
[ "${rc}" -eq 0 ] && echo "done." || echo "one or more caches failed to build."
exit "${rc}"
