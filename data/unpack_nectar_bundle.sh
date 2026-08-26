#!/usr/bin/env bash
# Move the downloaded NECTAR Kaggle bundle into the on-disk paths the
# food-similarity pipeline expects. Run from the project root
# (the `tastebench/` directory after unzipping the supplementary).
#
# Usage:
#     bash data/unpack_nectar_bundle.sh <path-to-extracted-bundle>
#
# Where <path-to-extracted-bundle> is the directory you get after
# extracting the Kaggle Dataset zip
# (e.g., ~/Downloads/tastebench-nectar-bundle/).
#
# The bundle layout is the Kaggle-conventional flat form:
#     <bundle>/
#       nectar_consolidated_ingredients_nutrition.csv
#       nectar_consolidated_sensory_rating.csv
#       nectar_product_labels.csv
#       product_labels_manually_cleaned.csv
#       product_code_map.json
#       taste_like_category_map.py   (Taste Like → NECTAR category mapping)
#       images.zip       (cropped/<year>/<category>/<code>/<view>.jpg)
#       taste_like.zip   (Taste Like CPG product directory CSVs)
#
# This script moves each artifact to its target path under the project root.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: bash data/unpack_nectar_bundle.sh <path-to-extracted-bundle>" >&2
    exit 1
fi

SRC="$(cd "$1" && pwd)"
NEURIPS_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Sanity-check the expected files are all present in the source.
# The four NECTAR CSVs are always required as files.
EXPECTED_CSVS=(
    "nectar_consolidated_ingredients_nutrition.csv"
    "nectar_consolidated_sensory_rating.csv"
    "nectar_product_labels.csv"
    "product_labels_manually_cleaned.csv"
)
for f in "${EXPECTED_CSVS[@]}"; do
    if [ ! -f "${SRC}/${f}" ]; then
        echo "ERROR: missing ${f} in ${SRC}" >&2
        exit 1
    fi
done

# The withheld product-code map is the fifth gated item. It is small and easy to
# leave behind, and a bundle without it unpacks into a tree where Tier 1 stops at
# Phase 4 with nothing joining anything — so say so here instead.
if [ ! -f "${SRC}/product_code_map.json" ]; then
    echo "ERROR: missing product_code_map.json in ${SRC}" >&2
    echo "       It maps the bundle's original NECTAR product codes to the codes" >&2
    echo "       the public release carries. Without it nothing in the release can" >&2
    echo "       be joined to this bundle and Tier 1 cannot run. It is item 5 of" >&2
    echo "       the gated bundle (data/GATED.md); re-download the bundle." >&2
    exit 1
fi

# images and taste_like may arrive either as .zip files (Kaggle's
# native bundle layout) or as pre-extracted directories (some Kaggle
# download paths auto-extract inner zips). Accept either form.
if [ ! -f "${SRC}/images.zip" ] && [ ! -d "${SRC}/images" ]; then
    echo "ERROR: missing both images.zip and images/ in ${SRC}" >&2
    exit 1
fi
if [ ! -f "${SRC}/taste_like.zip" ] && [ ! -d "${SRC}/taste_like" ]; then
    echo "ERROR: missing both taste_like.zip and taste_like/ in ${SRC}" >&2
    exit 1
fi

# Make target directories (most exist already, but be safe).
mkdir -p "${NEURIPS_DIR}/data/consolidated_datasets"
mkdir -p "${NEURIPS_DIR}/shared/data"
mkdir -p "${NEURIPS_DIR}/food_similarity/zero_shot_baselines/data"
mkdir -p "${NEURIPS_DIR}/data/product_images"
mkdir -p "${NEURIPS_DIR}/data/taste_like"

echo "[1/8] consolidated_datasets/ CSVs..."
cp "${SRC}/nectar_consolidated_ingredients_nutrition.csv" "${NEURIPS_DIR}/data/consolidated_datasets/"
cp "${SRC}/nectar_consolidated_sensory_rating.csv"        "${NEURIPS_DIR}/data/consolidated_datasets/"

echo "[2/8] shared/data/nectar_product_labels.csv..."
cp "${SRC}/nectar_product_labels.csv" "${NEURIPS_DIR}/shared/data/"

echo "[3/8] food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv..."
cp "${SRC}/product_labels_manually_cleaned.csv" "${NEURIPS_DIR}/food_similarity/zero_shot_baselines/data/"

echo "[4/8] data/product_images/cropped/ from images..."
if [ -f "${SRC}/images.zip" ]; then
    unzip -o -q "${SRC}/images.zip" -d "${NEURIPS_DIR}/data/product_images/"
else
    cp -R "${SRC}/images/." "${NEURIPS_DIR}/data/product_images/"
fi

echo "[5/8] data/taste_like/ from taste_like..."
if [ -f "${SRC}/taste_like.zip" ]; then
    unzip -o -q "${SRC}/taste_like.zip" -d "${NEURIPS_DIR}/data/taste_like/"
else
    cp -R "${SRC}/taste_like/." "${NEURIPS_DIR}/data/taste_like/"
fi

echo "[6/8] data/product_code_map.json (WITHHELD — never publish this)..."
cp "${SRC}/product_code_map.json" "${NEURIPS_DIR}/data/product_code_map.json"

echo "[7/8] kaggle_tastebench/generate_data/taste_like_category_map.py..."
if [ -f "${SRC}/taste_like_category_map.py" ]; then
    cp "${SRC}/taste_like_category_map.py" "${NEURIPS_DIR}/kaggle_tastebench/generate_data/"
else
    echo "  WARNING: taste_like_category_map.py not in bundle — Kaggle competition" >&2
    echo "  regeneration (generate_kaggle_datasets.ipynb) will not work." >&2
    echo "  This does not affect model training or table reproduction." >&2
fi

echo "[8/8] verifying placements..."
for path in \
    "data/consolidated_datasets/nectar_consolidated_ingredients_nutrition.csv" \
    "data/consolidated_datasets/nectar_consolidated_sensory_rating.csv" \
    "shared/data/nectar_product_labels.csv" \
    "food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv" \
    "data/product_code_map.json"
do
    [ -f "${NEURIPS_DIR}/${path}" ] || { echo "ERROR: missing after copy: ${path}" >&2; exit 1; }
done
N_IMG=$(find "${NEURIPS_DIR}/data/product_images/cropped" -name '*.jpg' | wc -l | tr -d ' ')
[ "${N_IMG}" -gt 0 ] || { echo "ERROR: no JPGs found under data/product_images/cropped/" >&2; exit 1; }
N_TL=$(find "${NEURIPS_DIR}/data/taste_like" -name '*.csv' | wc -l | tr -d ' ')
[ "${N_TL}" -gt 0 ] || { echo "ERROR: no Taste Like CSVs found under data/taste_like/" >&2; exit 1; }

echo
echo "OK. Bundle unpacked into ${NEURIPS_DIR}/. Found ${N_IMG} cropped JPGs and ${N_TL} Taste Like CSVs."
echo "Tier 1 retraining is now runnable: see README.md."
