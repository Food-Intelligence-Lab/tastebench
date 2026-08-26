# Kaggle TasteBench (food-similarity competition)

Public Kaggle competition derived from the food-similarity task. The
release is a deliberately obfuscated mix of (a) NECTAR plant-based
products and (b) Taste Like distractor products, with all product
codes renumbered. Test labels are withheld; submissions are scored via
the Kaggle leaderboard.

> Live at: <https://www.kaggle.com/competitions/tastebench-challenge-2026>

The back-mapping (`product_code_map.csv`) is gitignored and never
shipped in this release. See `data/GATED.md` and the paper
for the privacy framing and its limitations.

## Layout

```
kaggle_tastebench/
├── generate_data/
│   ├── README.md
│   ├── generate_kaggle_datasets.ipynb   Build products.csv, ranking_pairs.csv, sample_submission.csv
│   ├── add_images_to_dataset.py         Copy renumbered product photos into dataset/images/
│   ├── taste_like_category_map.py       (gated — contains brand names; see data/GATED.md)
│   └── dataset/
│       ├── croissant.json               Croissant 1.0 metadata + RAI fields (generated, not committed)
│       ├── products.csv                 (committed, public on Kaggle)
│       ├── ranking_pairs.csv            (committed, public on Kaggle)
│       ├── sample_submission.csv        (committed, public on Kaggle)
│       ├── images/{category}/{code}.jpg (committed, public on Kaggle)
│       ├── solution.csv                 (gitignored — test labels)
│       └── product_code_map.csv         (gitignored — Kaggle ↔ NECTAR back-mapping)
└── predict/
    ├── llm_to_kaggle_submission.py      Convert LLM OOFs (NECTAR-keyed) → Kaggle submission
    │                                     (added so anyone can submit LLM baselines without
    │                                      re-running the LLM models; needs product_code_map.csv)
    └── gemini_{modality}.csv            Pre-generated Kaggle submissions for Gemini 3.1 Pro
                                          across all 7 modality combos. Requires
                                          product_code_map.csv (gated) to generate; not
                                          committed in the public release.
```

## Generating the competition data (first time)

Requires the gated NECTAR data and the product_code_map.csv, neither
of which ship in this release. From the project root:

```bash
jupyter nbconvert --to notebook --execute \
    kaggle_tastebench/generate_data/generate_kaggle_datasets.ipynb
python3 kaggle_tastebench/generate_data/add_images_to_dataset.py
```

## Generating an LLM submission from existing OOFs

The LLM baselines were run as part of the food-similarity area
(`food_similarity/zero_shot_baselines/`). To replay those predictions
as a Kaggle submission without re-running the LLMs:

```bash
python3 kaggle_tastebench/predict/llm_to_kaggle_submission.py \
    --model gemini_3_1_pro_preview \
    --modality ingredients_image \
    --out submission_gemini.csv
```

The script reads
`food_similarity/results/oof_predictions/llm_<model>_<modality>.csv`,
maps each NECTAR product code to its Kaggle code via
`generate_data/dataset/product_code_map.csv`, and writes a
`(test_id, higher_rated_product)` CSV that can be uploaded directly
to the Kaggle competition.

All 7 modality combos for Gemini 3.1 Pro are pre-generated and
committed under `predict/gemini_{modality}.csv` (one per row of the
LLM-ablation table in the paper):

| File | Modality |
|---|---|
| `gemini_ingredients.csv`                 | Ingr.        |
| `gemini_nutrition.csv`                   | Nutr.        |
| `gemini_image.csv`                       | Img.         |
| `gemini_ingredients_nutrition.csv`       | Ingr.+Nutr.  |
| `gemini_ingredients_image.csv`           | Ingr.+Img.   |
| `gemini_nutrition_image.csv`             | Nutr.+Img.   |
| `gemini_ingredients_nutrition_image.csv` | All          |

See the paper for per-modality pairwise accuracy and BCa CIs.

Each is 4006 pairs; ~23% of those are real LLM predictions (within
NECTAR), the remaining ~77% are deterministic (seed=42) coin flips
for pairs that involve at least one Taste Like distractor product.
