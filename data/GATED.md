# Gated data

Five inputs are not redistributable in the public release:

1. **NECTAR sensory + ingredient + nutrition CSVs** — `data/consolidated_datasets/nectar_consolidated_*.csv`, plus the derivative label files `shared/data/nectar_product_labels.csv` and `food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv`. These contain participating brand identities and are NDA-protected by NECTAR.
2. **NECTAR product images** — `data/product_images/cropped/`. The packaging-cropped photos are NECTAR-owned and fall under the same NDA.
3. **Taste Like CPG product directory** — `data/taste_like/*.csv`. NECTAR-acquired (November 2024). Public availability of this data via the original taste-like.com website is uncertain post-acquisition; we ship it gated alongside NECTAR's own data so that the cross-reference deanonymization attack against the public Kaggle competition (matching public Taste Like features against the obfuscated bundle to identify NECTAR products by elimination) is not made trivial by this repository.

These files are deliberately excluded from version control (see `.gitignore`) and are not part of the released artifact.

4. **Derived feature matrix** — `food_similarity/data/product_features.pkl` (5.3 MB).
   Ships inside the gated bundle rather than the public release. It is the single
   input that lets Tier 1 run with no downloads at all: it already contains the
   compound, text and image embeddings, so FoodAtlas, FooDB, SMILES resolution,
   FART, Qwen3-Embedding and DINOv3 are all unnecessary. It is gated because its
   nutrition block holds the per-100g macros keyed by product code, and the
   public out-of-fold files carry the panel scores on that same key — publishing
   both would let a product's composition be matched to public CPG listings and
   its score attributed to a brand.

5. **Withheld product-code map** — `data/product_code_map.json` (4.5 KB).
   Maps each product's ORIGINAL NECTAR code to the code it carries in the public
   release. Step 4aa of the release build re-keys every product code from a
   cryptographically random per-category permutation, so that a published row
   cannot be walked back to a manufacturer's product. This file is the only way
   back across that re-key, and it exists so Tier 1 can join at all: without it
   `data/pairs.csv` and the LLM out-of-fold files share no key with
   `food_similarity/data/product_features.pkl`, and the retraining path stops
   dead. It carries the same NDA handling as the four items above and **must
   never be published** — releasing it would undo the de-identification of every
   artifact in every version already distributed, retroactively. `.gitignore`
   refuses it by name at any depth, so unpacking the bundle inside a working copy
   cannot commit it by accident, and the release build scans for it by name.

## What is *not* gated

- The **TasteBench Kaggle competition** dataset (`tastebench-challenge-2026`), which is a deliberately obfuscated, zero-shot release: NECTAR products are interleaved with samples from the Taste Like CPG product directory, all product codes are renumbered, and labels are withheld. See `kaggle_tastebench/generate_data/generate_kaggle_datasets.ipynb` for the construction.
- **All rendered tables and analysis artifacts**: out-of-fold prediction CSVs (`food_similarity/results/oof_predictions/`), human-baseline tables and figures (`human_baseline/results/`), CI tables, and rendered LaTeX (`paper/`). These are sufficient to verify every numerical claim and re-render every table and figure without access to the gated data.
- **Public datasets** the pipeline depends on: the FART splits (fetched via `python3 -m molecular.src.data.download`), FoodAtlas and FooDB (fetched via `data/download_public.sh`).

## Panelist privacy: what is aggregated before release

The gating above is about NECTAR's commercial data. This section is about the
people who did the tasting, which is a separate question.

The raw sensory ratings (`data/consolidated_datasets/nectar_consolidated_sensory_rating.csv`)
are per-panelist and carry a `respondent` column. That file is gated, so those
ids are not redistributed. But one *derived* file was individual-level and was
released: `human_baseline/results/individual_loo_accuracy.csv`, one row per
panelist per block, 4,664 rows, giving each panelist's pairwise accuracy,
Spearman correlation, and recall against the leave-one-out panel mean.

It shipped with the `respondent` column, and those ids were global rather than
block-local. 627 distinct panelists appeared across the 49 blocks, 278 of them in
more than one product category and one in 23 of the 24. A reader could therefore
assemble a per-person performance profile spanning most of the benchmark. There
are no demographics and no direct identifiers anywhere in the release, so the
re-identification risk was low; the objection is narrower and does not depend on
re-identification. Panelists consented to a sensory study whose published output
is aggregate agreement statistics. A durable per-person skill ranking across 24
product categories is not that, and we should not have been the ones to decide it
was close enough.

**The `respondent` column is no longer released.** Every row and every value is
unchanged — the median panelist accuracy (.650), the interquartile range
(.50–.75), the 4,664 observations, and the inter-rater and group-size statistics
are all exactly as previously published. What is gone is the ability to join two
rows as the same person. Dropping the column costs no analysis, because
`(category, block_id, respondent)` was unique across all 4,664 rows: each row was
already exactly one panelist-block observation, so within a block the column was a
bare row key with no analytic content. Its only function in the released file was
the cross-block join.

Two details worth stating, since both are the kind of thing that quietly undoes a
de-identification:

- **Row order.** The rows were previously ordered by ascending global respondent
  id within each block, so the row order alone republished the ranking the column
  was meant to withhold (Spearman of position against original id was +1.00 in all
  49 blocks). Rows are now sorted lexicographically on the released columns only,
  making the order a function of data the release already publishes. Re-deriving
  the file from a randomly shuffled source produces byte-identical output, so the
  released file encodes nothing about the original ordering. A seeded shuffle was
  rejected instead: the generator is public, so a published seed would invert
  straight back to the original order.
- **The panelist count.** The paper reports *n* = 627 panelists. A count of
  distinct people is by definition not recoverable from a file with no person key,
  so `render_table_human_baseline.py` now reads it from
  `human_baseline/results/summary.json` (`individual_panelist.n_panelists`) rather
  than recomputing it. That one number is attested rather than recomputed. The
  alternative was to keep the identifier, and re-deriving the count from anything
  else in the release would have silently printed a different *n*.

`human_baseline/human_panelist_baseline.py` drops the column at write time, so
regenerating from the gated bundle reproduces the de-identified file rather than
reintroducing the key. The pre-release NDA disclosure scan enforces this:
it scans every released CSV for panelist keys and pins this file's schema to its
eight columns, so a re-added or renamed key fails the scan.

Every other released human-baseline artifact was checked and is aggregated:
`inter_rater_reliability.csv` (per-block α and W, with `n_respondents` as a
headcount), `group_size_curve.csv` (simulated panels of size *k*, no ids and no
block labels), `per_category_comparison.csv`, `pair_difficulty.csv`,
`meta_category.csv`, `comparison_table.csv`, and the `results/verification/`
sidecars `rater_pair_signs.csv`, `alpha_coincidence.csv`,
`split_half_iterations.csv`, and `block_group_stats.csv`. None carries a panelist
key or any column that joins rows of the same person.

## How to request access

Request access via the Google Form linked in the paper's Datasets section. Approval grants access to a private Kaggle Dataset holding the four NECTAR CSVs plus the cropped product images and the Taste Like CSVs. The Croissant metadata for that bundle ships at `croissant/nectar.json`.

## Downloading the bundle

1. Open the Kaggle Dataset URL in your browser. (Complete the Google Form first to receive the URL.)
2. Click **Download** on the Kaggle Dataset page. Depending on your browser and Kaggle's delivery path, you will end up with one of:
   - A folder named `archive/` directly (some setups auto-extract on download), or
   - A zip file `archive.zip`, which you'll need to extract: `unzip archive.zip -d archive`.

   In either case, the resulting `archive/` folder contains four NECTAR CSVs plus image and Taste Like contents. The image and Taste Like contents may appear as `images.zip` and `taste_like.zip` (Kaggle's native bundle layout) or as already-extracted `images/` and `taste_like/` directories — the unpack script in the next step accepts both forms.

## Unpacking into the pipeline

The Kaggle Dataset uses the flat layout convention: four CSVs at the top level plus `images.zip` (cropped product photos) and `taste_like.zip` (Taste Like CPG directory). The recommended path is the one-command helper script.

From the project root (the `tastebench/` directory):

```bash
bash data/unpack_nectar_bundle.sh ~/Downloads/archive
```

The script copies each CSV to its target subdirectory, places the image and Taste Like contents (extracting `images.zip`/`taste_like.zip` or copying the already-extracted directories), and verifies every expected file is present. If you would rather copy by hand (e.g., to debug a failure), the file-to-folder mapping is:

| File in the downloaded bundle | Target folder (relative to project root) |
|---|---|
| `nectar_consolidated_ingredients_nutrition.csv` | `data/consolidated_datasets/` |
| `nectar_consolidated_sensory_rating.csv` | `data/consolidated_datasets/` |
| `nectar_product_labels.csv` | `shared/data/` |
| `product_labels_manually_cleaned.csv` | `food_similarity/zero_shot_baselines/data/` |
| `images.zip` (extract contents) | `data/product_images/` (creates `cropped/<year>/<category>/<product_code>/<view>.jpg`) |
| `taste_like.zip` (extract contents) | `data/taste_like/` (5 CSVs from the Taste Like CPG product directory) |

After unpacking, the artifacts sit at:

```
data/consolidated_datasets/nectar_consolidated_ingredients_nutrition.csv
data/consolidated_datasets/nectar_consolidated_sensory_rating.csv
data/product_images/cropped/<year>/<category>/<product_code>/<view>.jpg
shared/data/nectar_product_labels.csv
food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv
data/taste_like/*.csv
```

These are the paths `food_similarity/prepare_data.py`, `human_baseline/human_panelist_baseline.py`, and the LLM zero-shot baselines read from. With the bundle in place, the Tier 1 retraining path (`bash food_similarity/reproduce.sh`) retrains every baseline and re-renders all artifacts. It has only been run in the environment this release was built in, so expect agreement at the three decimals the tables print rather than byte-identical files: the BCa bootstrap is seeded (42), but retraining picks up whatever BLAS and LightGBM build the environment supplies.

The Tier 0 verification path (`bash verify_paper.sh`) does NOT require this bundle — it works from the committed out-of-fold prediction CSVs.

## Verifying without the bundle

Every rendered number can be checked from the committed artifacts alone:

- `food_similarity/results/oof_predictions/` contains every leave-one-out prediction the food-similarity tables depend on (including the LLM predictions, which cost real money to regenerate via the OpenRouter API).
- `food_similarity/scripts/render_*.py` and `molecular/scripts/render_*.py` re-render every table from those OOFs and the molecular parquet predictions.
- `human_baseline/results/` contains the human-baseline summary, group-size curve data, per-category comparisons, and split-half reliability JSON used to render the human-baseline table and the group-size curve figure.
- `molecular/results/` contains the GNN grid-search outputs (12 trained checkpoints + per-run metrics) used for the molecular tables.

The entire numerical content of every rendered table is committed and rerunnable from public inputs. The gated NECTAR raw ratings are needed only to regenerate the OOFs from scratch.
