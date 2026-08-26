# TasteBench (NeurIPS 2026)

Code and rendering scripts accompanying the TasteBench paper
(NeurIPS 2026 Evaluations and Datasets Track).

> All commands below are run from inside this directory (the
> `tastebench/` folder). `cd` into it before running anything.

## Reproduction guide

Every numerical claim is reproducible at four tiers, each requiring
strictly more than the last. Tier 0 has zero prerequisites and verifies
all rendered tables and figures from committed artifacts; tiers 1–3 add
the dependencies needed to regenerate those artifacts from scratch.

**This release ships all model artifacts needed for inference**:
the 12 trained GNN checkpoints (one per hyperparameter-grid configuration)
at `molecular/results/grid/run_*/ckpt.pt`, all leave-one-out prediction
CSVs from the food-similarity baselines (including the paid LLM
predictions) at `food_similarity/results/oof_predictions/`, the
human-baseline analysis artifacts at `human_baseline/results/`, and the
rendered LaTeX tables and figures at `paper/`. Tier 0 verification is
self-contained: no downloads, no GPU, no API calls required.

### Tier 0: Verify rendered numbers (no downloads, no API spend)

```bash
bash verify_paper.sh
```

Re-renders every table and the human-baseline figure from the committed
OOF prediction CSVs in `food_similarity/results/oof_predictions/`, the
parquet predictions in `molecular/results/`, and the human-baseline
analysis artifacts in `human_baseline/results/`. Runtime is dominated by
the BCa bootstrap (10,000 resamples), with `render_table_ablation_features.py`
as the long pole. Pass `--check-diff` to additionally verify all 20
re-rendered outputs: every `paper/**/*.tex` table plus the five chart
files: against the SHA-256 manifest shipped at
`paper/RENDERED_SHA256.txt`. It fails on the first byte that differs, and
it needs no git checkout: the manifest is generated from the shipped
outputs, so it works in a plain unpacked copy. The figures are covered
because their renderers omit `/CreationDate`, which makes the PDFs
byte-reproducible.

### Tier 1: Verify or retrain from `reproduce.sh`

`reproduce.sh` adapts to whatever data is available:

```bash
cd food_similarity && bash reproduce.sh
```

**Without NECTAR** (bare public clone): training phases (3–6) are
skipped. The pipeline independently recomputes every BCa CI from the
committed out-of-fold predictions, then re-renders all tables — a
stronger check than `verify_paper.sh` because it re-derives the CIs
rather than reading cached ones. No downloads, no API key.

**With NECTAR**: retrains every supervised LOOCV baseline (Bradley–Terry,
Hierarchical BT, Ridge, Kernel RankSVM, LightGBM), refits the nested
NNLS meta-learner, recomputes BCa CIs, and re-renders the
food-similarity tables. Distance baselines (cosine, L2 MMRF) are
included. Requires NECTAR access, requested through the Google Form
linked in the Datasets section of the paper:

```bash
# 1. Drop the gated bundle into the right place:
bash data/unpack_nectar_bundle.sh <path-to-downloaded-bundle>

# 2. Fetch public datasets and run the pipeline:
bash data/download_public.sh                       # FoodAtlas v4.0 + FooDB
cd food_similarity && bash reproduce.sh
```

See [`data/GATED.md`](data/GATED.md) for the full file-by-file
breakdown, NDA terms, and the four post-unpack paths.

The human-baseline analysis is a separate script that runs on the raw
NECTAR ratings:

```bash
cd human_baseline
python3 human_panelist_baseline.py
python3 plot_human_baseline.py
```

### Tier 2: Retrain the GNN from scratch (needs network access; no GPU)

The FART split CSVs are **not** in this release: they are third-party
data fetched on demand and gitignored, so the first step is to download
them (~2 MB from `raw.githubusercontent.com`, pinned to a git SHA and
verified against the hashes and per-class counts in
`molecular/data/PROVENANCE.md`):

```bash
python3 -m molecular.src.data.download \
    --sha bde90e6562ce5d248e76af791fab29ffc9ae901b   # -> molecular/data/splits/
```

Then the 12-configuration hyperparameter grid:

```bash
bash molecular/scripts/submit_grid.sh dmpnn_grid
python3 -m molecular.src.train.select_best_and_evaluate \
    --results_dir <grid-output-dir> --skip_nectar
bash molecular/scripts/aggregate_grid.sh
```

`--results_dir` is required; the shipped grid is at
`molecular/results/grid/`. Without `--skip_nectar`, the script continues
into the NECTAR encoder-transfer step, which is Tier 1 and needs the
gated bundle.

The D-MPNN itself trains on CPU; no GPU is required. Regenerating the
FART Augmented reference row of Table 2 is a separate step and pulls the
`FartLabs/FART_Augmented` weights from HuggingFace:

```bash
python3 -m molecular.src.eval.evaluate --model_type fart_augmented \
    --test_csv molecular/data/splits/fart_test.csv \
    --output_dir molecular/results/fart_augmented_test
```

**All 12 trained GNN checkpoints are shipped** at
`molecular/results/grid/run_*/ckpt.pt` (~1.2 MB each, ~14.4 MB total),
together with the per-run `config.yaml` and `val_metrics.json`, and both
sets of test predictions are committed as parquet. This tier is only
needed if you want to verify training from the raw FART splits;
inference and embedding extraction with the committed checkpoints run on
CPU and need no download.

### Tier 3: Regenerate LLM baselines (needs OpenRouter API; ~$300)

Gemini 3.1 Pro and Qwen 3.5 397B-A17B predictions are generated via the
OpenRouter API. The committed OOFs in
`food_similarity/results/oof_predictions/llm_*.csv` are the source of
the LLM numbers in every table. Regenerating them from scratch cost
approximately $300 in OpenRouter API spend across both models and the
LLM modality ablation grid (at the pricing in force when those runs were made).

There is one config per (model, input modality). The modality names are
`ingredients`, `nutrition`, `image`, `ingredients_nutrition`,
`ingredients_image`, `nutrition_image` and
`ingredients_nutrition_image`: seven configs per model, which together
are the modality ablation grid:

```bash
export OPENROUTER_API_KEY=...
cd food_similarity/zero_shot_baselines
python3 run.py --config configs/llm/gemini_3_1_pro_preview/ingredients_nutrition_image.yaml
python3 run.py --config configs/llm/qwen3_5_397b_a17b/ingredients_nutrition_image.yaml

# ...or the whole grid for one model:
for c in configs/llm/gemini_3_1_pro_preview/*.yaml; do python3 run.py --config "$c"; done
```

### Summary

| Component | Tier | Needs NECTAR | Notes |
|---|---|---|---|
| Re-render every table & figure | 0 | no | runs from committed OOFs |
| Distance baselines (MMRF cos / L2) | 1 | yes | |
| Supervised pairwise baselines (BT, HBT, Ridge, KSVM, LGBM) | 1 | yes | |
| NNLS meta-learner ensembles | 1 | yes | |
| Human-panelist baseline analysis | 1 | yes | |
| GNN inference / embedding extraction | 1 | yes | uses shipped ckpts |
| GNN training from scratch | 2 | no | CPU only, but needs network: FART splits from GitHub, and HuggingFace weights for the FART Augmented row |
| LLM baselines (Gemini, Qwen) regeneration | 3 | yes | ~$300 OpenRouter API spend |

## Layout

The repo is split into five areas, one per paper section, plus shared
infrastructure. Each area has a top-level `README.md` describing its
inputs, outputs, and verify path.

```
tastebench/
├── verify_paper.sh           Top-level umbrella: re-renders every table & figure
│                              from committed OOFs (no NECTAR, no training)
│
├── food_similarity/          Food-level pairwise ranking (results + ablations)
│   ├── README.md
│   ├── verify.sh             Re-render this area's tables/figures only
│   ├── reproduce.sh          End-to-end retrain (heavyweight; needs NECTAR)
│   ├── prepare_data.py       Build product_features.pkl from raw NECTAR
│   ├── data/                 LOOCV loaders + cached features
│   ├── models/               BT, hierarchical-BT, ridge, kernel-RankSVM, LightGBM, NNLS
│   ├── train/                LOOCV runners + nested NNLS meta-learner + LLM integration
│   ├── evaluation/           Metrics + BCa bootstrap
│   ├── zero_shot_baselines/  LLM (Gemini, Qwen) + distance (cosine, L2) baselines
│   ├── scripts/              Render scripts for this area's tables/figures
│   └── results/              oof_predictions/, verification/, cis_*.csv,
│                              input_cache_sha256.txt (rendered tables land in paper/)
│
├── molecular/                Molecular taste classification (results + ablations)
│   ├── README.md
│   ├── verify.sh
│   ├── data/                 PROVENANCE.md for the FART splits; splits/ appears
│   │                          here after `molecular.src.data.download` (gitignored)
│   ├── src/                  D-MPNN data, models, train, eval, embed
│   ├── configs/              dmpnn_{base,grid,scaffold}.yaml
│   ├── scripts/              Render scripts + training shell drivers
│   └── results/              grid/, fart_augmented_test/, tables_csv/, cis_*.csv
│
├── kaggle_tastebench/        Public Kaggle competition for the food-similarity task
│   │                        https://www.kaggle.com/c/tastebench-challenge-2026
│   ├── README.md
│   ├── generate_data/        Obfuscated NECTAR + Taste Like construction; competition CSVs
│   └── predict/              LLM → Kaggle-format submission converter (added in phase D)
│
├── kaggle_molecular_taste/   Public Kaggle competition for the molecular task
│   │                        https://www.kaggle.com/c/molecular-taste-classification-2026
│   ├── README.md
│   ├── generate_kaggle_datasets.py
│   └── dataset/              train.csv, val.csv, test.csv, sample_submission.csv (solution.csv gated)
│
├── human_baseline/           Untrained human-panelist baseline on the food-similarity task
│   ├── README.md
│   ├── verify.sh             Re-render figures from committed CSVs
│   ├── human_panelist_baseline.py   → results/{summary,split_half_reliability}.json, group_size_curve.csv,
│   │                                  human_baseline_table.tex (needs NECTAR)
│   ├── plot_human_baseline.py       → paper/human_baseline/group_size_curve.pdf
│   └── results/
│
├── paper/          Rendered table & figure outputs consumed by the paper
│   ├── human_baseline/       human_baseline_table.tex, group_size_curve.pdf
│   ├── model_results_tables/ table_results, table_per_category, table_ablation_*, …
│   ├── molecular_prediction/ table_molecular_prediction, table_gnn_*, …
│   ├── appendix_tables/      table_micro_macro, table_brand_blind, …
│   ├── charts/               chart_gnn_per_model, chart_molecular_prediction
│   └── two_column/           two-column layout variants of main tables
│
├── shared/                   Cross-area code + cached embeddings
│   ├── oof_io.py             read_oof() — the ONE way to read an out-of-fold file
│   ├── check_oof_readers.py  gate: every OOF reader accepts both target forms
│   ├── cache_provenance.py   CI-cache sha256 sidecars + exact-float CSV I/O
│   ├── compound_mapping/     FoodAtlas → SMILES + FooDB concentrations
│   ├── scripts/              prepare_smiles_cache.py, prepare_fart_embeddings.py, …
│   └── data/                 Manually-curated labels + cached embeddings (gitignored;
│                             not hosted anywhere — rebuilt locally, see "Caches")
│
└── data/                     Inputs (see "Data: public vs. gated" below)
    ├── GATED.md              Gating policy + Google Form access flow
    ├── download_public.sh    Fetches FoodAtlas + FooDB
    ├── taste_like/           Taste Like CSVs (gated — see data/GATED.md)
    ├── food_atlas/v4.0/      FoodAtlas v4.0 (82 MB — download)
    ├── foodb_2020_04_07_csv/ FooDB CSV release (953 MB — download)
    ├── consolidated_datasets/  NECTAR CSVs (gated — request via Form)
    └── product_images/       NECTAR product photos (gated — request via Form)
```

## Artifact-to-renderer mapping

| Artifact | Producer |
|---|---|
| Combined food-similarity results (Pw.Acc + ρ + R@k) | `food_similarity/scripts/render_table_results.py` |
| Molecular taste classification | `molecular/scripts/render_table_molecular_prediction.py` |
| GNN per-model transfer | `molecular/scripts/render_table_gnn_per_model.py` |
| Per-category model comparison | `food_similarity/scripts/render_table_per_category.py` |
| Per-category NNLS metrics | `food_similarity/scripts/render_table_per_category_nnls.py` |
| Per-model NNLS swap | `food_similarity/scripts/render_table_per_model_nnls.py` |
| Feature-subset ablation | `food_similarity/scripts/render_table_ablation_features.py` |
| LLM input ablation | `food_similarity/scripts/render_table_ablation_llm.py` |
| GNN hyperparameter grid | `molecular/scripts/render_table_gnn_grid.py` |
| Molecular per-class breakdown | `molecular/scripts/render_table_molecular_per_class.py` |
| Human-panelist baseline | `human_baseline/human_panelist_baseline.py` |
| Group-size curve figure | `human_baseline/plot_human_baseline.py` |

## Data: public vs. gated

NECTAR ratings, ingredient/nutrition CSVs, product images, and the
Taste Like CPG product directory are not redistributed with this
artifact; see `data/GATED.md` for the access procedure.

**Public, fetched by `data/download_public.sh`:**

| Dataset | Source | Target path |
|---|---|---|
| FoodAtlas v4.0 | https://www.foodatlas.ai/food-composition-downloads | `data/food_atlas/v4.0/` |
| FooDB 2020-04-07 | https://foodb.ca/downloads | `data/foodb_2020_04_07_csv/` |

Data derived from these sources and redistributed inside this release is attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).


**Public, fetched on demand (gitignored, so not in this tree):**

| Dataset | Fetched by | Target path |
|---|---|---|
| FART train/val/test splits (~2 MB) | `python3 -m molecular.src.data.download --sha bde90e65…` | `molecular/data/splits/` |
| FART Augmented weights | `transformers` `from_pretrained("FartLabs/FART_Augmented")` | HuggingFace cache |

Hashes, row counts and per-class counts for the splits are pinned in
`molecular/data/PROVENANCE.md`, and the downloader verifies against them.

**Public, vendored in repo:**

| Dataset | Where |
|---|---|
| Kaggle competition obfuscated dataset | `kaggle_tastebench/generate_data/dataset/` |

**Gated, see [`data/GATED.md`](data/GATED.md):** NECTAR sensory + ingredient
+ nutrition CSVs, NECTAR product images, the derivative label files
(`nectar_product_labels.csv`, `product_labels_manually_cleaned.csv`), and the
Taste Like CPG product directory CSVs. Access is requested through the Google
Form linked in the paper's Datasets section; you then receive a bundle that
unpacks into the existing `data/` paths, after which the pipeline retrains and
re-renders every result.

Tier 1 has been run end to end against the gated bundle, and it reproduces the
published numbers **exactly**. All 110 supervised and distance out-of-fold files
and all 6 ensembles come back at |delta| = 0.000000 on every metric: pairwise
accuracy, Spearman, Kendall tau and recall@1/2/3: including the headline
BT+Gemini NNLS at .682888, whose nested base was refit from scratch (23,005 inner
folds, cache bypassed) rather than read from cache.

One caveat, and it is about WHERE you run it. The out-of-fold files here carry the
ORIGINAL NECTAR product codes; the released tree carries re-keyed public ones, and
`data/loocv.py` re-keys the feature matrix on load to match. That changes the row
order the leave-one-out loop sees, and three models are sensitive to it:
Bradley-Terry (.610160 -> .609091), Kernel RankSVM (.617647 -> .619786) and
LightGBM (.557754 -> .547059). The shifts are exactly -1/935, +2/935 and -10/935
pairwise comparisons. Ridge is permutation-invariant and never moves, which is the
tell: this is row order, not training noise.

So retraining inside a built public tree reproduces the paper to three decimals,
and retraining in this source layout reproduces it exactly. It is not
nondeterminism: LightGBM already pins `random_state`, `deterministic`,
`force_row_wise` and `num_threads=1`, and three fits of the same data give
byte-identical predictions.

One thing to expect when comparing a Tier 1 rerun against the shipped files: the
target column here is **`true_rank`**, not the panel mean. The released
out-of-fold files carry each product's within-category rank, because every metric
reported in the paper depends only on the order of the targets and not their
values, so publishing the ordering and withholding the ratings costs nothing
(verified: all six metrics identical, max |difference| 0.000e+00). A rerun from
raw NECTAR naturally produces the panel means, so rank-transform them within each
category before diffing: `df.groupby("category")["true_score"].rank(method="average")`,
which is exactly what the release build applies. Use `method="average"`: it
preserves ties, and 6 of the committed out-of-fold files contain real ones.

`predicted_score` is untouched, so predictions compare directly.

Without the gated bundle, the Tier 0 path (`bash verify_paper.sh`)
still re-renders every table from the committed OOF predictions and
parquet predictions in this repository. See `data/GATED.md`.

## Caches

`shared/data/caches/` holds derived embeddings consumed by
`food_similarity/prepare_data.py` and `molecular/`. Most are hundreds of MB and
are excluded from git.

**None of them is hosted for download.** There is no mirror, no release asset and
no bucket: every one is either shipped in this repository, rebuilt locally by a
script below, or, for four of them, not obtainable at all. Earlier revisions of
this file described the cache directory as "hosted"; that was never true and the
word has been removed rather than a link invented for it.

None of this is needed for Tier 0. `bash verify_paper.sh` renders every table
from the committed out-of-fold predictions and touches no cache.

### What the manifest reports

In this release, `food_similarity/results/input_cache_sha256.txt` carries 24
SHA-256 entries and `food_similarity/scripts/check_input_caches.py` checks them.
On a clean checkout it prints `verified: 15   missing: 9`, which is the expected
state, not a problem to fix:

* **15 verified**: 14 committed artifacts (the eight verification-bundle files
  and the six category-balanced appendix CSVs; these are results, not caches, and
  need no build step) plus `compound_slice.parquet`, the one cache that ships.
* **9 missing**: every other `shared/data/caches/` entry. Five can be rebuilt,
  four cannot; see below.

### The ten cache entries

| Cache file | How to obtain it | Needs | Size |
|---|---|---|---|
| `compound_slice.parquet` | **already shipped** — no action | — | 34 KB |
| `smiles_cache.csv` | rebuild: `prepare_smiles_cache.py`, then `resolve_chebi_smiles.py` appends to it | FoodAtlas v4.0 + network access to PubChem PUG-REST (free; ~6 min) | 13 MB |
| `chebi_smiles_cache.csv` | rebuild: `resolve_chebi_smiles.py` | same as above | 179 KB |
| `fart_compound_embeddings.pkl` | rebuild: `prepare_fart_embeddings.py` | `smiles_cache.csv` + `FartLabs/FART_Augmented` from HuggingFace (free) | 431 MB |
| `taste_gnn_best_compound_embeddings.pkl` | rebuild: `molecular.src.embed.generate_cache` (command below) | `fart_compound_embeddings.pkl` + the shipped checkpoint `molecular/results/grid/best/ckpt.pt` | 177 MB |
| `sensory_descriptions.csv` | **already shipped** — rebuild requires gated data and paid API calls (`generate_sensory_descriptions.py` reads the NDA-protected `product_labels_manually_cleaned.csv` and calls OpenRouter) | — | 1.2 MB |
| `fart_classifier_dense_embeddings.pkl` | **not obtainable** | — | 431 MB |
| `fart_taste_probs_embeddings.pkl` | **not obtainable** | — | 16 MB |
| `taste_gnn_centered_compound_embeddings.pkl` | **not obtainable** | — | 177 MB |
| `taste_gnn_regression_compound_embeddings.pkl` | **not obtainable** | — | 177 MB |

The last four are exploratory caches left over from development. No script in
this repository builds them and no code in this repository reads them: they back
no table, figure or ablation that is published. Their hashes stay in the manifest
as a record of the authoring environment, which is why `check_input_caches.py`
reports nine missing rather than five. Nothing in the release needs them.

`sensory_descriptions.csv` is shipped so that Tier 1 runs use the same
descriptions as the paper. Rebuilding it requires the NDA-protected ingredient
labels and an OpenRouter API call, but the shipped copy makes that unnecessary.
It backs no paper table, and `prepare_data.py` warns and continues when it is
absent, so a Tier 1 run without it is expected rather than degraded.

### Rebuilding

Build everything that can be built, skipping anything already matching the
manifest:

```bash
bash shared/scripts/prepare_caches/build_all_caches.sh          # build
bash shared/scripts/prepare_caches/build_all_caches.sh --check  # report only
```

Or step by step, in order, each one consumes the previous:

```bash
python3 shared/scripts/prepare_caches/prepare_smiles_cache.py   # FoodAtlas CIDs -> SMILES
python3 shared/scripts/prepare_caches/resolve_chebi_smiles.py   # ChEBI-only compounds -> SMILES
python3 shared/scripts/prepare_caches/prepare_fart_embeddings.py

python3 -m molecular.src.embed.generate_cache \
    --checkpoint   molecular/results/grid/best/ckpt.pt \
    --source_cache shared/data/caches/fart_compound_embeddings.pkl \
    --output_pkl   shared/data/caches/taste_gnn_best_compound_embeddings.pkl
```

`generate_cache` takes three required arguments; earlier revisions of this file
and of `build_all_caches.sh` invoked it bare, which exits 2 at argument parsing
without building anything. Each script above stops with a message naming the
input it is missing, `data/download_public.sh` fetches FoodAtlas v4.0, rather
than failing partway through.

One caveat on the two SMILES caches: they are built from live PubChem responses,
so a rebuild years later can differ from the recorded hash. `check_input_caches.py`
will then report a mismatch and exit 1. That is drift in the upstream service, not
corruption; the manifest records what this release was built from.

## Pre-computed predictions

`food_similarity/results/oof_predictions/` ships every leave-one-out prediction
CSV the paper depends on, including the `llm_*` predictions. Regenerating the
LLM predictions costs real $$ via the OpenRouter API, and the preview endpoints
drift between runs, so they are committed deliberately. The raw per-pair
reasoning logs those CSVs were derived from are not redistributed: they name
brands: so the released tree carries de-identified per-pair flags instead.

## Anonymization

The public Kaggle competition uses renumbered product codes; the
back-mapping (`kaggle_tastebench/generate_data/dataset/product_code_map.csv`)
is gitignored and not redistributed. See `data/GATED.md` and the paper
for the privacy framing.
