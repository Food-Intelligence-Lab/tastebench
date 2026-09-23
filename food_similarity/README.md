# Food-similarity ranking

This area covers the food-level pairwise sensory ranking task: given two
plant-based products in the same category (e.g. two bacons), predict
which one was rated as more similar to the animal-based reference. The
task is evaluated leave-one-product-out (LOOCV) over 215 NECTAR
products in 24 categories.

All food-similarity tables are produced from this folder. The tables
cite four families of models — supervised (Bradley–Terry, hierarchical
BT, kernel RankSVM, ridge, LightGBM), unsupervised distance baselines
(cosine, L2 over multimodal embeddings), zero-shot LLMs (Gemini 3.1
Pro, Qwen 3.5 397B-A17B), and the NNLS ensemble.

Rendered `.tex` outputs land in
[`../paper/model_results_tables/`](../paper/model_results_tables/).
See the top-level [README](../README.md) for the artifact-to-renderer
mapping.

## Verifying numbers (no NECTAR required)

```bash
bash food_similarity/verify.sh
```

Re-renders every food-similarity table from the committed OOF
prediction CSVs in `results/oof_predictions/`. Runtime is dominated by
`render_table_ablation_features.py`, which sweeps BCa CIs over 105
(model, feature-subset) pairs using the full Python-loop bootstrap.
No data downloads, no model training. This script takes no arguments; to
check the re-rendered tables against the shipped ones byte for byte, run
the top-level `bash verify_paper.sh --check-diff`.

## Full retrain (heavyweight; needs gated NECTAR data)

```bash
bash food_similarity/reproduce.sh
```

This regenerates `prepare_data.py`'s `product_features.pkl` from raw
NECTAR ingredient/nutrition CSVs, then retrains every supervised
baseline LOOCV, every NNLS ensemble (nested LOOCV), recomputes BCa
CIs, and re-renders the tables.

## Layout

```
food_similarity/
├── prepare_data.py             Build product_features.pkl from NECTAR
├── data/                       LOOCV loaders + cached features
├── models/                     BT, ridge, LightGBM, NNLS, … (one file per model)
├── train/                      Training entry points (LOOCV runners, NNLS meta-learner,
│                               LLM integration into ensemble)
├── evaluation/                 Metrics + BCa bootstrap (compute_bca_pw_acc, …)
├── zero_shot_baselines/        LLM and distance-predictor baseline pipelines
│   ├── configs/{llm,cosine_dist,l2_dist}/*.yaml
│   ├── lib/                    Shared embedding + preprocessor utilities
│   └── run.py                  Single entry point
├── scripts/                    Render scripts for paper tables/figures
└── results/
    ├── oof_predictions/        Per-model LOOCV predictions (committed; canonical input
    │                           for all render scripts)
    ├── cis_*.csv               Precomputed BCa CIs per table
    └── llm_bootstrap_cis.csv   Precomputed BCa CIs for all 14 LLM modality combos
```

### Reading an out-of-fold file

Always `from oof_io import read_oof` (`shared/oof_io.py`), never a bare
`pd.read_csv`. The target column is `true_score` (the panel's mean similarity
rating) when the file was built from the gated NECTAR bundle and `true_rank`
(the within-category rank) when it came out of the public release, and a Tier 1
reproduction sees both — it starts on released files and overwrites them with
retrained ones at Phase 5. `read_oof` presents `true_score` either way; every
metric depends only on within-category order, so the two are identical to the
last bit. Writers keep writing `true_score`.

`python3 shared/check_oof_readers.py` fails if any script reads an OOF without
it. `verify_paper.sh`, `reproduce.sh` Phase 1 and the release build all run it.

`zero_shot_baselines/` writes its raw per-pair LLM logs to a `results/`
folder that is **not** part of this release: the logs name brands, so
they are not redistributed. The distilled per-model LOOCV predictions in
`results/oof_predictions/` are what every render script reads.

Rendered `.tex` tables land in
[`../paper/model_results_tables/`](../paper/model_results_tables/),
the single location consumed by `\input{...}` from the paper source.
