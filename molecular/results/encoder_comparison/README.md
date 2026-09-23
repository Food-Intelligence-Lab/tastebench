# Corrected Table 3 — reproduction bundle

Regenerate the corrected Table 3 (FART / GNN / ChemBERTa-2 as the compound encoder,
fair FoodAtlas-v4.0 / `log_weighted_average` / main-pipeline footing) **and** the
ChemBERTa-2 row of molecular Table 2 — byte-identically, from cache, with **no API
and no gated NECTAR data**. Mirrors the paper's `results/cis_*.csv` + `oof_predictions/*`
convention (`verify_paper.sh` renders tables from committed OOFs, not the retrain path).

## Run

```bash
TASTEBENCH_REPO=/path/to/tastebench \
  python regenerate_table3.py
```

Only external dependency: the repo's two deterministic, committed helpers
`evaluation.metrics` and `evaluation.bootstrap` (read-only). Writes
`cis_encoder_comparison.csv` (points + 95% BCa CIs for all 30 downstream cells),
fills the derived ensemble OOFs, asserts the two canonical gates (FART NNLS .6829,
FART BT .6102), and diffs against the shipped `downstream_results.csv`
(expected `max |Δpairwise| = 0`).

## Contents

| Path | What |
|---|---|
| `bases/{c5_base_FART,c5_base_GNN,enc_base_ChemBERTa}.npz` | nested BT bases: `bt_outer` (LOOCV), `bt_inner` (N×N nested pairs), `gem_oof` (frozen Gemini), `y_all`, `categories`, `product_keys`. Each ~370 KB. Gemini is frozen inside `gem_oof`, so no API is ever needed. |
| `oof/oof_<enc>_<model>.csv` | per-cell out-of-fold predictions. 19 individual-model cells ship as-is; the 11 BT/ensemble cells are written by `regenerate_table3.py` (bit-exactly derived from the bases). |
| `molecular/` | ChemBERTa-2 Table-2 inputs: `c3b_chemberta_predictions.parquet` (hard labels), `c3b_chemberta_hpo_result.csv` (AUROC point), `c3d_significance.csv` (paired-gap CIs). FART/GNN Table-2 rows read from the committed `molecular/results/cis_molecular_prediction.csv`. |
| `downstream_results.csv`, `molecular_results.csv` | the final shipped table values (diff target). |
| `cis_encoder_comparison.csv` | produced by the regen script: 30 downstream cells + ChemBERTa molecular row, points + BCa intervals. |
| `input_sha256.txt` | sha256 of every bundle file (mirrors `results/input_cache_sha256.txt`). |
| `REPRO_AUDIT.md` | full reproducibility audit + provenance + determinism evidence. |

## Provenance & determinism (verified)

- **Faithful & leak-free:** the scoring pipeline is a line-for-line reimplementation of
  the canonical `compute_per_model_nnls` / `nested_meta_all`; the nested LOOCV is
  leak-free by construction; all three encoders share ONE code path (the bases differ
  only in `bt_outer`, with identical `product_keys`/`gem_oof`/`y_all`/`categories`).
- **Gate:** FART reproduces the archived canonical `.6829` (NNLS) and `.6102` (BT) — the
  same values in the paper's headline results — from both cache and a from-scratch rebuild
  (`max|Δ| = 0`). This ties the corrected table to the committed archive.
- **GNN & ChemBERTa determinism:** each base rebuilt from scratch is bit-identical to the
  shipped cache (`max|Δbt_outer| = max|Δbt_inner| = 0`); all four ensemble numbers match.
- **Fully deterministic:** all models `random_state=42`, bootstrap `default_rng(42)`,
  LightGBM `deterministic/force_row_wise/num_threads=1`; metric ties get deterministic 0.5
  credit; joblib results are index-keyed (order-independent). The only non-bit-stable link
  is ~1e-6 cross-BLAS PCA drift on a *from-scratch rebuild* — neutralized by shipping these
  caches.

## Scoring-function note (important)

Two NNLS ensembles exist in the codebase. Table 3 uses the **main pipeline**
(`nested_bt_gemini`), the canonical scorer that produces the paper's headline FART
`.6829`. A separate **transfer script** (`nested_l2_bt_nnls`, L2-BT + custom gesvd PCA)
scores the *same* v4.0 features differently — FART `.642`, GNN `.654`, ChemBERTa `.635` —
and was the source of the original Table-3 bug (used only for the GNN column). It scores
FART at `.642`, **not** the published `.683`, so it is **not** the canonical Table-3 scorer.
All cells here use the main pipeline.

## Rebuild-from-raw (Tier B, optional — not needed to reproduce the table)

Rebuilding the `.npz` bases and 19 individual OOFs from raw v4.0 features additionally
needs `pf_canonical.pkl`, the FART/GNN/ChemBERTa embedding caches, and gated FoodAtlas v4.0,
via `../ensemble_focused.py` / `../downstream_embedding_table.py`. This is the provenance
chain behind the caches, documented in `REPRO_AUDIT.md`; you reproduce from cache
(above) instead.
