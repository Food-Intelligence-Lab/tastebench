# Release and versioning policy

## What v1 is

TasteBench v1 is the benchmark reported in the paper: 215 plant-based products in
24 categories, 935 within-category ranking pairs, plus the 15,025-molecule taste
classification task. The public competitions are
`tastebench-challenge-2026` (food-level) and
`molecular-taste-classification-2026` (molecular).

Each release is **frozen**. New NECTAR data: additional categories, and
additional geographic markets: is published as a new version with its own
identifier rather than by modifying an existing one, so numbers reported against
v1 stay comparable to v1 forever.

## What is and is not applied

This tree is the submitted artifact plus every camera-ready change that is ready. The rest
are listed here so nothing is silently missing.

Applied:

| Commitment | Change |
|---|---|
| C22 | `resolve_chebi_smiles.py` reads `entities.parquet` |
| C24, C25 | Python 3.11 pinned; `shared/check_env.py` guards the interpreter, a pyarrow floor, and packages whose version silently changes results |
| C02 | The panel design is described as an incomplete block design. It is not balanced: only 25 of 247 products span two or more blocks. Corrected in the paper, both Croissant files, and the human-baseline docs |
| C10, C11, C12, C13 | Corrected Table 3: three encoders through one matched pipeline. The renderer refuses to emit the table unless all ten FART cells reproduce their archived values. ChemBERTa-2 row added to Table 2. |
| C14 | Table 6 caption states the within-noise reading |
| C16 | Human-baseline table has a caption; `verify_paper.sh` fails if any rendered table lacks one |
| C01, C19, C26 | Croissant: `prov:wasDerivedFrom` / `prov:wasGeneratedBy`, corrected panel-diet wording, image provenance |
| C20, C21 | Tier 1 needs no API key. `reproduce.sh` defaults to the feature matrix behind `REBUILD_FEATURES`; the precomputed sensory descriptions ship |
| C23 | Cache-to-source table in the README covering every manifest entry; `build_all_caches.sh` |
| C28, C30, C31 | This file, and `food_similarity/results/oof_predictions/PROVENANCE.md` (endpoints, temperature, run window, the ~29% drift measurement) |
| C29 | Embedding caches ship as slices: 738 SMILES, 9.9 MB instead of 1.9 GB, vectors copied verbatim |
| verification | 18 published numbers recomputable with no gated data |

Not applied, and why:

- Naming the planned Europe and India collections in the paper's future-work section. Omitted until the collections are confirmed.
- The permanent identifier. See the section above.
- The paper-side changes: two new appendices, the abstract and framing revisions, the expanded Limitations, and the transfer-focused rewrite of the molecular section. Those are edits to the paper source, so they do not appear in this tree.

Optional, not required by anything: a FoodAtlas slice keyed by ingredient name would replace
the 82 MB FoodAtlas and 953 MB FooDB downloads that a from-scratch feature rebuild needs. It
freezes eight mapper knobs, so any FoodAtlas ablation must fall back to the full download.

## Tier 1 and the two code spaces

Tier 1 retrains the non-LLM baselines and completes on this tree, given the gated bundle.
One thing to understand before running it:

- Every artifact here is re-keyed to per-category surrogates. The gated bundle carries the original manufacturer codes.
- The map between the two keyings ships as the fifth item of the gated bundle, so it reaches only people who already hold the ratings. `food_similarity/data/loocv.py` applies it on load, in one place.
- Without the map, which is the state of every public Tier 0 user, every code path involved is a no-op. It is decided by inspecting the codes, not by a flag.
- A Tier 1 join with gated data but no map fails with a message naming the map and `data/GATED.md`.
- The `taste_gnn` row skips: its compound-embedding cache is derived by the molecular grid and is not shipped.
- The LLM out-of-fold files are frozen API predictions and are not retrained by any phase.

## Permanent identifier

**DOI: not yet minted.** v1 is currently distributed from this repository. The archival
deposit is made when it is needed to keep a *superseded* version resolvable: that is,
when v2 exists: so that numbers reported against v1 stay checkable against the v1 files
even after the benchmark moves on. Until then there is nothing for a permanent identifier
to protect that this repository does not already serve.

When it is minted, cite the **version** DOI for v1, not the concept DOI. The concept DOI
always resolves to the latest version, so citing it would break the guarantee below that
numbers reported against v1 stay comparable to v1.

Nothing in this release depends on the DOI: Tier 0 verification, the verification bundle,
and every rendered table reproduce from the files here alone.

## What the archive contains

- the public competition files for both tasks
- Croissant metadata (`croissant/nectar.json`)
- the frozen out-of-fold predictions for every baseline, including the LLMs
  (see `food_similarity/results/oof_predictions/PROVENANCE.md`)
- the per-cell encoder-comparison bundle backing Table 3
- the verification bundle backing the human-agreement and contamination numbers
- `food_similarity/results/input_cache_sha256.txt`, the checksum manifest

## What stays gated, and why

| Withheld | Reason |
|---|---|
| NECTAR sensory, ingredient and nutrition CSVs | contain participating brand identities; NDA-protected |
| NECTAR product photographs | NECTAR-owned, same NDA |
| Taste Like CPG directory | prevents a cross-reference de-anonymisation of the obfuscated competition set |
| Kaggle answer keys (`solution.csv`, `product_code_map.csv`) | publishing them would break both public leaderboards |
| Raw model reasoning logs | the model names a real brand in roughly 10% of pairs; the de-identified per-pair flags ship instead |

Access to the gated bundle: see `data/GATED.md`.

## Product codes are surrogates, not the original NECTAR codes

- `product_code` in the out-of-fold files is a per-category surrogate.
- The original code is a manufacturer slot. Across 215 products there were only 24 distinct codes, one appearing in 19 of the 24 categories, and 14 of the 24 had a unique category-incidence signature. That made "which company sells products across these categories" answerable from public retail knowledge, with the panel score in the same row: a route from rating to brand that this data must not provide.
- No reported metric reads the code's value. Pairwise accuracy, Spearman, Kendall and Recall@k all work within a category on (target, prediction) pairs. Every number is unchanged and all fifteen paper tables render byte-identically either way.
- The surrogate is drawn from a cryptographically random per-category permutation held outside the release. It is independent of the target, so recovering it reveals nothing about a product's rating, and it cannot be re-derived from anything published.
- Rows are sorted by that public code, so row position carries no trace of the original ordering.
- Reproducing from the gated bundle gives the original codes and identical numbers. The two keyings cannot be aligned from the release alone.
- If you hold an earlier copy of this release, the numbers match but the row keys do not.

## Targets ship as within-category ranks, not panel means

The out-of-fold files carry `true_rank`: each product's **within-category rank**
among its analogues (average rank, so ties stay tied). The panel's mean similarity
rating is **not** in this release, and `y_rank` in
`molecular/results/encoder_comparison/bases/*.npz` is likewise the rank vector
rather than the ratings.

Nothing published is lost. Every metric the paper reports: pairwise accuracy,
Spearman, Kendall, Recall@k: reads only the *sign* of a within-category target
difference, so an average-rank transform is exactly metric-preserving. Verified
across all 182 out-of-fold files: max $|\Delta|$ = 0 on all six metrics, and all
twenty rendered tables and figures are byte-identical either way.

Two places needed the actual scale, and both are handled without shipping it:

- The per-category NNLS table (`table_per_category_nnls.tex`) reports a
  coefficient of variation $\sigma_S/\mu_S$ and a Pearson $r$. Those are
  precomputed into `food_similarity/results/per_category_scale.csv`: 24
  per-category aggregates instead of 215 product means.
- The `.npz` bases' NNLS ensemble was *fitted* on the ratings, not merely scored
  against them. The per-fold weights are therefore shipped as `nnls_weights`, so
  the ensemble reconstructs bit-identically with no ratings present.

What this does and does not protect, stated plainly:

- **Why the ratings moved.** The product compositions: verbatim ingredient
  lists, nutrition panels, photographs: are already public through a live Kaggle
  competition. Brand identity is therefore inferable without this release, so
  de-identifying `product_code` can no longer be the thing that protects the
  ratings. The ratings had to move further downstream instead, and they did.
- **Rank is still ordinal information.** Anyone who establishes which product a
  `(category, surrogate)` row refers to learns that product's *position* among
  its analogues. That is real per-product information about the panel and this
  release does not claim otherwise; what it no longer learns is any product's
  rating, the gap between two products, or where a category sits on the scale.
- **`predicted_score` still ships**, because publishing the models' out-of-fold
  predictions is the point of the artifact. For the regression baselines that
  column is calibrated to the original scale and is therefore an *estimate* of a
  product's panel mean: a model output, with the models' documented error, not
  panel data.
- The gate is `assert_no_ratings.py` in the build script, run by every build: it
  scans every `.csv`, `.parquet`, `.npz` and `.json` for a per-product rating by
  name and by shape, and fails the build rather than letting one through in a
  container nobody thought to grep.

## Before mirroring this repository

The reasoning logs are excluded from archive builds by `.gitattributes`
(`export-ignore`), which is what keeps them out of the released artifact. That
mechanism protects an NDA commitment, so:

- do not remove those `export-ignore` lines without a replacement, and
- a **git mirror** of the development repository would carry the logs even though
  an archive build does not. Strip them, or take the decision deliberately.
