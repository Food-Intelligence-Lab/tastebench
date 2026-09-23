# Verification bundle

Derived statistics that let anyone recompute the numbers we published in the rebuttal **without
the NECTAR NDA**. Built as a self-contained unit for the public release;
`MANIFEST.tsv` gives each file's destination within this tree.

## Status

- **18/18** published numbers recompute from these files alone (`verify_bundle.py`)
- An adversarial NDA disclosure scan was run before release (13 pass, 0 failures).
- **131 KB** total, 8 files, no gated data, no API, no model downloads

## Use

```bash
python verify_bundle.py     # recompute every published number from data/ only
```

`verify_bundle.py` is what an outside reader runs. It recomputes every published statistic
from the data files in this directory alone — no gated data, no API, no model downloads.

## What verifies what

| File | Rows | Recomputes |
|---|---|---|
| `data/human_agreement/alpha_coincidence.csv` | 2,348 | Krippendorff's α = .077 (r2) |
| `data/human_agreement/rater_pair_signs.csv` | 394 | rater-vs-rater agreement = .547 (r2) |
| `data/human_agreement/block_group_stats.csv` | 54 | 25 units, 5 raw p<.05, 1 after BH-FDR, median 0.24 (r1) |
| `data/human_agreement/split_half_iterations.csv` | 1,000 | split-half .825 — **spread only, not reproduction** (r2) |
| `data/recognition/recognition_flags.csv` | 931 | identification 12.5%, brand 10.2%, recall 2.4%, accuracy-by-flag +.042 n.s., margin +.026 vs +.052 (r1) |
| `data/oof/c2_oof_*.csv` | 3 × 215 | additives: micro +.000, macro +.0005 (r1) |

Already verifiable from the submitted artifact, so not duplicated here: α per block, median
panelist .650, IQR, n=627, and the panel-size curve all recompute from the committed
`human_baseline/results/*.csv`. The corrected Table 3 has its own bundle at
`../encoder_comparison/repro_bundle/`.

## Design rule

Ship the smallest statistic that makes the number recomputable, and strip every column that
computation does not need. Two consequences worth knowing:

- **Sufficient, not approximate.** α is a function of the coincidence matrix; `pair_credit_counts`
  consumes only `(n_pos, n_neg, n)`; a one-way ANOVA F is closed-form in group `(n, mean, sd)`.
  Each file is mathematically equivalent to the raw data *for its statistic*, which is why the
  recomputed values match to four decimals rather than approximately.
- **Invariances are free privacy.** The ANOVA is shift-invariant, so group means are centered and
  no absolute rating level ships. The agreement statistic is invariant to item labelling, so item
  labels are permuted and no product pair is identifiable. Neither change costs a digit.

## One file this bundle exists to replace

`analysis/B_contamination/recognition_audit_release/data/flags_deidentified.csv` — prepared as the
"de-identified, NDA-safe" substitute — pairs the competition product codes with `prediction` and
`correct`, from which the true pairwise winner is recoverable for every pair: verified to
reconstruct 100% of the 932 scored rows of the withheld Kaggle `solution.csv`. Not an NDA breach,
but disqualifying for the public leaderboard, which is why it must never ship.
`data/recognition/recognition_flags.csv` replaces it, keyless and shuffled.

The NDA disclosure scan was tested against that known-bad file before release and correctly
flagged it. `data/recognition/recognition_flags.csv` replaces it, keyless and shuffled.

## What still cannot be verified without the NDA

Shipping this bundle does not make everything reproducible. Four things stay gated:

1. **Auditing the recognition regexes in context.** The rates are checkable by anyone, but
   confirming that a specific pair was correctly flagged needs the reasoning text.
2. **Rebuilding features from raw ingredient text and product photographs.** The inputs are the
   gated asset; irreducible.
3. **The brand-blind probe** (−.005, McNemar p = .71). Needs API spend, and cannot be reproduced
   exactly at any price — the endpoint drifted ~29% at temperature 0 over three months.
4. **Verifying the de-identification itself.** By construction, that requires the linkage.

A longer internal note (`NDA_REVIEW.md`, file-by-file, with the open judgement calls to put to
NECTAR) is kept beside this bundle in the development repository and is deliberately **not** part of
the public release.

## Integration into the release tree

`MANIFEST.tsv` has one row per file with `rows`, `bytes`, `sha256`, what it verifies, and its
destination path. The build script's task `16-verification-bundle` copies them in, adds the
hashes to `food_similarity/results/input_cache_sha256.txt`, and wires `verify_bundle.py` into
`verify_paper.sh`.

Ship `verify_bundle.py` alongside the data. `make_bundle.py` (the provenance record for how
each statistic was derived) is kept in the development repository — it requires gated inputs
and is not part of the public verification path.
