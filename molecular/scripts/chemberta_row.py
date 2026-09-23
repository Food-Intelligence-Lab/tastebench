"""ChemBERTa-2's Table 2 row, in one place.

Both the one- and two-column renderers need these numbers. They were hard-coded
in the one-column renderer only, so the two-column variant silently shipped
without the row — the same drift that let a stale CSV sidecar keep the retracted
Table 3 values. One definition, two importers.

Values are the best-validation single run: the figure quoted in the rebuttal
(.842) and the run the paired significance tests are computed on. The 3-seed
means are accuracy .847 and macro-F1 .670 +/- .014 — report those in prose, not
here, and do not mix the two.

Source: analysis/C_molecular/c3b_chemberta_predictions.parquet, percentile
bootstrap, 10,000 resamples, seed 42. AUROC has no interval because that file
stores hard labels only.
"""

LABEL = "ChemBERTa-2 (fine-tuned)"

# metric -> (point, ci_lo, ci_hi); None means no interval is available
ROW = {
    "accuracy": (0.842, 0.827, 0.857),
    "precision": (0.643, 0.590, 0.709),
    "recall": (0.710, 0.635, 0.802),
    "f1": (0.671, 0.609, 0.731),
    "auroc": (0.950, None, None),
}

SEED_MEANS = {"accuracy": 0.847, "f1_mean": 0.670, "f1_std": 0.014}
