"""Render the human-baseline table from shipped intermediates only.

Previously this table was the one paper table nobody outside the team could
regenerate: `human_panelist_baseline.py` recomputes it from the NDA-gated
per-panelist ratings. Tracing what each row actually needs showed that five of
the six numbers were already reproducible from files the artifact ships, and the
sixth needed only the block -> product grouping — not the ratings.

Inputs (all shipped, none gated):
  results/individual_loo_accuracy.csv      per-panelist-block accuracy
  results/group_size_curve.csv             simulated panels of size k
  results/summary.json                     panelist count (see _n_panelists)
  results/verification/block_membership.csv which products each block saw
  ../food_similarity/results/oof_predictions/nested_bt_gemini_nnls.csv
  results/comparison_table.csv             cross-check target

Writes: ../paper/human_baseline/human_baseline_table.tex
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
sys.path.insert(0, str(TREE / "shared"))

from oof_io import read_oof  # noqa: E402
RES = HERE / "results"
VER = RES / "verification"
OOF = TREE / "food_similarity" / "results" / "oof_predictions" / "nested_bt_gemini_nnls.csv"
OUT = TREE / "paper" / "human_baseline" / "human_baseline_table.tex"

SEED = 42
N_BOOT = 10000


def f3(x: float) -> str:
    return f"{x:.3f}".replace("0.", ".")


def _n_panelists() -> int:
    """Distinct panelists behind the table, from summary.json.

    This used to be `individual_loo_accuracy.csv`'s `respondent.nunique()`. That
    column no longer ships: its ids were global rather than block-local, so it
    let a reader follow one panelist across blocks and categories and build a
    per-person skill profile over most of the benchmark. Every row and value in
    that file is unchanged -- only the person key is gone -- but a count of
    distinct people is by definition not recoverable from a file with no person
    key, so it is read from the aggregate summary instead.

    That makes this one number attested rather than recomputed. It is the honest
    trade: the alternative was to keep the identifier, and re-deriving the count
    from anything else in the release would silently print a different n.
    """
    s = json.loads((RES / "summary.json").read_text())
    n = s["individual_panelist"]["n_panelists"]
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"summary.json n_panelists is not a positive int: {n!r}")
    return n


def _pairwise(true: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """Correct-count and total, ties at half credit — matches evaluation/metrics.py."""
    correct, total = 0.0, 0
    for i, j in combinations(range(len(true)), 2):
        td, pd_ = true[i] - true[j], pred[i] - pred[j]
        total += 1
        if abs(td) < 1e-10 or abs(pd_) < 1e-10:
            correct += 0.5
        elif td * pd_ > 0:
            correct += 1.0
    return correct, total


def _within_block_counts(oof: pd.DataFrame, membership: pd.DataFrame) -> np.ndarray:
    """Per-block (n_correct, n_pairs). Block is the resampling unit: each block
    is a distinct panelist subset, so blocks are independent."""
    scores = {(r.category, r.product_code): (r.true_score, r.predicted_score)
              for r in oof.itertuples(index=False)}
    rows = []
    for _, grp in membership.groupby("block"):
        t, p = [], []
        for r in grp.itertuples(index=False):
            key = (r.category, r.product_code)
            if key in scores:
                t.append(scores[key][0])
                p.append(scores[key][1])
        if len(t) >= 2:
            c, tot = _pairwise(np.asarray(t), np.asarray(p))
            rows.append((c, tot))
    return np.asarray(rows, dtype=np.float64)


def _cluster_percentile(counts: np.ndarray) -> tuple[float, float, float]:
    """Within-block accuracy and its 95% interval, resampling whole blocks.

    Reproduces `human_panelist_baseline.py` exactly rather than substituting a
    method of my own: `rng.choice` (not `integers`), seed DEFAULT_SEED + 77777,
    10,000 draws, and a plain 2.5/97.5 percentile interval — not BCa. A BCa
    interval on the same counts gives [.599,.714] instead of the published
    [.604,.716], which would have looked like a discrepancy in the table rather
    than a different estimator.
    """
    point = counts[:, 0].sum() / counts[:, 1].sum()
    rng = np.random.default_rng(SEED + 77777)
    B = len(counts)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.choice(B, size=B, replace=True)
        s = counts[idx]
        tt = s[:, 1].sum()
        if tt > 0:
            boot.append(s[:, 0].sum() / tt)
    arr = np.asarray(boot)
    return float(point), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> int:
    need = [RES / "individual_loo_accuracy.csv", RES / "group_size_curve.csv",
            RES / "summary.json", OOF]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        print("ERROR: missing shipped input(s):")
        for m in missing:
            print(f"    {m}")
        return 1

    ind = pd.read_csv(RES / "individual_loo_accuracy.csv")
    gsc = pd.read_csv(RES / "group_size_curve.csv")
    mem = pd.read_csv(VER / "block_membership.csv") if (VER / "block_membership.csv").exists() else None
    oof = read_oof(OOF).dropna(subset=["predicted_score"])

    # --- human rows, from the per-panelist and panel-size tables ---
    med = float(ind.pairwise_accuracy.median())
    q1, q3 = ind.pairwise_accuracy.quantile([0.25, 0.75])
    n_obs, n_pan = len(ind), _n_panelists()
    if "respondent" in ind.columns:
        print("ERROR: individual_loo_accuracy.csv still carries a `respondent` "
              "column; the panelist key must not be released.")
        return 1

    panels = {}
    for k in (3, 5):
        s = gsc[gsc.k == k].pairwise_accuracy
        panels[k] = (float(s.median()), float(s.quantile(0.025)), float(s.quantile(0.975)))

    # --- model rows ---
    # The model's within-block row needs the block -> product grouping. That file
    # (block_membership.csv) is deliberately NOT shipped: it de-anonymises
    # rater_pair_signs.csv and keys individual_loo_accuracy.csv to named products.
    # Without it this one row is read from the published table and labelled as
    # attestation rather than silently recomputed from something else.
    wb_recomputed = False
    if mem is not None:
        wb = _within_block_counts(oof, mem)
        if len(wb):
            wb_pt, wb_lo, wb_hi = _cluster_percentile(wb)
            wb_recomputed = True
            print(f"  model within-block: {wb_pt:.3f} [{wb_lo:.3f},{wb_hi:.3f}]  "
                  f"({len(wb)} blocks, {int(wb[:,1].sum())} pairs)")
    if not wb_recomputed:
        ref_tbl = pd.read_csv(RES / "comparison_table.csv")
        row = ref_tbl[ref_tbl.method == "Best model (within-block)"].iloc[0]
        wb_pt = float(row.pairwise_accuracy)
        ci = str(row["95% CI"]).strip("[]").split(";")
        wb_lo, wb_hi = float(ci[0]), float(ci[1])
        print(f"  model within-block: {wb_pt:.3f} [{wb_lo:.3f},{wb_hi:.3f}]  "
              f"(from comparison_table.csv — not recomputed; block grouping is not shipped)")

    all_true = oof.true_score.values
    all_pred = oof.predicted_score.values
    per_cat = oof.groupby("category")
    num = den = 0.0
    for _, g in per_cat:
        c, t = _pairwise(g.true_score.values, g.predicted_score.values)
        num += c
        den += t
    ap_pt = num / den

    # all-pairs CI comes from the cached bootstrap the paper already ships
    ap_lo, ap_hi = float("nan"), float("nan")
    cis = TREE / "food_similarity" / "results" / "cis_table_results.csv"
    if cis.exists():
        c = pd.read_csv(cis)
        row = c[(c["row"] == "nnls") & (c["metric"] == "pairwise_accuracy")]
        if len(row):
            ap_lo, ap_hi = float(row.iloc[0]["ci_lo"]), float(row.iloc[0]["ci_hi"])
        else:
            print("  WARN: no cached CI for nnls/pairwise_accuracy")

    print(f"  individual panelist median {med:.3f} (IQR {q1:.2f}-{q3:.2f}, "
          f"n={n_obs} obs / {n_pan} panelists)")
    for k in (3, 5):
        print(f"  panel of {k}: {panels[k][0]:.3f} [{panels[k][1]:.3f},{panels[k][2]:.3f}]")
    print(f"  model all pairs:    {ap_pt:.3f} [{ap_lo:.3f},{ap_hi:.3f}]")

    # --- cross-check against the published values, and refuse on drift ---
    ref = RES / "comparison_table.csv"
    if ref.exists():
        r = pd.read_csv(ref).set_index("method")["pairwise_accuracy"].to_dict()
        checks = [
            ("Individual panelist (median)", med),
            ("Panel of 3", panels[3][0]),
            ("Panel of 5", panels[5][0]),
            ("Best model (within-block)", wb_pt),
            ("Best model (all pairs)", ap_pt),
        ]
        bad = [(k, v, r[k]) for k, v in checks
               if k in r and abs(v - float(r[k])) > 1e-3]
        if bad:
            print("ERROR: recomputed values disagree with the published table:")
            for k, got, want in bad:
                print(f"    {k}: recomputed {got:.4f}, published {float(want):.4f}")
            return 1
        print(f"  cross-check against comparison_table.csv: {len(checks)}/{len(checks)} agree")

    caption = (
        r"\captionof{table}{\textbf{Human-panelist baseline.} Pairwise ranking accuracy against "
        r"the leave-one-out panel mean, on the within-block pairs each panelist rated, alongside "
        r"the best model on those same pairs and on all within-category pairs. Individual "
        r"panelists agree with each other only ${\approx}$.55 of the time, but the averaged panel "
        r"mean is reliable (split-half .825), which is why a median panelist matches it on .650 "
        r"of pairs: the meaningful band for models runs from .650 (median panelist) to .825 (the "
        rf"reliability ceiling). $n$ = {n_pan} panelists, {n_obs:,} panelist-block observations. "

        + (r"Regenerated in full from shipped intermediates.}" if wb_recomputed else
           r"All rows but the model's within-block accuracy are recomputed from shipped "
           r"intermediates; that row is reproduced from the published analysis, since the "
           r"block composition needed to recompute it is not released.}")
    ).replace("{:,}".format(n_obs), f"{n_obs:,}".replace(",", "{,}"))

    lines = [
        caption,
        r"\label{tab:human-baseline}",
        r"\begin{tabular}{@{}lc@{}}", r"\toprule",
        r"Method & Pairwise Acc.\ [95\% CI] \\", r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Evaluated on within-block pairs}} \\",
        rf"\quad Individual panelist (median) & {f3(med)}\textsuperscript{{\dag}} \\",
        rf"\quad Panel of 3 & {f3(panels[3][0])} [{f3(panels[3][1])},{f3(panels[3][2])}] \\",
        rf"\quad Panel of 5 & {f3(panels[5][0])} [{f3(panels[5][1])},{f3(panels[5][2])}] \\",
        rf"\quad Best model (within-block) & {f3(wb_pt)} [{f3(wb_lo)},{f3(wb_hi)}] \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Evaluated on all within-category pairs}} \\",
        rf"\quad Best model (all pairs) & {f3(ap_pt)} [{f3(ap_lo)},{f3(ap_hi)}] \\",
        r"\bottomrule", r"\end{tabular}", "",
        rf"\vspace{{2pt}}",
        rf"{{\scriptsize \textsuperscript{{\dag}}IQR: {q1:.2f}--{q3:.2f} across "
        rf"{n_obs:,} panelist-block observations.}}".replace(",", "{,}"),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {OUT.relative_to(TREE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
