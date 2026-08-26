"""Two-column workshop variant of ``table_molecular_prediction.tex``.

The 6-column FART vs GNN comparison is too wide for ``\\columnwidth`` in
a typical two-column layout when CIs are inline. This renderer keeps
all five metrics in a single consolidated table by stacking each
FART/GNN cell as point-estimate above ``[lo, hi]``; baselines without
published CIs render as plain points. Output:

    table_molecular_prediction.tex  Acc, Precision, Recall, F1, AUROC

Reads the same BCa-CI cache (``cis_molecular_prediction.csv``) populated
by ``render_table_molecular_prediction.py`` -- no recomputation. McNemar's
on FART vs GNN accuracy still needs the parquet predictions, so those
are loaded once.

Font tuning:
    --outer  size of the table body (default scriptsize)
    --inner  size of the bracketed CI under each FART/GNN cell
             (default tiny)

Drop in any of: tiny, scriptsize, footnotesize, small. The whole table
is wrapped in ``\\resizebox{\\columnwidth}{!}{...}``, so font choices
trade off scaling vs natural line height -- ``footnotesize/scriptsize``
gives a less aggressive rescale and a heavier-looking table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from render_table_molecular_prediction import (  # noqa: E402
    BASELINES,
    COLUMNS,
    MOLECULAR_DIR,
    _baseline_cell,
    _overlaps,
    _short,
    load_run,
)
sys.path.insert(0, str(MOLECULAR_DIR.parent))
from molecular.src.eval.metrics import mcnemar_accuracy  # noqa: E402

OUT_DIR = MOLECULAR_DIR.parent / "paper" / "two_column" / "molecular_prediction"

METRICS = [
    ("Accuracy",  "accuracy"),
    ("Precision", "precision"),
    ("Recall",    "recall"),
    ("F1",        "f1"),
    ("AUROC",     "auroc"),
]


def _stacked_with_dagger(point: float, lo: float, hi: float,
                         bold: bool, overlap: bool) -> str:
    """Stacked makecell cell with optional bold + dagger.

    Inner CI line is wrapped in ``\\tabci`` (defined as a macro at the
    top of the table) so its size can be changed in one place.
    """
    if np.isnan(lo) or np.isnan(hi):
        s = _short(point)
        return rf"\textbf{{{s}}}" if bold else s
    pt = _short(point)
    if bold:
        pt = rf"\textbf{{{pt}}}"
    elif overlap:
        pt = pt + r"$^\dag$"
    return (rf"\makecell{{{pt} \\ "
            rf"{{\tabci [{_short(lo)},{_short(hi)}]}}}}")


def _render_combined(fart_d: dict, gnn_d: dict,
                     caption: str, label: str, out_path: Path,
                     outer_size: str, inner_size: str) -> None:
    """Render the consolidated 6-column table into ``out_path``."""
    col_max: dict[str, float] = {}
    leader_ci: dict[str, tuple[float, float]] = {}
    for _, m in METRICS:
        baseline_pts = []
        for _, acc, prec, rec, f1, auroc in BASELINES:
            baseline_pts.append({"accuracy": acc, "precision": prec,
                                 "recall": rec, "f1": f1, "auroc": auroc}[m])
        all_pts = baseline_pts + [fart_d[m]["point"], gnn_d[m]["point"]]
        col_max[m] = round(max(all_pts), 3)
        for d in (fart_d, gnn_d):
            if round(d[m]["point"], 3) == col_max[m]:
                leader_ci[m] = (d[m]["ci_lo"], d[m]["ci_hi"])
                break

    n_cols = len(METRICS)
    header = " & ".join(["Model"] + [h for h, _ in METRICS]) + r" \\"

    def baseline_row(label: str, vals: dict[str, float]) -> str:
        cells = [_baseline_cell(vals[m], bold=round(vals[m], 3) == col_max[m])
                 for _, m in METRICS]
        return rf"\quad {label} & " + " & ".join(cells) + r" \\"

    def ours_row(label: str, d: dict) -> str:
        cells = []
        for _, m in METRICS:
            p, lo, hi = d[m]["point"], d[m]["ci_lo"], d[m]["ci_hi"]
            is_best = round(p, 3) == col_max[m]
            ovl = (not is_best) and m in leader_ci \
                and _overlaps(lo, hi, *leader_ci[m])
            cells.append(_stacked_with_dagger(p, lo, hi,
                                              bold=is_best, overlap=ovl))
        return rf"\quad {label} & " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        # Font knobs -- edit these in Overleaf to retune for the workshop
        # template. \tabci controls the bracketed-CI font (one place,
        # affects every CI in the table). The outer size is the line
        # below; both honor \providecommand so paper preamble overrides.
        rf"\providecommand{{\tabci}}{{\{inner_size}}}",
        rf"\{outer_size}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        # Auto-scale to whatever workshop column width is in force.
        # The natural width is ~3.7'' at scriptsize; \columnwidth is
        # ~3.3-3.5'' in most two-column templates, so ~95% scale.
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "c" * n_cols + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
        rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}"
        r"{\textit{Reported by~\citet{zimmermann2025chemical}}} \\",
    ]
    for label_b, acc, prec, rec, f1, auroc in BASELINES:
        vals = {"accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "auroc": auroc}
        lines.append(baseline_row(label_b, vals))
    lines.append(ours_row("FART", fart_d))
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}"
                 r"{\textit{This work}} \\")
    lines.append(ours_row("GNN", gnn_d))
    # ChemBERTa-2 (see molecular/scripts/chemberta_row.py). Added because
    # this variant previously omitted a row the one-column table had.
    from chemberta_row import LABEL as _CB_LABEL, ROW as _CB_ROW
    _cb = " & ".join(
        ("--" if _CB_ROW[_m][0] is None else f"{_CB_ROW[_m][0]:.3f}".replace("0.", "."))
        for _m in ("accuracy", "precision", "recall", "f1", "auroc"))
    lines.append(rf"\quad {_CB_LABEL} & {_cb} \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path.relative_to(MOLECULAR_DIR.parent)}")


def _mcnemar_chi2_p(fart_d: dict, gnn_d: dict) -> tuple[float, str]:
    """McNemar's chi^2 + already-formatted ``= 0.07`` / ``< 0.001`` p-string.

    Returns the operator-and-value half so callers can compose the math
    expression they want (e.g., ``$p {p_str}$`` -> ``$p = 0.07$``).
    """
    if not np.array_equal(fart_d["smiles"], gnn_d["smiles"]):
        raise SystemExit("SMILES mismatch between FART and GNN predictions.parquet")
    mcn = mcnemar_accuracy(fart_d["correct"], gnn_d["correct"])
    p = mcn["p_value"]
    if p < 1e-3:
        return mcn["chi2"], "< 0.001"
    if p < 1e-2:
        return mcn["chi2"], f"= {p:.3f}"
    return mcn["chi2"], f"= {p:.2f}"


FONT_CHOICES = ("tiny", "scriptsize", "footnotesize", "small")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outer", default="footnotesize", choices=FONT_CHOICES,
                        help="font size for the table body (default: footnotesize)")
    parser.add_argument("--inner", default="footnotesize", choices=FONT_CHOICES,
                        help="font size for the bracketed CI line (default: footnotesize)")
    args = parser.parse_args()

    columns: list[tuple[str, dict]] = []
    for label, spec in COLUMNS:
        columns.append((label, load_run(spec, label=label)))
    fart_d, gnn_d = columns[0][1], columns[1][1]

    chi2, p_str = _mcnemar_chi2_p(fart_d, gnn_d)

    caption = (
        r"\textbf{The trained GNN matches FART's overall accuracy but "
        r"exhibits a distinct precision-recall tradeoff on umami.} "
        r"Taste classification on the 2{,}254-molecule FART test split. "
        r"The ``FART augmented + confidence'' test-time augmentation "
        r"variant is excluded for single-checkpoint comparability. "
        r"fp $=$ Morgan fingerprints; desc $=$ molecular descriptors. "
        r"\textbf{Bold} = best in column; $^\dag$ = CI overlaps leader. "
        rf"McNemar's test on accuracy gap: $\chi^2 = {chi2:.2f}, p {p_str}$."
    )

    _render_combined(fart_d, gnn_d,
                     caption=caption,
                     label="tab:molecular-prediction",
                     out_path=OUT_DIR / "table_molecular_prediction.tex",
                     outer_size=args.outer,
                     inner_size=args.inner)
    print(f"  fonts: outer=\\{args.outer}  inner=\\{args.inner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
