"""Render the corrected Table 3 (three molecular encoders, matched pipeline).

Replaces the two-column FART-vs-GNN table. The published version compared a FART
column scored by the main pipeline on FoodAtlas v4.0 against a GNN column scored
by a different function on the preceding FoodAtlas version, so the reported
gap confounded the encoder with the scoring pipeline and the data version
(rebuttal r1). All three encoders now go through one pipeline, differing only in
the compound-embedding cache.

Reads the shipped reproduction bundle — no API, no gated data, no training:
    molecular/results/encoder_comparison/downstream_results.csv

Writes: paper/molecular_prediction/table_gnn_per_model.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MOL = HERE.parent
TREE = MOL.parent
SRC = MOL / "results" / "encoder_comparison" / "downstream_results.csv"
OUT = TREE / "paper" / "molecular_prediction" / "table_gnn_per_model.tex"

# archived FART values: the control that licenses the other two columns
GATE = {
    "MMRF (cosine)": 0.574332, "MMRF (L2)": 0.570588, "Ridge": 0.579144,
    "Bradley-Terry": 0.610160, "Hierarchical BT": 0.613369,
    "Kernel RankSVM": 0.617647, "LightGBM": 0.557754, "NNLS ens": 0.682888,
    "Rank ens": 0.670588, "Mean ens": 0.622995,
}

GROUPS = [
    ("Unsupervised", ["MMRF (cosine)", "MMRF (L2)"]),
    ("Supervised -- linear", ["Ridge"]),
    ("Supervised -- pairwise", ["Bradley-Terry", "Hierarchical BT", "Kernel RankSVM"]),
    ("Supervised -- nonlinear", ["LightGBM"]),
    ("Ensemble (BT + Gemini)", ["NNLS ens", "Rank ens", "Mean ens"]),
]
DISPLAY = {"Bradley-Terry": "Bradley--Terry", "NNLS ens": "NNLS",
           "Rank ens": "Rank average", "Mean ens": "Mean"}
ENCODERS = ["FART", "GNN", "ChemBERTa"]
ENC_HEAD = {"FART": "FART", "GNN": "GNN", "ChemBERTa": "ChemBERTa-2"}

CAPTION = (
    r"\textbf{Molecular-encoder quality does not transfer to product-level ranking.} "
    r"Pairwise accuracy on 215 NECTAR products (935 within-category pairs; LOOCV) with the "
    r"compound block supplied by each of three encoders, every other component held fixed: "
    r"FoodAtlas v4.0, log-concentration ingredient aggregation, and the same scoring pipeline "
    r"throughout. The three span .84--.90 molecular-classification accuracy (Table~\ref{tab:molecular-prediction}) "
    r"yet are statistically indistinguishable here. 95\% BCa CIs (10{,}000 resamples). "
    r"$\Delta$ is relative to FART. Every $\Delta$ lies inside the corresponding interval and the "
    r"direction is not consistent across models, so these are ties rather than a ranking. "
    r"The encoders are not tuned symmetrically --- FART is pretrained and untuned, the D-MPNN grid "
    r"swept depth, dropout and class weighting at fixed learning rate, and ChemBERTa-2 swept "
    r"learning rate --- but the intervals ($\approx\pm.05$) dwarf any plausible tuning effect."
)


def f3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.3f}".replace("0.", ".").replace("-0.", "-.")


def delta(x):
    if np.isnan(x):
        return "--"
    sign = "$+$" if x >= 0 else "$-$"
    return sign + f"{abs(x):.3f}".replace("0.", ".")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found (ship the encoder-comparison bundle)")
        return 1
    d = pd.read_csv(SRC)
    cell = {(r.model, r.encoder): r for r in d.itertuples()}

    # The FART column must reproduce the archived paper values, or the comparison
    # is not controlled. Fail loudly rather than render a table we cannot defend.
    for model, want in GATE.items():
        got = cell.get((model, "FART"))
        if got is None:
            print(f"ERROR: missing FART/{model}")
            return 1
        if abs(got.pw - want) > 5e-4:
            print(f"ERROR: FART/{model} = {got.pw:.6f}, archived {want:.6f} — "
                  "the control column has drifted; do not publish this table")
            return 1
    print(f"  FART control column reproduces all {len(GATE)} archived values")

    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{0.95}",
        r"\caption{" + CAPTION + "}",
        r"\label{tab:encoder-transfer}",
        r"\begin{tabular}{l" + "c" * len(ENCODERS) + "cc}", r"\toprule",
        "Model & " + " & ".join(rf"C $=$ {ENC_HEAD[e]}" for e in ENCODERS)
        + r" & $\Delta_{\text{GNN}}$ & $\Delta_{\text{Chem}}$ \\", r"\midrule",
    ]
    for gname, models in GROUPS:
        lines.append(rf"\multicolumn{{{len(ENCODERS)+3}}}{{@{{}}l}}{{\emph{{{gname}}}}} \\")
        for m in models:
            if not all((m, e) in cell for e in ENCODERS):
                continue
            pws = {e: cell[(m, e)].pw for e in ENCODERS}
            best = max(pws.values())
            cells = []
            for e in ENCODERS:
                r = cell[(m, e)]
                pt = f3(r.pw)
                if abs(r.pw - best) < 1e-9:
                    pt = rf"\textbf{{{pt}}}"
                cells.append(pt + rf" {{\scriptsize [{f3(r.ci_lo)},{f3(r.ci_hi)}]}}")
            lines.append(
                f"\\quad {DISPLAY.get(m, m)} & " + " & ".join(cells)
                + f" & {delta(pws['GNN'] - pws['FART'])}"
                + f" & {delta(pws['ChemBERTa'] - pws['FART'])} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {OUT.relative_to(TREE)}")

    # Emit the CSV sidecar too. The renderer this replaces wrote one, and leaving
    # it behind meant the release shipped the retracted numbers beside the
    # corrected table.
    csv_out = MOL / "results" / "tables_csv" / "table_gnn_per_model.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gname, models in GROUPS:
        for m in models:
            if not all((m, e) in cell for e in ENCODERS):
                continue
            r_ = {e: cell[(m, e)] for e in ENCODERS}
            rows.append({
                "group": gname, "model": DISPLAY.get(m, m),
                "fart": r_["FART"].pw, "fart_lo": r_["FART"].ci_lo, "fart_hi": r_["FART"].ci_hi,
                "gnn": r_["GNN"].pw, "gnn_lo": r_["GNN"].ci_lo, "gnn_hi": r_["GNN"].ci_hi,
                "chemberta": r_["ChemBERTa"].pw, "chemberta_lo": r_["ChemBERTa"].ci_lo,
                "chemberta_hi": r_["ChemBERTa"].ci_hi,
                "delta_gnn": r_["GNN"].pw - r_["FART"].pw,
                "delta_chemberta": r_["ChemBERTa"].pw - r_["FART"].pw,
            })
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"  Wrote {csv_out.relative_to(TREE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
