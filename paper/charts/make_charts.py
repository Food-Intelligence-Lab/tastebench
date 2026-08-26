"""Generate NeurIPS-quality charts for the molecular prediction tables (v4).

Chart 1: Single-panel grouped bar chart (Acc, Prec, Rec, F1; AUROC in caption),
         read from Table 2's CSV sidecar plus the shared ChemBERTa-2 row.
Chart 2: Single-panel horizontal grouped bar chart (three compound encoders)
         with inline deltas, read from the encoder-comparison results CSV.

Neither chart holds a number of its own. Each shares its data source with the
table printed beside it — Chart 1 with molecular/scripts/
render_table_molecular_prediction.py, Chart 2 with molecular/scripts/
render_table_encoder_comparison.py; see the comments above each section.
"""

import csv
import importlib.util
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
TREE = OUT.parents[1]          # the repository root

# Omitting /CreationDate makes the PDFs byte-reproducible, so
# `verify_paper.sh --check-diff` can tell a real chart change from a re-render.
PDF_META = {"CreationDate": None}

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe-Ito colorblind-safe palette. ChemBERTa-2 keeps #009E73 in both figures,
# so the encoder reads the same colour in Chart 1 and Chart 2; XGBoost (fp+desc)
# used to share that hex, which was harmless only while ChemBERTa-2 was absent
# from Chart 1 and became two identically-coloured bars in one panel the moment
# it was added. It moves to the palette's unused sky blue.
C_GNN     = "#D55E00"
C_FART    = "#0072B2"
C_CHEM    = "#009E73"
C_XGB_FPD = "#56B4E9"
C_XGB_FP  = "#CC79A7"
C_BRF     = "#E69F00"


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 – Taste classification: single-panel grouped bar chart
#           (Accuracy, Precision, Recall, F1 — AUROC goes in caption)
# ═══════════════════════════════════════════════════════════════════════════

# Every bar and whisker below is READ from the two files Table 2 renders from.
# This section carried the same kind of literal table Chart 2 did, and it went
# stale the same way: the camera-ready correction added a ChemBERTa-2 row to
# Table 2, table_molecular_prediction.tex re-rendered and grew it, and the
# figure printed beside it could not — so the panel showed five models against a
# six-row table. The literals that were here were not *wrong*, which is the
# trap: a figure that agrees with today's results by coincidence is one
# correction away from contradicting them, and nothing was watching.
#
# Two sources, because Table 2 itself has two. The cited tree baselines carry no
# published CIs and FART/GNN carry BCa intervals, and the renderer emits all five
# of those rows to its CSV sidecar in the same call that writes the .tex, so the
# sidecar cannot drift from the table. ChemBERTa-2's row lives in
# molecular/scripts/chemberta_row.py, which exists for exactly this reason: it
# was hard-coded into the one-column renderer only, so the two-column variant
# shipped without the row, and camera-ready task 25 pulled it out into one
# definition with two importers. This chart is the third importer.
#
# check_charts.py fails the build if a number is inlined here again and re-reads
# the rendered PDF against both sources cell by cell.
SRC_C1  = TREE / "molecular" / "results" / "tables_csv" / "table_molecular_prediction.csv"
CHEM_C1 = TREE / "molecular" / "scripts" / "chemberta_row.py"

METRICS_C1 = ["accuracy", "precision", "recall", "f1"]
METRIC_LABEL = {"accuracy": "Accuracy", "precision": "Precision",
                "recall": "Recall", "f1": "F1"}

if not SRC_C1.exists():
    raise SystemExit(f"ERROR: {SRC_C1} not found — Table 2's CSV sidecar is this "
                     "chart's data source; render the table, do not inline it")
if not CHEM_C1.exists():
    raise SystemExit(f"ERROR: {CHEM_C1} not found — the ChemBERTa-2 row lives "
                     "there so the tables and this figure share one definition")

with SRC_C1.open(newline="") as fh:
    tbl2 = {r["model"]: r for r in csv.DictReader(fh)}

_spec = importlib.util.spec_from_file_location("chemberta_row", CHEM_C1)
chemberta_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chemberta_row)
CHEM = chemberta_row.LABEL

# Row order: the three encoders we trained or re-evaluated, then the cited tree
# baselines in the order Table 2 lists them.
ROWS_C1 = ["GNN", "FART", CHEM,
           "XGBoost (fp+desc)", "XGBoost (fp)", "Balanced RF (fp)"]
DISPLAY_C1 = {"GNN": "GNN (ours)", CHEM: "ChemBERTa-2",
              "Balanced RF (fp)": "Balanced RF"}
COLOR_C1 = {"GNN": C_GNN, "FART": C_FART, CHEM: C_CHEM,
            "XGBoost (fp+desc)": C_XGB_FPD, "XGBoost (fp)": C_XGB_FP,
            "Balanced RF (fp)": C_BRF}


def cell_c1(row, metric):
    """(point, ci_lo, ci_hi) for one cell; lo/hi are None when unpublished."""
    if row == CHEM:
        return chemberta_row.ROW[metric]
    r = tbl2[row]
    lo, hi = r[f"{metric}_ci_lo"], r[f"{metric}_ci_hi"]
    return (float(r[metric]),
            float(lo) if lo else None,
            float(hi) if hi else None)


absent = [r for r in ROWS_C1 if r != CHEM and r not in tbl2]
if absent:
    raise SystemExit(f"ERROR: {SRC_C1.name} is missing rows: {absent}")
absent = [m for m in METRICS_C1 + ["auroc"] if m not in chemberta_row.ROW]
if absent:
    raise SystemExit(f"ERROR: {CHEM_C1.name} is missing metrics: {absent}")

vals_c1 = np.array([[cell_c1(r, m)[0] for m in METRICS_C1] for r in ROWS_C1])

# A model gets whiskers only if every metric it is drawn for has an interval.
# A half-populated row would silently draw whiskers on some bars and not others
# and read as a missing interval rather than a missing column, so refuse it
# outright rather than guess which metric the gap belongs to.
cis_c1 = {}
for i, r in enumerate(ROWS_C1):
    bounds = [cell_c1(r, m)[1:] for m in METRICS_C1]
    have = [b for b in bounds if b[0] is not None and b[1] is not None]
    if len(have) == len(bounds):
        cis_c1[i] = np.array(bounds, dtype=float)
    elif have:
        raise SystemExit(f"ERROR: {r} has CIs for only {len(have)} of "
                         f"{len(bounds)} metrics; do not invent the rest")
    # A point outside its own interval means the sources disagree with each other
    # — one of them has moved and the other has not. Say that, rather than letting
    # matplotlib report it downstream as a negative error bar.
    for m, (lo, hi) in zip(METRICS_C1, bounds):
        p = cell_c1(r, m)[0]
        if lo is not None and not lo <= p <= hi:
            raise SystemExit(f"ERROR: {r} {m} = {p} lies outside its own CI "
                             f"[{lo}, {hi}]; the sources have drifted apart")

best_c1 = np.argmax(vals_c1, axis=0)
nm = len(ROWS_C1)
metrics = [METRIC_LABEL[m] for m in METRICS_C1]
n_met = len(metrics)

fig1, ax = plt.subplots(figsize=(5.5, 3.0))

x = np.arange(n_met)
w = 0.13                                 # six bars per group now, not five
offsets = np.linspace(-(nm - 1) / 2 * w, (nm - 1) / 2 * w, nm)

for i, r in enumerate(ROWS_C1):
    yerr = None
    if i in cis_c1:
        lo = vals_c1[i] - cis_c1[i][:, 0]
        hi = cis_c1[i][:, 1] - vals_c1[i]
        yerr = np.array([lo, hi])

    ax.bar(x + offsets[i], vals_c1[i], w,
           color=COLOR_C1[r], edgecolor="white", linewidth=0.3,
           yerr=yerr, error_kw=dict(lw=0.6, capsize=1.5, capthick=0.5),
           zorder=3)

    # Every bar carries its own value, turned on its side to fit the pitch. Only
    # the per-metric best used to be labelled; text is the one part of a
    # matplotlib PDF a checker can read back without reconstructing the axis
    # transform, so labelling all of them is what makes each cell auditable.
    # The best per metric keeps its bold, larger annotation.
    for k in range(n_met):
        top = cis_c1[i][k, 1] if i in cis_c1 else vals_c1[i, k]
        is_best = best_c1[k] == i
        ax.text(x[k] + offsets[i], top + 0.014,
                f"{vals_c1[i, k]:.3f}"[1:],
                ha="center", va="bottom", rotation=90,
                fontsize=6 if is_best else 5,
                color=COLOR_C1[r],
                fontweight="bold" if is_best else "normal", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_ylim(0.50, 1.06)
ax.set_yticks(np.arange(0.50, 1.05, 0.10))
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
ax.grid(axis="y", ls="--", lw=0.3, alpha=0.5)
ax.set_axisbelow(True)

handles = [mpatches.Patch(facecolor=COLOR_C1[r], edgecolor="white",
                          label=DISPLAY_C1.get(r, r)) for r in ROWS_C1]
fig1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
            ncol=nm, frameon=False,
            columnspacing=0.8, handletextpad=0.3, handlelength=1.2)

fig1.savefig(OUT / "chart_molecular_prediction.pdf", metadata=PDF_META)
fig1.savefig(OUT / "chart_molecular_prediction.png")
plt.close(fig1)

# AUROC is charted nowhere — it goes in the caption, so print it for whoever
# writes that caption rather than leaving them to re-derive it from the table.
auroc_c1 = "  ".join("{}={}".format(DISPLAY_C1.get(r, r),
                                   f"{cell_c1(r, 'auroc')[0]:.3f}"[1:])
                     for r in ROWS_C1)
print(f"Saved chart_molecular_prediction (from {SRC_C1.relative_to(TREE)} "
      f"+ {CHEM_C1.relative_to(TREE)})")
print(f"  caption AUROC: {auroc_c1}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2 – Encoder transfer: single-panel horizontal grouped bars + inline Δ
# ═══════════════════════════════════════════════════════════════════════════
#
# Every bar, whisker and Δ below is READ from downstream_results.csv. It used to
# be a literal table in this file, and that is exactly how this figure came to
# ship numbers the paper had already retracted: the published FART-vs-GNN
# comparison scored the two columns with different functions on different
# FoodAtlas versions, so all three encoders were re-run through one matched
# pipeline (rebuttal r1). That moved NNLS/GNN from .635 to .673 and its Δ from
# −.048 to −.010, and added a third encoder. table_gnn_per_model.tex renders
# from this CSV so it picked the correction up; the chart's literals could not
# be re-rendered into agreement, so the figure kept contradicting the table
# printed beside it and nothing detected the drift. Reading the same file the
# table reads is what makes the two artifacts unable to disagree — do not
# reintroduce a hard-coded value here (check_charts.py fails the build if you
# do, and re-checks the rendered PDF against this CSV cell by cell).
#
# Grouping, row order and display names mirror the table's renderer,
# render_table_encoder_comparison.py — added to molecular/scripts/ by
# the release build step that deranks the public tree.
SRC = TREE / "molecular" / "results" / "encoder_comparison" / "downstream_results.csv"

ENCODERS = ["FART", "GNN", "ChemBERTa"]
ENC_LABEL = {"FART": "C = FART", "GNN": "C = GNN",
             "ChemBERTa": "C = ChemBERTa-2"}
ENC_COLOR = {"FART": C_FART, "GNN": C_GNN, "ChemBERTa": C_CHEM}
DELTA_OF = ["GNN", "ChemBERTa"]          # Δ columns, both relative to FART
DELTA_HEAD = {"GNN": "$\\Delta_{\\mathrm{GNN}}$",
              "ChemBERTa": "$\\Delta_{\\mathrm{Chem}}$"}

cats = [
    ("Unsupervised", ["MMRF (cosine)", "MMRF (L2)"]),
    ("Supervised (linear)", ["Ridge"]),
    ("Supervised (pairwise)", ["Bradley-Terry", "Hierarchical BT",
                               "Kernel RankSVM"]),
    ("Supervised (nonlinear)", ["LightGBM"]),
    ("Ensemble (BT + Gemini)", ["NNLS ens", "Rank ens", "Mean ens"]),
]
DISPLAY = {"Bradley-Terry": "Bradley–Terry", "NNLS ens": "NNLS",
           "Rank ens": "Rank average", "Mean ens": "Mean"}

if not SRC.exists():
    raise SystemExit(f"ERROR: {SRC} not found — the encoder-comparison results "
                     "are the chart's only data source; do not inline them")
with SRC.open(newline="") as fh:
    cell = {(r["model"], r["encoder"]):
            (float(r["pw"]), float(r["ci_lo"]), float(r["ci_hi"]))
            for r in csv.DictReader(fh)}
missing = [(m, e) for _, ms in cats for m in ms for e in ENCODERS
           if (m, e) not in cell]
if missing:
    raise SystemExit(f"ERROR: {SRC.name} is missing cells: {missing}")

# Within a group, order rows by the FART column (the reference encoder), best
# first — same convention as before the rewrite.
for _, models in cats:
    models.sort(key=lambda m: cell[(m, "FART")][0], reverse=True)

# Build y-positions with category header rows
all_labels, all_y, is_hdr = [], [], []
data_y, data_rows = [], []

ROW_H = 1.15
HDR_SPACE = 0.55
CAT_GAP = 0.45
BAR_H = 0.30                             # three bars per row, not two
BAR_PITCH = BAR_H + 0.05
# offsets run FART, GNN, ChemBERTa top-to-bottom (the axis is inverted below)
BAR_OFF = {e: (i - (len(ENCODERS) - 1) / 2) * BAR_PITCH
           for i, e in enumerate(ENCODERS)}

yy = 0.0
for ci_idx, (cat, models) in enumerate(cats):
    if ci_idx > 0:
        yy += CAT_GAP
    all_labels.append(cat)
    all_y.append(yy)
    is_hdr.append(True)
    yy += HDR_SPACE
    for m in models:
        all_labels.append(DISPLAY.get(m, m))
        all_y.append(yy)
        is_hdr.append(False)
        data_y.append(yy)
        data_rows.append(m)
        yy += ROW_H

all_y  = np.array(all_y)
data_y = np.array(data_y)
nd = len(data_y)

# pw / CI / Δ arrays, one column per encoder, in row order
pw  = {e: np.array([cell[(m, e)][0] for m in data_rows]) for e in ENCODERS}
ci  = {e: np.array([cell[(m, e)][1:] for m in data_rows]) for e in ENCODERS}
delta = {e: pw[e] - pw["FART"] for e in DELTA_OF}

# Three groups are enough that the row is still legible at ROW_H, but the
# margin text needs a little more vertical room than the two-bar version had.
fig2, ax2 = plt.subplots(figsize=(5.5, 5.8))

# Solid bars
for e in ENCODERS:
    ax2.barh(data_y + BAR_OFF[e], pw[e], height=BAR_H,
             color=ENC_COLOR[e], edgecolor="white", linewidth=0.3,
             label=ENC_LABEL[e], zorder=3)

# Faint CI whiskers (drawn separately so they don't fight with text)
for i in range(nd):
    for e in ENCODERS:
        yc, arr, col = data_y[i] + BAR_OFF[e], ci[e], ENC_COLOR[e]
        ax2.plot([arr[i, 0], arr[i, 1]], [yc, yc],
                 color=col, lw=0.8, alpha=0.4, zorder=2)
        for ep in [0, 1]:
            ax2.plot([arr[i, ep]] * 2,
                     [yc - BAR_H * 0.35, yc + BAR_H * 0.35],
                     color=col, lw=0.6, alpha=0.4, zorder=2)

# Value + delta columns in right margin (past all CIs)
val_x = max(np.max(ci[e][:, 1]) for e in ENCODERS) + 0.015
delta_x = {e: val_x + 0.042 + 0.038 * i for i, e in enumerate(DELTA_OF)}

# Column headers
hdr_y = all_y[0] - 0.5
ax2.text(val_x, hdr_y, "Acc.",
         ha="left", va="bottom", fontsize=7, color="#555555", fontweight="bold")
for e in DELTA_OF:
    ax2.text(delta_x[e], hdr_y, DELTA_HEAD[e],
             ha="left", va="bottom", fontsize=7, color="#555555",
             fontweight="bold")

for i in range(nd):
    for e in ENCODERS:
        ax2.text(val_x, data_y[i] + BAR_OFF[e],
                 f"{pw[e][i]:.3f}"[1:],
                 ha="left", va="center", fontsize=6, color=ENC_COLOR[e],
                 fontweight="bold", zorder=5)

    # Two Δ columns now, so they are coloured by the encoder they belong to;
    # the sign carries the direction (a single Δ column could colour by sign).
    for e in DELTA_OF:
        s = f"{delta[e][i]:+.3f}"
        s = s[0] + s[2:]
        s = s.replace("-", "−")
        ax2.text(delta_x[e], data_y[i], s,
                 ha="left", va="center", fontsize=6, color=ENC_COLOR[e],
                 fontweight="bold")

ax2.set_xlabel("Pairwise accuracy (LOOCV)")
ax2.set_xlim(0.47, max(delta_x.values()) + 0.024)
ax2.axvline(0.5, ls=":", lw=0.8, color="#bbbbbb", zorder=0)
ax2.grid(axis="x", ls="--", lw=0.3, alpha=0.5)
ax2.set_axisbelow(True)
# Three entries no longer fit inside the axes without covering the top group's
# CI whiskers, so the legend moves above the panel — same placement Chart 1 uses.
ax2.legend(handles=[mpatches.Patch(facecolor=ENC_COLOR[e], edgecolor="white",
                                   label=ENC_LABEL[e]) for e in ENCODERS],
           loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=len(ENCODERS),
           frameon=False, fontsize=7, columnspacing=1.4, handletextpad=0.4,
           handlelength=1.2)

ax2.set_yticks(all_y)
ytl = ax2.set_yticklabels(all_labels)
for i, t in enumerate(ytl):
    if is_hdr[i]:
        t.set_fontstyle("italic")
        t.set_fontsize(6.5)
        t.set_color("#666666")
    else:
        t.set_fontsize(7.5)

ax2.invert_yaxis()

fig2.savefig(OUT / "chart_gnn_per_model.pdf", metadata=PDF_META)
fig2.savefig(OUT / "chart_gnn_per_model.png")
plt.close(fig2)


print(f"Saved chart_gnn_per_model (from {SRC.relative_to(TREE)})")
