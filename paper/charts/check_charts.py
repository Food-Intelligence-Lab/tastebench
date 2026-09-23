"""Assert the paper/charts figures still agree with the results they chart.

Run after make_charts.py; verify_paper.sh chains the two.

WHY THIS EXISTS. Both figures used to carry their numbers as Python literals, and
both went stale, for the same reason: a .tex table renders from a results file
and picks a correction up automatically, and a literal cannot be re-rendered.

  * chart_gnn_per_model: when the FART-vs-GNN comparison was re-run through one
    matched pipeline (rebuttal r1) the shipped PDF kept showing NNLS/GNN .635
    and Δ = −.048 beside a table that said .673 and −.010.
  * chart_molecular_prediction: the camera-ready correction added a ChemBERTa-2
    row to Table 2, so the figure showed five models beside a six-row table.

No check looked at either figure, so nothing caught either one. make_charts.py
now reads the tables' own sources; this script is the tripwire that keeps it that
way. Per chart, cheapest first:

  1. no literal accuracy table in that chart's section of make_charts.py
  2. the source's control values have not drifted, so the comparison the chart
     draws is still the one the paper licenses — the archived FART column for
     Chart 2 (the same gate molecular/scripts/render_table_encoder_comparison.py
     applies to the table), and for Chart 1 the sidecar's agreement with the BCa
     cache Table 2 bootstraps from, which is what stops a stale sidecar from
     moving the chart and this checker together
  3. every number printed in the committed PDF is the source's number, matched
     to its row geometrically — no ordering assumptions, so this does not have
     to be kept in sync with either chart's layout
  4. mutation test: re-render against a perturbed copy of the source in a scratch
     tree and confirm the figure follows it. This is what makes check 1 redundant
     rather than load-bearing — a literal that happened to agree with today's
     source would still fail here.

Check 3 reads the committed PDF and check 4 re-renders, so between them a value
is never compared against itself: a stale PDF fails 3 while a chart that ignores
its source fails 4. Also note that Chart 1 labels every bar rather than only the
per-metric best, because text is the one part of a matplotlib PDF check 3 can
read back without reconstructing the axis transform.

Usage:  python3 check_charts.py        (exit 0 = charts match the results)
"""

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
SCRIPT = HERE / "make_charts.py"
PDF = HERE / "chart_gnn_per_model.pdf"
REL_CSV = Path("molecular/results/encoder_comparison/downstream_results.csv")
SRC = TREE / REL_CSV

ENCODERS = ["FART", "GNN", "ChemBERTa"]
DELTA_OF = ["GNN", "ChemBERTa"]          # left-to-right, as make_charts.py draws them

PDF_C1 = HERE / "chart_molecular_prediction.pdf"
REL_C1 = Path("molecular/results/tables_csv/table_molecular_prediction.csv")
REL_C1_CHEM = Path("molecular/scripts/chemberta_row.py")
REL_C1_CACHE = Path("molecular/results/cis_molecular_prediction.csv")
REL_C1_RENDERER = Path("molecular/scripts/render_table_molecular_prediction.py")
SRC_C1 = TREE / REL_C1
CHEM_C1 = TREE / REL_C1_CHEM
CACHE_C1 = TREE / REL_C1_CACHE

C1_METRICS = ["accuracy", "precision", "recall", "f1"]
# The rows Table 2 bootstraps rather than cites, i.e. the ones whose CIs the
# sidecar and the cache both carry.
C1_OURS = ["FART", "GNN"]

# Archived FART column from the published table: the control for the other two.
GATE = {
    "MMRF (cosine)": 0.574332, "MMRF (L2)": 0.570588, "Ridge": 0.579144,
    "Bradley-Terry": 0.610160, "Hierarchical BT": 0.613369,
    "Kernel RankSVM": 0.617647, "LightGBM": 0.557754, "NNLS ens": 0.682888,
    "Rank ens": 0.670588, "Mean ens": 0.622995,
}

failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def f3(x):
    """Chart's own value format: .683, and no leading zero."""
    return f"{x:.3f}"[1:]


def d3(x):
    """Chart's own delta format, minus the typographic minus (see pdf_text)."""
    s = f"{x:+.3f}"
    return (s[0] + s[2:]) if x >= 0 else s[2:]


def read_rows(path):
    import csv
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_csv(path):
    """Encoder-comparison cells: {(model, encoder): (pw, ci_lo, ci_hi)}."""
    return {(r["model"], r["encoder"]):
            (float(r["pw"]), float(r["ci_lo"]), float(r["ci_hi"]))
            for r in read_rows(path)}


def load_module(path):
    """Import a declarative data module (chemberta_row) by path, not by name.

    make_charts.py runs from paper/charts and the table renderers run from
    molecular/, so neither can reach the other's sys.path; both load this file
    the same way, which is the point of it existing.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# matplotlib emits each text run as `<matrix> cm / BT / <font> Tf / <off> Td /
# [ (chars) ] TJ`. The matrix is `1 0 -0 1 x y` for upright text and `0 1 -1 0
# x y` for the quarter-turned bar labels in Chart 1, so only its last two
# components — the position — are captured. The typographic minus is drawn as a
# glyph path rather than a character, so a negative delta arrives as a bare
# ".048" — absence of a leading "+" is what marks it negative, which d3()
# mirrors.
TEXT_OP = re.compile(
    r"(?:-?[\d.]+ ){4}(-?[\d.]+) (-?[\d.]+) cm\s*\nBT\s*\n/\w+ [\d.]+ Tf\s*\n"
    r"-?[\d.]+ -?[\d.]+ Td\s*\n\[ \((.*?)\) \] TJ")


def pdf_text(path):
    """Return [(x, y, text)] for every text run in a matplotlib PDF."""
    raw = path.read_bytes()
    blobs = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        try:
            blobs.append(zlib.decompress(m.group(1)))
        except zlib.error:
            continue          # uncompressed / font stream
    body = b"\n".join(blobs).decode("latin-1")
    return [(float(x), float(y), t) for x, y, t in TEXT_OP.findall(body)]


def numeric_columns(items):
    """Split the right-margin text into (value column, [delta columns])."""
    cols = defaultdict(list)
    for x, y, t in items:
        if re.fullmatch(r"\+?\.\d{3}", t):
            cols[round(x, 1)].append((y, t))
    vals = [x for x, c in cols.items() if len(c) == len(ENCODERS) * len(GATE)]
    deltas = sorted(x for x, c in cols.items() if len(c) == len(GATE))
    if len(vals) != 1 or len(deltas) != len(DELTA_OF):
        raise SystemExit(
            f"ERROR: expected one {len(ENCODERS) * len(GATE)}-cell value column and "
            f"{len(DELTA_OF)} {len(GATE)}-cell delta columns in {PDF.name}; found "
            f"value={vals} delta={deltas}. The chart's margin layout changed — "
            "update this checker deliberately, do not delete it.")
    return cols[vals[0]], [cols[x] for x in deltas]


def check_pdf_against_csv(pdf, cell):
    """Match each charted row to a CSV model by geometry, then compare cells."""
    n0 = len(failures)
    val_col, delta_cols = numeric_columns(pdf_text(pdf))

    # The delta labels sit on the row centre, which is also the middle bar's
    # centre, so a delta's y identifies its row; the three encoder values are
    # that y and its two nearest neighbours in the value column.
    by_y = {round(y, 1): t for y, t in val_col}
    ys = sorted(by_y)
    charted = {}
    for y, t in sorted(delta_cols[0]):
        i = ys.index(round(y, 1))
        if not 0 < i < len(ys) - 1:
            fail(f"delta at y={y} has no bar above/below it")
            return
        # PDF y grows upward and the chart's y axis is inverted, so the value
        # column reads ChemBERTa, GNN, FART bottom-to-top.
        triple = tuple(by_y[ys[j]] for j in (i + 1, i, i - 1))
        charted[triple] = [dict((round(yy, 1), tt) for yy, tt in col)[round(y, 1)]
                           for col in delta_cols]

    expected = {}
    for model in GATE:
        pw = {e: cell[(model, e)][0] for e in ENCODERS}
        expected[tuple(f3(pw[e]) for e in ENCODERS)] = \
            [d3(pw[e] - pw["FART"]) for e in DELTA_OF]

    for triple, got in charted.items():
        want = expected.get(triple)
        if want is None:
            fail(f"charted row {triple} matches no row of {REL_CSV.name} "
                 "(the figure is showing numbers the results do not contain)")
        elif got != want:
            fail(f"row {triple}: charted Δ {got}, results say {want}")
    for triple in expected:
        if triple not in charted:
            fail(f"{REL_CSV.name} row {triple} is missing from the figure")
    if len(failures) == n0:
        ok(f"all {len(expected)} rows × {len(ENCODERS)} encoders + "
           f"{len(DELTA_OF)} Δ columns match {REL_CSV}")


def check_no_literals(script, section, srcs, until=None):
    """Reject a re-inlined data table in one chart's section of make_charts.py.

    Two shapes are rejected, one per chart's original sin: a quoted label sitting
    next to a float (Chart 2's ``("NNLS ens", .635, ...)`` rows) and a bracket
    opening on two floats (Chart 1's ``np.array([[.904, .825, ...]])`` matrix and
    its ``[[.890,.915], ...]`` CI pairs). ``srcs`` is [(name, token)]; each token
    must still appear in the section, so deleting the read along with the
    literals does not pass by default.
    """
    body = script.read_text()
    start = body.index(f"# {section}")
    stop = body.index(f"# {until}", start + 1) if until else len(body)
    sect = body[start:stop]
    rows = re.findall(r"""\(\s*["'][^"']+["']\s*,\s*-?\.?\d""", sect)
    rows += re.findall(r"\[\s*-?\d*\.\d+\s*,\s*-?\d*\.\d+", sect)
    names = ", ".join(n for n, _ in srcs)
    if rows:
        fail(f"{script.name} {section} lists {len(rows)} literal data row(s) "
             f"({rows[0].strip()}...); it must read {names} instead")
        return
    absent = [n for n, tok in srcs if tok not in sect]
    if absent:
        fail(f"{script.name} {section} no longer reads {', '.join(absent)}")
    else:
        ok(f"{script.name} {section} holds no literal data table (reads {names})")


def c1_expected():
    """{model: (acc, prec, rec, f1)} in the chart's own format, from the sources.

    Table 2's two sources: the sidecar its renderer writes in the same call as
    the .tex, and chemberta_row.py, the single definition of the ChemBERTa-2 row.
    A ChemBERTa row in the sidecar would collide with the module's on the same
    label, so single-sourcing it stays single-sourced.
    """
    exp = {r["model"]: tuple(f3(float(r[m])) for m in C1_METRICS)
           for r in read_rows(SRC_C1)}
    chem = load_module(CHEM_C1)
    exp[chem.LABEL] = tuple(f3(chem.ROW[m][0]) for m in C1_METRICS)
    return exp


def check_c1_control(expected):
    """The sidecar's bootstrapped rows must still match Table 2's BCa cache.

    Chart and checker both read the sidecar, so a stale sidecar would move both
    and agree with itself — the one failure mode a self-consistent check cannot
    see. The cache is what render_table_molecular_prediction.py actually
    bootstraps from, so it is the independent anchor. The cited tree baselines
    have no cache entry and no published CIs; they are literals in the renderer
    because they are another paper's numbers, not ours to re-run.
    """
    if not CACHE_C1.exists():
        fail(f"{REL_C1_CACHE} not found; {REL_C1.name} has nothing to be "
             "checked against and could be stale without detection")
        return
    cache = {(r["label"], r["metric"]): float(r["point"])
             for r in read_rows(CACHE_C1)}
    side = {r["model"]: r for r in read_rows(SRC_C1)}
    drift = []
    for label in C1_OURS:
        for m in C1_METRICS:
            want, got = cache.get((label, m)), side.get(label, {}).get(m)
            if want is None or got is None:
                drift.append(f"{label}/{m}: absent")
            elif abs(float(got) - want) > 5e-4:
                drift.append(f"{label}/{m}: {float(got):.6f} vs cache {want:.6f}")
    if drift:
        fail(f"{REL_C1.name} has drifted from {REL_C1_CACHE.name}, so the "
             "sidecar is stale and the figure is charting superseded numbers: "
             + "; ".join(drift))
    else:
        ok(f"{REL_C1.name} reproduces all {len(C1_OURS) * len(C1_METRICS)} "
           f"bootstrapped cells in {REL_C1_CACHE.name}")


def check_pdf_against_source(pdf, expected):
    """Match each charted series to a source row by geometry, then compare cells.

    Chart 1 labels all of its bars, which cluster into one x-group per metric.
    One slot of each group belongs to one model, so reading slot i across the
    groups rebuilds that model's row whichever slot the chart drew it in — the
    same ordering-free match check_pdf_against_csv makes for Chart 2. Only
    left-to-right group order is assumed to be the metric order, and that is a
    caption-level convention rather than a layout detail.
    """
    n0 = len(failures)
    labels = sorted((x, t) for x, _, t in pdf_text(pdf)
                    if re.fullmatch(r"\.\d{3}", t))
    n_met, n_row = len(C1_METRICS), len(expected)
    if len(labels) != n_met * n_row:
        raise SystemExit(
            f"ERROR: expected {n_met} x {n_row} labelled bars in {pdf.name}, "
            f"found {len(labels)}. Either the chart stopped labelling every bar "
            "or its sources changed shape — update this checker deliberately, "
            "do not delete it.")
    splits = sorted(sorted(range(1, len(labels)),
                           key=lambda i: labels[i][0] - labels[i - 1][0]
                           )[-(n_met - 1):])
    bounds = [0] + splits + [len(labels)]
    groups = [labels[a:b] for a, b in zip(bounds, bounds[1:])]
    if any(len(g) != n_row for g in groups):
        raise SystemExit(
            f"ERROR: {pdf.name}'s bar labels split into groups of "
            f"{[len(g) for g in groups]}, not {n_met} x {n_row}; the chart's "
            "grouping changed — update this checker deliberately.")
    charted = {tuple(g[i][1] for g in groups) for i in range(n_row)}

    by_row = {v: k for k, v in expected.items()}
    matched = set()
    for row in sorted(charted):
        if row in by_row:
            matched.add(by_row[row])
            continue
        # Name the cell, not the row: the nearest source row by Hamming distance
        # is the one the figure was trying to draw.
        name, want = min(expected.items(),
                         key=lambda kv: sum(a != b for a, b in zip(kv[1], row)))
        matched.add(name)
        fail(f"{name}: " + "; ".join(
            f"charted {C1_METRICS[i]} {row[i]}, sources say {want[i]}"
            for i in range(n_met) if row[i] != want[i]))
    for name in expected:
        if name not in matched:
            fail(f"{name} is missing from {pdf.name}, but "
                 f"{REL_C1.name}/{REL_C1_CHEM.name} carry it")
    if len(failures) == n0:
        ok(f"all {n_row} rows x {n_met} metrics match {REL_C1} + {REL_C1_CHEM}")


def check_chemberta_single_source():
    """If Table 2's renderer still inlines the ChemBERTa row, it must agree.

    Camera-ready task 25 pulled the row into chemberta_row.py because it had been
    hard-coded in the one-column renderer only, so the two-column variant shipped
    without it — but it rewired only the two-column renderer, leaving the inline
    copy in place. Chart 1 reads the module, so if the two ever diverge the figure
    contradicts the table again. In-repo the renderer has no inline copy and this
    is a no-op; it fires in the built tree, where task 18 puts one there.
    """
    r = TREE / REL_C1_RENDERER
    if not r.exists():
        return
    m = re.search(r"\bchemberta\s*=\s*\{(.*?)\}", r.read_text(), re.S)
    if not m:
        ok(f"{REL_C1_RENDERER.name} inlines no ChemBERTa row "
           f"({REL_C1_CHEM.name} is the only definition)")
        return
    inline = dict(re.findall(r'"(\w+)":\s*\(([^)]*)\)', m.group(1)))
    row = load_module(CHEM_C1).ROW
    drift = []
    for metric, txt in inline.items():
        got = tuple(None if v.strip() == "None" else float(v)
                    for v in txt.split(","))
        if metric not in row:
            drift.append(f"{metric}: only in the renderer")
        elif got != tuple(row[metric]):
            drift.append(f"{metric}: renderer {got} vs {tuple(row[metric])}")
    if drift:
        fail(f"{REL_C1_RENDERER.name}'s inline ChemBERTa row disagrees with "
             f"{REL_C1_CHEM.name}, which Chart 1 charts: " + "; ".join(drift))
    else:
        ok(f"{REL_C1_RENDERER.name}'s inline ChemBERTa row matches "
           f"{REL_C1_CHEM.name}")


def check_control_column(cell):
    drift = []
    for model, want in GATE.items():
        got = cell.get((model, "FART"))
        if got is None:
            drift.append(f"{model}: absent")
        elif abs(got[0] - want) > 5e-4:
            drift.append(f"{model}: {got[0]:.6f} vs archived {want:.6f}")
    if drift:
        fail("the FART control column has drifted, so the GNN/ChemBERTa "
             "columns are no longer a controlled comparison: " + "; ".join(drift))
    else:
        ok(f"FART control column reproduces all {len(GATE)} archived values")


# make_charts.py renders both charts in one pass, so a mirror has to carry every
# file either chart reads even when only one of them is being probed.
MIRRORED = [REL_CSV, REL_C1, REL_C1_CHEM]


def render_in_mirror(edits, pdf_name):
    """Render both charts from a scratch mirror of the tree, and read one back.

    make_charts.py resolves both its inputs and its outputs from __file__, so a
    mirror of those paths is enough to redirect it — no flags to keep in sync.
    ``edits`` maps a mirrored relative path to replacement text. Every render
    happens inside the mirror, so this says nothing about whether the committed
    PDF is current (check 3's job) and stays valid when it is not.

    Returns (Counter of the PDF's text runs, None) or (None, stderr).
    """
    with tempfile.TemporaryDirectory() as td:
        mirror = Path(td)
        (mirror / "paper/charts").mkdir(parents=True)
        shutil.copy2(SCRIPT, mirror / "paper/charts/make_charts.py")
        for rel in MIRRORED:
            (mirror / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(TREE / rel, mirror / rel)
        for rel, text in edits.items():
            (mirror / rel).write_text(text)
        r = subprocess.run([sys.executable, "make_charts.py"],
                           cwd=mirror / "paper/charts",
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr.strip()[-300:]
        return Counter(t for _, _, t in
                       pdf_text(mirror / "paper/charts" / pdf_name)), None


def report_mutation(label, moved, want, rel, pdf_name):
    if moved != want:
        fail(f"perturbing {label} changed {pdf_name} by +{dict(moved[0])} "
             f"-{dict(moved[1])}, expected +{dict(want[0])} -{dict(want[1])} — "
             f"the chart is not reading {rel.name}, or a stale literal is "
             "overriding it")
    else:
        ok(f"mutation test: {label} moves {sum(want[0].values())} label(s) in "
           f"{pdf_name} and nothing else")


def check_data_driven(cell):
    """Perturb the encoder CSV in a scratch mirror; the figure must follow."""
    probe_model, probe_enc = "NNLS ens", "GNN"
    real = cell[(probe_model, probe_enc)][0]
    fake = round(real - 0.037, 6)          # .673 -> .636
    pristine = SRC.read_text()
    perturbed = pristine.replace(f",{real},", f",{fake},", 1)
    if f",{fake}," not in perturbed:
        raise SystemExit(f"ERROR: could not perturb {probe_model}/{probe_enc} "
                         f"in {REL_CSV.name}")

    shown = []
    for text in (pristine, perturbed):
        counts, err = render_in_mirror({REL_CSV: text}, PDF.name)
        if err:
            fail(f"mutation test could not re-render: {err}")
            return
        shown.append(counts)

    # Two labels must move and no others: the probed cell and its Δ, which is
    # derived from the same number. Counting rather than testing membership
    # matters — .641 and .635 each occur twice in this figure, so "the old value
    # is gone" would be the wrong question.
    fart = cell[(probe_model, "FART")][0]
    report_mutation(
        f"{probe_model}/{probe_enc} {f3(real)} -> {f3(fake)}",
        (shown[1] - shown[0], shown[0] - shown[1]),
        (Counter([f3(fake), d3(fake - fart)]),
         Counter([f3(real), d3(real - fart)])),
        REL_CSV, PDF.name)


def check_data_driven_c1():
    """Perturb Table 2's sidecar in a scratch mirror; the figure must follow.

    The probe is a cited baseline cell, which carries no CI. Moving a
    bootstrapped point estimate away from its own interval would make the
    whiskers negative and the render would fail for a reason that has nothing to
    do with whether the chart read the file.
    """
    probe_row, probe_metric = "XGBoost (fp)", "accuracy"
    row = next((r for r in read_rows(SRC_C1) if r["model"] == probe_row), None)
    if row is None:
        raise SystemExit(f"ERROR: {REL_C1.name} has no {probe_row} row to probe")
    real, fake = row[probe_metric], "0.8588"        # .899 -> .859

    pristine = SRC_C1.read_text()
    lines = pristine.splitlines(keepends=True)
    hit = [i for i, ln in enumerate(lines)
           if ln.startswith(probe_row + ",") and f",{real}," in ln]
    if len(hit) != 1:
        raise SystemExit(f"ERROR: could not perturb {probe_row}/{probe_metric} "
                         f"in {REL_C1.name}")
    lines[hit[0]] = lines[hit[0]].replace(f",{real},", f",{fake},", 1)

    shown = []
    for text in (pristine, "".join(lines)):
        counts, err = render_in_mirror({REL_C1: text}, PDF_C1.name)
        if err:
            fail(f"mutation test could not re-render: {err}")
            return
        shown.append(counts)

    report_mutation(
        f"{probe_row}/{probe_metric} {f3(float(real))} -> {f3(float(fake))}",
        (shown[1] - shown[0], shown[0] - shown[1]),
        (Counter([f3(float(fake))]), Counter([f3(float(real))])),
        REL_C1, PDF_C1.name)


def main() -> int:
    print("Checking paper/charts against the sources the paper tables render from")
    for p in (SRC, SRC_C1, CHEM_C1):
        if not p.exists():
            print(f"  FAIL  {p} not found; a chart has no data source")
            return 1
    for p in (PDF, PDF_C1):
        if not p.exists():
            print(f"  FAIL  {p} not found; run make_charts.py first")
            return 1
    cell = read_csv(SRC)

    print("\nChart 1 – chart_molecular_prediction vs Table 2")
    check_no_literals(SCRIPT, "Chart 1",
                      [(REL_C1.name, "tables_csv"),
                       (REL_C1_CHEM.name, "chemberta_row")], until="Chart 2")
    check_c1_control(c1_expected())
    check_chemberta_single_source()
    check_pdf_against_source(PDF_C1, c1_expected())
    check_data_driven_c1()

    print("\nChart 2 – chart_gnn_per_model vs Table 3")
    check_no_literals(SCRIPT, "Chart 2", [(REL_CSV.name, "encoder_comparison")])
    check_control_column(cell)
    check_pdf_against_csv(PDF, cell)
    check_data_driven(cell)
    print()

    if failures:
        print(f"{len(failures)} check(s) failed: a figure and the results "
              "disagree. Re-run make_charts.py; if that does not fix it, the "
              "chart is not reading the results file.")
        return 1
    print("All checks passed: neither figure can disagree with the table beside "
          "it — table_molecular_prediction.tex and table_gnn_per_model.tex "
          "render from the same sources the charts do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
