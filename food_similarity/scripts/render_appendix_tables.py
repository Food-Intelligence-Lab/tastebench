#!/usr/bin/env python3
"""Render the six appendix tables (E.7 and E.8) from committed analysis CSVs.

E.7  Category-balanced evaluation and human comparison   (Tables 14–16)
E.8  Recognition audit and brand-blind probe             (Tables 17–19)

Every cell is read from a committed CSV or JSON. Nothing is modeled or re-fit
here; this script formats numbers that already exist.

Three values cannot be recomputed from released artifacts and are carried as
named constants, each with its source:

  * rater-vs-rater pairwise agreement (.547) -- needs per-panelist NECTAR ratings
  * the E.8b recognition-interaction bootstrap CI -- re-running would drift on seed
  * tau^2 and the shrinkage fraction -- used in the E.7 prose, in no table

Usage:
    python3 food_similarity/scripts/render_appendix_tables.py
    python3 food_similarity/scripts/render_appendix_tables.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent                   # the repository root
ANALYSIS = HERE / "appendix_data"
HB = TREE / "human_baseline" / "results"
OUT = TREE / "paper" / "appendix_tables"

# --------------------------------------------------------------------------
# constants that cannot be recomputed from released artifacts
# --------------------------------------------------------------------------

RATER_VS_RATER = 0.547
# Needs per-panelist ratings (NDA-gated), so this is not recomputable here.

INTERACTION = "$-$.026, 95\\% CI [$-$.159,$+$.108]"
# Bootstrap; re-running under a different seed would move the interval,
# so the published one is carried verbatim.

TAU2, SHRINKAGE_PCT = 0.00208, 91     # E.7 prose only; from hierarchical_estimation.py

# --------------------------------------------------------------------------
# formatting -- paper house style: .XXX with no leading zero, [.lo,.hi] no space
# --------------------------------------------------------------------------


def f3(x: float) -> str:
    s = f"{abs(float(x)):.3f}"
    return s[1:] if s.startswith("0") else s


def s3(x: float) -> str:
    v = float(x)
    sign = "$-$" if v < 0 else "$+$"
    return f"{sign}{f3(v)}"


def pct1(x: float) -> str:
    return f"{100 * float(x):.1f}\\%"


def p3(x: float) -> str:
    return f3(x)


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# E.7a -- micro vs macro vs partial pooling
# --------------------------------------------------------------------------

E7A_GROUPS = [
    ("Ensemble (BT $+$ Gemini)", [
        ("NNLS (BT+Gemini)", "NNLS"),
        ("Rank avg.", "Rank average"),
        ("Mean", "Mean"),
    ]),
    ("Zero-shot LLMs", [
        ("Gemini 3.1 Pro", "Gemini 3.1 Pro"),
        ("Qwen 3.5 397B", "Qwen 3.5 397B-A17B"),
    ]),
    ("Supervised", [
        ("Kernel RankSVM", "Kernel RankSVM"),
        ("Hierarchical BT", "Hierarchical BT"),
        ("Bradley-Terry", "Bradley--Terry"),
        ("Ridge", "Ridge"),
        ("LightGBM", "LightGBM"),
    ]),
    ("Unsupervised", [
        ("MMRF (cosine)", "MMRF (cosine)"),
        ("MMRF (L2)", "MMRF (L2)"),
    ]),
]

EXPECTED_E7A = {
    "NNLS (BT+Gemini)": (".683", ".641", "$-$.042", ".682"),
    "Rank avg.":        (".671", ".629", "$-$.041", ".668"),
    "Gemini 3.1 Pro":   (".654", ".633", "$-$.021", ".666"),
    "Qwen 3.5 397B":    (".630", ".611", "$-$.019", ".639"),
    "Mean":             (".623", ".595", "$-$.028", ".613"),
    "Kernel RankSVM":   (".618", ".587", "$-$.031", ".612"),
    "Hierarchical BT":  (".613", ".558", "$-$.055", ".595"),
    "Bradley-Terry":    (".610", ".567", "$-$.043", ".567"),
    "Ridge":            (".579", ".537", "$-$.042", ".540"),
    "MMRF (cosine)":    (".574", ".573", "$-$.002", ".589"),
    "MMRF (L2)":        (".571", ".570", "$-$.000", ".600"),
    "LightGBM":         (".558", ".532", "$-$.026", ".555"),
}


def build_e7a(fail):
    mm = {r["model"]: r for r in read_csv(ANALYSIS / "A_category_weighted/micro_vs_macro.csv")}
    eb = {r["model"]: r for r in read_csv(ANALYSIS / "A_category_weighted/hierarchical_estimation.csv")}
    corr = {r[""]: float(r["spearman_vs_micro_all"])
            for r in read_csv(ANALYSIS / "A_category_weighted/ranking_correlation.csv")}
    spearman = f3(corr["macro_all"])
    if spearman != ".881":
        fail("E.7a", "Spearman(macro, micro)", spearman, ".881")

    rows = []
    for group, models in E7A_GROUPS:
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{group}}}}} \\\\")
        for key, display in models:
            r, e = mm[key], eb[key]
            cells = (f3(r["micro"]), f3(r["macro"]), s3(r["delta"]), f3(e["hierarchical"]))
            if key in EXPECTED_E7A and cells != EXPECTED_E7A[key]:
                fail("E.7a", key, str(cells), str(EXPECTED_E7A[key]))
            micro, macro, delta, pooled = cells
            if key == "NNLS (BT+Gemini)":
                micro, macro, pooled = (f"\\textbf{{{v}}}" for v in (micro, macro, pooled))
            rows.append(f"\\quad {display} & {micro} & {macro} & {delta} & {pooled} \\\\")

    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{0.95}}
\\caption{{\\textbf{{Category-balanced evaluation preserves the ranking and the best
model.}} Pairwise accuracy on the 935 within-category pairs under three aggregation
schemes. Micro is the paper's reported metric (Table~\\ref{{tab:results}}); macro weights
each of the 24 categories equally; EB is DerSimonian--Laird partial pooling toward the
precision-weighted random-effects grand mean, with each category's sampling variance
taken from its product count. $\\Delta$ $=$ macro $-$ micro. Point estimates only:
intervals for the macro and EB columns come from a within-category percentile bootstrap
rather than the BCa procedure used elsewhere in the paper, so they are not comparable to
the intervals in Table~\\ref{{tab:results}} and are omitted here. Spearman correlation
between the macro and micro orderings is {spearman}.}}
\\label{{tab:micro-macro}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
Model & Micro & Macro & $\\Delta$ & Partial pooling (EB) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# --------------------------------------------------------------------------
# E.7b -- small-category exclusion sweep
# --------------------------------------------------------------------------

BEST = "NNLS (BT+Gemini)"

EXPECTED_E7B = {
    5:  ("24", ".683", ".641", "1 / 1"),
    6:  ("21", ".689", ".661", "1 / 1"),
    7:  ("20", ".694", ".674", "1 / 1"),
    8:  ("18", ".694", ".672", "1 / 1"),
    9:  ("16", ".701", ".682", "1 / 1"),
    10: ("12", ".700", ".674", "1 / 1"),
}


def build_e7b(fail):
    sweep = read_csv(ANALYSIS / "A_category_weighted/exclusion_sweep.csv")
    by_threshold: dict[int, list[dict]] = {}
    for r in sweep:
        by_threshold.setdefault(int(r["min_products"]), []).append(r)

    rows = []
    for threshold in sorted(by_threshold):
        group = by_threshold[threshold]
        kept = group[0]["kept_categories"]
        best = next(r for r in group if r["model"] == BEST)
        rank_micro = 1 + sum(1 for r in group if float(r["micro"]) > float(best["micro"]))
        rank_macro = 1 + sum(1 for r in group if float(r["macro"]) > float(best["macro"]))
        cells = (kept, f3(best["micro"]), f3(best["macro"]), f"{rank_micro} / {rank_macro}")
        if threshold in EXPECTED_E7B and cells != EXPECTED_E7B[threshold]:
            fail("E.7b", f"min_products={threshold}", str(cells), str(EXPECTED_E7B[threshold]))
        rows.append(f"{threshold} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")

    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{0.95}}
\\caption{{\\textbf{{Dropping small categories does not change the best model.}} The best
model (NNLS over Bradley--Terry and Gemini 3.1 Pro) as categories below each
product-count threshold are excluded. Rank is among the twelve models of
Table~\\ref{{tab:micro-macro}}, under micro and macro averaging respectively. The
smallest category has 5 products, so the first row is the full benchmark.}}
\\label{{tab:exclusion-sweep}}
\\begin{{tabular}}{{@{{}}ccccc@{{}}}}
\\toprule
Min.\\ products & Categories kept & Micro & Macro & Rank (micro / macro) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# --------------------------------------------------------------------------
# E.7c -- matched human comparison
# --------------------------------------------------------------------------


def build_e7c(fail):
    summary = read_json(HB / "summary.json")
    split_half = read_json(HB / "split_half_reliability.json")
    comparison = {r["method"]: r for r in read_csv(HB / "comparison_table.csv")}
    blocks = read_csv(HB / "inter_rater_reliability.csv")
    eb = {r["model"]: r for r in read_csv(ANALYSIS / "A_category_weighted/hierarchical_estimation.csv")}

    ind = summary["individual_panelist"]
    median_panelist = f3(ind["median_pairwise_accuracy"])
    iqr_lo, iqr_hi = f3(ind["percentiles"]["25"]), f3(ind["percentiles"]["75"])
    n_panelists = ind["n_panelists"]
    alpha = f3(statistics.mean(float(b["krippendorff_alpha"]) for b in blocks))
    kendall = f3(statistics.median(float(b["kendall_w"]) for b in blocks))
    reliability = f3(split_half["pairwise_accuracy"])
    within_block = f3(comparison["Best model (within-block)"]["pairwise_accuracy"])
    all_pairs = f3(comparison["Best model (all pairs)"]["pairwise_accuracy"])
    pooled = f3(eb[BEST]["hierarchical"])
    rater = f3(RATER_VS_RATER)

    for label, got, want in [
        ("median panelist", median_panelist, ".650"),
        ("IQR low", iqr_lo, ".500"),
        ("IQR high", iqr_hi, ".750"),
        ("n panelists", str(n_panelists), "627"),
        ("Krippendorff alpha (mean)", alpha, ".077"),
        ("Kendall W (median)", kendall, ".108"),
        ("split-half reliability", reliability, ".825"),
        ("model within-block", within_block, ".661"),
        ("model all pairs", all_pairs, ".683"),
        ("model partial-pooled", pooled, ".682"),
    ]:
        if got != want:
            fail("E.7c", label, got, want)

    return f"""\\begin{{table}}[t]
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{0.95}}
\\caption{{\\textbf{{The model against individual panelists, on matched pairs.}}
Agreement quantities for the panel and for the best model. The panelist rows are
computed on the within-block pairs each panelist actually rated; the model is scored on
those same pairs in the first model row, and on all 935 within-category pairs in the
second. The category-balanced row is the partial-pooled figure from
Table~\\ref{{tab:micro-macro}}. Rows are not mutually comparable: the panelist median is
a per-panelist quantity on within-block pairs, not a category-macro one.}}
\\label{{tab:human-comparison}}
\\begin{{tabular}}{{@{{}}lc@{{}}}}
\\toprule
Quantity & Value \\\\
\\midrule
\\multicolumn{{2}}{{@{{}}l}}{{\\emph{{Panel}}}} \\\\
\\quad Panelist vs.\\ another single panelist, pairwise ordering & {rater} \\\\
\\quad Median panelist vs.\\ panel mean, within block & {median_panelist} (IQR {iqr_lo}--{iqr_hi}, $n = {n_panelists}$) \\\\
\\quad Krippendorff's $\\alpha$ (ordinal, per block) & {alpha} (mean) \\\\
\\quad Kendall's $W$ (rank concordance, per block) & {kendall} (median) \\\\
\\quad Split-half reliability of the panel mean & {reliability} \\\\
\\midrule
\\multicolumn{{2}}{{@{{}}l}}{{\\emph{{Best model}}}} \\\\
\\quad Within-block pairs & {within_block} \\\\
\\quad All within-category pairs & {all_pairs} \\\\
\\quad Category-balanced, partial-pooled & {pooled} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# --------------------------------------------------------------------------
# E.8a -- what surfaces in the reasoning
# --------------------------------------------------------------------------

E8A_ROWS = [
    ("identification", "Product identification"),
    ("brand", "Brand mention (mostly analogy)"),
    ("external_recall", "External-performance recall"),
    ("contamination_sig", "Any signature"),
]

EXPECTED_E8A = {
    "identification":    ("12.5\\%", "$+$.042", ".403"),
    "brand":             ("10.2\\%", "$+$.013", ".820"),
    "external_recall":   ("2.4\\%", "$-$.026", ".821"),
    "contamination_sig": ("14.2\\%", "$+$.032", ".489"),
}


def build_e8a(fail):
    flags = {r["flag"]: r for r in read_csv(ANALYSIS / "B_contamination/accuracy_by_flag.csv")}

    totals = {int(r["n_flagged"]) + int(r["n_clean"]) for r in flags.values()}
    if len(totals) != 1:
        fail("E.8a", "denominator", f"inconsistent: {sorted(totals)}", "one value")
    n_total = totals.pop()

    rows = []
    for key, display in E8A_ROWS:
        r = flags[key]
        prevalence = pct1(int(r["n_flagged"]) / n_total)
        cells = (prevalence, s3(r["delta"]), p3(r["fisher_p"]))
        if key in EXPECTED_E8A and cells != EXPECTED_E8A[key]:
            fail("E.8a", key, str(cells), str(EXPECTED_E8A[key]))
        rows.append(f"{display} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    body = "\n".join(rows)
    return n_total, f"""\\begin{{table}}[t]
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{0.95}}
\\caption{{\\textbf{{Recognition rarely surfaces, and it does not track accuracy.}}
Prevalence of each signature in the saved reasoning for the ingredients-plus-image
configuration, and the accuracy difference between the pairs where it appears and the
rest. Scored on the {n_total} pairs for which a graded judgment and a Bradley--Terry
comparison both exist: of the 935 within-category pairs, three have tied panel means and
are not gradable, and one more is unmatched. Fisher's exact test, two-sided; no
signature reaches significance. Signatures are not mutually exclusive.}}
\\label{{tab:recognition-signatures}}
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Signature & Prevalence & $\\Delta$ accuracy (flagged $-$ clean) & Fisher $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# --------------------------------------------------------------------------
# E.8b -- is the edge concentrated where it recognized the product?
# --------------------------------------------------------------------------


def build_e8b(fail):
    rows = read_csv(ANALYSIS / "B_contamination/b5_margin_by_recognition.csv")
    n_total = len(rows)

    def acc(subset, column):
        return sum(float(r[column]) for r in subset) / len(subset)

    recognized = [r for r in rows if r["identification"] == "True"]
    clean = [r for r in rows if r["identification"] != "True"]

    gem_all, bt_all = acc(rows, "gem_correct"), acc(rows, "bt_correct")
    adv_rec = acc(recognized, "gem_correct") - acc(recognized, "bt_correct")
    adv_clean = acc(clean, "gem_correct") - acc(clean, "bt_correct")
    bt_rec, bt_clean = acc(recognized, "bt_correct"), acc(clean, "bt_correct")

    for label, got, want in [
        ("n recognized", str(len(recognized)), "116"),
        ("n not recognized", str(len(clean)), "815"),
        ("recognized margin", s3(adv_rec), "$+$.026"),
        ("not-recognized margin", s3(adv_clean), "$+$.052"),
        ("Gemini overall", f3(gem_all), ".662"),
        ("BT overall", f3(bt_all), ".613"),
        ("overall margin", s3(gem_all - bt_all), "$+$.048"),
        ("BT on recognized", f3(bt_rec), ".672"),
        ("BT on not recognized", f3(bt_clean), ".605"),
    ]:
        if got != want:
            fail("E.8b", label, got, want)

    return f"""\\begin{{table}}[t]
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{0.95}}
\\caption{{\\textbf{{The LLM's margin is not concentrated where it recognized the
product.}} Per-pair margin of Gemini 3.1 Pro over Bradley--Terry, split by whether the
reasoning identified the product. Bradley--Terry sees only features, so it holds pair
difficulty constant. Over these {n_total} pairs Gemini scores {f3(gem_all)} and
Bradley--Terry {f3(bt_all)}, a margin of {s3(gem_all - bt_all)}; both differ slightly
from Table~\\ref{{tab:results}}, which scores the same models over all 935 pairs
including the three ties. Bradley--Terry also improves on the recognized subset
({f3(bt_clean)} to {f3(bt_rec)}), which is what identifies those pairs as easier rather
than leaked. The interaction interval is a within-category bootstrap.}}
\\label{{tab:recognition-margin}}
\\begin{{tabular}}{{@{{}}lcc@{{}}}}
\\toprule
Split & $n$ & Gemini $-$ Bradley--Terry \\\\
\\midrule
Recognized & {len(recognized)} & {s3(adv_rec)} \\\\
Not recognized & {len(clean)} & {s3(adv_clean)} \\\\
\\midrule
Interaction & & {INTERACTION} (n.s.) \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# --------------------------------------------------------------------------
# E.8c -- interventional brand-blind re-run
# --------------------------------------------------------------------------


def build_e8c(fail):
    logs = ANALYSIS / "B4_formulation_only"
    results = sorted(logs.glob("b4_faithful_result_*.json"))
    if not results:
        fail("E.8c", "result json", "missing", "b4_faithful_result_*.json")
        return ""
    d = read_json(results[-1])
    stamp = results[-1].stem.replace("b4_faithful_result_", "")

    paired = read_csv(logs / f"b4_faithful_paired_{stamp}.csv")
    n_paired = len(paired)
    n_correct_o = sum(float(r["correct_o"]) for r in paired)
    n_correct_f = sum(float(r["correct_f"]) for r in paired)
    n_flips = sum(1 for r in paired if r["pred_o"] != r["pred_f"])

    original, formulation = f3(n_correct_o / n_paired), f3(n_correct_f / n_paired)
    delta = s3((n_correct_f - n_correct_o) / n_paired)
    flip = pct1(n_flips / n_paired)
    mcnemar = p3(d["mcnemar_p"])
    brand_before, brand_after = pct1(d["brand_rate_original"]), pct1(d["brand_rate_formulation"])

    if n_paired != d["n_paired"]:
        fail("E.8c", "n_paired", str(n_paired), str(d["n_paired"]))

    for label, got, want in [
        ("original", original, ".639"),
        ("brand-blind", formulation, ".634"),
        ("delta", delta, "$-$.005"),
        ("flip rate", flip, "12.6\\%"),
        ("McNemar p", mcnemar, ".712"),
        ("brand rate before", brand_before, "10.8\\%"),
        ("brand rate after", brand_after, "1.7\\%"),
    ]:
        if got != want:
            fail("E.8c", label, got, want)

    return f"""\\begin{{table}}[t]
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}
\\renewcommand{{\\arraystretch}}{{0.95}}
\\caption{{\\textbf{{Forbidding brand reasoning barely moves accuracy.}} Original versus
brand-blind system prompt, both re-run fresh in the same session so that model drift
cancels; the user prompt is byte-identical between arms, and {n_paired} pairs are scored
in both. Flip rate is the fraction of pairs whose predicted winner changes. The
brand-mention rate is measured on the same reasoning scan as
Table~\\ref{{tab:recognition-signatures}}.}}
\\label{{tab:brand-blind}}
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
Arm & Original & Brand-blind & $\\Delta$ acc.\\ & Flip rate & McNemar $p$ & Brand mentions \\\\
\\midrule
Gemini 3.1 Pro & {original} & {formulation} & {delta} & {flip} & {mcnemar} & {brand_before} $\\to$ {brand_after} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify values and write nothing")
    args = ap.parse_args()

    mismatches: list[str] = []

    def fail(table, label, got, want):
        mismatches.append(f"  {table}  {label}: got {got}, expected {want}")

    tables = {
        "table_micro_macro.tex": build_e7a(fail),
        "table_exclusion_sweep.tex": build_e7b(fail),
        "table_human_comparison.tex": build_e7c(fail),
    }
    n_audited, e8a = build_e8a(fail)
    tables["table_recognition_signatures.tex"] = e8a
    tables["table_recognition_margin.tex"] = build_e8b(fail)
    tables["table_brand_blind.tex"] = build_e8c(fail)

    if mismatches:
        print("VALUE MISMATCHES:", file=sys.stderr)
        print("\n".join(mismatches), file=sys.stderr)
        return 1

    print(f"all values verified (audit set: n = {n_audited})")
    print(f"carried as constants: rater-vs-rater {f3(RATER_VS_RATER)} "
          f"(needs gated ratings); the E.8b interaction CI (bootstrap)")

    if args.check:
        print("--check: nothing written")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in tables.items():
        (OUT / name).write_text(body)
        print(f"  wrote {(OUT / name).relative_to(TREE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
