"""Render the food-similarity results table split for a two-column workshop.

The single 7-column ``table_results.tex`` is wider than ``\\columnwidth``
in a typical two-column layout. This renderer splits it by metric family
into two single-column tables that share the same row set, the same
cache, and the same bolding/CI-overlap conventions:

    table_results_ranking.tex     Pw.Acc, Spearman, Kendall
    table_results_retrieval.tex   R@1, R@2, R@3

Reads the same BCa-CI cache (``cis_table_results.csv``) populated by
``render_table_results.py`` -- no recomputation. If the cache is
missing, run the original renderer first.

Font tuning:
    --outer  size of the table body (default scriptsize)
    --inner  size of the bracketed CI under each cell
             (default scriptsize)

Drop in any of: tiny, scriptsize, footnotesize, small. The inner size
is also exposed as the ``\\tabci`` macro at the top of each emitted
.tex, so you can hand-tune in Overleaf by editing one line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from render_table_results import (  # noqa: E402
    CACHE,
    MAIN_ROWS,
    SUPERVISED_DIR,
    _load_cache,
    _overlaps,
    _short,
    theoretical_random_baseline,
)

OUT_DIR = SUPERVISED_DIR.parent / "paper" / "two_column" / "model_results_tables"

METRICS_RANKING = [
    ("pairwise_accuracy", r"Pw.\ Acc."),
    ("spearman",          r"$\rho_{\mathrm{S}}$"),
    ("kendall_tau",       r"$\tau_{\mathrm{K}}$"),
]
METRICS_RETRIEVAL = [
    ("recall_at_1", "R@1"),
    ("recall_at_2", "R@2"),
    ("recall_at_3", "R@3"),
]


def _stack_cell(p: float, lo: float, hi: float,
                bold: bool, overlap: bool) -> str:
    """Stacked makecell with optional bold + dagger marker.

    Inner CI line wraps in ``\\tabci`` (defined at the top of the table)
    so its size can be retuned in one place.
    """
    if np.isnan(p):
        return "--"
    pt = _short(p)
    if bold:
        pt = r"\textbf{" + pt + "}"
    elif overlap:
        pt = pt + r"$^\dag$"
    if np.isnan(lo) or np.isnan(hi):
        return pt
    return r"\makecell{" + pt + r" \\ {\tabci [" + _short(lo) + "," + _short(hi) + r"]}}"


def render_split(rows_data: dict, metrics: list[tuple[str, str]],
                 label: str, caption: str, out_path: Path,
                 outer_size: str, inner_size: str) -> None:
    """Render one metric-family split into ``out_path``."""
    best_per: dict[str, set] = {}
    leader_ci: dict[str, tuple[float, float]] = {}
    for metric, _ in metrics:
        max_v = max(d["metrics"][metric] for d in rows_data.values())
        best_per[metric] = {key for key, d in rows_data.items()
                            if abs(d["metrics"][metric] - max_v) < 1e-9}
        for key, d in rows_data.items():
            if abs(d["metrics"][metric] - max_v) < 1e-9:
                leader_ci[metric] = d["cis"][metric]
                break

    def cell(key: str, metric: str) -> str:
        d = rows_data[key]
        is_best = key in best_per[metric]
        lo, hi = d["cis"][metric]
        overlap = (not is_best) and metric in leader_ci \
            and _overlaps(lo, hi, *leader_ci[metric])
        return _stack_cell(d["metrics"][metric], lo, hi,
                           bold=is_best, overlap=overlap)

    header = " & ".join(["Model"] + [h for _, h in metrics]) + r" \\"
    n_cols = len(metrics)

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
        r"\begin{tabular}{@{}l" + "c" * n_cols + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    random_baseline = theoretical_random_baseline()
    for key, lbl, f in MAIN_ROWS:
        if key == "midrule":
            lines.append(r"\midrule"); continue
        if key == "random":
            cells = [random_baseline[m] for m, _ in metrics]
            lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")
            continue
        cells = [cell(key, m) for m, _ in metrics]
        lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path.relative_to(SUPERVISED_DIR.parent)}")


CAPTION_RANKING = (
    r"\textbf{Ensembles of supervised models and LLMs deliver the "
    r"strongest continuous ranking on all three metrics.} "
    r"Pairwise accuracy and rank-correlation metrics on 215 NECTAR "
    r"plant-based products (935 within-category pairs; LOOCV) with "
    r"95\% BCa CIs. \textbf{Bold} = best in column; $^\dag$ = CI "
    r"overlaps the leader (no significant difference at 95\%). "
    r"Top-$k$ retrieval in Table~\ref{tab:results-retrieval}."
)

CAPTION_RETRIEVAL = (
    r"\textbf{Recall metrics indicate the practical utility of these "
    r"models for recommending the best formulations.} "
    r"R@$k$ is the fraction of categories whose top-truth product is "
    r"among the model's top $k$ predictions. Random rates are "
    r"macro-averaged $\min(k, n_c)/n_c$ over 24 categories. "
    r"\textbf{Bold} = best in column; $^\dag$ = CI overlaps the leader. "
    r"Continuous ranking metrics in Table~\ref{tab:results-ranking}."
)


FONT_CHOICES = ("tiny", "scriptsize", "footnotesize", "small")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outer", default="footnotesize", choices=FONT_CHOICES,
                        help="font size for the table body (default: footnotesize)")
    parser.add_argument("--inner", default="footnotesize", choices=FONT_CHOICES,
                        help="font size for the bracketed CI line (default: footnotesize)")
    args = parser.parse_args()

    rows_with_files = [(key, f) for key, _, f in MAIN_ROWS if f]
    rows_data = _load_cache(rows_with_files)
    if rows_data is None:
        print(
            f"Cache {CACHE.relative_to(SUPERVISED_DIR.parent)} is missing or "
            f"incomplete. Run render_table_results.py first to populate it.",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded BCa CIs for {len(rows_data)} rows from "
          f"{CACHE.relative_to(SUPERVISED_DIR.parent)}.")

    render_split(
        rows_data, METRICS_RANKING,
        label="tab:results-ranking",
        caption=CAPTION_RANKING,
        out_path=OUT_DIR / "table_results_ranking.tex",
        outer_size=args.outer, inner_size=args.inner,
    )
    render_split(
        rows_data, METRICS_RETRIEVAL,
        label="tab:results-retrieval",
        caption=CAPTION_RETRIEVAL,
        out_path=OUT_DIR / "table_results_retrieval.tex",
        outer_size=args.outer, inner_size=args.inner,
    )
    print(f"  fonts: outer=\\{args.outer}  inner=\\{args.inner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
