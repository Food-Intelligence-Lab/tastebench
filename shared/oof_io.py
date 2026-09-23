"""One way to read an out-of-fold prediction file, whichever target form it has.

WHY THIS EXISTS. An out-of-fold file carries one target column, and there are two
legitimate spellings of it:

``true_score``
    The panel's MEAN SIMILARITY RATING for a product. Written by everything that
    builds an OOF from the gated NECTAR bundle (``run_loocv.py``,
    ``regenerate_distance_oofs.py``, ``paper_table.py``, ``nested_meta_all.py``,
    ``compute_per_model_nnls.py``, ``human_panelist_baseline.py``, ...). A
    rebuild from raw data produces this, and that is correct.

``true_rank``
    The within-category average rank of the same value. This is what the public
    release SHIPS: the ratings are withheld under NDA and only their order is
    published (see the release build script that converts scores to ranks). Every metric the
    paper reports reads only the sign of a within-category target difference, so
    the two forms give bit-identical numbers — verified across all 182
    rating-bearing OOF files, max |delta| = 0.000e+00 on all six metrics.

The mixture is unavoidable: a Tier 1 reproduction starts from a released tree
(``true_rank`` on disk) and *overwrites* those files with freshly trained ones
(``true_score``), so a single script can see either form depending on which
phase has run. The first attempt at handling this patched each call site
individually at release-build time from a hand-maintained list, which went stale
immediately: ``train/run_taste_gnn_nectar.py`` reads an archived OOF before
retraining and was never on the list, so ``reproduce.sh`` Phase 4 died with
``KeyError: ['true_score']`` in a reviewer's checkout.

So: READ through :func:`read_oof`, WRITE ``true_score``.

    from oof_io import read_oof
    df = read_oof(OOF_DIR / "bradley_terry_SNCTI_bench.csv")
    df["true_score"]        # present whichever form the file used

A writer that derives its target from the gated ratings keeps writing
``true_score`` and must NOT use this module to name its output column. A script
that reads ranks and writes them back out again (only
``molecular/results/encoder_comparison/regenerate_table3.py`` does) must write
back the name the data actually has — use :func:`target_column` to recover it,
because :func:`read_oof` deliberately forgets.

``shared/check_oof_readers.py`` is the gate: it fails if a module reads a path
under an OOF directory without going through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

#: The name every consumer sees, and the name a rebuild from gated data writes.
TARGET = "true_score"

#: The de-identified form the public release ships.
RANK_TARGET = "true_rank"

#: Both spellings, most-canonical first.
TARGET_COLUMNS = (TARGET, RANK_TARGET)


def target_column(source) -> str | None:
    """Return which target spelling ``source`` uses, or ``None`` if neither.

    ``source`` may be a DataFrame, an iterable of column names, or a path to a
    CSV (only its header is read).
    """
    if isinstance(source, pd.DataFrame):
        columns = source.columns
    elif isinstance(source, (str, Path)):
        columns = pd.read_csv(source, nrows=0).columns
    else:
        columns = list(source)
    for name in TARGET_COLUMNS:
        if name in columns:
            return name
    return None


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Present the target as ``true_score`` regardless of how it was stored.

    A no-op (returns the same object) when the frame already uses ``true_score``
    or carries no target at all — several OOF consumers only want
    ``predicted_score``, and those must not start failing on a target-free file.
    """
    if TARGET not in df.columns and RANK_TARGET in df.columns:
        return df.rename(columns={RANK_TARGET: TARGET})
    return df


def read_oof(path, **kwargs) -> pd.DataFrame:
    """Read an OOF CSV and present its target column as ``true_score``.

    Any keyword argument is forwarded to :func:`pandas.read_csv`. Note the
    absence of ``float_precision="round_trip"``, which
    ``cache_provenance.read_cache_csv`` does use: the CI caches are written by
    this codebase and must round-trip bit-exactly, whereas the OOF files are
    read by the published render scripts with plain ``read_csv``, and switching
    parsers here would move predictions in the last ULP and with them the
    rendered tables.
    """
    return normalize_target(pd.read_csv(path, **kwargs))


def read_oofs(paths: Iterable) -> list[pd.DataFrame]:
    """:func:`read_oof` over several paths."""
    return [read_oof(p) for p in paths]
