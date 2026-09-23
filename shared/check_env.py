"""Fail fast on the environment problems that actually break this artifact.

Reviewers hit a Parquet read error reproducing this artifact, and a version
mismatch is the likely cause. Checking up front turns that into one clear
message instead of a traceback several minutes into a render.

Only the checks tied to an observed failure are fatal: a missing or too-old
pyarrow (the reviewer's Parquet error) stops the run. The interpreter version
is a warning, because two interpreters have been measured to produce the same
numbers and neither can be called *the* pinned one without overstating what
was tested.

numpy 1.26.4 publishes no cp313 wheel (its wheel tags stop at cp312), so a
Python 3.13 environment builds numpy from source and needs a C compiler. That
is how the shipped artifacts were built, and it is why environment.yml asks
for 3.11 instead: every pinned wheel exists there.

Usage:
    python3 shared/check_env.py            # assert; exit 1 on a fatal problem
    python3 shared/check_env.py --report   # print findings, exit 0 unless fatal
    python3 shared/check_env.py --warn     # never exit non-zero
"""

import sys

# Interpreters this artifact has actually been exercised on. 3.13.12 produced
# every shipped number; 3.11.14 is what environment.yml resolves to, and it
# re-renders the tables and figures byte-identically (see
# food_similarity/results/reproducibility.md). Anything else is untested, which
# is worth saying but not worth refusing to run over.
PY_TESTED = ((3, 13), (3, 11))

# The one that produced the committed artifacts.
PY_ARTIFACTS = (3, 13)

# Minimum pyarrow for the committed molecular/results/**/predictions.parquet.
#
# Chosen from the files' own metadata rather than guessed: they are Parquet
# format 2.6, SNAPPY-compressed, using PLAIN / RLE / RLE_DICTIONARY encodings,
# written by parquet-cpp-arrow 23.0.1. None of that is a recent feature, so the
# reviewer's read error was almost certainly a missing or very old pyarrow rather
# than an unsupported format. 8.0 is a conservative floor for this combination;
# the exact pin in requirements.txt is the real mechanism, and this guard exists
# to catch an environment that ignored it.
PYARROW_MIN = (8, 0)

# Packages whose version silently changes results rather than erroring.
# LightGBM is here because the rebuttal encoder comparison was run on 4.7.0
# against the pinned 4.6.0 and every prediction shifted.
EXACT = {
    "lightgbm": "4.6.0",
    "numpy": "1.26.4",
    "scikit-learn": "1.7.1",
}


def _version(mod_name: str):
    try:
        from importlib.metadata import version
        return version(mod_name)
    except Exception:
        return None


def _tuple(v: str):
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def check():
    """Return (fatal, warnings) as lists of human-readable strings."""
    fatal, warn = [], []

    here = sys.version_info[:2]
    tested = ", ".join(f"{a}.{b}" for a, b in PY_TESTED)
    artifacts = f"{PY_ARTIFACTS[0]}.{PY_ARTIFACTS[1]}"
    if here not in PY_TESTED:
        warn.append(
            f"Python {here[0]}.{here[1]} has not been exercised on this "
            f"artifact. Tested: {tested} ({artifacts} produced the shipped "
            f"artifacts). Create a tested env with: "
            "conda env create -f environment.yml"
        )
    elif here == PY_ARTIFACTS:
        warn.append(
            f"Python {artifacts} produced the shipped artifacts, but numpy "
            f"{EXACT['numpy']} has no cp313 wheel, so installing it here "
            "builds numpy from source and needs a C compiler. "
            "environment.yml asks for 3.11, where every pinned wheel exists."
        )

    pa = _version("pyarrow")
    if pa is None:
        fatal.append("pyarrow is not installed; it is required to read the committed "
                     "molecular predictions. pip install -r requirements.txt")
    elif _tuple(pa)[:2] < PYARROW_MIN:
        fatal.append(
            f"pyarrow>={PYARROW_MIN[0]}.{PYARROW_MIN[1]} required to read "
            f"molecular/results/**/predictions.parquet; found {pa}. "
            "pip install -r requirements.txt"
        )

    for pkg, want in EXACT.items():
        got = _version(pkg)
        if got is None:
            warn.append(f"{pkg} is not installed (pinned {want})")
        elif got != want:
            warn.append(
                f"{pkg} {got} != pinned {want}; results may differ from the "
                "committed artifacts (re-render with --check-diff to confirm)"
            )

    return fatal, warn


def main(argv):
    mode = argv[1] if len(argv) > 1 else "--assert"
    fatal, warn = check()

    for m in warn:
        print(f"WARNING: {m}", file=sys.stderr)
    for m in fatal:
        print(f"ERROR:   {m}", file=sys.stderr)

    if mode == "--report" and not fatal:
        # "OK" is reserved for a clean run: printing it under a warning is how a
        # reviewer learns to skim past the warnings.
        verdict = "environment OK" if not warn else f"no fatal problems ({len(warn)} warning(s))"
        print(f"{verdict} (python {sys.version.split()[0]}, "
              f"pyarrow {_version('pyarrow')}, lightgbm {_version('lightgbm')})")
    if mode == "--warn":
        return 0
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
