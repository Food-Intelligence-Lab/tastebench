"""Verify re-rendered paper tables against the hashes shipped with them.

`verify_paper.sh --check-diff` used to run `git diff -- paper/`. The released
tree has no `.git`, so that command printed nothing and the check reported
"byte-identical" unconditionally — a check that could not fail. This replaces it
with a comparison against a committed SHA-256 manifest, which works with or
without a git checkout and fails on the first differing byte.

Scope is `paper/**/*.tex` plus the figures in FIGURES. The figures were excluded
at first because matplotlib stamps a CreationDate into every PDF, so two runs of
the same code differed and a hash would have failed for a reason unrelated to the
numbers. Their renderers now pass `metadata={"CreationDate": None}` and are
byte-reproducible, and covering them is the point: chart_gnn_per_model.pdf once
shipped hard-coded numbers the corrected .tex table beside it contradicted, and
no check looked at the figure. A .tex-only manifest reproduces that blind spot.

Only figures a verify_paper.sh step re-renders belong here; hand-drawn artwork
under paper/ has no render step, so a hash of it would assert nothing.

Usage:
    python3 shared/check_rendered_outputs.py            # verify; exit 1 on any diff
    python3 shared/check_rendered_outputs.py --write     # (re)generate the manifest
    python3 shared/check_rendered_outputs.py --list      # show what is covered
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MANIFEST = PAPER / "RENDERED_SHA256.txt"

HEADER = """\
# SHA-256 of every rendered table and figure shipped under paper/.
#
# Written by shared/check_rendered_outputs.py --write, and checked by
# `bash verify_paper.sh --check-diff`. The figures are covered because a figure
# that silently disagreed with the table beside it is the failure this manifest
# exists to catch; their renderers omit /CreationDate so they hash stably.
#
# Format: sha256  path-relative-to-repo-root
"""

# Figures with a verify_paper.sh render step, and therefore a reproducible hash.
FIGURES = [
    "charts/chart_gnn_per_model.pdf",
    "charts/chart_gnn_per_model.png",
    "charts/chart_molecular_prediction.pdf",
    "charts/chart_molecular_prediction.png",
    "human_baseline/group_size_curve.pdf",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rendered_outputs() -> list[Path]:
    if not PAPER.is_dir():
        return []
    figs = [PAPER / rel for rel in FIGURES]
    return sorted(list(PAPER.rglob("*.tex")) + [f for f in figs if f.exists()])


def read_manifest() -> dict[str, str]:
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not rel:
            raise SystemExit(f"ERROR: malformed manifest line: {line!r}")
        entries[rel.strip()] = digest
    return entries


def cmd_write() -> int:
    files = rendered_outputs()
    if not files:
        print("ERROR: nothing found under paper/ to record", file=sys.stderr)
        return 1
    # A figure that is absent at write time would silently drop out of the
    # manifest, which is how a covered output stops being covered without anyone
    # noticing. Refuse instead.
    absent = [rel for rel in FIGURES if not (PAPER / rel).exists()]
    if absent:
        print("ERROR: expected rendered figure(s) missing; run verify_paper.sh "
              "first, or drop them from FIGURES if they are gone for good:",
              file=sys.stderr)
        for rel in absent:
            print(f"  paper/{rel}", file=sys.stderr)
        return 1
    lines = [HEADER]
    for f in files:
        lines.append(f"{sha256(f)}  {f.relative_to(ROOT)}\n")
    MANIFEST.write_text("".join(lines))
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(files)} rendered output(s))")
    return 0


def cmd_verify() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: {MANIFEST.relative_to(ROOT)} is missing, so the rendered "
              f"tables and figures cannot be checked. Regenerate it with\n"
              f"  python3 shared/check_rendered_outputs.py --write", file=sys.stderr)
        return 1

    expected = read_manifest()
    on_disk = {str(f.relative_to(ROOT)): f for f in rendered_outputs()}

    changed, missing = [], []
    for rel, digest in sorted(expected.items()):
        f = on_disk.pop(rel, None)
        if f is None:
            missing.append(rel)
        elif sha256(f) != digest:
            changed.append(rel)
    unlisted = sorted(on_disk)

    for rel in changed:
        print(f"DIFFERS:  {rel}", file=sys.stderr)
    for rel in missing:
        print(f"MISSING:  {rel}", file=sys.stderr)
    for rel in unlisted:
        print(f"UNLISTED: {rel}  (rendered but not in the manifest)", file=sys.stderr)

    n_bad = len(changed) + len(missing) + len(unlisted)
    if n_bad:
        print(f"\nFAIL: {n_bad} of {len(expected)} rendered output(s) do not match "
              f"{MANIFEST.relative_to(ROOT)}.\n"
              f"      Re-rendering did not reproduce the shipped outputs. Check the "
              f"environment first (python3 shared/check_env.py --report); if the "
              f"change is intended, refresh the manifest with --write.",
              file=sys.stderr)
        return 1

    print(f"All {len(expected)} rendered output(s) match "
          f"{MANIFEST.relative_to(ROOT)} byte for byte.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true",
                   help="regenerate the manifest from the tables now on disk")
    g.add_argument("--list", action="store_true",
                   help="list the rendered tables and figures the manifest covers")
    args = ap.parse_args()

    if args.list:
        for f in rendered_outputs():
            print(f.relative_to(ROOT))
        return 0
    return cmd_write() if args.write else cmd_verify()


if __name__ == "__main__":
    sys.exit(main())
