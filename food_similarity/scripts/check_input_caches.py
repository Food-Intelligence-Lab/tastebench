"""Verify the derived caches against the manifest, and say what was checked.

`reproduce.sh` used to run:

    sha256sum -c results/input_cache_sha256.txt --ignore-missing

and print "input caches match manifest". Two problems, both observed:

  * every manifest line carries a trailing `# NNN bytes` comment, which
    sha256sum folds into the filename — so those entries are silently skipped
    and it warns "3 lines are improperly formatted";
  * with `--ignore-missing` and NO caches on disk it verifies nothing, exits 0,
    and the script announces success. Phase 4 then dies twenty minutes later on
    a cache that was never there.

This parses the manifest properly, reports present/missing/mismatched counts, and
fails when nothing was verified — so the failure lands in Phase 1 with a usable
message instead of mid-run.

Usage:
    python3 scripts/check_input_caches.py              # require >=1 verified
    python3 scripts/check_input_caches.py --require 0  # report only, never fail
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUP = HERE.parent
ROOT = SUP.parent
MANIFEST = SUP / "results" / "input_cache_sha256.txt"

LINE = re.compile(r"^([0-9a-fA-F]{64})\s+(\S+)")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", type=int, default=1,
                    help="minimum number of caches that must verify (0 = report only)")
    a = ap.parse_args()

    if not MANIFEST.exists():
        print(f"  ERROR: manifest not found at {MANIFEST}")
        return 1

    entries = []
    unparsed = 0
    for raw in MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if m:
            entries.append((m.group(1).lower(), m.group(2)))
        else:
            unparsed += 1

    ok, missing, bad = [], [], []
    for want, rel in entries:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif sha256(p) == want:
            ok.append(rel)
        else:
            bad.append(rel)

    print(f"  manifest entries: {len(entries)}"
          + (f"  (unparsed lines: {unparsed})" if unparsed else ""))
    print(f"  verified: {len(ok)}   missing: {len(missing)}   MISMATCHED: {len(bad)}")
    for rel in bad:
        print(f"    mismatch: {rel}")
    for rel in missing[:6]:
        print(f"    missing:  {rel}")
    if len(missing) > 6:
        print(f"    ... and {len(missing)-6} more")

    if bad:
        print("  ERROR: cache contents differ from the manifest; results would not match.")
        return 1
    if len(ok) < a.require:
        print(f"  ERROR: {len(ok)} cache(s) verified, need at least {a.require}.")
        print("  Build them with: bash shared/scripts/prepare_caches/build_all_caches.sh")
        print("  Or skip the phases that need them — Tier 0 (verify_paper.sh) needs none.")
        return 1
    if missing:
        print("  note: missing caches are only needed by the phases that consume them;"
              " see README.md -> Caches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
