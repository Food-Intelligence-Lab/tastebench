"""Join the gated NECTAR bundle to the public release: original codes -> public codes.

WHY THIS EXISTS. Everything in the public release carries a re-keyed
`product_code`: a cryptographically random per-category permutation, drawn once
and held outside the release, that exists so a released row cannot be walked back
to a manufacturer's product. The gated NECTAR bundle carries the ORIGINAL
manufacturer codes, because that is what NECTAR ships and what the raw CSVs and
the image directory tree are keyed on.

Tier 1 — retrain from the gated data and compare against the released artifacts —
therefore has to join two different code spaces. Before this module it did not:
`data/pairs.csv` and the LLM out-of-fold files (public codes) shared no key with
`product_features.pkl` (original codes), so `0 of 932` pairs matched and the run
died several minutes later inside an empty index array.

WHICH DIRECTION, AND WHY. Translation runs ONE way — the gated bundle is read
into the PUBLIC code space — never the reverse:

  * only one artifact needs translating (the feature matrix), against a dozen
    released files that would otherwise all have to be rewritten;
  * nothing on disk in the release is modified, so a Tier 1 run still diffs
    against exactly the bytes that were published;
  * a Tier 1 tree never acquires a file holding original codes that the release
    format says should not exist. Rewriting the release into the original space
    would create one, and a reviewer who then pushed that tree would publish the
    de-identification in reverse.

The 215 analog products are exactly the map's domain. The 32 animal-reference
products are not in the map — no released artifact mentions them — so they are
numbered after the analogs within their category (`n_analogs+1`, `n_analogs+2`,
... in ascending original-code order). That keeps every category's code set a
contiguous `1..N`, which is what `is_public_space()` tests, and it cannot collide
with a mapped code.

WHAT IT DOES NOT CHANGE. Every reported metric is computed within a category over
(target, prediction) pairs, so re-keying moves no point estimate; the release's
own `remint_product_codes.py` asserts this when it re-keys.

ROW ORDER, AND WHY IT IS PINNED. Re-keying used to reorder training as well, and
that was not a last-bit effect. `data/loocv.py` orders products by
(Category, Product_Code), so relabelling the code reorders the leave-one-out
loop; Bradley-Terry, kernel RankSVM and LightGBM are order-sensitive and moved by
exactly -1/935, +2/935 and -10/935 pairs — .610160 -> .609091, .617647 ->
.619786, .557754 -> .547059. That is a THIRD-decimal change, so a public-tree
Tier 1 did not reproduce the paper, and the README's "agreement to three
decimals" was describing a defect.

`translate_product_features` now records each product's pre-re-key code in
`ORDER_CODE_FIELD`, and loocv.py sorts by that when it is present. A re-keyed
matrix therefore trains in the same order as the gated one and reproduces the
published values exactly. A tree that never re-keys records nothing and sorts by
`product_code` as before, so no published number moves. The published order is
preserved, not replaced by a new canonical one — sorting on a data value such as
`mean_similarity` would have reordered the source tree too and invalidated every
number in the paper.

ABSENT MAP = NO-OP. A public Tier 0 reader has neither the map nor the gated
bundle, and must not be affected: with nothing to translate, every entry point
here returns its input unchanged. The map is only ever REQUIRED when a join is
actually attempted across the two spaces — i.e. when an artifact that is still in
the original code space is about to meet a released one.

The map itself is the fifth item of the gated NECTAR bundle (see data/GATED.md).
It is not in this release and must never be published.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ENV_VAR = "TASTEBENCH_PRODUCT_CODE_MAP"
DEFAULT_REL = "data/product_code_map.json"

# Entry field carrying the code a product held BEFORE the re-key. Written by
# `translate_product_features` below; read by `order_key` in
# food_similarity/data/loocv.py, which is where it decides the LOOCV row order.
#
# The literal is duplicated there rather than imported, because loocv.py has to
# work in a source tree where this module does not exist at all (it arrives with
# task 32). The two are held in agreement by task 32's `check`, which fails the
# build if the strings ever drift apart.
ORDER_CODE_FIELD = "original_product_code"

ROOT = Path(__file__).resolve().parents[1]

ProductKey = Tuple[str, int]


def map_path() -> Path:
    """Where the map is looked for. `ENV_VAR` wins, else the gated-bundle path."""
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else ROOT / DEFAULT_REL


def _missing_message(what: str) -> str:
    return (
        f"\nERROR: {what} still uses the ORIGINAL NECTAR product codes, and the\n"
        f"       withheld product-code map is not here — so it cannot be joined to\n"
        f"       anything in this release.\n"
        f"  looked for  {map_path()}\n"
        f"              (override the location with {ENV_VAR}=/path/to/map.json)\n"
        f"  The public release re-keys every product code from a random per-category\n"
        f"  permutation; the gated bundle keeps the original manufacturer codes. The\n"
        f"  map between them is the fifth item in the gated NECTAR bundle — see\n"
        f"  data/GATED.md — and ships only to NECTAR-licensed users.\n"
        f"  If you have the bundle, place the map with:\n"
        f"      bash data/unpack_nectar_bundle.sh <path-to-extracted-bundle>\n"
        f"  Tier 0 (verify_paper.sh) needs none of this and is unaffected."
    )


class ProductCodeMap:
    """`{category: {original_code: public_code}}`, plus the join it enables."""

    def __init__(self, per_category: Dict[str, Dict[int, int]], path: Path,
                 created: str = "?"):
        self.per_category = per_category
        self.path = path
        self.created = created
        self._public_sets = {c: set(d.values()) for c, d in per_category.items()}

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["ProductCodeMap"]:
        """Return the map, or None when it is not present (never raises for absence)."""
        p = Path(path) if path else map_path()
        if not p.exists():
            return None
        raw = json.loads(p.read_text())
        if "map" not in raw:
            raise SystemExit(
                f"ERROR: {p} is not a product-code map (no 'map' key). It is the "
                f"file described in data/GATED.md item 5.")
        per_cat = {str(c): {int(k): int(v) for k, v in d.items()}
                   for c, d in raw["map"].items()}
        return cls(per_cat, p, str(raw.get("created", "?")))

    @property
    def n_products(self) -> int:
        return sum(len(d) for d in self.per_category.values())

    # -- code-space questions ------------------------------------------------
    def space_of(self, keys: Iterable[ProductKey]) -> str:
        """'original', 'public', or 'unknown' for a set of (category, code) keys.

        Decided by MEMBERSHIP, not by shape: the original codes (manufacturer
        slots, 164-923 in the shipped data) and the public codes (1..n per
        category) are disjoint, so a key can only be in one space. Keys that are
        in neither — the animal references, which the map does not cover — do not
        vote; they follow whatever the mapped products say.
        """
        orig = pub = 0
        for cat, code in keys:
            d = self.per_category.get(str(cat))
            if d is None:
                continue
            if int(code) in d:
                orig += 1
            elif int(code) in self._public_sets[str(cat)]:
                pub += 1
        if orig and pub:
            return "unknown"
        if orig:
            return "original"
        if pub:
            return "public"
        return "unknown"

    def public_for(self, category: str, code: int) -> int:
        return self.per_category[str(category)][int(code)]

    # -- translation ---------------------------------------------------------
    def _reference_codes(self, keys: Iterable[ProductKey]) -> Dict[ProductKey, int]:
        """Number the products the map does not cover, after the ones it does."""
        extra: Dict[str, List[int]] = {}
        for cat, code in keys:
            c = str(cat)
            if int(code) not in self.per_category.get(c, {}):
                extra.setdefault(c, []).append(int(code))
        out: Dict[ProductKey, int] = {}
        for c, codes in extra.items():
            base = len(self.per_category.get(c, {}))
            for i, code in enumerate(sorted(set(codes)), start=1):
                out[(c, code)] = base + i
        return out

    def translate_product_features(self, pf: dict) -> dict:
        """Re-key `product_features.pkl` into the public code space.

        Idempotent: a matrix already in the public space is returned untouched.
        """
        space = self.space_of(pf.keys())
        if space == "public":
            return pf
        if space == "unknown":
            raise SystemExit(
                f"\nERROR: product_features.pkl and the product-code map disagree.\n"
                f"  map: {self.path} ({self.n_products} products, created {self.created})\n"
                f"  No product code in the matrix is either an original or a public\n"
                f"  code this map knows, or the matrix mixes the two. Do not guess —\n"
                f"  rebuild the matrix from the gated bundle, or restore the map that\n"
                f"  was used to build this release.")

        unknown = self._reference_codes(pf.keys())
        out = {}
        for (cat, code), v in pf.items():
            c, code = str(cat), int(code)
            new = self.per_category[c][code] if code in self.per_category.get(c, {}) \
                else unknown[(c, code)]
            entry = dict(v)
            if "product_code" in entry:
                entry["product_code"] = new
            # Pin the row order to the code space the matrix ARRIVED in.
            #
            # The LOOCV sites in data/loocv.py order products by
            # (Category, Product_Code), so re-keying reorders the leave-one-out
            # loop. Ridge did not care, but Bradley-Terry, kernel RankSVM and
            # LightGBM are order-sensitive and moved by -1/935, +2/935 and
            # -10/935 pairs -- enough to change a published third decimal, which
            # is why a public-tree Tier 1 did not reproduce the paper exactly.
            #
            # Recording the original code here, and sorting by it there, makes
            # the training order a property of the data rather than of the code
            # space. A tree that never re-keys records nothing and keeps sorting
            # by product_code, so the published numbers do not move.
            entry[ORDER_CODE_FIELD] = code
            out[(cat, new)] = entry
        if len(out) != len(pf):
            raise SystemExit(
                f"ERROR: re-keying product_features.pkl collapsed "
                f"{len(pf)} products into {len(out)}; the map is not injective "
                f"over this matrix.")
        return out

    def translate_frame(self, df, columns: Iterable[str]):
        """Re-key a DataFrame's product-code columns in place-safe fashion.

        Used for `data/pairs.csv`, the one released CSV that a Tier 1 run
        REGENERATES from the gated bundle (REBUILD_FEATURES=1). Without this it
        would be written back in the original space and stop joining the
        out-of-fold files that cannot be regenerated without API spend.
        """
        cols = [c for c in columns if c in df.columns]
        if not cols or "category" not in df.columns:
            return df
        keys = [(cat, code) for col in cols
                for cat, code in zip(df["category"], df[col])]
        if self.space_of(keys) == "public":
            return df
        unknown = self._reference_codes(keys)
        df = df.copy()
        for col in cols:
            df[col] = [
                self.per_category[str(cat)][int(code)]
                if int(code) in self.per_category.get(str(cat), {})
                else unknown[(str(cat), int(code))]
                for cat, code in zip(df["category"], df[col])
            ]
        return df


# --- module-level entry points ---------------------------------------------

def load(path: Optional[Path] = None) -> Optional[ProductCodeMap]:
    """The map, or None. Absence is never an error here — see `to_public_features`."""
    return ProductCodeMap.load(path)


def is_public_space(keys: Iterable[ProductKey]) -> bool:
    """True when every category's codes are a contiguous 1..N — the public shape.

    This is the only question that can be answered WITHOUT the map, and it is
    asked for exactly one reason: to decide whether a map is needed at all. The
    original codes are sparse manufacturer slots (24 distinct values in the range
    164-923 across 24 categories), so they never form 1..N.
    """
    per_cat: Dict[str, set] = {}
    for cat, code in keys:
        per_cat.setdefault(str(cat), set()).add(int(code))
    if not per_cat:
        return True
    return all(codes == set(range(1, len(codes) + 1)) for codes in per_cat.values())


def to_public_features(pf: dict, what: str = "the feature matrix") -> dict:
    """Bring `product_features.pkl` into the public code space, or explain why not.

    The three states, in the order they are tested:

      already public   -> returned unchanged. This is a Tier 0 tree, or a second
                          call on an already-translated matrix. No map needed,
                          and none is read.
      original + map    -> translated.
      original, no map  -> SystemExit naming the map and data/GATED.md. This is
                          the case that used to run for nine minutes and then die
                          in an IndexError from an empty pair array.
    """
    if not pf or is_public_space(pf.keys()):
        return pf
    m = load()
    if m is None:
        raise SystemExit(_missing_message(what))
    return m.translate_product_features(pf)


def to_public_frame(df, columns=("product_code", "product_code_1", "product_code_2",
                                 "higher_rated_product")):
    """Same for a DataFrame; a no-op when the map is absent.

    Deliberately softer than `to_public_features`: this is called on the WRITE
    side (regenerating pairs.csv), where the alternative to writing original
    codes is writing nothing at all. The read side is where a cross-space join
    is actually attempted, and that is where the hard failure belongs.
    """
    m = load()
    if m is None:
        return df
    return m.translate_frame(df, columns)
