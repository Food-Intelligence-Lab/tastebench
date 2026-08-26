"""Build a compact ingredient->compound slice so the pipeline needs no bulk downloads.

The compound feature is produced by: ingredient string -> FoodAtlas food entity ->
`contains` edges -> chemical entities -> SMILES -> embedding -> concentration-weighted
mean. Only the resolved (ingredient, compound, SMILES, concentration) rows can
affect the result, and there are few of them relative to FoodAtlas (222k entities,
594k triplets, 1.06M attestations) and FooDB (953 MB).

Emitting just those rows gives a slice of a few tens of kilobytes that reproduces
the compound vectors exactly, with no FoodAtlas or FooDB download.

KEYED BY INGREDIENT NAME, deliberately. The slice is a corpus-wide vocabulary: it
records that an ingredient string occurs somewhere in the panel, never in which
product. Keying by product code would make it a per-product composition
fingerprint, which the source data's terms forbid.

THE SLICE FREEZES THE MAPPER CONFIGURATION. Eight environment knobs change what
the mapper returns. The sidecar JSON records all of them plus the FoodAtlas file
hashes; the loader must refuse the slice when they differ and fall back to the
full source, or an ablation would silently reuse the wrong mapping.

Usage (needs FoodAtlas, FooDB and the gated product list — run once):
    python3 shared/compound_mapping/build_compound_slice.py
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "food_similarity"))

OUT = ROOT / "shared" / "data" / "caches" / "compound_slice.parquet"
SIDECAR = OUT.with_suffix(".config.json")

def config_fingerprint(food_atlas_dir: Path, knobs) -> dict:
    files = {}
    for name in ("entities.parquet", "triplets.parquet", "attestations.parquet"):
        p = food_atlas_dir / name
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            files[name] = h.hexdigest()[:16]
    import prepare_data as P
    return {
        # What the loader actually gates on: resolved settings, so that editing an
        # in-code default invalidates the slice just as an env override does.
        "effective_config": P.foodatlas_effective_config(),
        # Raw env values, recorded for humans reading the sidecar.
        "knobs": {k: os.environ.get(k) for k in knobs},
        "foodatlas_files": files,
    }


def main() -> int:
    import prepare_data as P
    from compound_mapping.smiles_resolver import SMILESResolver

    if not P.FOOD_ATLAS_DIR.exists():
        print(f"ERROR: FoodAtlas not found at {P.FOOD_ATLAS_DIR}")
        print("  fetch it with data/download_public.sh — needed once, to build this slice")
        return 1

    products = P.load_nectar_products()
    # Same constructor the pipeline uses, so the slice is built under exactly the
    # configuration it will be consumed under.
    mapper = P.build_foodatlas_mapper()
    resolver = SMILESResolver(str(P.SMILES_CACHE), cache_only=True)

    foodb = None
    try:
        from compound_mapping.foodb_concentrations import FooDBConcentrations
        if P.FOODB_DIR.exists():
            foodb = FooDBConcentrations(str(P.FOODB_DIR))
    except Exception:
        pass

    rows, seen = [], set()
    for _, r in products.iterrows():
        for ing, mapping in mapper.map_product(r.get("cleaned_ingredients", "") or "").items():
            if mapping is None or not mapping.compounds:
                continue
            key = ing.strip()
            if key in seen:
                continue
            seen.add(key)
            food = (mapping.matched_entity_name or "").split(" (+")[0].strip()
            for c in mapping.compounds:
                # Mirror prepare_data.resolve_smiles EXACTLY: PubChem first, then
                # ChEBI. An earlier version tried only PubChem and silently
                # dropped every ChEBI-only compound, which would have produced a
                # slice that quietly disagreed with the full source.
                smiles = None
                if c.pubchem_cid is not None:
                    smiles = resolver.resolve(c.pubchem_cid)
                if smiles is None and getattr(c, "chebi_id", None) is not None:
                    smiles = resolver.resolve_chebi(c.chebi_id)
                if smiles is None:
                    continue
                conc = c.concentration
                # bake in the FooDB fallback so the 953 MB source is not needed
                if conc is None and foodb is not None:
                    conc = foodb.get_concentration(smiles, food_name=food)
                rows.append({"ingredient": key,
                             "compound_id": getattr(c, "foodatlas_id", None) or str(c),
                             "smiles": smiles,
                             "concentration": conc})

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT, index=False, compression="zstd")
    except Exception:
        df.to_parquet(OUT, index=False)
    # Recorded for humans; gating is by effective_config (see prepare_data).
    SIDECAR.write_text(
        json.dumps(config_fingerprint(P.FOOD_ATLAS_DIR, P.SIDECAR_KNOB_NAMES), indent=2) + "\n")

    print(f"  {len(df)} rows, {df.ingredient.nunique()} ingredients, "
          f"{df.smiles.nunique()} distinct SMILES")
    print(f"  concentrations present: {df.concentration.notna().sum()}")
    print(f"  wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"  wrote {SIDECAR.relative_to(ROOT)}  (mapper config + FoodAtlas hashes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
