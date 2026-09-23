# Third-party data notices

This release redistributes small amounts of data derived from third-party
sources. This file records what is included, where it came from, and under what
terms, so that downstream users inherit correct attribution.

The benchmark's own sensory data (NECTAR) is **not** redistributed here; it is
gated and covered separately — see `data/GATED.md` and `croissant/nectar.json`.

---

## FoodAtlas v4.0 — Apache-2.0

**Included in this release:** `shared/data/caches/compound_slice.parquet`
(14,677 rows, 33 KB).

**What it is:** a derived subset mapping ingredient names to the chemical
compounds attested for them, with SMILES strings and, where available,
concentrations. It contains only the (ingredient, compound, SMILES,
concentration) tuples the benchmark's compound features actually consume.

**Why it is here:** it replaces an 82 MB FoodAtlas download plus a 953 MB FooDB
download, so the compound features can be rebuilt with no external data. The
derived subset reproduces the published compound vectors exactly (243/243
product vectors bit-identical; see
`shared/compound_mapping/verify_compound_slice.py`).

**Source:** FoodAtlas v4.0, https://www.foodatlas.ai/food-composition-downloads

**License:** Apache License 2.0, as stated in the FoodAtlas v4.0 distribution's
own `README.md`. Apache-2.0 permits redistribution of derivative works provided
attribution and the license notice are retained, and modifications are stated.

**Statement of changes (as Apache-2.0 §4(b) requires):** the redistributed data
is a filtered projection of the FoodAtlas v4.0 release, not a copy. Specifically:
restricted to food entities matched from this benchmark's ingredient
vocabulary; restricted to `contains` edges; concentration attestations filtered
to the units `mg/100g` and `%`; attestations from the sources
`lit2kg:gpt-5.2`, `lit2kg:gpt-4` and `lit2kg:gpt-3.5-finetuned` excluded;
compounds retained only where a SMILES string resolved via PubChem or ChEBI.
The exact filter configuration is recorded in
`shared/data/caches/compound_slice.config.json`, and the generator is
`shared/compound_mapping/build_compound_slice.py`.

No FoodAtlas source file is redistributed in full.

---

## FooDB 2020-04-07

**Included in this release:** 349 concentration values, inside the same
`compound_slice.parquet`. They are numeric concentrations only, used as a
fallback where FoodAtlas records a compound for an ingredient but no
concentration for it. For reference, of the slice's 14,677 rows, 2,103
concentrations come from FoodAtlas, 349 from FooDB, and 12,225 rows carry no
concentration at all.

No FooDB compound records, descriptions, identifiers, or source files are
redistributed.

**Source:** FooDB 2020-04-07 CSV release, https://foodb.ca/downloads

**Terms:** the CSV distribution ships no license file, but foodb.ca states the
terms: FooDB "is offered to the public as a freely available resource", and
"use and re-distribution of the data, in whole or in part, for **commercial
purposes** requires explicit permission of the authors and explicit
acknowledgment of the source material (FooDB) and the original publication."

Non-commercial redistribution of the derived values above is therefore permitted
under those terms, and this notice is the required acknowledgment.

**If you intend to use this release commercially**, the 349 FooDB-derived
concentration values are not covered by this repository's MIT code license, and
you need permission from the Metabolomics Innovation Centre before
redistributing them: metabolomicsinnovations@gmail.com.

To avoid the condition instead, rebuild the slice with FooDB absent:
`build_compound_slice.py` treats it as optional (it sets `foodb = None` when
`FOODB_DIR` is missing) and emits a FoodAtlas-only slice, which is Apache-2.0
and carries no commercial restriction. Note this drops those 349 concentrations,
so the compound vectors will differ from the published ones — the bit-exactness
claim above holds only for the slice as shipped. `DISABLE_COMPOUND_SLICE=1` is
NOT the way to do this: it ignores the shipped slice and reads the full upstream
sources, FooDB included.

FooDB also asks that users who take significant portions of the database cite
the FooDB paper. 349 concentration values is a small fraction of FooDB rather
than a significant portion, but anyone building on this slice should cite FooDB;
the current citation is given on https://foodb.ca.

---

## Regenerating rather than relying on the redistributed subset

Anyone preferring to derive the subset themselves, from the upstream sources
under those sources' own terms, can do so:

```bash
bash data/download_public.sh                              # FoodAtlas v4.0 + FooDB
python3 shared/compound_mapping/build_compound_slice.py    # rewrites the slice
python3 shared/compound_mapping/verify_compound_slice.py    # confirms it matches
```

Setting `DISABLE_COMPOUND_SLICE=1` ignores the redistributed subset entirely and
reads the full upstream sources instead.
