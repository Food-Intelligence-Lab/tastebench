# Reproducibility — NeurIPS 2026 supervised pipeline

## Git state
- Commit SHA recorded in `repro_git_sha.txt` after each `reproduce.sh` run

## System
- OS: Darwin 25.3.0 (arm64)
- Python: 3.13.12 (main, Feb 12 2026, 01:06:02) [Clang 21.1.4 ]

This is the interpreter that produced every artifact in this directory. The
`numpy.show_config` block below is its captured output, which is why it reports
`version: '3.13'` and `blas: accelerate`. `environment.yml` asks for Python
3.11 instead: numpy 1.26.4 publishes no cp313 wheel (its tags stop at cp312),
so a 3.13 environment compiles numpy from source and needs a C compiler, while
3.11 installs every pin from a published wheel. The two are not the same build
— see "Cross-environment check" below for what that does and does not change.

- numpy.show_config:
  ```
  Build Dependencies:
    blas:
      detection method: system
      found: true
      include directory: unknown
      lib directory: unknown
      name: accelerate
      openblas configuration: unknown
      pc file directory: unknown
      version: unknown
    lapack:
      detection method: internal
      found: true
      include directory: unknown
      lib directory: unknown
      name: dep4368281552
      openblas configuration: unknown
      pc file directory: unknown
      version: 1.26.4
  Compilers:
    c:
      commands: cc
      linker: ld64
      name: clang
      version: 17.0.0
    c++:
      commands: c++
      linker: ld64
      name: clang
      version: 17.0.0
    cython:
      commands: cython
      linker: cython
      name: cython
      version: 3.0.12
  Machine Information:
    build:
      cpu: aarch64
      endian: little
      family: aarch64
      system: darwin
    host:
      cpu: aarch64
      endian: little
      family: aarch64
      system: darwin
  Python Information:
    version: '3.13'
  SIMD Extensions:
    baseline:
    - NEON
    - NEON_FP16
    - NEON_VFPV4
    - ASIMD
    found:
    - ASIMDHP
    not found:
    - ASIMDFHM
  
  
  ```

## Hardware (inferred)
- MPS available: True
- CUDA available: False

## Cross-environment check

Measured, not assumed. A Python 3.11.14 environment was built from
`requirements.txt` (every pin resolved to a published wheel; `uv pip
compile --python-version 3.11` resolves all 129 requirements), and four
renders were re-run there **with their CI caches deleted**, so every
number was recomputed rather than read back:

- `food_similarity/scripts/render_table_results.py` — 12 models x 6
  metrics, 10,000-resample BCa via the Python-loop bootstrap
- `food_similarity/scripts/render_table_per_category.py` — 72
  per-category cells via the vectorised bootstrap
- `molecular/scripts/render_table_molecular_prediction.py` — pyarrow
  parquet read plus sklearn metrics, BCa CIs, ECE, McNemar's
- `human_baseline/plot_human_baseline.py` — the matplotlib PDF

All seven output files (three CI caches, three .tex tables, one PDF) are
**byte-identical** to the shipped artifacts, and identical to a 3.13.12
re-render of the same tree. The comparison was checked against a
perturbed seed first, which does move the CI bounds, so "identical" here
is a measurement rather than a comparison that cannot fail.

The two numpy builds do differ — 3.13 links Accelerate, the 3.11 wheel
links OpenBLAS 0.3.23 — but these renders rank, resample and average
committed predictions instead of multiplying matrices, and the bootstrap
RNG is seeded, so the backend has nothing to change.

Not covered: `render_table_ablation_features.py`, the encoder-comparison
tables, and the Tier 1 retraining path. Tier 1 retrains models on gated
data and has only ever run in the environment recorded above; expect it
to agree to the three decimals the tables print, not byte for byte.

## Pipeline-specific notes
- BLAS backend differences: ~1e-6 drift in PCA; pairwise accuracy
  unaffected at 3-decimal precision.
- See input_cache_sha256.txt for the canonical input-artifact hashes;
  any divergence from those is a provenance red flag.
