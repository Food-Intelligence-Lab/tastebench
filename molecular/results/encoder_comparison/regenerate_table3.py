"""Regenerate the corrected Table 3 (+ ChemBERTa molecular row) byte-identically
from the shipped cache — NO API, NO gated NECTAR data, NO product-features pkl.

Inputs (all in this bundle): bases/*.npz  +  oof/*.csv  +  molecular/*.
The only external dependency is the repo's two DETERMINISTIC, committed helpers
evaluation.metrics and evaluation.bootstrap (read-only). Point REPO at the repo:

    TASTEBENCH_REPO=/path/to/tastebench \
      python regenerate_table3.py

Outputs (written next to this script): cis_encoder_comparison.csv (points + 95%
BCa CIs for all 30 downstream cells + the ChemBERTa molecular row). Verifies the
two canonical gates (FART NNLS .6829, FART BT .6102) and diffs against the shipped
downstream_results.csv.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent


def _find_food_similarity() -> Path:
    """Locate food_similarity/, whichever layout we are running in.

    Works from inside the release tree (molecular/results/encoder_comparison/
    -> ../../..) and from an explicit TASTEBENCH_REPO override. No absolute
    path is baked in:
    one would break every other machine and would leak an author path into a
    public artifact.
    """
    env = os.environ.get("TASTEBENCH_REPO")
    candidates = []
    if env:
        candidates += [Path(env) / "food_similarity"]
    for d in [HERE, *HERE.parents]:
        candidates += [d / "food_similarity"]
    for c in candidates:
        if (c / "evaluation" / "metrics.py").exists():
            return c
    raise SystemExit(
        "Could not locate food_similarity/. Run from inside the code tree, or set "
        "TASTEBENCH_REPO=/path/to/repo."
    )


SUP = _find_food_similarity()
REPO = SUP.parent
sys.path.insert(0, str(SUP))
sys.path.insert(0, str(SUP.parent / "shared"))
from evaluation.metrics import compute_all_metrics          # deterministic
from oof_io import RANK_TARGET, TARGET, read_oof
from evaluation.bootstrap import compute_bca_cis            # np.random.default_rng(42)

BASES = {"FART": "bases/c5_base_FART.npz", "GNN": "bases/c5_base_GNN.npz",
         "ChemBERTa": "bases/enc_base_ChemBERTa.npz"}
INDIV = ["MMRF (cosine)", "MMRF (L2)", "Ridge", "Hierarchical BT", "Kernel RankSVM", "LightGBM"]
ENSEMBLE = ["Bradley-Terry", "NNLS ens", "Mean ens", "Rank ens"]
# All ten archived FART values, not two. A two-cell gate is what let the MMRF and
# LightGBM divergences through in the first place
GATE = {
    ("FART", "MMRF (cosine)"): 0.574332, ("FART", "MMRF (L2)"): 0.570588,
    ("FART", "Ridge"): 0.579144, ("FART", "Bradley-Terry"): 0.610160,
    ("FART", "Hierarchical BT"): 0.613369, ("FART", "Kernel RankSVM"): 0.617647,
    ("FART", "LightGBM"): 0.557754, ("FART", "NNLS ens"): 0.682888,
    ("FART", "Rank ens"): 0.670588, ("FART", "Mean ens"): 0.622995,
}


def _check_not_stale():
    """Warn if this bundle's copies have drifted from the generator's outputs.

    The bundle duplicates the per-cell OOFs so it can ship standalone, which means
    they can fall behind `../downstream_embedding_table.py`. That bit once: after
    the B1 fix the parent had corrected MMRF and LightGBM cells while the bundle
    still held the old ones, and regenerating here reproduced the stale numbers
    while reporting success. Only runs when the parent directory is present.
    """
    parent = HERE.parent
    if not (parent / "downstream_embedding_table.py").exists():
        return                                    # shipped standalone; nothing to compare
    stale = []
    for f in sorted(parent.glob("oof_*.csv")):
        mine = HERE / "oof" / f.name
        if mine.exists() and mine.read_bytes() != f.read_bytes():
            stale.append(f.name)
    res = parent / "downstream_results.csv"
    if res.exists() and (HERE / "downstream_results.csv").read_bytes() != res.read_bytes():
        stale.append("downstream_results.csv")
    if stale:
        print("STALE BUNDLE — these copies differ from the generator's current outputs:")
        for x in stale:
            print(f"    {x}")
        print("  Re-sync before trusting the result:")
        print("    python -c \"import shutil,pathlib as p; h=p.Path('.'); "
              "[shutil.copy2(f, h/'oof'/f.name) for f in p.Path('..').glob('oof_*.csv')]; "
              "shutil.copy2('../downstream_results.csv','downstream_results.csv')\"")
        raise SystemExit(1)


def oof_file(enc, label):
    return HERE / "oof" / f"oof_{enc}_{label.replace(' ','_').replace('(','').replace(')','')}.csv"


def dfify(keys, y, pred, target=TARGET):
    return pd.DataFrame({"category": [k[0] for k in keys], "product_code": [int(k[1]) for k in keys],
                        target: y, "predicted_score": pred})


def ensemble_oof(b, kind):
    # The public release ships within-category ranks plus the precomputed
    # per-fold NNLS weights rather than the panel-mean ratings the weights were
    # fitted on; either form reconstructs the same ensemble, bit for bit.
    y = b["y_all"] if "y_all" in b.files else b["y_rank"]
    W = b["nnls_weights"] if "nnls_weights" in b.files else None
    bo, bi, gem, cats = b["bt_outer"], b["bt_inner"], b["gem_oof"], b["categories"]
    n = len(bo)
    if kind == "Bradley-Terry":
        return bo
    if kind == "NNLS ens":
        o = np.zeros(n)
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            w = W[i] if W is not None else nnls(
                np.column_stack([bi[i, m], gem[m]]), y[m])[0]
            o[i] = np.array([bo[i], gem[i]]) @ w
        return o
    if kind == "Mean ens":
        return 0.5 * (bo + gem)
    d = pd.DataFrame({"c": cats, "bt": bo, "g": gem})
    return 0.5 * (d.groupby("c")["bt"].rank(pct=True).values + d.groupby("c")["g"].rank(pct=True).values)


rows = []
_check_not_stale()

for enc, bp in BASES.items():
    b = np.load(HERE / bp, allow_pickle=True)
    keys = [tuple(k) for k in b["product_keys"]]
    y = b["y_all"] if "y_all" in b.files else b["y_rank"]
    # This script REWRITES the per-cell ensemble OOFs it derives, so the written
    # column has to carry whichever name the data actually has — writing ranks
    # out as `true_score` would be both a lie and a no-ratings-gate failure.
    WRITE_TARGET = TARGET if "y_all" in b.files else RANK_TARGET
    for label in INDIV:                                   # load committed per-cell OOFs
        df = read_oof(oof_file(enc, label))
        rows.append((enc, label, df))
    for label in ENSEMBLE:                                # derive from base (bit-exact), also write OOF
        df = dfify(keys, y, ensemble_oof(b, label), target=WRITE_TARGET)
        oof_file(enc, label).parent.mkdir(exist_ok=True)
        df.to_csv(oof_file(enc, label), index=False)
        rows.append((enc, label, df.rename(columns={RANK_TARGET: TARGET})))

out = []
for enc, label, df in rows:
    df = df.dropna(subset=["predicted_score"])
    pw = compute_all_metrics(df)["pairwise_accuracy"]
    lo, hi = compute_bca_cis(df, n_bootstrap=10000)["pairwise_accuracy"]
    out.append({"encoder": enc, "model": label, "pw": pw, "ci_lo": lo, "ci_hi": hi})
    g = GATE.get((enc, label))
    if g is not None:
        assert abs(pw - g) < 5e-4, f"GATE FAIL {enc}/{label}: {pw:.6f} != {g}"
        print(f"GATE OK  {enc} {label} = {pw:.4f} (canonical {g})")

cis = pd.DataFrame(out)
cis.to_csv(HERE / "cis_encoder_comparison.csv", index=False)

# diff against shipped table
ship = pd.read_csv(HERE / "downstream_results.csv")
m = cis.merge(ship, on=["encoder", "model"], suffixes=("_re", "_ship"))
m["dpw"] = (m["pw_re"] - m["pw_ship"]).abs()
maxd = m["dpw"].max()
print(f"\n30-cell max |Δpairwise| vs shipped downstream_results.csv = {maxd:.2e}  "
      f"({'REPRODUCED' if maxd < 1e-9 else 'DIFF — investigate'})")
print(f"saved -> {HERE/'cis_encoder_comparison.csv'}")
