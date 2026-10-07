"""Paired AUROC significance tests (DeLong, via MLstatkit) for the graph-effect contrasts.

  uv run python scripts/delong.py main [membrane ...]   ->  results/<exp>/delong.csv

Within one seed, two runs that differ only in the graph condition (or a graph model vs the
no-graph RF) are scored on exactly the same test genes, so their AUROCs are correlated and
DeLong's test applies. Per (universe, graph, labels, feats, split, loss, model, regime,
contrast) it reports:
  delta_auroc     mean over seeds of AUROC(real) - AUROC(control)
  z_seeds         per-seed DeLong z (positive = real better); for "rewired" the z against
                  each rewired copy is averaged within the seed
  stouffer_z, p   seeds combined with Stouffer's method (two-sided p)
  p_bh            Benjamini-Hochberg adjusted within the experiment
Test genes per regime are those evaluate.by_regime scores (random: all test positives vs all
test unlabelled; matched regimes: the pooled matched sets). Labels are positive vs
unlabelled, so these compare rankings of known targets against unlabelled genes.
Needs the processed data the runs used ($GHELPS_DATA), so it runs in the summarise job.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from MLstatkit.stats import Delong_test
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghelps import paths  # noqa: E402

RES = ROOT / "results"
KEYS = ["universe", "graph", "labels", "feats", "split", "seed", "model", "loss"]
BASELINE = "rf_feat"  # no-graph reference for "does the graph model beat features alone?"


def regime_genes(universe: str, labels: str, split: str, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """regime -> (gene indices, 0/1 labels) exactly as ghelps.evaluate.by_regime scores them."""
    neg = pd.read_parquet(paths.PROCESSED / "splits" / universe / labels / split / f"seed{seed}.neg.parquet")
    out = {}
    for regime, g in neg.groupby("regime"):
        pos = g.pos[g.pos >= 0].unique()
        ng = g.neg[g.neg >= 0].unique()
        if len(pos) and len(ng):
            out[regime] = (np.r_[pos, ng], np.r_[np.ones(len(pos)), np.zeros(len(ng))])
    return out


def delong_z(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """DeLong z for AUROC(a) vs AUROC(b), signed so positive means a is better. MLstatkit's
    Delong_test returns (z, p); only |z| is taken from it and the sign comes from the AUROC
    difference itself, so the result doesn't depend on the package's sign convention."""
    z, _ = Delong_test(y, a, b)
    if not np.isfinite(z):  # identical rankings -> no difference
        return 0.0
    return float(np.sign(roc_auc_score(y, a) - roc_auc_score(y, b)) * abs(z))


def run(exp: str) -> pd.DataFrame:
    out = RES / exp
    df = pd.read_csv(out / "results.csv").drop_duplicates("run_id")
    df["copy"] = df["copy"].fillna("none")
    df["feats"] = df["feats"].fillna("all")
    scores = lambda rid: np.load(out / "scores" / f"{rid}.npy")  # noqa: E731
    cache: dict = {}

    rows = []
    real = df[df["copy"] == "real"]
    base = df[df.model == BASELINE].set_index(["universe", "labels", "feats", "split", "seed", "loss"]).run_id
    cells = {k: g for k, g in df[df["copy"] != "none"].groupby(KEYS)}  # one cell = all conditions of a run
    for _, r in real.iterrows():
        cell = cells[tuple(r[k] for k in KEYS)]
        controls = {c: cell[cell["copy"] == c].run_id.tolist() for c in ("empty", "shufattr", "pwsub")}
        controls["rewired"] = cell[cell["copy"].str.startswith("rw")].run_id.tolist()
        bkey = (r.universe, r.labels, r.feats, r.split, r.seed, r.loss)
        if r.model != BASELINE and bkey in base.index:
            controls[f"vs_{BASELINE}"] = [base.loc[bkey]]
        ck = (r.universe, r.labels, r.split, r.seed)
        if ck not in cache:
            cache[ck] = regime_genes(*ck)
        s_real = scores(r.run_id)
        for regime, (idx, y) in cache[ck].items():
            for contrast, rids in controls.items():
                if not rids:
                    continue
                zs, deltas = [], []
                for rid in rids:
                    s_c = scores(rid)
                    zs.append(delong_z(y, s_real[idx], s_c[idx]))
                    deltas.append(roc_auc_score(y, s_real[idx]) - roc_auc_score(y, s_c[idx]))
                rows.append({**{k: r[k] for k in KEYS}, "regime": regime, "contrast": contrast,
                             "z": float(np.mean(zs)), "delta": float(np.mean(deltas))})
    per_seed = pd.DataFrame(rows)
    group = [k for k in KEYS if k != "seed"] + ["regime", "contrast"]
    g = per_seed.groupby(group)
    res = g.agg(n_seeds=("z", "size"), delta_auroc=("delta", "mean"),
                stouffer_z=("z", lambda z: z.sum() / np.sqrt(len(z))),
                z_seeds=("z", lambda z: " ".join(f"{v:+.2f}" for v in z))).reset_index()
    res["p"] = 2 * norm.sf(np.abs(res.stouffer_z))
    res["p_bh"] = multipletests(res.p, method="fdr_bh")[1]
    res.to_csv(out / "delong.csv", index=False)
    sig = res[(res.regime == "random") & (res.p_bh < 0.05)]
    print(f"{exp}: {len(res)} tests, {len(sig)} significant at BH 5% (random regime) -> {out / 'delong.csv'}")
    return res


if __name__ == "__main__":
    for e in sys.argv[1:] or ["main", "membrane", "nonsense_labels", "nonsense_feats"]:
        if (RES / e / "results.csv").exists():
            run(e)
        else:
            print(f"skip {e}: no results.csv")
