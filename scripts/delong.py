"""Paired AUROC significance tests (DeLong, via MLstatkit) for the graph-effect contrasts.

  uv run python scripts/delong.py main [membrane ...]   ->  results/<exp>/delong.csv

Within one seed, two runs that differ only in the graph condition (or a graph model vs the
no-graph RF) are scored on exactly the same test genes, so their AUROCs are correlated and
DeLong's test applies. Per (universe, graph, labels, feats, split, loss, model, regime,
contrast) it reports:
  delta_auroc     mean over seeds of AUROC(real) - AUROC(control)
  z_seeds         per-seed DeLong z (positive = real better); for "rewired" the z against
                  each rewired copy is averaged within the seed; "vs_minikel_label" compares a
                  run trained on another label set with its Minikel twin (positive = the other
                  label better) on the test genes both share (genetic/time)
  stouffer_z, p   seeds combined with Stouffer's method (two-sided p)
  p_bh            Benjamini-Hochberg adjusted within the experiment
effects.csv: effect sizes instead of counts of significant cells. Cells share seeds, genes and
configurations, so counting them overstates the evidence. Per contrast (and per contrast x
feature set / universe / split / loss / label set; random regime), the mean AUROC difference
weighting every model equally (mean over models of the mean over that model's cells of the
mean over seeds), with a 95% hierarchical bootstrap CI: resample models, then cells within each
model, then seeds within each cell (2000 draws). The three universes are too few to resample,
so they are reported separately. An experiment with several graphs on one universe (dated
graphs) also gets per graph and per graph x feature set rows (by "graph|feats", level
"<graph>|<feats>"). Per-seed deltas: delong_seeds.csv.gz.
Test genes per regime are those evaluate.by_regime scores (random: all test positives vs all
test unlabelled; matched regimes: the pooled matched sets). Labels are positive vs
unlabelled, so these compare rankings of known targets against unlabelled genes.
Needs the processed data the runs used ($GHELPS_DATA), so it runs in the summarise job.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from MLstatkit import Delong_test
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghelps import paths  # noqa: E402

RES = ROOT / "results"
KEYS = ["universe", "graph", "labels", "feats", "split", "seed", "model", "loss"]
BASELINE = "rf_feat"  # no-graph reference for "does the graph model beat features alone?"


@lru_cache(maxsize=None)
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
    """DeLong z for AUROC(a) vs AUROC(b), signed so positive means a is better.
    MLstatkit 0.1.91: Delong_test(..., return_ci=False, return_auc=False) -> (z, p) with
    z = (AUC_B - AUC_A) / se. Only |z| is used; the sign comes from the AUROC difference, and
    p is recomputed after combining seeds (its p for identical scores is 0, not 1).
    Identical rankings (e.g. a graph condition that leaves a model's scores unchanged) have
    zero variance, for which MLstatkit falls back to a slow unseeded 5000-sample bootstrap;
    they are no difference by definition, so z = 0. Other degenerate cases use a seeded,
    smaller bootstrap."""
    if np.array_equal(rankdata(a), rankdata(b)):
        return 0.0
    z = Delong_test(y, a, b, return_ci=False, return_auc=False, n_boot=1000, random_state=0)[0]
    if not np.isfinite(z):  # degenerate variance -> no difference
        return 0.0
    return float(np.sign(roc_auc_score(y, a) - roc_auc_score(y, b)) * abs(z))


def _compare(task: tuple) -> list[dict]:
    """All contrasts of one real-graph run, in every evaluation regime."""
    exp, keys, real_rid, controls, ck = task
    scores_dir = RES / exp / "scores"
    s_real = np.load(scores_dir / f"{real_rid}.npy")
    s_ctrl = {rid: np.load(scores_dir / f"{rid}.npy") for rids in controls.values() for rid in rids}
    rows = []
    for regime, (idx, y) in regime_genes(*ck).items():
        for contrast, rids in controls.items():
            zs = [delong_z(y, s_real[idx], s_ctrl[rid][idx]) for rid in rids]
            deltas = [roc_auc_score(y, s_real[idx]) - roc_auc_score(y, s_ctrl[rid][idx]) for rid in rids]
            rows.append({**keys, "regime": regime, "contrast": contrast,
                         "z": float(np.mean(zs)), "delta": float(np.mean(deltas))})
    return rows


EFFECT_BY = ["feats", "universe", "split", "loss", "labels"]


def _hier_boot(g: pd.DataFrame, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    """Mean delta over models (each model's mean over its cells, each cell's mean over seeds)
    with a model > cell > seed bootstrap CI."""
    cells = [k for k in KEYS if k != "seed"] + ["regime", "contrast"]
    model_boot, model_pt = [], []
    for _, gm in g.groupby("model"):
        cb, cp = [], []
        for _, gc in gm.groupby(cells):
            d = gc.delta.values
            cb.append(d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1))
            cp.append(d.mean())
        cb = np.stack(cb, 1)  # (n_boot, n_cells)
        pick = rng.integers(0, cb.shape[1], cb.shape)
        model_boot.append(np.take_along_axis(cb, pick, 1).mean(1))
        model_pt.append(np.mean(cp))
    mb = np.stack(model_boot, 1)  # (n_boot, n_models)
    boot = np.take_along_axis(mb, rng.integers(0, mb.shape[1], mb.shape), 1).mean(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(mean_delta=float(np.mean(model_pt)), ci_lo=lo, ci_hi=hi, n_models=mb.shape[1],
                n_cells=int(g.groupby(cells).ngroups), n_seed_deltas=len(g))


def effects(per_seed: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    d = per_seed[per_seed.regime == "random"]
    rows = []
    bys = [[]] + [[a] for a in EFFECT_BY if d[a].nunique() > 1]
    if d.groupby("universe").graph.nunique().max() > 1:  # several graphs on one universe (dated graphs)
        bys += [["graph"], ["graph", "feats"]]
    for by in bys:
        for key, g in d.groupby(["contrast"] + by):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"contrast": key[0], "by": "|".join(by) or "-", "level": "|".join(map(str, key[1:])) or "-",
                         **_hier_boot(g, rng)})
    return pd.DataFrame(rows)


def run(exp: str) -> pd.DataFrame:
    out = RES / exp
    df = pd.read_csv(out / "results.csv").drop_duplicates("run_id")
    df["copy"] = df["copy"].fillna("none")
    df["feats"] = df["feats"].fillna("all")
    tasks = []
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
        controls = {c: rids for c, rids in controls.items() if rids}
        tasks.append((exp, {k: r[k] for k in KEYS}, r.run_id, controls, (r.universe, r.labels, r.split, int(r.seed))))
    # label contrast: the same model / graph condition / split / seed trained on another label
    # set than Minikel, where that label's split shares Minikel's test genes (genetic/time)
    twin_keys = ["universe", "graph", "copy", "feats", "split", "seed", "model", "loss"]
    mk = df[df.labels == "minikel"].fillna({"graph": "-"}).set_index(twin_keys).run_id
    for _, r in df[(df.labels != "minikel") & df["copy"].isin(["real", "none"])].fillna({"graph": "-"}).iterrows():
        twin = tuple(r[k] for k in twin_keys)
        if twin in mk.index:
            tasks.append((exp, {k: r[k] for k in KEYS}, r.run_id, {"vs_minikel_label": [mk.loc[twin]]},
                          (r.universe, r.labels, r.split, int(r.seed))))
    workers = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as ex:
        rows = [row for rs in ex.map(_compare, tasks, chunksize=16) for row in rs]
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "delong_seeds.csv.gz", index=False)
    effects(per_seed).to_csv(out / "effects.csv", index=False)
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
    for e in sys.argv[1:] or ["main", "waves", "membrane", "nonsense_labels", "nonsense_feats", "novelty"]:
        if (RES / e / "results.csv").exists():
            run(e)
        else:
            print(f"skip {e}: no results.csv")
