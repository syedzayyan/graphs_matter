"""Expand an experiment config into runs, execute them in parallel, skip finished ones.

Each run writes results/<exp>/runs/<run_id>.json (config + metrics) and
results/<exp>/scores/<run_id>.npy (one score per universe gene).
"""
from __future__ import annotations

import hashlib
import itertools
import multiprocessing as mp
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from ghelps.paths import PROCESSED as _PROCESSED  # noqa: E402
P_SPLITS = _PROCESSED / "splits"
AXES = ["universe", "graph", "copy", "labels", "feats", "split", "seed", "model", "loss"]
AXIS_DEFAULTS = {"feats": ["all"]}  # axes an experiment config may omit


def tuning_condition(copy: str | None) -> str:
    """Which tuned parameter set a graph condition uses (see tune.py)."""
    if copy is None:
        return "none"
    if copy.startswith("rw"):
        return "rewired"
    if copy == "empty":
        return "empty"
    return "real"  # shufattr and pwsub are variants of the real graph


def tuning_loss(loss: str) -> str:
    """Hard-negative regimes are PN with different negatives: they reuse PN's params."""
    return "pn" if loss.startswith("pn") else loss


def tuned_params(best: dict, universe: str, model: str, loss: str, copy: str | None) -> dict:
    """best_params.yaml lookup: {universe: {model: {loss: {condition: params}}}}. A missing
    entry means DEFAULTS; params from another condition are never borrowed. Params are tuned
    on Minikel labels with the full feature set and reused for the other label sets / feature
    sets of the same universe."""
    from ghelps.train import LADDER
    if LADDER[model][0] == "sklr":
        return {}  # untuned by design
    if best and not set(best) & set(_universes()):
        raise ValueError("best_params.yaml predates per-graph tuning (no universe level); "
                         "re-run tune.py or pass --no-tuned")
    per_cond = best.get(universe, {}).get(model, {}).get(tuning_loss(loss), {})
    params = per_cond.get(tuning_condition(copy), {})
    return {k: v for k, v in params.items() if not k.startswith("_")}


def _universes() -> dict:
    import yaml
    return yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["universes"]


def expand(exp: dict, best: dict | None = None) -> list[dict]:
    """Grid -> run configs. Hyperparameters: the tuned params for (universe, model, loss,
    graph condition) if given, then the experiment's own `hparams` on top. Combinations whose
    graph is not built on the universe are skipped."""
    from ghelps.train import LADDER, uses_features
    grid = {**AXIS_DEFAULTS, **exp["grid"]}
    best = best or {}
    universes = _universes()
    runs, seen = [], set()
    for combo in itertools.product(*(grid[a] for a in AXES)):
        r = dict(zip(AXES, combo))
        if r["feats"] != "all" and not uses_features(r["model"]):
            continue                           # feature sets are moot for feature-free models
        conditions = LADDER[r["model"]][2]
        if not conditions:                     # graph-free: one run per (labels, split, seed, loss)
            r["graph"], r["copy"] = None, None
        elif r["copy"] not in conditions:      # e.g. shuffled edge vectors for a non-edge model
            continue
        elif r["graph"] not in universes[r["universe"]]:
            continue
        if not (P_SPLITS / r["universe"] / r["labels"] / r["split"]).exists():
            continue                           # split not built for this universe (e.g. too few wave genes)
        r.update(tuned_params(best, r["universe"], r["model"], r["loss"], r["copy"]), **exp.get("hparams", {}))
        key = run_id(r)
        if key not in seen:
            seen.add(key)
            runs.append(r)
    return runs


def run_id(r: dict) -> str:
    base = "__".join(str(r.get(a)) for a in AXES)
    extra = {k: v for k, v in r.items() if k not in AXES}
    if extra:
        base += "__" + hashlib.md5(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:8]
    return base


def execute(r: dict, out: Path) -> str:
    import torch
    torch.set_num_threads(1)
    from ghelps import data, evaluate, train

    rid = run_id(r)
    if (out / "runs" / f"{rid}.json").exists():
        return f"skip {rid}"
    t0 = time.time()
    try:
        hp = {k: v for k, v in r.items() if k not in AXES}
        p = data.load(r["universe"], r["graph"], r["copy"] or "real", r["labels"], r["split"], r["seed"],
                      feats=r["feats"])
        s, extras = train.score(p, r["model"], r["loss"], r["seed"], **hp)
        metrics = evaluate.by_regime(s, p.negatives)
        # degree dependence is always measured against the *real* graph's degree so that
        # graph-free and rewired runs are comparable
        deg = None
        if r["graph"] is not None:
            deg = data._struct(r["universe"], r["graph"], "real")[1]
        extra = evaluate.degree_dependence(s, deg, p.mask("test"))
        extra["val_auprc"] = evaluate.val_auprc(s, p.y, p.fold)
        if "gp_parts" in extras:
            extra.update(_gp_summary(extras, s, p))
            pd.DataFrame({**extras["gp_parts"], "score": s, "sd": extras["gp_sd"]}).astype(np.float32) \
                .to_parquet(out / "scores" / f"{rid}.gp.parquet")
        np.save(out / "scores" / f"{rid}.npy", s.astype(np.float32))
        rec = {"config": r, "metrics": metrics, **extra, "seconds": time.time() - t0}
        (out / "runs" / f"{rid}.json").write_text(json.dumps(rec, default=float))
        return f"done {rid} ({time.time() - t0:.0f}s) random auroc={metrics['random']['auroc']:.3f}"
    except Exception:
        (out / "errors").mkdir(exist_ok=True)
        (out / "errors" / f"{rid}.txt").write_text(traceback.format_exc())
        return f"FAIL {rid}"


def _gp_summary(extras: dict, s: np.ndarray, p) -> dict:
    """share[b] = cov(f_b, f) / var(f) over test genes: sums to 1 exactly even when the
    blocks are correlated. Plus whether uncertainty tracks degree / study intensity."""
    from scipy.stats import spearmanr
    from ghelps.data import P
    te = p.mask("test")
    tot = s[te] - s[te].mean()
    share = {b: float(np.dot(v[te] - v[te].mean(), tot) / np.dot(tot, tot)) for b, v in extras["gp_parts"].items()}
    pubmed = pd.read_parquet(P / "features" / p.universe / "pubmed.parquet").pubmed_count.values
    sd = extras["gp_sd"]
    return {"gp_share": share, "gp_hyper": extras["gp_hyper"],
            "gp_sd_vs_degree": float(spearmanr(sd[te], p.raw_degree[te]).statistic),
            "gp_sd_vs_pubmed": float(spearmanr(sd[te], pubmed[te]).statistic)}


def run(exp: dict, workers: int, best: dict | None = None, shard: tuple[int, int] = (0, 1)) -> Path:
    """Run the grid (or shard i of n of it, for HPC array jobs; shards are disjoint and
    together cover the grid). Finished runs are skipped, so re-running resumes."""
    out = ROOT / "results" / exp["name"]
    for d in ("runs", "scores"):
        (out / d).mkdir(parents=True, exist_ok=True)
    runs = expand(exp, best)
    # results.csv / summaries only cover the current grid, not runs left over from an earlier
    # grid or earlier tuned params (their run ids differ via the hyperparameter hash)
    (out / "manifest.txt").write_text("\n".join(run_id(r) for r in runs) + "\n")
    i, n = shard
    runs = runs[i::n]
    # slowest first so the pool drains evenly
    runs.sort(key=lambda r: r["model"].startswith("gat"), reverse=True)
    print(f"{len(runs)} runs -> {out}", flush=True)
    # spawn, not fork: forked children must not inherit a CUDA context
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(execute, r, out) for r in runs]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(runs)}] {f.result()}", flush=True)
    if n == 1:  # sharded runs are collected once at the end (main.py summarise)
        collect(out)
    return out


def collect(out: Path) -> pd.DataFrame:
    """Finished runs of the current grid (manifest.txt) -> <out>/results.csv, one row per
    run x evaluation regime, with the full config (incl. hyperparameters) and run-level
    extras (degree dependence, GP shares)."""
    rows = []
    manifest = out / "manifest.txt"
    files = sorted((out / "runs").glob("*.json"))
    if manifest.exists():
        keep = set(manifest.read_text().split())
        stale = [f for f in files if f.stem not in keep]
        files = [f for f in files if f.stem in keep]
        if stale:
            print(f"collect: ignoring {len(stale)} runs not in the current grid", flush=True)
    for f in files:
        r = json.loads(f.read_text())
        c = r["config"]
        hp = {k: v for k, v in c.items() if k not in AXES}
        base = {**{a: c.get(a) for a in AXES}, "run_id": f.stem, "hparams": json.dumps(hp, sort_keys=True),
                "rho_degree_test": r.get("spearman_degree_test"), "val_auprc": r.get("val_auprc"),
                "seconds": r.get("seconds")}
        base.update({f"gp_share_{k}": v for k, v in r.get("gp_share", {}).items()})
        if "gp_sd_vs_degree" in r:
            base.update(gp_sd_vs_degree=r["gp_sd_vs_degree"], gp_sd_vs_pubmed=r["gp_sd_vs_pubmed"])
        for regime, m in r["metrics"].items():
            rows.append({**base, "regime": regime, **m})
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    print(f"wrote {out / 'results.csv'} ({len(df)} rows)", flush=True)
    return df
