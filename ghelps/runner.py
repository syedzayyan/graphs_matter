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
AXES = ["universe", "graph", "copy", "labels", "split", "seed", "model", "loss"]


def expand(exp: dict, best: dict | None = None) -> list[dict]:
    """Grid -> run configs. Hyperparameters: tuned best params for (model, loss) if given,
    then the experiment's own `hparams` on top."""
    from ghelps.train import LADDER
    grid = exp["grid"]
    best = best or {}
    runs, seen = [], set()
    for combo in itertools.product(*(grid[a] for a in AXES)):
        r = dict(zip(AXES, combo))
        tuned = best.get(r["model"], {}).get(r["loss"], {})
        r.update({k: v for k, v in tuned.items() if not k.startswith("_")}, **exp.get("hparams", {}))
        conditions = LADDER[r["model"]][2]
        if not conditions:                     # graph-free: one run per (labels, split, seed, loss)
            r["graph"], r["copy"] = None, None
        elif r["copy"] not in conditions:      # e.g. shuffled edge vectors for a non-edge model
            continue
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
        p = data.load(r["universe"], r["graph"], r["copy"] or "real", r["labels"], r["split"], r["seed"])
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


def run(exp: dict, workers: int, best: dict | None = None) -> Path:
    out = ROOT / "results" / exp["name"]
    for d in ("runs", "scores"):
        (out / d).mkdir(parents=True, exist_ok=True)
    runs = expand(exp, best)
    # slowest first so the pool drains evenly
    runs.sort(key=lambda r: r["model"].startswith("gat"), reverse=True)
    print(f"{len(runs)} runs -> {out}", flush=True)
    # spawn, not fork: forked children must not inherit a CUDA context
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(execute, r, out) for r in runs]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(runs)}] {f.result()}", flush=True)
    collect(out)
    return out


def collect(out: Path) -> pd.DataFrame:
    """All finished runs -> <out>/results.csv, one row per run x evaluation regime, with the
    full config (incl. hyperparameters) and run-level extras (degree dependence, GP shares)."""
    rows = []
    for f in sorted((out / "runs").glob("*.json")):
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
