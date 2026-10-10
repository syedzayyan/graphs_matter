"""Make sure raw data and the processed benchmark exist before tuning or running.

ensure_data(): fetch missing raw sources, then run every build step whose outputs are
missing (and every step after it, since later steps depend on earlier ones). Guarded by
a file lock so parallel jobs (e.g. a tuning array) don't fetch or build at the same time;
the first job does the work and the others wait, then find everything in place.
"""
from __future__ import annotations

import fcntl
import runpy
import shutil
from contextlib import contextmanager
from pathlib import Path

import yaml

from ghelps import fetch, paths

ROOT = Path(__file__).resolve().parents[1]
P = paths.PROCESSED
STEPS = ["graphs", "structure", "edge_attr", "features", "splits"]


def _universes() -> dict[str, list[str]]:
    return yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["universes"]


def _done(step: str) -> bool:
    u = _universes()
    if step == "graphs":
        from ghelps.graphs import universe_graphs
        cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
        return (P / "graphs" / "summary.tsv").exists() and all(
            (P / "graphs" / n / f"{g}.{c}.npy").exists() for n in u for g in universe_graphs(cfg, n)
            for c in ("real", "pwsub"))
    if step == "structure":
        want = [g.name.replace(".npy", ".parquet") for n in u for g in (P / "graphs" / n).glob("*.npy")
                if ".empty." not in g.name]
        return bool(want) and all((P / "structure" / n / w).exists() for n in u for w in want
                                  if (P / "graphs" / n / w.replace(".parquet", ".npy")).exists())
    if step == "edge_attr":
        from ghelps.graphs import universe_graphs
        cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
        return all((P / "edge_attr" / n / f"{g}.pwsub.npy").exists() and
                   (P / "edge_attr" / n / f"{g}.shufattr.npy").exists() and
                   (P / "graphs" / n / f"{g}.empty.npy").exists() for n in u for g in universe_graphs(cfg, n))
    if step == "features":
        return all((P / "features" / n / f"{b}.parquet").exists() for n in u
                   for b in ("pubmed", "esm", "seqfeat", "gtex"))
    if step == "splits":
        sc = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["splits"]
        return (P / "splits" / "summary.tsv").exists() and all(
            (P / "splits" / n / ls / "random" / f"seed{sc['n_seeds'] - 1}.neg.parquet").exists()
            for n in u for ls in sc["label_sets"])
    raise ValueError(step)


@contextmanager
def _lock():
    paths.DATA.mkdir(parents=True, exist_ok=True)
    with open(paths.DATA / ".ensure.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def build(steps: list[str] | None = None, force: bool = False) -> None:
    """Run build steps in dependency order. Without `force`, start from the first step
    whose outputs are missing; with `force`, run exactly `steps`."""
    if force:
        todo = [s for s in STEPS if s in (steps or STEPS)]
    else:
        first = next((i for i, s in enumerate(STEPS) if not _done(s)), None)
        todo = [] if first is None else STEPS[first:]
    fresh = not all((P / "graphs" / n / "genes.txt").exists() for n in _universes())
    if "graphs" in todo and fresh and not force:
        shutil.rmtree(P, ignore_errors=True)  # partial first build: start clean
    # otherwise build steps are deterministic and rewrite identical files; structure stats
    # (the slow step) skip graphs that already have them
    for step in todo:
        print(f"[build] {step}", flush=True)
        runpy.run_path(str(ROOT / "scripts" / f"build_{step}.py"), run_name="__main__")


def ensure_data() -> None:
    with _lock():
        if fetch.missing():
            fetch.fetch_all()
        if not all(_done(s) for s in STEPS):
            build()
