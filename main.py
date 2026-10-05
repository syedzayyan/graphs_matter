"""Single entry point.

  uv run python main.py device                                   # which device will be used
  uv run python main.py fetch                                    # download missing raw sources
  uv run python main.py build [--steps graphs structure ...]     # stages 1-2 (data release)
  uv run python main.py run configs/exp_headline.yaml [--params results/tuning/best_params.yaml]
  uv run python main.py summarise headline [--metric auroc|auprc]

`run` (and tune.py) first fetch any missing raw data and build any missing processed data.
Device: CUDA when available, otherwise CPU (override with --device or GHELPS_DEVICE).
Hyperparameters: the best params written by tune.py, if present (--no-tuned for defaults).
Results: results/<exp>/results.csv (every run x regime) and summary_*.csv from `summarise`.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                    help="compute device (default: GHELPS_DEVICE or auto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("device", help="print the device runs will use")
    sub.add_parser("fetch", help="download any missing raw sources into data/raw")

    b = sub.add_parser("build", help="fetch + build graphs, structural stats, edge vectors, features, splits")
    b.add_argument("--steps", nargs="+", choices=["graphs", "structure", "edge_attr", "features", "splits"],
                   help="rebuild exactly these steps (default: whatever is missing)")

    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("config")
    r.add_argument("--workers", type=int, default=None,
                   help="parallel runs (default: 4 on CUDA, cpu_count - 2 on CPU)")
    r.add_argument("--name", help="override the experiment name")
    r.add_argument("--params", default="results/tuning/best_params.yaml",
                   help="tuned hyperparameters from tune.py (used if the file exists)")
    r.add_argument("--no-tuned", action="store_true", help="ignore tuned params, use DEFAULTS")

    s = sub.add_parser("summarise", help="aggregate an experiment's results")
    s.add_argument("exp")
    s.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])

    a = ap.parse_args()
    if a.device:
        os.environ["GHELPS_DEVICE"] = a.device  # read by ghelps.device in every worker
    sys.path.insert(0, str(ROOT))
    from ghelps import device, ensure, fetch

    if a.cmd == "device":
        import torch
        dev = device.get()
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
        print(f"{dev} ({name})")

    elif a.cmd == "fetch":
        fetch.fetch_all()

    elif a.cmd == "build":
        if a.steps:
            fetch.fetch_all()
            ensure.build(a.steps, force=True)
        else:
            ensure.ensure_data()

    elif a.cmd == "run":
        import yaml
        from ghelps import runner
        ensure.ensure_data()
        exp = yaml.safe_load(open(a.config))
        if a.name:
            exp["name"] = a.name
        workers = a.workers or (4 if device.get().type == "cuda" else max(1, (os.cpu_count() or 2) - 2))
        params = ROOT / a.params
        best = None
        if not a.no_tuned and params.exists():
            best = yaml.safe_load(open(params))
            print(f"tuned params: {params} ({sum(len(v) for v in best.values())} model/loss pairs)")
        else:
            print("tuned params: none, using ghelps.train.DEFAULTS")
        print(f"device={device.get()} workers={workers}", flush=True)
        runner.run(exp, workers, best)

    elif a.cmd == "summarise":
        sys.argv = ["summarise.py", a.exp, a.metric]
        runpy.run_path(str(ROOT / "scripts" / "summarise.py"), run_name="__main__")


if __name__ == "__main__":
    main()
