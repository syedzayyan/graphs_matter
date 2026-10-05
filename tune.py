"""Hyperparameter tuning with Optuna. Writes the best params that main.py runs with.

  uv run python tune.py configs/tune.yaml [--models gcn_feat rf_feat] [--trials 50] [--device cuda]

One study per (model, loss). Objective: mean val AUPRC (val positives vs val unlabelled)
over the configured tuning seeds, on the real graph. The test fold is never touched, and
the tuned params are reused unchanged for the rewired / empty / shuffled conditions so the
graph is the only thing that varies between them.

Studies live in an SQLite file, so several tune.py processes (e.g. one per GPU job) can
work on the same study in parallel and an interrupted run resumes where it stopped.
Output: <out>/best_params.yaml   {model: {loss: {param: value}}}  (+ trial tables as CSV)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

NN = lambda t: {  # noqa: E731  shared by every torch model
    "lr": t.suggest_float("lr", 1e-3, 3e-2, log=True),
    "wd": t.suggest_float("wd", 1e-6, 1e-2, log=True),
    "hidden": t.suggest_categorical("hidden", [32, 64, 128, 256]),
    "layers": t.suggest_int("layers", 1, 3),
    "dropout": t.suggest_float("dropout", 0.0, 0.7),
}
GAT = lambda t: {**NN(t), "heads": t.suggest_categorical("heads", [1, 2, 4, 8])}  # noqa: E731
RF = lambda t: {  # noqa: E731
    "rf_trees": t.suggest_categorical("rf_trees", [200, 500, 1000]),
    "rf_leaf": t.suggest_categorical("rf_leaf", [1, 2, 5, 10, 20]),
    "rf_max_features": t.suggest_categorical("rf_max_features", ["sqrt", "0.1", "0.3"]),
}

# Search space per model kind (see ghelps.train.LADDER). GNN depth starts at 1 so that
# "0 extra hops" style solutions are reachable; MLP depth 1 is logistic regression.
SPACES = {
    "linear": lambda t: {"lr": t.suggest_float("lr", 1e-3, 3e-2, log=True),
                         "wd": t.suggest_float("wd", 1e-6, 1e-2, log=True)},
    "mlp": NN, "sign": NN, "gcn": NN, "gat": GAT, "gat_e": GAT,
    "cs": lambda t: {**NN(t), "cs_alpha": t.suggest_float("cs_alpha", 0.5, 0.99),
                     "cs_layers": t.suggest_int("cs_layers", 10, 50)},
    "rf": RF,
    "lp": lambda t: {"lp_alpha": t.suggest_float("lp_alpha", 0.5, 0.99),
                     "lp_layers": t.suggest_int("lp_layers", 5, 50)},
    "gp": lambda t: {"gp_noise_floor": t.suggest_float("gp_noise_floor", 0.05, 0.5),
                     "gp_n_unl": t.suggest_categorical("gp_n_unl", [1000, 2000, 3000])},
}
NNPU = {"prior": lambda t: t.suggest_float("prior", 0.03, 0.3, log=True)}


def _clean(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if k == "rf_max_features" and v != "sqrt":
            v = float(v)
        out[k] = v
    return out


def objective_for(model: str, loss: str, cfg: dict):
    from ghelps import data, evaluate, train
    kind, _, conditions = train.LADDER[model]
    graph = cfg["graph"] if conditions else None
    problems = [data.load(cfg["universe"], graph, "real", cfg["labels"], cfg["split"], s) for s in cfg["seeds"]]
    uses_prior = loss == "nnpu" and kind not in ("rf", "lp", "gp")

    def objective(trial: optuna.Trial) -> float:
        hp = SPACES[kind](trial)
        if uses_prior:
            hp["prior"] = NNPU["prior"](trial)
        vals = []
        for s, p in zip(cfg["seeds"], problems):
            scores, _ = train.score(p, model, loss, s, **_clean(hp))
            vals.append(evaluate.val_auprc(scores, p.y, p.fold))
        return float(np.mean(vals))

    return objective


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--models", nargs="+", help="subset of the config's models")
    ap.add_argument("--losses", nargs="+", help="subset of the config's losses")
    ap.add_argument("--trials", type=int, help="trials per study (overrides config)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()
    if a.device:
        os.environ["GHELPS_DEVICE"] = a.device
    from ghelps import device, ensure

    ensure.ensure_data()  # fetch + build anything missing (e.g. on a fresh HPC checkout)
    cfg = yaml.safe_load(open(a.config))
    out = ROOT / cfg.get("out", "results/tuning")
    out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out / 'optuna.db'}"
    n_trials = a.trials or cfg["trials"]
    print(f"device={device.get()} storage={storage}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for model in a.models or cfg["models"]:
        for loss in a.losses or cfg["losses"]:
            name = f"{cfg['universe']}__{cfg['graph']}__{cfg['labels']}__{cfg['split']}__{model}__{loss}"
            study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=cfg.get("sampler_seed", 0)),
                                        load_if_exists=True)
            todo = n_trials - len([t for t in study.trials if t.state.is_finished()])
            if todo > 0:
                study.optimize(objective_for(model, loss, cfg), n_trials=todo,
                               timeout=cfg.get("timeout_per_study"), catch=(RuntimeError, ValueError))
            study.trials_dataframe().to_csv(out / f"trials__{name}.csv", index=False)
            print(f"{model:16s} {loss:5s} best val AUPRC {study.best_value:.4f}  {study.best_params}", flush=True)

    write_best(out, storage, cfg)


def write_best(out: Path, storage: str, cfg: dict) -> None:
    """Collect the best params of every finished study into best_params.yaml."""
    prefix = f"{cfg['universe']}__{cfg['graph']}__{cfg['labels']}__{cfg['split']}__"
    best: dict = {}
    for s in optuna.get_all_study_summaries(storage):
        if not s.study_name.startswith(prefix) or s.best_trial is None:
            continue
        model, loss = s.study_name[len(prefix):].rsplit("__", 1)
        best.setdefault(model, {})[loss] = {**_clean(s.best_trial.params),
                                            "_val_auprc": round(s.best_trial.value, 4)}
    with open(out / "best_params.yaml", "w") as f:
        yaml.safe_dump(best, f, sort_keys=True)
    print(f"wrote {out / 'best_params.yaml'} ({sum(len(v) for v in best.values())} model/loss pairs)")


if __name__ == "__main__":
    main()
