"""Hyperparameter tuning with Optuna. Writes the best params that main.py runs with.

  uv run python tune.py configs/tune.yaml [--models gcn_feat rf_feat] [--trials 50] [--device cuda]

One study per (model, loss, graph condition). Each condition gets its own best attempt, so
"real vs empty vs rewired" compares tuned-for-that-graph models; reusing real-graph params
on the controls had collapsed them (e.g. GAT on a rewired graph 0.83 -> 0.54 AUROC).
  real     the graph                     (also used for shufattr runs)
  rewired  rewired copy rw0              (used for rw0, rw1, rw2 runs)
  empty    no edges                      (propagation / GNN models only)
  none     graph-free models

Objective: mean val AUPRC over dedicated *tuning* split seeds (configs/tune.yaml), which
are never used for evaluation (main.py experiments use seeds 0-4). Test folds are never
touched. Studies live in SQLite, so parallel tune.py processes share them and an
interrupted run resumes.
Output: <out>/best_params.yaml  {model: {loss: {condition: {param: value}}}}  + trial CSVs
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
STUDY_VERSION = "v2"  # v1 studies (real-only, eval seeds, narrow ranges) are ignored
CONDITION_COPY = {"real": "real", "rewired": "rw0", "empty": "empty", "none": "real"}

NN = lambda t: {  # noqa: E731  shared by every torch model
    "lr": t.suggest_float("lr", 3e-4, 3e-2, log=True),
    "wd": t.suggest_float("wd", 1e-6, 5e-2, log=True),
    "hidden": t.suggest_categorical("hidden", [32, 64, 128, 256]),
    "layers": t.suggest_int("layers", 1, 3),
    "dropout": t.suggest_float("dropout", 0.0, 0.8),
}
GAT = lambda t: {**NN(t), "heads": t.suggest_categorical("heads", [1, 2, 4, 8])}  # noqa: E731
RF = lambda t: {  # noqa: E731
    "rf_trees": t.suggest_categorical("rf_trees", [200, 500, 1000]),
    "rf_leaf": t.suggest_categorical("rf_leaf", [1, 2, 5, 10, 20]),
    "rf_max_features": t.suggest_categorical("rf_max_features", ["sqrt", "0.1", "0.3"]),
}

# Search space per model kind (see ghelps.train.LADDER); MLP depth 1 is logistic regression.
SPACES = {
    "linear": lambda t, loss: {"lr": t.suggest_float("lr", 3e-4, 3e-2, log=True),
                               "wd": t.suggest_float("wd", 1e-6, 5e-2, log=True)},
    "mlp": lambda t, loss: NN(t), "sign": lambda t, loss: NN(t), "gcn": lambda t, loss: NN(t),
    "sage": lambda t, loss: NN(t),
    "gat": lambda t, loss: GAT(t), "gat_e": lambda t, loss: GAT(t),
    "cs": lambda t, loss: {**NN(t), "cs_alpha": t.suggest_float("cs_alpha", 0.3, 0.99),
                           "cs_layers": t.suggest_int("cs_layers", 5, 50)},
    "rf": lambda t, loss: RF(t),
    "lp": lambda t, loss: {"lp_alpha": t.suggest_float("lp_alpha", 0.3, 0.99),
                           "lp_layers": t.suggest_int("lp_layers", 2, 50)},
    # under PN the GP already trains on a 1:1 sample, so the unlabelled sample size is moot
    "gp": lambda t, loss: {"gp_noise_floor": t.suggest_float("gp_noise_floor", 0.05, 0.9),
                           **({"gp_n_unl": t.suggest_categorical("gp_n_unl", [500, 1000, 2000, 3000])}
                              if loss == "nnpu" else {})},
}
PRIOR = lambda t: t.suggest_float("prior", 0.005, 0.3, log=True)  # noqa: E731


def _clean(params: dict) -> dict:
    return {k: (float(v) if k == "rf_max_features" and v != "sqrt" else v) for k, v in params.items()}


def conditions_for(model: str) -> list[str]:
    from ghelps.train import LADDER
    conds = LADDER[model][2]
    if not conds:
        return ["none"]
    return [c for c in ("real", "rewired", "empty") if CONDITION_COPY[c] in conds]


def task_list(config: str = "configs/tune.yaml") -> list[tuple[str, str]]:
    """(model, condition) pairs, the unit of one HPC array task (see scripts/hpc/tune.sbatch)."""
    cfg = yaml.safe_load(open(ROOT / config))
    return [(m, c) for m in cfg["models"] for c in conditions_for(m)]


def objective_for(model: str, loss: str, condition: str, cfg: dict):
    from ghelps import data, evaluate, train
    kind, _, conds = train.LADDER[model]
    graph = cfg["graph"] if conds else None
    problems = [data.load(cfg["universe"], graph, CONDITION_COPY[condition], cfg["labels"], cfg["split"], s)
                for s in cfg["seeds"]]
    uses_prior = loss == "nnpu" and kind not in ("rf", "lp", "gp")

    def objective(trial: optuna.Trial) -> float:
        hp = SPACES[kind](trial, loss)
        if uses_prior:
            hp["prior"] = PRIOR(trial)
        vals = []
        for s, p in zip(cfg["seeds"], problems):
            scores, _ = train.score(p, model, loss, s, **_clean(hp))
            vals.append(evaluate.val_auprc(scores, p.y, p.fold))
        return float(np.mean(vals))

    return objective


def _prefix(cfg: dict) -> str:
    return f"{STUDY_VERSION}__{cfg['universe']}__{cfg['graph']}__{cfg['labels']}__{cfg['split']}__"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--models", nargs="+", help="subset of the config's models")
    ap.add_argument("--losses", nargs="+", help="subset of the config's losses")
    ap.add_argument("--conditions", nargs="+", choices=list(CONDITION_COPY), help="subset of conditions")
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
    print(f"device={device.get()} storage={storage} tuning seeds={cfg['seeds']}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for model in a.models or cfg["models"]:
        for loss in a.losses or cfg["losses"]:
            for cond in conditions_for(model):
                if a.conditions and cond not in a.conditions:
                    continue
                name = f"{_prefix(cfg)}{model}__{loss}__{cond}"
                study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                            sampler=optuna.samplers.TPESampler(seed=cfg.get("sampler_seed", 0)),
                                            load_if_exists=True)
                todo = n_trials - len([t for t in study.trials if t.state.is_finished()])
                if todo > 0:
                    study.optimize(objective_for(model, loss, cond, cfg), n_trials=todo,
                                   timeout=cfg.get("timeout_per_study"), catch=(RuntimeError, ValueError))
                study.trials_dataframe().to_csv(out / f"trials__{name}.csv", index=False)
                print(f"{model:16s} {loss:5s} {cond:8s} best val AUPRC {study.best_value:.4f}  "
                      f"{study.best_params}", flush=True)

    write_best(out, storage, cfg)


def write_best(out: Path, storage: str, cfg: dict) -> None:
    """Collect the best params of every finished v2 study into best_params.yaml."""
    prefix = _prefix(cfg)
    best: dict = {}
    for s in optuna.get_all_study_summaries(storage):
        if not s.study_name.startswith(prefix) or s.best_trial is None:
            continue
        model, loss, cond = s.study_name[len(prefix):].rsplit("__", 2)
        best.setdefault(model, {}).setdefault(loss, {})[cond] = {
            **_clean(s.best_trial.params), "_val_auprc": round(s.best_trial.value, 4)}
    with open(out / "best_params.yaml", "w") as f:
        yaml.safe_dump(best, f, sort_keys=True)
    n = sum(len(c) for m in best.values() for c in m.values())
    print(f"wrote {out / 'best_params.yaml'} ({n} model/loss/condition studies)")


if __name__ == "__main__":
    main()
