"""Turn one run config into scores for every gene.

Run config keys: universe, graph, copy, labels, split, seed, model, loss (nnpu | pn), plus
optional hyperparameter overrides (see DEFAULTS).
"""
from __future__ import annotations

import copy as _copy

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ghelps import device, gp, losses, models
from ghelps.data import Problem, gp_inputs

# Graph conditions each model is run under (see ghelps.data). "struct" models only see
# structural statistics, so the empty graph and shuffled edge vectors are meaningless for
# them; edge vectors only matter to the edge-aware model.
NOGRAPH: tuple[str, ...] = ()
STRUCT = ("real", "rw0", "rw1", "rw2", "pwsub")
PROP = STRUCT + ("empty",)
EDGE = PROP + ("shufattr",)

# model -> (kind, input blocks, conditions)
LADDER = {
    "deg_lr":          ("sklr",   ["degree"],          STRUCT),
    "rf_struct":       ("rf",     ["struct"],          STRUCT),
    "labelprop":       ("lp",     [],                  STRUCT),
    "mlp_feat":        ("mlp",    ["feat"],            NOGRAPH),
    "rf_feat":         ("rf",     ["feat"],            NOGRAPH),
    "mlp_feat_struct": ("mlp",    ["feat", "struct"],  STRUCT),
    "rf_feat_struct":  ("rf",     ["feat", "struct"],  STRUCT),
    "sign_feat":       ("sign",   ["feat"],            PROP),
    "cs_feat":         ("cs",     ["feat"],            PROP),
    "gcn_none":        ("gcn",    ["ones"],            STRUCT),
    "gcn_feat":        ("gcn",    ["feat"],            PROP),
    "gat_none":        ("gat",    ["ones"],            STRUCT),
    "sage_feat":       ("sage",   ["feat"],            PROP),
    "gat_feat":        ("gat",    ["feat"],            PROP),
    "gat_edge_feat":   ("gat_e",  ["feat"],            EDGE),
    "gp":              ("gp",     [],                  STRUCT),
}

# Every tunable knob, with the defaults used when no tuned value exists (see tune.py).
def uses_features(model: str) -> bool:
    kind, blocks, _ = LADDER[model]
    return "feat" in blocks or kind == "gp"


DEFAULTS = dict(hidden=64, layers=2, dropout=0.5, lr=0.01, wd=5e-4, epochs=300, patience=30,
                min_epochs=100, prior=0.1, heads=4,
                rf_trees=500, rf_leaf=5, rf_max_features="sqrt",
                lp_alpha=0.85, lp_layers=50, cs_alpha=0.8, cs_layers=50,
                gp_n_unl=2000, gp_noise_floor=0.25)


def inputs(p: Problem, blocks: list[str]) -> np.ndarray:
    parts = {"feat": lambda: p.feat, "struct": lambda: p.struct,
             "degree": lambda: p.struct[:, :1],  # z-scored log degree
             "ones": lambda: np.ones((len(p.genes), 1), dtype=np.float32)}
    return np.hstack([parts[b]() for b in blocks]).astype(np.float32)


LOSSES = ("nnpu", "pn", "pn_pathway", "pn_pubmed")


def _hard_negatives(p: Problem, pos: np.ndarray, unl: np.ndarray, kind: str, rng) -> np.ndarray:
    """Two-step-PU style negative selection, 1:1 with the positives, without replacement.
    pathway: unlabelled genes sharing a specific pathway with some positive (hard negatives
             from the same modules), topped up at random if there are too few.
    pubmed:  for each positive the unlabelled gene closest in log PubMed count, so negatives
             are as well studied as positives (removes the study-intensity shortcut)."""
    n = min(len(pos), len(unl))
    if kind == "pathway":
        m = p.pathways
        shares = np.asarray((m[unl] @ m[pos].T).sum(1)).ravel() > 0
        hard = rng.permutation(unl[shares])[:n]
        rest = rng.permutation(np.setdiff1d(unl, hard))[: n - len(hard)]
        return np.r_[hard, rest]
    lv = np.log1p(p.pubmed)
    free = np.ones(len(unl), dtype=bool)
    picks = []
    for q in rng.permutation(pos)[:n]:
        d = np.abs(lv[unl] - lv[q])
        d[~free] = np.inf
        j = int(np.argmin(d))
        free[j] = False
        picks.append(unl[j])
    return np.array(picks, dtype=int)


def training_set(p: Problem, loss: str, seed: int, fold: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Indices and 0/1 targets for fitting (or, with fold="val", for early stopping).
    nnpu: every gene in the fold (P vs U). pn: the fold's positives + an equal-sized random
    draw of its unlabelled genes as negatives (the MORGaN-style 1:1 setup). pn_pathway /
    pn_pubmed: the same 1:1 PN but with hard negatives (see _hard_negatives)."""
    tr = np.flatnonzero(p.mask(fold))
    pos, unl = tr[p.y[tr] == 1], tr[p.y[tr] == 0]
    rng = np.random.default_rng(seed)
    if loss == "pn":
        unl = rng.choice(unl, size=min(len(pos), len(unl)), replace=False)
    elif loss in ("pn_pathway", "pn_pubmed"):
        unl = _hard_negatives(p, pos, unl, loss.split("_")[1], rng)
    idx = np.r_[pos, unl]
    return idx, p.y[idx].astype(np.int64)


def _objective(logits, t, loss, prior):
    return losses.nnpu(logits, t.bool(), prior) if loss == "nnpu" else losses.pn(logits, t)  # pn*: BCE


def _fit_torch(net, forward, p: Problem, idx, t, loss, hp, seed) -> torch.Tensor:
    """Full-batch training. Early stopping and checkpoint selection use the training
    objective on the val fold: rank metrics (val AUPRC) are flat while a weight drifts
    through zero, which froze low-capacity models at sign-inverted solutions."""
    dev = device.get()
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    idx_t, t_t = torch.as_tensor(idx, device=dev), torch.as_tensor(t, device=dev)
    vi, vt = (torch.as_tensor(a, device=dev) for a in training_set(p, loss, seed, fold="val"))
    best, best_state, bad = np.inf, None, 0
    for epoch in range(hp["epochs"]):
        net.train()
        opt.zero_grad()
        _objective(forward()[idx_t], t_t, loss, hp["prior"]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = _objective(forward()[vi], vt, loss, hp["prior"]).item()
        if v < best:
            best, best_state, bad = v, _copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"] and epoch >= hp["min_epochs"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return forward()


def score(p: Problem, model: str, loss: str, seed: int, **over) -> tuple[np.ndarray, dict]:
    """Scores for every gene, plus model-specific extras (e.g. GP attribution)."""
    hp = {**DEFAULTS, **over}
    kind, blocks, _ = LADDER[model]
    dev = device.get()
    idx, t = training_set(p, loss, seed)
    train_mask = torch.zeros(len(p.genes), dtype=torch.bool, device=dev)
    train_mask[torch.as_tensor(idx, device=dev)] = True
    y_t = torch.as_tensor(p.y.astype(np.int64), device=dev)  # copy: parquet arrays are read-only
    ei = p.edge_index.to(dev) if p.edge_index is not None else None
    ea = p.edge_attr.to(dev) if p.edge_attr is not None else None

    if kind == "lp":
        return models.label_propagation(y_t, ei, train_mask, hp["lp_layers"], hp["lp_alpha"]).cpu().numpy(), {}

    if kind == "sklr":
        # Degree-only baseline: convex, deterministic logistic regression fit like the RF
        # (P vs U, balanced). A one-weight torch model under nnPU converged to the inverted
        # ranking on some seeds; with one feature only the sign matters, so nothing to tune.
        x = inputs(p, blocks)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(x[idx], t)
        return lr.decision_function(x), {}

    if kind == "rf":
        x = inputs(p, blocks)
        rf = RandomForestClassifier(n_estimators=hp["rf_trees"], min_samples_leaf=hp["rf_leaf"],
                                    max_features=hp["rf_max_features"], class_weight="balanced_subsample",
                                    n_jobs=1, random_state=seed)
        rf.fit(x[idx], t)
        return rf.predict_proba(x)[:, 1], {}

    if kind == "gp":
        # P-vs-U regression on all train positives + a seeded sample of train unlabelled
        rng = np.random.default_rng(seed)
        pos, unl = idx[t == 1], idx[t == 0]
        sub = np.sort(np.r_[pos, rng.choice(unl, size=min(hp["gp_n_unl"], len(unl)), replace=False)])
        x, cols = gp_inputs(p)
        mean, parts, sd, hyper = gp.fit_predict(x, p.y.astype(float), sub, cols, hp["gp_noise_floor"],
                                                seed=seed, device=dev)
        return mean, {"gp_parts": parts, "gp_sd": sd, "gp_hyper": hyper}

    torch.manual_seed(seed)  # before construction, so initial weights are reproducible
    x = torch.as_tensor(inputs(p, blocks), device=dev)
    if kind in ("mlp", "cs"):
        net = models.mlp(x.size(1), hp["hidden"], hp["layers"], hp["dropout"])
        forward = lambda: net(x).squeeze(-1)  # noqa: E731
    elif kind == "sign":
        xs = models.sign_features(x, ei)
        net = models.mlp(xs.size(1), hp["hidden"], hp["layers"], hp["dropout"])
        forward = lambda: net(xs).squeeze(-1)  # noqa: E731
    elif kind == "gcn":
        net = models.gcn(x.size(1), hp["hidden"], hp["layers"], hp["dropout"])
        forward = lambda: net(x, ei).squeeze(-1)  # noqa: E731
    elif kind == "sage":
        net = models.sage(x.size(1), hp["hidden"], hp["layers"], hp["dropout"])
        forward = lambda: net(x, ei).squeeze(-1)  # noqa: E731
    elif kind == "gat":
        net = models.gat(x.size(1), hp["hidden"], hp["layers"], hp["dropout"], heads=hp["heads"])
        forward = lambda: net(x, ei).squeeze(-1)  # noqa: E731
    elif kind == "gat_e":
        net = models.gat(x.size(1), hp["hidden"], hp["layers"], hp["dropout"], heads=hp["heads"],
                         edge_dim=ea.size(1))
        forward = lambda: net(x, ei, edge_attr=ea).squeeze(-1)  # noqa: E731
    else:
        raise ValueError(kind)
    net.to(dev)
    s = _fit_torch(net, forward, p, idx, t, loss, hp, seed)

    if kind == "cs":
        s = models.correct_and_smooth(s, y_t, ei, train_mask, hp["cs_layers"], hp["cs_alpha"])
    return s.cpu().numpy(), {}
