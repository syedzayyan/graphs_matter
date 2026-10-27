"""Feature baselines for a release: fit on each seed's training folds, score every gene.

    targetsplit baseline --release R --universe U --features X.tsv --out scores.tsv
                         [--label minikel] [--model pubag|rf]

X.tsv: column gene (or ensg) and one numeric column per feature. Use features dated before the
cut-off, or say that they leak (README, "Rules").

Models:
  rf     random forest, positives vs unlabelled (class-balanced): the paper's RF.
  pubag  bagging PU (Mordelet & Vert 2013; pulearn.BaggingPuClassifier) over decision trees.
         Each tree sees all positives and a bootstrap of as many unlabelled genes, so the
         unlabelled set is never treated as one negative class. Needs `pip install targetsplit[pu]`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from targetsplit.release import Release


def make(model: str, seed: int, n_pos: int, trees: int = 500, leaf: int = 5):
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(trees, min_samples_leaf=leaf, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if model == "pubag":
        try:
            from pulearn import BaggingPuClassifier
        except ImportError as e:
            raise SystemExit("model pubag needs pulearn: pip install 'targetsplit[pu]'") from e
        from sklearn.tree import DecisionTreeClassifier
        return BaggingPuClassifier(DecisionTreeClassifier(min_samples_leaf=leaf, max_features="sqrt"),
                                   n_estimators=trees, max_samples=n_pos, n_jobs=-1, random_state=seed)
    raise ValueError(model)


def fit_score(rel: Release, x: np.ndarray, label: str | None = None, model: str = "pubag",
              seeds: list[int] | None = None, **kw) -> pd.DataFrame:
    """Scores per seed (columns gene, seed, score), each model fit on that seed's train/val genes."""
    out = []
    for s in seeds or rel.seeds:
        idx, y = rel.train(s, label)
        m = make(model, s, int(y.sum()), **kw).fit(x[idx], y)
        out.append(pd.DataFrame({"gene": rel.G, "seed": s, "score": m.predict_proba(x)[:, 1]}))
    return pd.concat(out, ignore_index=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="targetsplit baseline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=".")
    ap.add_argument("--universe", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--label")
    ap.add_argument("--model", default="pubag", choices=["pubag", "rf"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rel = Release(a.release, a.universe)
    f = pd.read_csv(a.features, sep="\t").rename(columns={"ensg": "gene"})
    f = f.assign(gene=f.gene.astype(str)).set_index("gene").reindex(rel.G)
    missing = f.isna().all(1).sum()
    if missing:
        print(f"{missing} genes have no features: filled with column means")
    x = f.fillna(f.mean()).values.astype(np.float32)
    fit_score(rel, x, a.label, a.model).to_csv(a.out, sep="\t", index=False)
    print(f"wrote {a.out}")
