"""Per-graph descriptive analyses (no model training beyond small logistic regressions).

  uv run python scripts/analysis.py     ->  results/analysis/*.csv

graph_sizes.csv   nodes / edges / degree / isolated nodes per graph and condition (real,
                  rewired, pathway subgraph), plus positives covered
hub_bias.csv      do Minikel positives sit on hubs? degree of positives vs unlabelled,
                  degree-only AUROC, share of positives in the top-10% degree, and whether
                  degree survives controlling for study intensity (PubMed count): degree AUROC
                  within PubMed quintiles, and PubMed-only vs PubMed+degree logistic regression
membrane.csv      fraction of membrane proteins among positives vs unlabelled, per graph

Uses $GHELPS_DATA, so on the cluster it describes exactly the data the runs used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghelps import data, labels, paths  # noqa: E402

OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def oof_lr(x, y):
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        s[te] = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return s


def main():
    pos_all = labels.minikel(CFG["labels"]["minikel_min_phase"])
    membrane = labels.membrane()
    sizes, hubs, mem = [], [], []
    for universe, members in CFG["universes"].items():
        genes = data.genes_of(universe)
        y = np.array([g in pos_all for g in genes], dtype=int)
        is_mem = np.array([g in membrane for g in genes])
        pm = data.pubmed_counts(universe)
        lpm = np.log1p(pm)
        quint = pd.qcut(pd.Series(lpm).rank(method="first"), 5, labels=False).values
        for graph in members:
            for copy in ("real", "rw0", "pwsub"):
                ei = np.load(paths.PROCESSED / "graphs" / universe / f"{graph}.{copy}.npy")
                deg = np.bincount(ei.ravel(), minlength=len(genes))
                sizes.append(dict(graph=graph, condition=copy, nodes=len(genes), edges=len(ei),
                                  mean_degree=deg.mean(), median_degree=np.median(deg), max_degree=int(deg.max()),
                                  isolated=int((deg == 0).sum()), positives=int(y.sum()),
                                  positives_covered=y.sum() / len(pos_all)))
            d = np.bincount(np.load(paths.PROCESSED / "graphs" / universe / f"{graph}.real.npy").ravel(),
                            minlength=len(genes))
            within = [auc(y[quint == k], d[quint == k]) for k in range(5)]
            hubs.append(dict(
                graph=graph, genes=len(genes), positives=int(y.sum()),
                median_deg_pos=np.median(d[y == 1]), median_deg_unl=np.median(d[y == 0]),
                deg_auroc=auc(y, d), mwu_p=mannwhitneyu(d[y == 1], d[y == 0]).pvalue,
                pos_in_top10pct_deg=y[d >= np.quantile(d, 0.9)].sum() / y.sum(),
                pubmed_auroc=auc(y, pm), rho_deg_pubmed=spearmanr(d, pm).statistic,
                deg_auroc_within_pubmed_quintiles=np.nanmean(within),
                deg_auroc_by_pubmed_quintile=" ".join(f"{w:.2f}" for w in within),
                lr_pubmed_auroc=auc(y, oof_lr(lpm[:, None], y)),
                lr_pubmed_plus_degree_auroc=auc(y, oof_lr(np.c_[lpm, np.log1p(d)], y))))
            mem.append(dict(graph=graph, membrane_frac_positives=is_mem[y == 1].mean(),
                            membrane_frac_unlabelled=is_mem[y == 0].mean(),
                            membrane_flag_auroc=auc(y, is_mem.astype(float)),
                            membrane_prevalence=is_mem.mean()))
    for name, rows in (("graph_sizes", sizes), ("hub_bias", hubs), ("membrane", mem)):
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"== {name}\n{t.round(3).to_string(index=False)}\n", flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()
