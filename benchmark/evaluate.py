"""Score any gene ranking on the time-split benchmark (see README.md). Needs numpy, pandas, scipy.

  python evaluate.py --release . --universe string --scores my_scores.tsv [--out report.tsv]

my_scores.tsv: columns `ensg` and `score` (higher = more target-like), and optionally `seed` (0-4)
when the model was trained once per seed on that seed's training genes. Without `seed`, the same
scores are evaluated on every seed's test set. Train only on genes whose fold is train/val in
splits/seed<k>.tsv, with y_minikel or y_genetic (or anything dated before the cut-off).

Reported, each with a 95% bootstrap CI over the future targets (2000 draws) and paired against
a ranker by PubMed count before the cut-off:
  auroc_<stratum>   future targets of the stratum vs the test genes that were never drugged
  recall_<stratum>  share of the stratum's future targets in the top 10% of the test ranking
Strata: all, novel (2 of 3) and its three criteria (future_targets.tsv). Then a top-100 audit of
the test ranking and, among test genes, selection (tried vs never) and outcome (succeeded vs
failed) AUROC.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

TOP_FRAC, TOP_K, BOOT = 0.10, 100, 2000


def auc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def wins(s_test: np.ndarray, fut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per future target: share of test negatives ranked below it (ties 1/2), and top-10% hit."""
    neg = np.sort(s_test[~fut])
    v = s_test[fut]
    lo, hi = np.searchsorted(neg, v, "left"), np.searchsorted(neg, v, "right")
    return (lo + 0.5 * (hi - lo)) / len(neg), v >= np.quantile(s_test, 1 - TOP_FRAC)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=".")
    ap.add_argument("--universe", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    u = Path(a.release) / a.universe
    genes = pd.read_csv(u / "genes.tsv", sep="\t")
    G = genes.ensg.values
    ft = pd.read_csv(u / "future_targets.tsv", sep="\t").set_index("ensg")
    fut_genes = ft.index.values
    sc = pd.read_csv(a.scores, sep="\t")
    pm = genes.pubmed_before_cutoff.values.astype(float)
    seeds = sorted(int(p.stem[4:]) for p in (u / "splits").glob("seed*.tsv"))
    win, hit, ref_win, ref_hit, tops, sel, out = [], [], [], [], [], [], []
    status = genes.programme_status.values
    for seed in seeds:
        sp = pd.read_csv(u / "splits" / f"seed{seed}.tsv", sep="\t")
        assert (sp.ensg.values == G).all()
        s = sc[sc.seed == seed] if "seed" in sc else sc
        s = s.set_index("ensg").score.reindex(G).values.astype(float)
        test = np.flatnonzero(sp.fold.values == "test")
        if np.isnan(s[test]).any():
            raise SystemExit(f"seed {seed}: {np.isnan(s[test]).sum()} test genes have no score")
        fut = sp.y_minikel.values[test] == 1
        assert (G[test][fut] == fut_genes).all(), "future targets differ from future_targets.tsv"
        for v, w_, h_ in ((s, win, hit), (pm, ref_win, ref_hit)):
            w, h = wins(v[test], fut)
            w_.append(w)
            h_.append(h)
        top = test[np.argsort(-s[test], kind="stable")[:TOP_K]]
        tops.append(dict(future_target=np.isin(G[top], fut_genes).mean(),
                         novel2_future_target=np.isin(G[top], ft.index[ft["novel (2 of 3)"]]).mean(),
                         top_study_decile=(genes.pubmed_pct_before_cutoff.values[top] >= 0.9).mean(),
                         in_programme=np.isin(genes.minikel_best_phase.values[top], [1, 2, 3]).mean(),
                         drugged_family=(genes.known_family_count.values[top] > 0).mean(),
                         one_hop_from_known=(genes.hops_to_known.values[top] <= 1).mean(),
                         genetic_before_cutoff=genes.genetic_before_cutoff.values[top].mean()))
        st = status[test]
        tried, never = (st != "never"), st == "never"
        succ, fail = fut, st == "failure"
        sel.append(dict(selection=auc(tried[tried | never].astype(int), s[test][tried | never]),
                        outcome=auc(succ[succ | fail].astype(int), s[test][succ | fail])))
    win, hit, ref_win, ref_hit = (np.mean(x, 0) for x in (win, hit, ref_win, ref_hit))
    masks = {"all": np.ones(len(ft), bool), "novel (2 of 3)": ft["novel (2 of 3)"].values, "novel": ft["novel"].values,
             "new family": ft["new family"].values}
    rng = np.random.default_rng(0)
    for name, m in masks.items():
        idx = np.flatnonzero(m)
        b = rng.integers(0, len(idx), (BOOT, len(idx)))
        for metric, v, r in (("auroc", win, ref_win), ("recall_top10pct", hit.astype(float), ref_hit.astype(float))):
            x, xr = v[idx], r[idx]
            d = x[b].mean(1) - xr[b].mean(1)
            out.append(dict(section="novelty", stratum=name, n=len(idx), metric=metric, value=x.mean(),
                            lo=np.quantile(x[b].mean(1), .025), hi=np.quantile(x[b].mean(1), .975),
                            vs_pubmed=x.mean() - xr.mean(), vs_pubmed_lo=np.quantile(d, .025),
                            vs_pubmed_hi=np.quantile(d, .975)))
    for k, v in pd.DataFrame(tops).mean().items():
        out.append(dict(section="top100", metric=k, value=v))
    for k, v in pd.DataFrame(sel).mean().items():
        out.append(dict(section="tried", metric=k, value=v))
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(res.round(3).fillna("").to_string(index=False))
    if a.out:
        res.to_csv(a.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
