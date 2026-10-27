"""Array-level pieces of the benchmark: novelty context, strata, per-target wins, scoring.

Everything here works on plain numpy arrays over one gene universe, so it applies to any
gene list, any dated label and any ranking (release.py and build.py feed it files).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata

TOP_FRAC, TOP_K, BOOT = 0.10, 100, 2000
# strata the scorer reports by default (any boolean column of the strata table can be scored)
PRIMARY_STRATA = ("all", "novel (2 of 3)", "novel", "new family")


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC with ties counted 1/2; nan when a class is empty."""
    y = np.asarray(y).astype(bool)
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def novelty_strata(study_pct: np.ndarray, drugged_family: np.ndarray | None = None,
                   hops: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Future targets by how novel they were at the cut-off: study then (percentile), whether
    their domain family already had a target, hops to the nearest known target. "novel" needs
    every criterion (not top-decile study, new family, >= 2 hops); "novel (2 of 3)" any two.
    A criterion passed as None (no domains, no graph) is left out: its strata are not made and
    "novel (2 of 3)" then needs both remaining criteria."""
    s = {"all": np.ones(len(study_pct), bool),
         "study<0.5": study_pct < 0.5, "study 0.5-0.9": (study_pct >= 0.5) & (study_pct < 0.9),
         "study>=0.9": study_pct >= 0.9}
    crit = [study_pct < 0.9]
    if drugged_family is not None:
        s |= {"drugged family": drugged_family, "new family": ~drugged_family}
        crit.append(~drugged_family)
    if hops is not None:
        s |= {"1 hop": hops <= 1, "2 hops": hops == 2, "3+ hops": hops >= 3}
        crit.append(hops >= 2)
    crit = np.c_[tuple(crit)]
    return s | {"novel": crit.all(1), "novel (2 of 3)": crit.sum(1) >= min(2, crit.shape[1])}


def adjacency(edges: np.ndarray, n: int) -> csr_matrix:
    """Undirected adjacency from an (m, 2) array of gene positions."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()


def context(known: np.ndarray, pubmed_before: np.ndarray, domains: np.ndarray | csr_matrix | None = None,
            edges: np.ndarray | None = None) -> pd.DataFrame:
    """Per gene, relative to the targets known at the cut-off (`known`, boolean):
      study_pct     percentile of PubMed count before the cut-off among all genes
      family_count  known targets sharing a domain (gene x domain 0/1 matrix); nan without domains
      hops          hops to the nearest known target in the graph (99 when unreachable); nan
                    without a graph
    """
    n = len(known)
    study_pct = rankdata(pubmed_before) / n
    if domains is None:
        fam = np.full(n, np.nan)
    else:
        d = csr_matrix(domains, dtype=np.int32)
        known_dom = np.asarray(d[known].sum(0)).ravel() > 0
        fam = np.asarray(d[:, known_dom].sum(1)).ravel().astype(float)
    if edges is None:
        hops = np.full(n, np.nan)
    elif not len(edges) or not known.any():
        hops = np.full(n, 99.0)
    else:
        hops = dijkstra(adjacency(edges, n), directed=False, unweighted=True, indices=np.flatnonzero(known),
                        min_only=True)
    return pd.DataFrame({"study_pct": study_pct, "family_count": fam, "hops": np.minimum(hops, 99)})


def strata_table(ctx: pd.DataFrame, future: np.ndarray) -> pd.DataFrame:
    """Strata of the future targets (rows in the order of np.flatnonzero(future))."""
    c = ctx.iloc[np.flatnonzero(future)]
    fam = None if ctx.family_count.isna().all() else c.family_count.values > 0
    hops = None if ctx.hops.isna().all() else c.hops.values
    return pd.DataFrame(novelty_strata(c.study_pct.values, fam, hops))


def wins(s_test: np.ndarray, fut: np.ndarray, top_frac: float = TOP_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Per future target: share of test negatives ranked below it (ties 1/2; a stratum's AUROC is
    the mean of these) and whether it is in the top `top_frac` of the test ranking."""
    neg = np.sort(s_test[~fut])
    v = s_test[fut]
    lo, hi = np.searchsorted(neg, v, "left"), np.searchsorted(neg, v, "right")
    return (lo + 0.5 * (hi - lo)) / len(neg), v >= np.quantile(s_test, 1 - top_frac)


def score(rankings: dict[str, dict[int, np.ndarray]], tests: dict[int, np.ndarray], future: np.ndarray,
          strata: pd.DataFrame, reference: str = "pubmed_before_cutoff", boot: int = BOOT,
          top_frac: float = TOP_FRAC, seed: int = 0) -> pd.DataFrame:
    """AUROC and recall in the top `top_frac` per ranking and stratum, averaged over seeds, with
    95% bootstrap CIs resampling the stratum's future targets (draws shared across rankings, so
    differences are paired) and the paired difference from the `reference` ranking.

    rankings  name -> seed -> score per gene (whole universe)
    tests     seed -> positions of that seed's test genes
    future    boolean over genes: the future targets (test positives; the same in every seed)
    strata    boolean columns over the future targets, in np.flatnonzero(future) order
    """
    fut_idx = np.flatnonzero(future)
    per = {}
    for name, by_seed in rankings.items():
        w, h = [], []
        for sd, test in tests.items():
            s = np.asarray(by_seed[sd] if sd in by_seed else by_seed[None], dtype=float)
            if np.isnan(s[test]).any():
                raise ValueError(f"{name}, seed {sd}: {int(np.isnan(s[test]).sum())} test genes have no score")
            f = future[test]
            assert (test[f] == fut_idx).all(), "future targets must be test genes in every seed"
            a, b = wins(s[test], f, top_frac)
            w.append(a)
            h.append(b)
        per[name] = (np.mean(w, 0), np.mean(h, 0).astype(float))
    rng = np.random.default_rng(seed)
    rows = []
    for st in strata.columns:
        idx = np.flatnonzero(strata[st].values)
        if not len(idx):
            continue
        b = rng.integers(0, len(idx), (boot, len(idx)))
        for name, (w, h) in per.items():
            for metric, v in (("auroc", w), ("recall_top", h)):
                x = v[idx]
                bx = x[b].mean(1)
                row = dict(ranking=name, stratum=st, n=len(idx), metric=metric, value=x.mean(),
                           lo=np.quantile(bx, .025), hi=np.quantile(bx, .975))
                if reference in per and name != reference:
                    r = (per[reference][0] if metric == "auroc" else per[reference][1])[idx]
                    d = bx - r[b].mean(1)
                    row |= dict(vs_reference=x.mean() - r.mean(), vs_reference_lo=np.quantile(d, .025),
                                vs_reference_hi=np.quantile(d, .975))
                rows.append(row)
    return pd.DataFrame(rows)


def top_audit(s: np.ndarray, test: np.ndarray, flags: dict[str, np.ndarray], k: int = TOP_K) -> dict[str, float]:
    """Share of the top-k test genes with each boolean gene flag."""
    top = test[np.argsort(-s[test], kind="stable")[:k]]
    return {name: float(f[top].mean()) for name, f in flags.items()}


def tried_aurocs(s: np.ndarray, test: np.ndarray, status: np.ndarray) -> dict[str, float]:
    """Among test genes: selection (tried vs never tried) and outcome (success vs resolved
    failure) AUROC. status: never / success / failure / censored."""
    st, v = status[test], s[test]
    tried, never, succ, fail = st != "never", st == "never", st == "success", st == "failure"
    return dict(selection=auc(tried[tried | never], v[tried | never]),
                outcome=auc(succ[succ | fail], v[succ | fail]))
