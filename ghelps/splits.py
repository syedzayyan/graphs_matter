"""Frozen train/val/test splits and evaluation negative sets.

A split is a DataFrame aligned to the universe gene order with columns
  fold: "train" | "val" | "test" | "excluded"
  y:    1 = positive, 0 = unlabelled
For splits that hold out a *different* positive set at test time (rcnt, pharos), `y` is
1 for those test positives even though they are not training positives.
"""
from __future__ import annotations

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp

TEST_FRAC, VAL_FRAC = 0.2, 0.1


# ---------------------------------------------------------------- helpers

def _random_folds(idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(idx)
    n_te, n_va = round(TEST_FRAC * len(idx)), round(VAL_FRAC * len(idx))
    out = np.empty(len(idx), dtype=object)
    out[:n_te], out[n_te:n_te + n_va], out[n_te + n_va:] = "test", "val", "train"
    return idx, out


def _frame(genes: list[str], y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"gene": genes, "fold": "train", "y": y.astype(np.int8)})


def _stratified(df: pd.DataFrame, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Random 70/10/20 split of the rows in `mask`, stratified by y. In place."""
    for cls in (0, 1):
        idx, folds = _random_folds(np.flatnonzero(mask & (df.y.values == cls)), rng)
        df.loc[idx, "fold"] = folds


def _grouped(df: pd.DataFrame, groups: np.ndarray, rng: np.random.Generator) -> None:
    """Assign whole groups to folds until test / val hold ~20% / ~10% of the positives
    (and, failing that, of all genes). In place."""
    y = df.y.values.astype(np.int64)
    order = rng.permutation(np.unique(groups))
    gpos = pd.Series(y).groupby(groups).sum().reindex(order).values
    gsize = pd.Series(1, index=range(len(y))).groupby(groups).sum().reindex(order).values
    P, N = y.sum(), len(y)
    fold_of = {}
    acc = {"test": [0, 0], "val": [0, 0]}
    target = {"test": TEST_FRAC, "val": VAL_FRAC}
    for g, p, s in zip(order, gpos, gsize):
        for f in ("test", "val"):
            if acc[f][0] < target[f] * P or (P == 0 and acc[f][1] < target[f] * N):
                # don't let one big group push the fold far past its budget, in genes
                # or in positives; skipped groups fall through to the next fold / train
                if acc[f][1] + s <= 1.5 * target[f] * N and acc[f][0] + p <= 1.25 * target[f] * P:
                    fold_of[g] = f
                    acc[f][0] += p
                    acc[f][1] += s
                    break
        else:
            fold_of[g] = "train"
    df["fold"] = pd.Series(groups).map(fold_of).values


# ---------------------------------------------------------------- splits

def random_split(genes, y, seed):
    rng = np.random.default_rng(seed)
    df = _frame(genes, y)
    _stratified(df, np.ones(len(df), bool), rng)
    return df


def pfam_groups(genes, pfam: pd.DataFrame, singletons: bool = True) -> np.ndarray:
    """Each gene's largest Pfam family (multi-domain proteins would otherwise chain most genes
    into one component). Genes without Pfam: their own singleton groups, or one shared group."""
    m = pfam.loc[genes].values
    has = m.sum(1) > 0
    primary = np.where(has, np.argmax(m * m.sum(0)[None, :], axis=1), -1)
    return np.where(has, primary, -(np.arange(len(genes)) + 1) if singletons else -1)


def permute_within(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Labels shuffled inside each group: group composition of positives is preserved, which
    gene within the group is positive is not."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[rng.permutation(idx)]
    return out


def pfam_split(genes, y, pfam: pd.DataFrame, seed):
    """Group genes by their largest Pfam family; genes without Pfam are singletons."""
    rng = np.random.default_rng(seed)
    df = _frame(genes, y)
    _grouped(df, pfam_groups(genes, pfam), rng)
    return df


def community_split(genes, y, communities: np.ndarray, seed):
    rng = np.random.default_rng(seed)
    df = _frame(genes, y)
    _grouped(df, communities, rng)
    return df


def held_out_positive_split(genes, y, test_pos: np.ndarray, seed, exclude_unl: np.ndarray | None = None):
    """Training positives = y minus test_pos, split train/val in the same 70:10 ratio as
    the random split. Test positives = test_pos. Unlabelled genes split 70/10/20; genes in
    exclude_unl are kept out of every fold except as test positives."""
    rng = np.random.default_rng(seed)
    y2 = (y.astype(bool) | test_pos).astype(np.int8)
    df = _frame(genes, y2)
    df.loc[test_pos, "fold"] = "test"
    trp = rng.permutation(np.flatnonzero(y.astype(bool) & ~test_pos))
    df.loc[trp[: round(len(trp) * VAL_FRAC / (1 - TEST_FRAC))], "fold"] = "val"
    unl = (y2 == 0)
    if exclude_unl is not None:
        df.loc[unl & exclude_unl, "fold"] = "excluded"
        unl &= ~exclude_unl
    idx, folds = _random_folds(np.flatnonzero(unl), rng)
    df.loc[idx, "fold"] = folds
    return df


# ---------------------------------------------------------------- communities

def leiden(a: sp.csr_matrix, resolution: float, seed: int) -> np.ndarray:
    coo = sp.triu(a, k=1).tocoo()
    g = ig.Graph(n=a.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=False)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed)
    return np.array(part.membership)


# ---------------------------------------------------------------- negatives

def _matched(pos: np.ndarray, cand: np.ndarray, sim, k: int, rng) -> list[tuple[int, int]]:
    """For each positive, draw up to k candidates for which sim(p) returns a pool;
    sampling is without replacement across positives so negatives are not reused."""
    used: set[int] = set()
    pairs = []
    cand_set = set(cand.tolist())
    for p in rng.permutation(pos):
        pool = [c for c in sim(p) if c in cand_set and c not in used]
        if not pool:
            continue
        pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
        used.update(pick.tolist())
        pairs += [(p, int(c)) for c in pick]
    return pairs


def negatives(split: pd.DataFrame, a_union: sp.csr_matrix, pfam: np.ndarray, pathway: np.ndarray,
              k: int, seed: int, exclude: np.ndarray | None = None,
              pubmed: np.ndarray | None = None) -> pd.DataFrame:
    """Long table (regime, pos, neg) of node indices for every test-time negative regime.
    `random` lists every unlabelled test gene against all test positives (pos = -1)."""
    rng = np.random.default_rng(seed)
    te = split.fold.values == "test"
    pos = np.flatnonzero(te & (split.y.values == 1))
    unl = te & (split.y.values == 0)
    if exclude is not None:
        unl &= ~exclude
    cand = np.flatnonzero(unl)

    deg = np.asarray(a_union.sum(1)).ravel()
    a2 = (a_union @ a_union).tocsr()
    pf = sp.csr_matrix(pfam)
    pw = sp.csr_matrix(pathway)
    pf_share = (pf @ pf.T).tocsr()
    pw_share = (pw @ pw.T).tocsr()

    def nearest(value: np.ndarray):
        """Candidates ordered by |log1p(value) difference| to the positive (study-intensity or
        degree matching)."""
        lv = np.log1p(value)
        order = cand[np.argsort(lv[cand])]

        def fn(p, width=50):
            i = np.searchsorted(lv[order], lv[p])
            window = order[max(0, i - width): i + width]
            return window[np.argsort(np.abs(lv[window] - lv[p]))]
        return fn

    nearest_degree = nearest(deg)

    def row(m, p):
        return m.indices[m.indptr[p]:m.indptr[p + 1]]

    hop1 = lambda p: row(a_union, p)
    hop2 = lambda p: np.setdiff1d(row(a2, p), np.r_[row(a_union, p), p])

    regimes = {
        "degree": lambda p: nearest_degree(p)[: 4 * k],  # nearest 4k, then sample k
        "pfam": lambda p: row(pf_share, p),
        "pathway": lambda p: row(pw_share, p),
        "hop1": hop1,
        "hop2": hop2,
    }
    if pubmed is not None:
        # study-intensity control: negatives as well studied as the positive
        nearest_pubmed = nearest(pubmed)
        regimes["pubmed"] = lambda p: nearest_pubmed(p)[: 4 * k]
    rows = [("random", -1, int(c)) for c in cand] + [("random", int(p), -1) for p in pos]
    for name, sim in regimes.items():
        rows += [(name, p, n) for p, n in _matched(pos, cand, sim, k, rng)]
    return pd.DataFrame(rows, columns=["regime", "pos", "neg"])
