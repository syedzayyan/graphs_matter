"""Per-gene structural statistics for one graph."""
from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp

K_PROP = 3


def adjacency(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    u, v = edge_index[:, 0], edge_index[:, 1]
    a = sp.coo_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    return a.tocsr()


def gcn_norm(a: sp.csr_matrix) -> sp.csr_matrix:
    """Â = D^-1/2 (A + I) D^-1/2, the propagation operator a GCN layer uses."""
    a = a + sp.eye(a.shape[0], format="csr")
    d = np.asarray(a.sum(1)).ravel() ** -0.5
    return sp.diags(d) @ a @ sp.diags(d)


def stats(edge_index: np.ndarray, n: int) -> pd.DataFrame:
    a = adjacency(edge_index, n)
    deg = np.asarray(a.sum(1)).ravel()
    out = {"degree": deg, "log_degree": np.log1p(deg), "sqrt_degree": np.sqrt(deg)}

    ah, x = gcn_norm(a), np.ones(n)
    for k in range(1, K_PROP + 1):
        x = ah @ x
        out[f"ahat{k}_1"] = x

    # neighbour-degree summaries (NaN-free: isolated nodes get 0)
    nd = a.multiply(deg[None, :]).tocsr()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(deg > 0, np.asarray(nd.sum(1)).ravel() / deg, 0.0)
        sq = np.asarray(nd.multiply(nd).sum(1)).ravel()
        out["nbr_deg_mean"] = mean
        out["nbr_deg_std"] = np.sqrt(np.maximum(np.where(deg > 0, sq / deg, 0.0) - mean**2, 0))
    out["nbr_deg_max"] = np.asarray(nd.max(1).todense()).ravel()
    out["nbr_deg_min"] = np.array([nd.data[nd.indptr[i]:nd.indptr[i + 1]].min() if deg[i] else 0
                                   for i in range(n)])
    out["nbr_deg_median"] = np.array([np.median(nd.data[nd.indptr[i]:nd.indptr[i + 1]]) if deg[i] else 0
                                      for i in range(n)])

    g = ig.Graph(n=n, edges=edge_index.tolist(), directed=False)
    out["kcore"] = np.array(g.coreness(), dtype=float)
    out["pagerank"] = np.array(g.pagerank(directed=False))
    out["clustering"] = np.nan_to_num(np.array(g.transitivity_local_undirected(mode="zero")))
    out["betweenness"] = np.array(g.betweenness(directed=False)) / max((n - 1) * (n - 2) / 2, 1)
    return pd.DataFrame(out)
