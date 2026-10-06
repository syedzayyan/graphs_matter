"""Load one (universe, graph, condition, label set, split, seed) problem.

Graph conditions ("copy"):
  real       the graph
  rw0..rw2   degree-preserving rewired copies (edge vectors permuted onto them)
  empty      no edges: same model with the graph removed
  shufattr   real topology, edge vectors permuted across edges
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import to_undirected

from ghelps import paths, structure

ROOT = Path(__file__).resolve().parents[1]
P = paths.PROCESSED
FEATURE_BLOCKS = ("go", "pfam", "pathway", "tract")
SVD_DIM = 256
GP_BLOCK_DIM = 32
GP_TOPO_DIM = 16

STRUCT_COLS = ["log_degree", "ahat1_1", "ahat2_1", "ahat3_1", "nbr_deg_mean", "nbr_deg_std",
               "nbr_deg_max", "nbr_deg_min", "nbr_deg_median", "kcore", "pagerank",
               "clustering", "betweenness"]
LOG_COLS = {"nbr_deg_mean", "nbr_deg_std", "nbr_deg_max", "nbr_deg_min", "nbr_deg_median",
            "kcore", "pagerank", "betweenness"}


@dataclass
class Problem:
    universe: str
    graph: str | None
    copy: str
    genes: list[str]
    y: np.ndarray                  # 1 positive / 0 unlabelled (split-specific)
    fold: np.ndarray               # train / val / test / excluded
    edge_index: torch.Tensor | None  # (2, 2m) undirected
    edge_attr: torch.Tensor | None   # (2m, 12)
    struct: np.ndarray | None
    raw_degree: np.ndarray | None
    feat: np.ndarray
    negatives: pd.DataFrame

    def mask(self, f: str) -> np.ndarray:
        return self.fold == f


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


@lru_cache(maxsize=4)
def genes_of(universe: str) -> list[str]:
    return (P / "graphs" / universe / "genes.txt").read_text().split()


def _svd(universe: str, blocks, dim: int, tag: str) -> np.ndarray:
    """Label-free TruncatedSVD of the concatenated binary blocks, z-scored, cached."""
    cache = P / "features" / universe / f"svd{dim}_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    genes = genes_of(universe)
    x = sp.hstack([sp.csr_matrix(pd.read_parquet(P / "features" / universe / f"{b}.parquet").loc[genes].values,
                                 dtype=np.float32) for b in blocks]).tocsr()
    z = _z(TruncatedSVD(min(dim, x.shape[1] - 1), random_state=0).fit_transform(x)).astype(np.float32)
    np.save(cache, z)
    return z


@lru_cache(maxsize=4)
def features(universe: str) -> np.ndarray:
    return _svd(universe, FEATURE_BLOCKS, SVD_DIM, "all")


def _edges(universe: str, graph: str, copy: str) -> tuple[np.ndarray, np.ndarray]:
    topo = "real" if copy == "shufattr" else copy
    ei = np.load(P / "graphs" / universe / f"{graph}.{topo}.npy")
    attr_path = P / "edge_attr" / universe / f"{graph}.{copy}.npy"
    attr = np.load(attr_path) if attr_path.exists() else np.zeros((len(ei), 12), np.float32)
    return ei, attr


@lru_cache(maxsize=64)
def _struct(universe: str, graph: str, copy: str) -> tuple[np.ndarray, np.ndarray]:
    path = P / "structure" / universe / f"{graph}.{'real' if copy == 'shufattr' else copy}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:  # e.g. the empty graph: cheap to compute on the fly
        df = structure.stats(_edges(universe, graph, copy)[0], len(genes_of(universe)))
    x = df[STRUCT_COLS].copy()
    for c in LOG_COLS:
        x[c] = np.log1p(x[c] * (1e4 if c in {"pagerank", "betweenness"} else 1))
    return _z(x.values).astype(np.float32), df.degree.values


def load(universe: str, graph: str | None, copy: str, labels: str, split: str, seed: int) -> Problem:
    genes = genes_of(universe)
    sdir = P / "splits" / universe / labels / split
    s = pd.read_parquet(sdir / f"seed{seed}.parquet")
    neg = pd.read_parquet(sdir / f"seed{seed}.neg.parquet")
    ei = ea = st = deg = None
    if graph is not None:
        e, a = _edges(universe, graph, copy)
        ei, ea = to_undirected(torch.as_tensor(e.T, dtype=torch.long), torch.as_tensor(a),
                               num_nodes=len(genes))
        st, deg = _struct(universe, graph, copy)
    return Problem(universe, graph, copy, genes, s.y.values, s.fold.values, ei, ea, st, deg,
                   features(universe), neg)


def gp_inputs(p: Problem) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Column blocks for the additive GP: own feature blocks (per-block SVD), log-degree,
    topology (Laplacian eigenvectors, PyG transform) and 2-hop neighbour features (SIGN)."""
    from ghelps.models import sign_features
    parts = {b: _svd(p.universe, (b,), GP_BLOCK_DIM, b) for b in FEATURE_BLOCKS}
    own = np.hstack(list(parts.values()))
    parts["degree"] = _z(np.log1p(p.raw_degree)[:, None]).astype(np.float32)
    cache = P / "structure" / p.universe / f"lappe{GP_TOPO_DIM}_{p.graph}.{p.copy}.npy"
    if not cache.exists():
        d = AddLaplacianEigenvectorPE(GP_TOPO_DIM, attr_name="pe", is_undirected=True)(
            Data(edge_index=p.edge_index, num_nodes=len(p.genes)))
        np.save(cache, d.pe.numpy().astype(np.float32))
    parts["topology"] = np.load(cache)
    nb = sign_features(torch.as_tensor(own), p.edge_index, hops=2)[:, -own.shape[1]:].numpy()
    parts["neighbour_feat"] = _z(nb).astype(np.float32)
    cols, start = {}, 0
    for k, v in parts.items():
        cols[k] = list(range(start, start + v.shape[1]))
        start += v.shape[1]
    return np.hstack(list(parts.values())), cols
