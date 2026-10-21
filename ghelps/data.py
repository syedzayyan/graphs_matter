"""Load one (universe, graph, condition, label set, split, seed) problem.

Graph conditions ("copy"):
  real       the graph
  rw0..rw2   degree-preserving rewired copies (edge vectors permuted onto them)
  empty      no edges: same model with the graph removed
  shufattr   real topology, edge vectors permuted across edges
  pwsub      pathway-coherent subgraph: real edges whose endpoints share a specific pathway
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
# Feature sets (experiment axis `feats`). no_tract drops Open Targets tractability entirely:
# even its non-clinical buckets (druggable family, ligands, pockets) partly reflect past
# drug programmes, so every result should also hold without it.
# no_loc drops GO and tractability (both encode subcellular localisation): the feature set
# for the membrane positive control. random is the nonsense test: Gaussian noise vectors.
FEATURE_SETS = {"all": FEATURE_BLOCKS, "no_tract": ("go", "pfam", "pathway"),
                "no_loc": ("pfam", "pathway"), "random": ("noise",),
                # attention-free: no GO / pathways / literature / curated structures; uniform
                # per gene (ESM-2 embeddings, sequence-derived features, GTEx expression)
                "attn_free": ("esm", "seqfeat", "gtex")}
# Study-residualised variants: every feature column replaced by its rank among genes with a
# similar PubMed count (50 equal-size bins), so each column has the same distribution at every
# level of study. Label-free. Diagnostic only (scripts/analysis.py: study_leak.csv): an RF
# still predicts log PubMed from them (STRING Spearman: attn_free 0.60 -> 0.53, all 0.79 ->
# 0.89), because study is concentrated in particular families and annotation patterns, which
# no per-column correction removes. Experiments remove the *incentive* instead (pn_pubmed).
FEATURE_SETS |= {f"{k}_resid": FEATURE_SETS[k] for k in ("all", "attn_free")}
# dated: only what could have been known before the time-split cut-off (2010). Attention-free
# blocks, Pfam (sequence-derived), GO as annotated by Dec 2009 and KEGG pathways as of Sep 2009
# (fetch.py). No tractability (it can't be dated) and no current GO / Reactome. For the time
# split: are the time-split results down to weak features (attn_free), or to the label?
FEATURE_SETS["dated"] = ("esm", "seqfeat", "gtex", "pfam", "go09", "kegg09")
# Sets mixing z-scored continuous blocks with sparse binary ones get one SVD per group (half the
# width each), so the continuous blocks' larger variance doesn't swamp the annotations.
SVD_GROUPS = {"dated": (("esm", "seqfeat", "gtex"), ("pfam", "go09", "kegg09"))}
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
    feats: str
    genes: list[str]
    y: np.ndarray                  # 1 positive / 0 unlabelled (split-specific)
    fold: np.ndarray               # train / val / test / excluded
    edge_index: torch.Tensor | None  # (2, 2m) undirected
    edge_attr: torch.Tensor | None   # (2m, 12)
    struct: np.ndarray | None
    raw_degree: np.ndarray | None
    feat: np.ndarray
    negatives: pd.DataFrame
    pubmed: np.ndarray             # PubMed count per gene (study intensity)

    @property
    def pathways(self) -> sp.csr_matrix:
        """Gene x specific-pathway (<= 100 genes) membership, for same-pathway hard negatives."""
        return pathway_membership(self.universe)

    def mask(self, f: str) -> np.ndarray:
        return self.fold == f


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


@lru_cache(maxsize=4)
def genes_of(universe: str) -> list[str]:
    return (P / "graphs" / universe / "genes.txt").read_text().split()


def _svd(universe: str, blocks, dim: int, tag: str) -> np.ndarray:
    """Label-free TruncatedSVD of the concatenated binary blocks, z-scored, cached. The
    pseudo-block "noise" gives seeded Gaussian vectors of the same width instead."""
    cache = P / "features" / universe / f"svd{dim}_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    genes = genes_of(universe)
    if tuple(blocks) == ("noise",):
        z = np.random.default_rng(0).standard_normal((len(genes), dim)).astype(np.float32)
        np.save(cache, z)
        return z
    x = sp.hstack([sp.csr_matrix(pd.read_parquet(P / "features" / universe / f"{b}.parquet").loc[genes].values,
                                 dtype=np.float32) for b in blocks]).tocsr()
    z = _z(TruncatedSVD(min(dim, x.shape[1] - 1), random_state=0).fit_transform(x)).astype(np.float32)
    np.save(cache, z)
    return z


@lru_cache(maxsize=8)
def pathway_membership(universe: str, max_genes: int = 100) -> sp.csr_matrix:
    pw = pd.read_parquet(P / "features" / universe / "pathway.parquet").loc[genes_of(universe)]
    return sp.csr_matrix(pw.loc[:, pw.sum() <= max_genes].values, dtype=np.float32)


@lru_cache(maxsize=8)
def pubmed_counts(universe: str) -> np.ndarray:
    return pd.read_parquet(P / "features" / universe / "pubmed.parquet").loc[genes_of(universe)].pubmed_count.values


def _resid(universe: str, x: np.ndarray, bins: int = 50) -> np.ndarray:
    """Each column -> normal scores of its rank within the gene's PubMed-count bin (ties
    broken at random, seeded)."""
    from scipy.stats import norm
    pm = pubmed_counts(universe)
    rng = np.random.default_rng(0)
    b = pd.qcut(pd.Series(pm).rank(method="first"), bins, labels=False).values
    out = np.empty(x.shape, np.float32)
    for k in range(bins):
        m = rng.permutation(np.flatnonzero(b == k))
        r = np.argsort(np.argsort(x[m], axis=0, kind="stable"), axis=0)
        out[m] = norm.ppf((r + 0.5) / len(m))
    return out


def _block(universe: str, blocks, dim: int, tag: str, feats: str) -> np.ndarray:
    """SVD features of `blocks` (cached under `tag`), residualised for the *_resid sets."""
    x = _svd(universe, blocks, dim, tag)
    return _resid(universe, x) if feats.endswith("_resid") else x


@lru_cache(maxsize=8)
def features(universe: str, feats: str = "all") -> np.ndarray:
    if feats in SVD_GROUPS:
        return np.hstack([_svd(universe, g, SVD_DIM // len(SVD_GROUPS[feats]), f"{feats}{i}")
                          for i, g in enumerate(SVD_GROUPS[feats])])
    return _block(universe, FEATURE_SETS[feats], SVD_DIM, feats.removesuffix("_resid"), feats)


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


def load(universe: str, graph: str | None, copy: str, labels: str, split: str, seed: int,
         feats: str = "all") -> Problem:
    genes = genes_of(universe)
    sdir = P / "splits" / universe / labels / split
    s = pd.read_parquet(sdir / f"seed{seed}.parquet")
    neg = pd.read_parquet(sdir / f"seed{seed}.neg.parquet")
    # a split with a cut-off date (time) ships the PubMed counts known at the cut-off, so
    # study-matched training negatives (pn_pubmed) don't use later papers
    pm = pd.read_parquet(sdir / "pubmed.parquet").loc[genes].pubmed_count.values \
        if (sdir / "pubmed.parquet").exists() else pubmed_counts(universe)
    ei = ea = st = deg = None
    if graph is not None:
        e, a = _edges(universe, graph, copy)
        ei, ea = to_undirected(torch.as_tensor(e.T, dtype=torch.long), torch.as_tensor(a),
                               num_nodes=len(genes))
        st, deg = _struct(universe, graph, copy)
    return Problem(universe, graph, copy, feats, genes, s.y.values, s.fold.values, ei, ea, st, deg,
                   features(universe, feats), neg, pm)


def gp_inputs(p: Problem) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Column blocks for the additive GP: own feature blocks (per-block SVD), log-degree,
    topology (Laplacian eigenvectors, PyG transform) and 2-hop neighbour features (SIGN)."""
    from ghelps.models import sign_features
    parts = {b: _block(p.universe, (b,), GP_BLOCK_DIM, b, p.feats) for b in FEATURE_SETS[p.feats]}
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
