"""Build the five gene-gene graphs as undirected ENSG edge lists, restrict them to a
shared gene universe, and make degree-preserving rewired copies."""
from __future__ import annotations

import zipfile
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

from ghelps import ids

GRAPHS = ["string", "string_exp", "intact", "reactome_fi", "huri"]


def _canon(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Undirected, deduplicated, loop-free, protein-coding-only edge list with u < v."""
    df = pd.DataFrame({"u": a.values, "v": b.values}).dropna()
    pc = ids.protein_coding()
    df = df[df.u.isin(pc) & df.v.isin(pc) & (df.u != df.v)]
    lo, hi = np.minimum(df.u, df.v), np.maximum(df.u, df.v)
    return pd.DataFrame({"u": lo, "v": hi}).drop_duplicates().reset_index(drop=True)


def _string_links(cfg: dict) -> pd.DataFrame:
    return _string_links_full(cfg, ["experiments"])


def _string_links_full(cfg: dict, channels: list[str]) -> pd.DataFrame:
    """STRING links with the requested channel columns, proteins mapped to ENSG (g1, g2)."""
    path = ids.RAW / "9606.protein.links.full.v12.0.txt.gz"
    df = pd.read_csv(path, sep=" ", usecols=["protein1", "protein2", "combined_score", *channels])
    al = pd.read_csv(ids.RAW / "string_aliases.txt.gz", sep="\t", names=["ensp", "alias", "src"], comment="#")
    al = al[al.src == "Ensembl_gene"].drop_duplicates("ensp")
    m = dict(zip(al.ensp, al.alias))
    df["g1"], df["g2"] = df.protein1.map(m), df.protein2.map(m)
    return df


def build_string(cfg: dict, links: pd.DataFrame) -> pd.DataFrame:
    d = links[links.combined_score >= cfg["graphs"]["string"]["combined_score_min"]]
    return _canon(d.g1, d.g2)


def build_string_exp(cfg: dict, links: pd.DataFrame) -> pd.DataFrame:
    d = links[links.experiments >= cfg["graphs"]["string_exp"]["experiments_min"]]
    return _canon(d.g1, d.g2)


def build_intact(cfg: dict) -> pd.DataFrame:
    c = cfg["graphs"]["intact"]
    df = pd.read_csv(ids.RAW / "intact_human_slim.tsv", sep="\t", names=["a", "b", "method", "type", "conf"])
    df["itype"] = df.type.str.extract(r"\((.*)\)$")[0]
    df["miscore"] = df.conf.str.extract(r"intact-miscore:([0-9.]+)")[0].astype(float)
    df = df[df.itype.isin(c["keep_types"]) & (df.miscore >= c["miscore_min"])]
    df = df[df.a.str.startswith("uniprotkb:") & df.b.str.startswith("uniprotkb:")]
    return _canon(ids.map_uniprot(df.a), ids.map_uniprot(df.b))


def build_reactome_fi(cfg: dict) -> pd.DataFrame:
    with zipfile.ZipFile(ids.RAW / "reactome_fi.zip") as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep="\t")
    if not cfg["graphs"]["reactome_fi"]["include_predicted"]:
        df = df[df.Annotation != "predicted"]
    return _canon(ids.map_symbols(df.Gene1), ids.map_symbols(df.Gene2))


def build_huri(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ids.RAW / "HuRI.tsv", sep="\t", names=["a", "b"])
    return _canon(df.a, df.b)


def build_all(cfg: dict) -> dict[str, pd.DataFrame]:
    links = _string_links(cfg)
    return {
        "string": build_string(cfg, links),
        "string_exp": build_string_exp(cfg, links),
        "intact": build_intact(cfg),
        "reactome_fi": build_reactome_fi(cfg),
        "huri": build_huri(cfg),
    }


def universe(edges: dict[str, pd.DataFrame]) -> list[str]:
    sets = [set(e.u) | set(e.v) for e in edges.values()]
    return sorted(set.intersection(*sets))


def induce(e: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    g = set(genes)
    return e[e.u.isin(g) & e.v.isin(g)].reset_index(drop=True)


def to_index(e: pd.DataFrame, genes: list[str]) -> np.ndarray:
    """(m, 2) int array of node indices into the universe ordering."""
    pos = {g: i for i, g in enumerate(genes)}
    return np.stack([e.u.map(pos).values, e.v.map(pos).values], axis=1).astype(np.int64)


def rewire(edge_index: np.ndarray, n: int, swaps_per_edge: int, seed: int) -> np.ndarray:
    """Degree-preserving double-edge-swap rewiring that keeps the graph simple."""
    import random
    random.seed(seed)  # igraph draws from Python's RNG
    g = ig.Graph(n=n, edges=edge_index.tolist(), directed=False)
    g.rewire(n=swaps_per_edge * g.ecount(), mode="simple")
    out = np.array(g.get_edgelist(), dtype=np.int64)
    assert (np.bincount(out.ravel(), minlength=n) == np.bincount(edge_index.ravel(), minlength=n)).all()
    return np.sort(out, axis=1)


EM_SUFFIX = "_em"


def universe_graphs(cfg: dict, universe: str) -> list[str]:
    """Every graph built on a universe: its member graphs plus any edge-matched copies."""
    out = list(cfg["universes"][universe])
    em = cfg.get("edge_matched")
    if em and em["universe"] == universe:
        out += [g + EM_SUFFIX for g in em["graphs"]]
    return out


def subsample_edges(edge_index: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Uniform random subset of m edges (all of them if the graph has <= m)."""
    if len(edge_index) <= m:
        return edge_index
    keep = np.random.default_rng(seed).choice(len(edge_index), size=m, replace=False)
    return edge_index[np.sort(keep)]
