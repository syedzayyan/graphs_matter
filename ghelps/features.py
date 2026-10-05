"""Node features (GO, Pfam, Reactome pathway membership, OT tractability) and PubMed counts.

All builders return a DataFrame indexed by ENSG over the protein-coding genes they cover;
alignment to a universe happens in scripts/build_features.py.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from ghelps import ids

# GO evidence codes that are themselves interaction data: using them would feed the PPI
# graph back in through the features.
GO_EXCLUDE_EVIDENCE = {"IPI", "ND"}
GO_EXCLUDE_TERMS = {"GO:0005515"}  # "protein binding" -- a PPI annotation in disguise
# Tractability buckets that encode clinical precedence, i.e. the label.
TRACT_EXCLUDE = {"Approved Drug", "Advanced Clinical", "Phase 1 Clinical"}


def _binary(pairs: pd.DataFrame, gene: str, term: str, min_genes: int, max_genes: int | None = None) -> pd.DataFrame:
    pairs = pairs[[gene, term]].dropna().drop_duplicates().reset_index(drop=True)
    sizes = pairs[term].value_counts()
    keep = sizes[(sizes >= min_genes) & ((sizes <= max_genes) if max_genes else True)].index
    pairs = pairs[pairs[term].isin(keep)]
    m = pd.crosstab(pairs[gene], pairs[term]).clip(upper=1).astype(np.uint8)
    m.index.name = "ensg"
    return m


def go(min_genes: int = 10) -> pd.DataFrame:
    cols = ["db", "acc", "sym", "qual", "go", "ref", "evidence"]
    df = pd.read_csv(ids.RAW / "goa_human.gaf.gz", sep="\t", comment="!", header=None,
                     usecols=range(7), names=cols, dtype=str)
    df = df[~df.evidence.isin(GO_EXCLUDE_EVIDENCE) & ~df.go.isin(GO_EXCLUDE_TERMS)
            & ~df.qual.str.startswith("NOT")]
    df["ensg"] = ids.map_uniprot(df.acc).fillna(ids.map_symbols(df.sym))
    return _binary(df, "ensg", "go", min_genes).add_prefix("go:")


def pfam(min_genes: int = 3) -> pd.DataFrame:
    df = pd.read_csv(ids.RAW / "uniprot_human.tsv.gz", sep="\t", dtype=str)
    df["ensg"] = ids.map_uniprot(df.Entry).fillna(ids.map_symbols(df["Gene Names (primary)"].fillna("")))
    df = df.assign(pfam=df.Pfam.str.rstrip(";").str.split(";")).explode("pfam")
    return _binary(df, "ensg", "pfam", min_genes).add_prefix("pfam:")


def pathways(min_genes: int = 5, max_genes: int = 500) -> pd.DataFrame:
    df = pd.read_csv(ids.RAW / "ensembl2reactome.txt", sep="\t", header=None,
                     names=["ens", "pw", "url", "name", "ev", "species"], usecols=[0, 1, 5])
    df = df[(df.species == "Homo sapiens") & df.ens.str.startswith("ENSG")]
    return _binary(df, "ens", "pw", min_genes, max_genes).rename_axis("ensg").add_prefix("rx:")


def tractability() -> pd.DataFrame:
    t = pd.read_parquet(ids.RAW / "ot" / "target_tractability")
    t = t[~t.category.isin(TRACT_EXCLUDE)]
    t["col"] = "tract:" + t.modality + ":" + t.category
    m = t.pivot_table(index="targetId", columns="col", values="value", aggfunc="max").fillna(False)
    m.index.name = "ensg"
    return m.astype(np.uint8)


def pubmed_counts() -> pd.Series:
    counts: dict[str, int] = {}
    with gzip.open(ids.RAW / "gene2pubmed.gz", "rt") as f:
        next(f)
        for line in f:
            if line.startswith("9606\t"):
                g = line.split("\t", 2)[1]
                counts[g] = counts.get(g, 0) + 1
    s = pd.Series(counts)
    s.index = s.index.map(ids.entrez_map())
    s = s[s.index.notna()].groupby(level=0).sum()
    s.index.name = "ensg"
    return s.rename("pubmed_count")
