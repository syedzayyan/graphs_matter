"""Gene identifier harmonisation. Canonical ID everywhere is the Ensembl gene ID
of an HGNC-approved protein-coding gene."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@lru_cache(maxsize=1)
def hgnc() -> pd.DataFrame:
    df = pd.read_csv(RAW / "hgnc.tsv", sep="\t", dtype=str, low_memory=False)
    df = df[(df.locus_group == "protein-coding gene") & (df.status == "Approved")]
    return df.dropna(subset=["ensembl_gene_id"]).reset_index(drop=True)


def protein_coding() -> set[str]:
    return set(hgnc().ensembl_gene_id)


def _explode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df[[col, "ensembl_gene_id"]].dropna()
    out = out.assign(**{col: out[col].str.split("|")}).explode(col)
    return out


@lru_cache(maxsize=1)
def symbol_map() -> dict[str, str]:
    """Symbol -> ENSG. Approved symbols win over previous symbols, which win over aliases;
    ambiguous previous/alias symbols are dropped."""
    h = hgnc()
    m: dict[str, str] = {}
    for col in ("alias_symbol", "prev_symbol"):
        e = _explode(h, col)
        e = e[~e[col].duplicated(keep=False)]
        m.update(dict(zip(e[col].str.upper(), e.ensembl_gene_id)))
    m.update(dict(zip(h.symbol.str.upper(), h.ensembl_gene_id)))
    return m


@lru_cache(maxsize=1)
def uniprot_map() -> dict[str, str]:
    e = _explode(hgnc(), "uniprot_ids")
    e = e[~e.uniprot_ids.duplicated(keep=False)]
    return dict(zip(e.uniprot_ids, e.ensembl_gene_id))


@lru_cache(maxsize=1)
def entrez_map() -> dict[str, str]:
    h = hgnc().dropna(subset=["entrez_id"])
    return dict(zip(h.entrez_id, h.ensembl_gene_id))


def map_symbols(s: pd.Series) -> pd.Series:
    return s.str.upper().map(symbol_map())


def map_uniprot(s: pd.Series) -> pd.Series:
    # strip isoform (-2) and chain (-PRO_xxx) suffixes
    return s.str.replace(r"^uniprotkb:", "", regex=True).str.split("-").str[0].map(uniprot_map())


def ensg_to_symbol() -> dict[str, str]:
    h = hgnc()
    return dict(zip(h.ensembl_gene_id, h.symbol))
