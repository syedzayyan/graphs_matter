"""Positive gene lists. Everything not positive is unlabelled (PU setting)."""
from __future__ import annotations

import pandas as pd

from ghelps import ids

PHASE_NUM = {"Preclinical": 1, "Phase I": 2, "Phase II": 3, "Phase III": 4, "Launched": 5}


def _minikel_pp() -> pd.DataFrame:
    p = pd.read_csv(ids.RAW / "minikel" / "pp.tsv", sep="\t")
    p["ensg"] = ids.map_symbols(p.gene)
    return p.dropna(subset=["ensg"])


def minikel(min_phase: str = "Phase III") -> set[str]:
    """Genes whose best target-indication pair reached >= min_phase (ccat = max of
    historical and active phase)."""
    p = _minikel_pp()
    return set(p.loc[p.ccatnum >= PHASE_NUM[min_phase], "ensg"])


def minikel_first_launch() -> pd.Series:
    """ENSG -> year of the gene's first launched indication (Minikel data runs to 2022)."""
    p = _minikel_pp().dropna(subset=["year_launch"])
    return p.groupby("ensg").year_launch.min()


def pharos_tdl() -> pd.Series:
    d = pd.read_csv(ids.RAW / "pharos_targets.tsv", sep="\t")
    d["ensg"] = ids.map_uniprot(d.uniprot).fillna(ids.map_symbols(d.sym))
    return d.dropna(subset=["ensg"]).drop_duplicates("ensg").set_index("ensg").tdl
