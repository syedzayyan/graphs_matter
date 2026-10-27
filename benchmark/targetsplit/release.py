"""A benchmark release on disk: one folder per gene universe (README.md, "Files").

    rel = Release("results/benchmark", "string")
    idx, y = rel.train(seed=0, label="minikel")      # fit on these genes only
    report = rel.score(my_scores)                   # DataFrame: novelty, top-100 audit, tried

The same layout is written by targetsplit.build from your own genes and dated label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from targetsplit import core

REFERENCE = "pubmed_before_cutoff"


def _read(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    return d.rename(columns={"ensg": "gene"})  # releases exported before 0.2 key genes by "ensg"


class Release:
    def __init__(self, root: str | Path, universe: str):
        self.dir = Path(root) / universe
        self.genes = _read(self.dir / "genes.tsv")
        self.G = self.genes.gene.astype(str).values
        self.pos = {g: i for i, g in enumerate(self.G)}
        self.seeds = sorted(int(p.stem[4:]) for p in (self.dir / "splits").glob("seed*.tsv"))
        ft = _read(self.dir / "future_targets.tsv")
        self.strata = ft.select_dtypes(bool).reset_index(drop=True)
        self.future = np.zeros(len(self.G), bool)
        self.future[[self.pos[g] for g in ft.gene.astype(str)]] = True
        assert (np.flatnonzero(self.future) == [self.pos[g] for g in ft.gene.astype(str)]).all(), \
            "future_targets.tsv must list genes in genes.tsv order"
        self._splits: dict[int, pd.DataFrame] = {}

    # ------------------------------------------------------------------ data

    def split(self, seed: int) -> pd.DataFrame:
        if seed not in self._splits:
            sp = _read(self.dir / "splits" / f"seed{seed}.tsv")
            assert (sp.gene.astype(str).values == self.G).all(), "split rows must follow genes.tsv"
            self._splits[seed] = sp
        return self._splits[seed]

    @property
    def labels(self) -> list[str]:
        return [c[2:] for c in self.split(self.seeds[0]).columns if c.startswith("y_")]

    def train(self, seed: int, label: str | None = None, folds=("train", "val")) -> tuple[np.ndarray, np.ndarray]:
        """Positions of the genes a model may be fit on, and their 0/1 training label."""
        sp = self.split(seed)
        label = label or self.labels[0]
        idx = np.flatnonzero(sp.fold.isin(folds).values)
        return idx, sp[f"y_{label}"].values[idx].astype(int)

    def test(self, seed: int) -> np.ndarray:
        return np.flatnonzero(self.split(seed).fold.values == "test")

    def edges(self, graph: str) -> np.ndarray:
        e = pd.read_csv(self.dir / "edges" / f"{graph}.tsv", sep="\t")
        keep = e.a.astype(str).isin(self.pos) & e.b.astype(str).isin(self.pos)
        return np.c_[e.a[keep].astype(str).map(self.pos), e.b[keep].astype(str).map(self.pos)].astype(int)

    @property
    def graphs(self) -> list[str]:
        return sorted(p.stem for p in (self.dir / "edges").glob("*.tsv"))

    # ------------------------------------------------------------------ rankings

    def baselines(self) -> dict[str, np.ndarray]:
        """Reference rankers needing no model: study before the cut-off, number of known targets
        in the gene's domain families, and closeness to known targets (PubMed breaks ties)."""
        g = self.genes
        pm = g.pubmed_before_cutoff.values.astype(float)
        tie = rankdata(pm) / len(pm)
        out = {REFERENCE: pm}
        if "known_family_count" in g and g.known_family_count.notna().any():
            out["known_family_count"] = g.known_family_count.values + 1e-3 * tie
        if "hops_to_known" in g and (g.hops_to_known < 99).any():
            out["closeness_to_known"] = -g.hops_to_known.values.astype(float) + 1e-3 * tie
        return out

    def as_vector(self, scores: pd.DataFrame | pd.Series | np.ndarray) -> dict[int | None, np.ndarray]:
        """seed -> score per gene, from an array over genes.tsv, a Series indexed by gene, or a
        table with columns gene (or ensg), score and optionally seed."""
        if isinstance(scores, np.ndarray):
            return {None: scores.astype(float)}
        if isinstance(scores, pd.Series):
            return {None: scores.reindex(self.G).values.astype(float)}
        d = scores.rename(columns={"ensg": "gene"})
        d = d.assign(gene=d.gene.astype(str))
        if "seed" not in d:
            return {None: d.set_index("gene").score.reindex(self.G).values.astype(float)}
        return {int(s): x.set_index("gene").score.reindex(self.G).values.astype(float) for s, x in d.groupby("seed")}

    # ------------------------------------------------------------------ scoring

    def flags(self) -> dict[str, np.ndarray]:
        """Boolean gene flags for the top-100 audit."""
        g = self.genes
        novel2 = np.zeros(len(g), bool)
        if "novel (2 of 3)" in self.strata:
            novel2[np.flatnonzero(self.future)[self.strata["novel (2 of 3)"].values]] = True
        f = {"future_target": self.future, "novel2_future_target": novel2,
             "top_study_decile": g.pubmed_pct_before_cutoff.values >= 0.9}
        if "minikel_best_phase" in g:  # preclinical-Phase II
            f["in_programme"] = g.minikel_best_phase.isin([1, 2, 3]).values
        elif "programme_status" in g:
            f["in_programme"] = g.programme_status.isin(["failure", "censored"]).values
        if "known_family_count" in g and g.known_family_count.notna().any():
            f["drugged_family"] = g.known_family_count.values > 0
        if "hops_to_known" in g and g.hops_to_known.notna().any():
            f["one_hop_from_known"] = g.hops_to_known.values <= 1
        for c in g.columns:
            if c == "genetic_before_cutoff" or c.startswith("flag_"):
                f[c.removeprefix("flag_")] = g[c].astype(bool).values
        return f

    def score(self, scores, name: str = "model", strata: list[str] | None = None, references: bool = True,
              boot: int = core.BOOT, top_frac: float = core.TOP_FRAC, top_k: int = core.TOP_K) -> pd.DataFrame:
        """Score one ranking (see as_vector for accepted forms); with references=True the
        baseline rankers are scored alongside and every row carries the paired difference from
        the PubMed ranker. strata: columns of future_targets.tsv (default: all of them)."""
        mine = self.as_vector(scores)
        rankings = {name: mine}
        if references:
            rankings |= {k: {None: v} for k, v in self.baselines().items()}
        seeds = [s for s in self.seeds if None in mine or s in mine]
        if not seeds:
            raise ValueError(f"scores have seeds {sorted(mine)}, release has {self.seeds}")
        tests = {s: self.test(s) for s in seeds}
        st = self.strata[strata] if strata else self.strata
        nov = core.score(rankings, tests, self.future, st, REFERENCE, boot, top_frac).assign(section="novelty")
        rows = []
        flags = self.flags()
        status = self.genes.programme_status.values if "programme_status" in self.genes else None
        for rname, by_seed in rankings.items():
            top, tried = [], []
            for s, test in tests.items():
                v = by_seed[s] if s in by_seed else by_seed[None]
                top.append(core.top_audit(v, test, flags, top_k))
                if status is not None:
                    tried.append(core.tried_aurocs(v, test, status))
            rows += [dict(section=f"top{top_k}", ranking=rname, metric=k, value=v)
                     for k, v in pd.DataFrame(top).mean().items()]
            rows += [dict(section="tried", ranking=rname, metric=k, value=v)
                     for k, v in pd.DataFrame(tried).mean().items()]
        return pd.concat([nov, pd.DataFrame(rows)], ignore_index=True)
