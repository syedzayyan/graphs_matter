"""Build a time-split release from your own genes and dated label (same layout as ours).

    targetsplit build --genes genes.tsv --cutoff 2010 --out my_release [--name custom]
                      [--domains domains.tsv] [--edges edges.tsv ...] [--seeds 5]

genes.tsv, one row per gene in your universe:
  gene                  any identifier (used everywhere else)
  first_year            year the gene first became a positive (e.g. first approval); empty if never
  positive              1 if the gene is a positive at any date (default: first_year is set).
                        Positives without first_year can't be placed in time and are excluded.
  pubmed_before_cutoff  papers about the gene before the cut-off (study intensity then)
  optional:
  label_<name>          another training label, already restricted to what was known before the
                        cut-off (e.g. label_genetic); written as y_<name> on the training folds
  programme_status      never / success / failure / censored, for selection and outcome AUROC
  flag_<name>           any 0/1 gene property to report in the top-100 audit
  further columns are copied to genes.tsv unchanged
domains.tsv: gene, domain (e.g. Pfam), one row per pair; defines "drugged family".
edges.tsv: a, b (gene identifiers); one file per graph; defines hops to known targets. Prefer a
graph dated before the cut-off: a current one links future targets through later knowledge.

Folds per seed follow the paper: test = every positive first seen from the cut-off on plus a
random 20% of the never-positive genes; the rest of the never-positive genes split 70:10 into
train / val, and the earlier positives in the same ratio.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from targetsplit import core

TRAIN, VAL, TEST = 0.7, 0.1, 0.2


def folds(known: np.ndarray, future: np.ndarray, excluded: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.full(len(known), "train", dtype=object)
    f[excluded] = "excluded"
    f[future] = "test"
    kp = rng.permutation(np.flatnonzero(known))
    f[kp[: round(len(kp) * VAL / (TRAIN + VAL))]] = "val"
    unl = rng.permutation(np.flatnonzero(~known & ~future & ~excluded))
    n_te, n_va = round(len(unl) * TEST), round(len(unl) * VAL)
    f[unl[:n_te]] = "test"
    f[unl[n_te:n_te + n_va]] = "val"
    return f


def build(genes: pd.DataFrame, cutoff: int, out: str | Path, name: str = "custom",
          domains: pd.DataFrame | None = None, edges: dict[str, pd.DataFrame] | None = None,
          seeds: int = 5, hops_graph: str | None = None) -> Path:
    g = genes.copy()
    g["gene"] = g.gene.astype(str)
    assert g.gene.is_unique, "genes.tsv: one row per gene"
    G = g.gene.values
    pos = {x: i for i, x in enumerate(G)}
    year = pd.to_numeric(g.first_year, errors="coerce").values
    positive = g.positive.astype(bool).values if "positive" in g else ~np.isnan(year)
    known, future = positive & (year < cutoff), positive & (year >= cutoff)
    excluded = positive & np.isnan(year)
    pm = pd.to_numeric(g.pubmed_before_cutoff).fillna(0).values.astype(float)

    dom = None
    if domains is not None:
        d = domains.assign(gene=domains.gene.astype(str))
        d = d[d.gene.isin(pos)].drop_duplicates(["gene", "domain"])
        codes, _ = pd.factorize(d.domain)
        dom = csr_matrix((np.ones(len(d)), (d.gene.map(pos).values, codes)), shape=(len(G), codes.max() + 1))
    edges = edges or {}
    eidx = {}
    for gname, e in edges.items():
        e = e.astype(str)
        keep = e.a.isin(pos) & e.b.isin(pos) & (e.a != e.b)
        eidx[gname] = np.c_[e.a[keep].map(pos), e.b[keep].map(pos)].astype(int)
    hg = hops_graph or (next(iter(eidx)) if eidx else None)
    ctx = core.context(known, pm, dom, eidx.get(hg))

    u = Path(out) / name
    (u / "splits").mkdir(parents=True, exist_ok=True)
    extra = [c for c in g.columns if c not in ("gene", "first_year", "positive", "pubmed_before_cutoff")
             and not c.startswith("label_")]
    pd.DataFrame({"gene": G, "first_year": year, "positive": positive.astype(int), "pubmed_before_cutoff": pm,
                  "pubmed_pct_before_cutoff": ctx.study_pct.values, "known_family_count": ctx.family_count.values,
                  "hops_to_known": ctx.hops.values, **{c: g[c].values for c in extra}}) \
        .to_csv(u / "genes.tsv", sep="\t", index=False)
    labels = {"target": known} | {c[6:]: g[c].astype(bool).values for c in g.columns if c.startswith("label_")}
    for s in range(seeds):
        f = folds(known, future, excluded, s)
        side = np.isin(f, ["train", "val"])
        ys = {f"y_{k}": np.where(side, v, f == "test") & (side | future) for k, v in labels.items()}
        pd.DataFrame({"gene": G, "fold": f, **{k: v.astype(int) for k, v in ys.items()}}) \
            .to_csv(u / "splits" / f"seed{s}.tsv", sep="\t", index=False)
    st = core.strata_table(ctx, future)
    st.insert(0, "gene", G[future])
    st.insert(1, "first_year", year[future])
    st.to_csv(u / "future_targets.tsv", sep="\t", index=False)
    if eidx:
        (u / "edges").mkdir(exist_ok=True)
        for gname, e in eidx.items():
            pd.DataFrame({"a": G[e[:, 0]], "b": G[e[:, 1]]}).to_csv(u / "edges" / f"{gname}.tsv", sep="\t", index=False)
    (Path(out) / "MANIFEST").write_text(
        f"time split cut-off: {cutoff}\nseeds: {list(range(seeds))}\n{name}: {len(G)} genes, {int(known.sum())} "
        f"known targets, {int(future.sum())} future targets, {int(excluded.sum())} undated positives excluded, "
        f"hops from {hg or 'no graph'}\n")
    return u


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="targetsplit build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="custom")
    ap.add_argument("--domains")
    ap.add_argument("--edges", nargs="*", default=[], help="edge files; the graph name is the file stem")
    ap.add_argument("--hops-graph", help="which graph defines hops (default: the first)")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)
    rd = lambda p: pd.read_csv(p, sep="\t")  # noqa: E731
    u = build(rd(a.genes), a.cutoff, a.out, a.name, rd(a.domains) if a.domains else None,
              {Path(p).stem: rd(p) for p in a.edges}, a.seeds, a.hops_graph)
    print((Path(a.out) / "MANIFEST").read_text(), f"written to {u}", sep="")
