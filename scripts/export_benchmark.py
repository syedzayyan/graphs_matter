"""Export the time-split benchmark as a self-contained release (benchmark/README.md).

  uv run python scripts/export_benchmark.py [--out results/benchmark]

Per universe (results/benchmark/<universe>/):
  genes.tsv          one row per gene: ids, study before the cut-off and now, Minikel best phase
                     and programme status (never / success / failure / censored), first approval
                     year, genetic evidence before the cut-off / any date, Pharos TDL, and the
                     novelty context (Pfam-family count and hops to the targets known at the cut-off)
  splits/seed<k>.tsv the time split: fold (train / val / test / excluded) and both training
                     labels (y_minikel: drugged before the cut-off; y_genetic: genetic evidence
                     before it). Test genes and their y are the same for both labels.
  future_targets.tsv the test positives (first drugged from the cut-off on) with their novelty strata
  edges/<graph>.tsv  the universe's graph(s) as ENSG pairs (real graph; dated graphs too)
plus benchmark/evaluate.py and benchmark/README.md copied to the release root, and a MANIFEST.
Uses $GHELPS_DATA, so it exports exactly the data the runs used.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analysis as A  # noqa: E402
from ghelps import data, features, graphs, ids, labels, paths  # noqa: E402


def export(out: Path) -> None:
    T = A.CFG["splits"]["time_split_year"]
    cum = features.pubmed_by_year()
    phase = labels.minikel_best_phase()
    approval = labels.target_first_approval()
    gen_T, gen_any = labels.genetic(before=T), labels.genetic()
    tdl = labels.pharos_tdl()
    sym = ids.ensg_to_symbol()
    seeds = range(A.CFG["splits"]["n_seeds"] - 2)  # 0-4: the evaluation seeds (5-6 are for tuning)
    manifest = []
    for universe, members in A.CFG["universes"].items():
        graph = members[0]
        genes = data.genes_of(universe)
        G = np.array(genes)
        udir = out / universe
        (udir / "splits").mkdir(parents=True, exist_ok=True)
        (udir / "edges").mkdir(exist_ok=True)
        pm_T = cum.reindex(genes, fill_value=0)[T - 1].values.astype(float)
        fut_idx, strata, fam_known, hops = A._future_context(universe, genes, graph, pm_T)
        gr = A.tried_groups(genes)
        status = np.select([gr.success, gr.failure, gr.never], ["success", "failure", "never"], "censored")
        pd.DataFrame({
            "ensg": G, "symbol": [sym.get(g, "") for g in genes],
            "pubmed_before_cutoff": pm_T.astype(int), "pubmed_now": data.pubmed_counts(universe).astype(int),
            "pubmed_pct_before_cutoff": rankdata(pm_T) / len(pm_T),
            "minikel_best_phase": phase.reindex(genes).fillna(0).astype(int).values, "programme_status": status,
            "first_approval_year": approval.reindex(genes).values,
            "genetic_before_cutoff": np.isin(G, list(gen_T)).astype(int),
            "genetic_any": np.isin(G, list(gen_any)).astype(int),
            "pharos_tdl": tdl.reindex(genes).fillna("").values,
            "known_family_count": fam_known, "hops_to_known": np.minimum(hops, 99).astype(int),
        }).to_csv(udir / "genes.tsv", sep="\t", index=False)
        for seed in seeds:
            m = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / "time" / f"seed{seed}.parquet")
            g = pd.read_parquet(paths.PROCESSED / "splits" / universe / "genetic" / "time" / f"seed{seed}.parquet")
            assert (m.gene.values == G).all() and (g.gene.values == G).all() and (m.fold.values == g.fold.values).all()
            pd.DataFrame({"ensg": G, "fold": m.fold.values, "y_minikel": m.y.values, "y_genetic": g.y.values}).to_csv(
                udir / "splits" / f"seed{seed}.tsv", sep="\t", index=False)
        strata.assign(ensg=G[fut_idx], symbol=[sym.get(x, "") for x in G[fut_idx]],
                      first_approval_year=approval.reindex(G[fut_idx]).values).drop(columns="gene").to_csv(
            udir / "future_targets.tsv", sep="\t", index=False)
        for gname in [*members, *graphs.dated_graphs(A.CFG, universe)]:
            ei = np.load(paths.PROCESSED / "graphs" / universe / f"{gname}.real.npy")
            pd.DataFrame({"a": G[ei[:, 0]], "b": G[ei[:, 1]]}).to_csv(udir / "edges" / f"{gname}.tsv", sep="\t",
                                                                      index=False)
        manifest.append(f"{universe}: {len(genes)} genes, {len(fut_idx)} future targets, graphs "
                        f"{', '.join([*members, *graphs.dated_graphs(A.CFG, universe)])}")
    for f in ("evaluate.py", "README.md"):
        shutil.copy(ROOT / "benchmark" / f, out / f)
    (out / "MANIFEST").write_text(f"time split cut-off: {T}\nseeds: {list(seeds)}\n" + "\n".join(manifest) + "\n")
    print((out / "MANIFEST").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()
    export(ROOT / a.out)
