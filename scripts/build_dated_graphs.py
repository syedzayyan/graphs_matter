"""Stage 1f: dated graphs on an existing universe (configs/data.yaml: dated_graphs).

The universe's genes.txt, features and splits are left as they are: a dated graph is induced
on those genes; genes it does not cover are isolated nodes. A graph with `sample_from` is a
density control: a uniform sample of another graph's edges, as many as `match_edges_of` has.
Per dated graph <g> on universe <u>:
  data/processed/graphs/<u>/<g>.{real,rw0..2,pwsub,empty}.npy   as for the other graphs
  data/processed/structure/<u>/<g>.<copy>.parquet                structural stats
  data/processed/graphs/dated_summary.tsv
No edge vectors are written (data.load falls back to zeros), so GAT + edge vec is not run on them.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghelps import features, graphs, paths, structure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
G, S = paths.PROCESSED / "graphs", paths.PROCESSED / "structure"
rw = cfg["rewire"]
pw = features.pathways(min_genes=5, max_genes=cfg["pathway_subgraph"]["max_pathway_genes"])
rows = []
for uname, extras in cfg.get("dated_graphs", {}).items():
    udir = G / uname
    genes = (udir / "genes.txt").read_text().split()
    for name in extras:
        gc = cfg["graphs"].get(name, {})
        if "sample_from" in gc:  # density control: a uniform sample of another graph's edges
            src = np.load(udir / f"{gc['sample_from']}.real.npy")
            m = len(np.load(udir / f"{gc['match_edges_of']}.real.npy"))
            ei = src[np.sort(np.random.default_rng(gc["seed"]).choice(len(src), m, replace=False))]
            e = pd.DataFrame({"u": np.array(genes)[ei[:, 0]], "v": np.array(genes)[ei[:, 1]]})
        else:
            e = graphs.DATED[name](cfg)
            ei = graphs.to_index(graphs.induce(e, genes), genes)
        e.to_csv(G / f"{name}.raw.tsv", sep="\t", index=False)
        copies = {"real": ei, "empty": np.zeros((0, 2), dtype=np.int64)}
        for k in range(rw["n_copies"]):
            seed = rw["seed"] + 1000 * k + zlib.crc32(f"{uname}/{name}".encode()) % 997
            copies[f"rw{k}"] = graphs.rewire(ei, len(genes), rw["swaps_per_edge"], seed=seed)
        copies["pwsub"] = graphs.pathway_subgraph(ei, sp.csr_matrix(pw.reindex(genes, fill_value=0).values))
        (S / uname).mkdir(parents=True, exist_ok=True)
        for c, x in copies.items():
            np.save(udir / f"{name}.{c}.npy", x)
            if c != "empty":
                structure.stats(x, len(genes)).to_parquet(S / uname / f"{name}.{c}.parquet")
        deg = np.bincount(ei.ravel(), minlength=len(genes))
        rows.append(dict(universe=uname, graph=name, raw_nodes=len(set(e.u) | set(e.v)), raw_edges=len(e),
                         n_genes=len(genes), univ_edges=len(ei), isolated=int((deg == 0).sum()),
                         mean_deg=deg.mean(), max_deg=int(deg.max()), pwsub_edges=len(copies["pwsub"])))
summary = pd.DataFrame(rows)
summary.to_csv(G / "dated_summary.tsv", sep="\t", index=False)
print(summary.round(3).to_string(index=False))
