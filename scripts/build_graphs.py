"""Stage 1a: raw edge lists -> shared universe -> real + rewired graphs on disk.

Outputs (data/processed/graphs/):
  <graph>.raw.tsv                 full edge list before universe restriction
  <universe>/genes.txt            universe ordering (ENSG), defines node indices everywhere
  <universe>/<graph>.real.npy     (m, 2) node-index edges on the universe
  <universe>/<graph>.rw{k}.npy    rewired copies
  <universe>/<graph>.pwsub.npy    pathway-coherent subgraph (configs/data.yaml: pathway_subgraph)
  summary.tsv
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghelps import features, graphs, paths  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
out = paths.PROCESSED / "graphs"
out.mkdir(parents=True, exist_ok=True)

edges = graphs.build_all(cfg)
rows = []
for name, e in edges.items():
    e.to_csv(out / f"{name}.raw.tsv", sep="\t", index=False)
    rows.append({"graph": name, "raw_nodes": len(set(e.u) | set(e.v)), "raw_edges": len(e)})

raw_rows = {r["graph"]: r for r in rows}
rw = cfg["rewire"]
rows = []
for uname, members in cfg["universes"].items():
    genes = graphs.universe({n: edges[n] for n in members})
    udir = out / uname
    udir.mkdir(exist_ok=True)
    (udir / "genes.txt").write_text("\n".join(genes) + "\n")
    for name in members:
        ei = graphs.to_index(graphs.induce(edges[name], genes), genes)
        np.save(udir / f"{name}.real.npy", ei)
        deg = np.bincount(ei.ravel(), minlength=len(genes))
        row = dict(raw_rows[name], universe=uname, n_genes=len(genes), univ_edges=len(ei),
                   isolated=int((deg == 0).sum()), mean_deg=deg.mean(), max_deg=int(deg.max()))
        orig = set(map(tuple, ei))
        for k in range(rw["n_copies"]):
            seed = rw["seed"] + 1000 * k + zlib.crc32(f"{uname}/{name}".encode()) % 997
            r = graphs.rewire(ei, len(genes), rw["swaps_per_edge"], seed=seed)
            np.save(udir / f"{name}.rw{k}.npy", r)
            # fraction of original edges that survived the swaps
            row[f"rw{k}_overlap"] = sum(tuple(t) in orig for t in r) / len(r)
        rows.append(row)

# Pathway-coherent subgraphs: same nodes, only edges inside a shared specific pathway.
pw = features.pathways(min_genes=5, max_genes=cfg["pathway_subgraph"]["max_pathway_genes"])
for row in rows:
    udir = out / row["universe"]
    genes = (udir / "genes.txt").read_text().split()
    member = sp.csr_matrix(pw.reindex(genes, fill_value=0).values)
    ei = np.load(udir / f"{row['graph']}.real.npy")
    sub = graphs.pathway_subgraph(ei, member)
    np.save(udir / f"{row['graph']}.pwsub.npy", sub)
    deg = np.bincount(sub.ravel(), minlength=len(genes))
    row.update(pwsub_edges=len(sub), pwsub_edge_frac=len(sub) / max(len(ei), 1),
               pwsub_isolated=int((deg == 0).sum()))

summary = pd.DataFrame(rows)
summary.to_csv(out / "summary.tsv", sep="\t", index=False)
print(summary.round(3).to_string(index=False))
