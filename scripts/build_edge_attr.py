"""Stage 1d: per-edge feature vectors + the extra graph conditions.

Edge vector (12-d, identical definition for every graph):
  7 STRING v12 channel scores for the pair / 1000 (0 if STRING has no evidence):
    neighborhood, fusion, cooccurence, coexpression, experiments, database, textmining
  5 bits: is the pair an edge in string / string_exp / intact / reactome_fi / huri (raw lists)

Graph conditions written next to the real/rewired edge lists in data/processed/graphs/<u>/:
  <g>.empty.npy       no edges (same model, graph removed)
Edge attributes in data/processed/edge_attr/<u>/<g>.<copy>.npy, row-aligned with the edges:
  real      true vectors
  rw{k}     the real vectors randomly permuted onto the rewired edges (attribute
            distribution kept, attachment to topology destroyed)
  shufattr  real topology, real vectors permuted across its edges
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghelps import graphs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
CHANNELS = ["neighborhood", "fusion", "cooccurence", "coexpression", "experiments", "database", "textmining"]
cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))

links = graphs._string_links_full(cfg, CHANNELS)
raw = {g: pd.read_csv(P / "graphs" / f"{g}.raw.tsv", sep="\t") for g in graphs.GRAPHS}

for uname, members in cfg["universes"].items():
    genes = (P / "graphs" / uname / "genes.txt").read_text().split()
    n = len(genes)
    pos = {g: i for i, g in enumerate(genes)}

    def keys(u, v):
        u, v = np.minimum(u, v), np.maximum(u, v)
        return u.astype(np.int64) * n + v

    s = links[links.g1.isin(pos) & links.g2.isin(pos)]
    sk = keys(s.g1.map(pos).values, s.g2.map(pos).values)
    string_attr = pd.DataFrame(s[CHANNELS].values / 1000.0, index=sk)
    string_attr = string_attr[~string_attr.index.duplicated()]
    member = {}
    for g, e in raw.items():
        e = e[e.u.isin(pos) & e.v.isin(pos)]
        member[g] = set(keys(e.u.map(pos).values, e.v.map(pos).values).tolist())

    out = P / "edge_attr" / uname
    out.mkdir(parents=True, exist_ok=True)
    for g in members:
        rng = np.random.default_rng(0)
        ei = np.load(P / "graphs" / uname / f"{g}.real.npy")
        k = keys(ei[:, 0], ei[:, 1])
        attr = np.hstack([
            string_attr.reindex(k).fillna(0).values,
            np.stack([np.isin(k, list(member[h])) for h in graphs.GRAPHS], axis=1),
        ]).astype(np.float32)
        np.save(out / f"{g}.real.npy", attr)
        np.save(out / f"{g}.shufattr.npy", attr[rng.permutation(len(attr))])
        for rw in sorted((P / "graphs" / uname).glob(f"{g}.rw*.npy")):
            m = len(np.load(rw))
            assert m == len(attr)
            np.save(out / f"{g}.{rw.stem.split('.')[-1]}.npy", attr[rng.permutation(m)])
        np.save(P / "graphs" / uname / f"{g}.empty.npy", np.zeros((0, 2), dtype=np.int64))
        cov = (attr[:, :7].sum(1) > 0).mean()
        print(f"{uname}/{g}: {len(attr)} edges, STRING evidence on {cov:.1%}, "
              f"mean channels {np.round(attr[:, :7].mean(0), 3).tolist()}")
