"""Stage 2: freeze labels, splits and evaluation negatives -- the benchmark release.

Output: data/processed/splits/<universe>/<labelset>/
  labels.parquet                   gene, y
  communities.parquet              Leiden membership on the union graph (computed once)
  <split>/seed<s>.parquet          gene, fold, y
  <split>/seed<s>.neg.parquet      regime, pos, neg  (node indices; -1 = "all")
  summary.tsv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghelps import labels, paths, splits, structure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = paths.PROCESSED
cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
sc = cfg["splits"]

pos_sets = {"minikel": labels.minikel(cfg["labels"]["minikel_min_phase"])}
first_launch = labels.minikel_first_launch()
tdl = labels.pharos_tdl()

rows = []
for uname, members in cfg["universes"].items():
    genes = (P / "graphs" / uname / "genes.txt").read_text().split()
    n = len(genes)
    ei = np.concatenate([np.load(P / "graphs" / uname / f"{g}.real.npy") for g in members])
    a_union = structure.adjacency(np.unique(ei, axis=0), n)
    a_union.data[:] = 1
    pfam = pd.read_parquet(P / "features" / uname / "pfam.parquet").loc[genes]
    pw = pd.read_parquet(P / "features" / uname / "pathway.parquet").loc[genes]
    pw = pw.loc[:, pw.sum() <= sc["pathway_match_max_genes"]]  # specific pathways only

    comm = splits.leiden(a_union, sc["leiden_resolution"], sc["leiden_seed"])
    csz = np.bincount(comm)
    print(f"[{uname}] leiden: {len(csz)} communities, largest {csz.max()} ({csz.max() / n:.1%})")

    recent = np.array([first_launch.get(g, 0) >= sc["recent_from_year"] for g in genes])
    t = np.array([tdl.get(g, "") for g in genes])

    for lname, pos in pos_sets.items():
        out = P / "splits" / uname / lname
        out.mkdir(parents=True, exist_ok=True)
        y = np.array([g in pos for g in genes], dtype=np.int8)
        pd.DataFrame({"gene": genes, "y": y}).to_parquet(out / "labels.parquet")
        pd.DataFrame({"gene": genes, "community": comm}).to_parquet(out / "communities.parquet")
        chem = (t == "Tchem") & (y == 0)
        clin = (t == "Tclin") & (y == 0)

        for seed in range(sc["n_seeds"]):
            made = {
                "random": (splits.random_split(genes, y, seed), None),
                "pfam": (splits.pfam_split(genes, y, pfam, seed), None),
                "community": (splits.community_split(genes, y, comm, seed), None),
                "rcnt": (splits.held_out_positive_split(genes, y, recent & (y == 1), seed), None),
                # Tchem genes become test positives; Tclin-but-unlabelled genes are ambiguous
                # so they are neither training unlabelled nor test negatives.
                "pharos": (splits.held_out_positive_split(genes, y, chem, seed, exclude_unl=clin), clin),
            }
            for sname, (df, excl) in made.items():
                d = out / sname
                d.mkdir(exist_ok=True)
                df.to_parquet(d / f"seed{seed}.parquet")
                neg = splits.negatives(df, a_union, pfam.values, pw.values, sc["neg_per_pos"], seed, excl)
                neg.to_parquet(d / f"seed{seed}.neg.parquet")
                fc = df.groupby(["fold", "y"]).size()
                row = {"universe": uname, "labels": lname, "split": sname, "seed": seed,
                       **{f"{f}_{c}": int(fc.get((f, c), 0)) for f in ("train", "val", "test") for c in (1, 0)}}
                npos = row["test_1"]
                for r, g in neg[neg.regime != "random"].groupby("regime"):
                    row[f"neg_{r}_poscov"] = g.pos.nunique() / max(npos, 1)
                rows.append(row)
        print(f"[{uname}/{lname}] positives {y.sum()}, recent {int((recent & (y == 1)).sum())}, "
              f"Tchem-unlabelled {int(chem.sum())}")

summary = pd.DataFrame(rows)
summary.to_csv(P / "splits" / "summary.tsv", sep="\t", index=False)
print(summary[summary.seed == 0].round(2).to_string(index=False))
