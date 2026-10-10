"""Stage 1c: node features + PubMed counts aligned to each universe.

Output: data/processed/features/<universe>/{go,pfam,pathway,tract}.parquet (binary),
        {esm,seqfeat,gtex}.parquet (attention-free, z-scored), pubmed.parquet
Genes missing from a source get an all-zero row (= the mean for z-scored blocks) and
pubmed_count 0. The ESM-2 model is set in configs/data.yaml (attention_free.esm_model).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghelps import features, paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G = paths.PROCESSED / "graphs"
OUT = paths.PROCESSED / "features"

blocks = {
    "go": features.go(),
    "pfam": features.pfam(),
    "pathway": features.pathways(),
    "tract": features.tractability(),
}
pubmed = features.pubmed_counts()
af = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["attention_free"]
continuous = {
    "esm": features.esm_embeddings(af["esm_model"]),
    "seqfeat": features.sequence_features(),
    "gtex": features.gtex_expression(),
}

for udir in sorted(p for p in G.iterdir() if p.is_dir()):
    genes = udir.joinpath("genes.txt").read_text().split()
    out = OUT / udir.name
    out.mkdir(parents=True, exist_ok=True)
    for name, m in blocks.items():
        a = m.reindex(genes, fill_value=0)
        a = a.loc[:, a.sum() >= 2]  # drop columns that are constant on this universe
        a.to_parquet(out / f"{name}.parquet")
        print(f"{udir.name}/{name}: {a.shape[1]} cols, coverage {(a.sum(axis=1) > 0).mean():.1%}")
    for name, m in continuous.items():
        a = m.reindex(genes).astype(np.float32)
        coverage = a.notna().all(axis=1).mean()
        a = a.fillna(0.0)
        a = a.loc[:, a.std() > 0]
        a.to_parquet(out / f"{name}.parquet")
        print(f"{udir.name}/{name}: {a.shape[1]} cols, coverage {coverage:.1%}")
    pm = pubmed.reindex(genes, fill_value=0).to_frame()
    pm.to_parquet(out / "pubmed.parquet")
    print(f"{udir.name}/pubmed: median {pm.pubmed_count.median():.0f}, zero {(pm.pubmed_count == 0).mean():.1%}")
