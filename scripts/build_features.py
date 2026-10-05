"""Stage 1c: node features + PubMed counts aligned to each universe.

Output: data/processed/features/<universe>/{go,pfam,pathway,tract}.parquet, pubmed.parquet
Genes missing from a source get an all-zero row (and pubmed_count 0).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghelps import features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "data" / "processed" / "graphs"
OUT = ROOT / "data" / "processed" / "features"

blocks = {
    "go": features.go(),
    "pfam": features.pfam(),
    "pathway": features.pathways(),
    "tract": features.tractability(),
}
pubmed = features.pubmed_counts()

for udir in sorted(p for p in G.iterdir() if p.is_dir()):
    genes = udir.joinpath("genes.txt").read_text().split()
    out = OUT / udir.name
    out.mkdir(parents=True, exist_ok=True)
    for name, m in blocks.items():
        a = m.reindex(genes, fill_value=0)
        a = a.loc[:, a.sum() >= 2]  # drop columns that are constant on this universe
        a.to_parquet(out / f"{name}.parquet")
        print(f"{udir.name}/{name}: {a.shape[1]} cols, coverage {(a.sum(1) > 0).mean():.1%}")
    pm = pubmed.reindex(genes, fill_value=0).to_frame()
    pm.to_parquet(out / "pubmed.parquet")
    print(f"{udir.name}/pubmed: median {pm.pubmed_count.median():.0f}, zero {(pm.pubmed_count == 0).mean():.1%}")
