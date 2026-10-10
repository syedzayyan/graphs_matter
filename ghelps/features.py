"""Node features (GO, Pfam, Reactome pathway membership, OT tractability) and PubMed counts.

All builders return a DataFrame indexed by ENSG over the protein-coding genes they cover;
alignment to a universe happens in scripts/build_features.py.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from ghelps import ids

# GO evidence codes that are themselves interaction data: using them would feed the PPI
# graph back in through the features.
GO_EXCLUDE_EVIDENCE = {"IPI", "ND"}
GO_EXCLUDE_TERMS = {"GO:0005515"}  # "protein binding" -- a PPI annotation in disguise
# Tractability buckets that encode clinical precedence, i.e. the label.
TRACT_EXCLUDE = {"Approved Drug", "Advanced Clinical", "Phase 1 Clinical"}


def _binary(pairs: pd.DataFrame, gene: str, term: str, min_genes: int, max_genes: int | None = None) -> pd.DataFrame:
    pairs = pairs[[gene, term]].dropna().drop_duplicates().reset_index(drop=True)
    sizes = pairs[term].value_counts()
    keep = sizes[(sizes >= min_genes) & ((sizes <= max_genes) if max_genes else True)].index
    pairs = pairs[pairs[term].isin(keep)]
    m = pd.crosstab(pairs[gene], pairs[term]).clip(upper=1).astype(np.uint8)
    m.index.name = "ensg"
    return m


def go(min_genes: int = 10) -> pd.DataFrame:
    cols = ["db", "acc", "sym", "qual", "go", "ref", "evidence"]
    df = pd.read_csv(ids.RAW / "goa_human.gaf.gz", sep="\t", comment="!", header=None,
                     usecols=range(7), names=cols, dtype=str)
    df = df[~df.evidence.isin(GO_EXCLUDE_EVIDENCE) & ~df.go.isin(GO_EXCLUDE_TERMS)
            & ~df.qual.str.startswith("NOT")]
    df["ensg"] = ids.map_uniprot(df.acc).fillna(ids.map_symbols(df.sym))
    return _binary(df, "ensg", "go", min_genes).add_prefix("go:")


def pfam(min_genes: int = 3) -> pd.DataFrame:
    df = pd.read_csv(ids.RAW / "uniprot_human.tsv.gz", sep="\t", dtype=str)
    df["ensg"] = ids.map_uniprot(df.Entry).fillna(ids.map_symbols(df["Gene Names (primary)"].fillna("")))
    df = df.assign(pfam=df.Pfam.str.rstrip(";").str.split(";")).explode("pfam")
    return _binary(df, "ensg", "pfam", min_genes).add_prefix("pfam:")


def pathways(min_genes: int = 5, max_genes: int = 500) -> pd.DataFrame:
    df = pd.read_csv(ids.RAW / "ensembl2reactome.txt", sep="\t", header=None,
                     names=["ens", "pw", "url", "name", "ev", "species"], usecols=[0, 1, 5])
    df = df[(df.species == "Homo sapiens") & df.ens.str.startswith("ENSG")]
    return _binary(df, "ens", "pw", min_genes, max_genes).rename_axis("ensg").add_prefix("rx:")


def tractability() -> pd.DataFrame:
    t = pd.read_parquet(ids.RAW / "ot" / "target_tractability")
    t = t[~t.category.isin(TRACT_EXCLUDE)]
    t["col"] = "tract:" + t.modality + ":" + t.category
    m = t.pivot_table(index="targetId", columns="col", values="value", aggfunc="max").fillna(False)
    m.index.name = "ensg"
    return m.astype(np.uint8)


def pubmed_counts() -> pd.Series:
    counts: dict[str, int] = {}
    with gzip.open(ids.RAW / "gene2pubmed.gz", "rt") as f:
        next(f)
        for line in f:
            if line.startswith("9606\t"):
                g = line.split("\t", 2)[1]
                counts[g] = counts.get(g, 0) + 1
    s = pd.Series(counts)
    s.index = s.index.map(ids.entrez_map())
    s = s[s.index.notna()].groupby(level=0).sum()
    s.index.name = "ensg"
    return s.rename("pubmed_count")


# ---------------------------------------------------------------------------------------
# Attention-free features: computed or measured the same way for every gene, independent of
# how much it has been studied (no GO, no pathways, no literature, no curated structures).

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9,
          M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)  # Kyte & Doolittle 1982
AA = "ACDEFGHIKLMNPQRSTVWY"


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std().replace(0, 1)
    return ((df - df.mean()) / sd).astype(np.float32)


def sequences() -> pd.Series:
    """ENSG -> canonical reviewed UniProt sequence (longest if a gene maps to several)."""
    seqs, acc, buf = {}, None, []
    with gzip.open(ids.RAW / "uniprot_human.fasta.gz", "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                acc, buf = line.split("|")[1], []
            else:
                buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    s = pd.Series(seqs)
    s.index = ids.map_uniprot(pd.Series(s.index)).values
    s = s[s.index.notna()]
    return s.groupby(level=0).agg(lambda x: max(x, key=len)).rename_axis("ensg")


def tm_helices(seq: str, window: int = 19, threshold: float = 1.6) -> tuple[int, float]:
    """Predicted transmembrane helices: runs of 19-residue windows whose mean Kyte-Doolittle
    hydropathy is >= 1.6 (the classic hydropathy-plot criterion). Returns (count, max window)."""
    h = np.array([KD.get(a, 0.0) for a in seq])
    if len(h) < window:
        return 0, float(h.mean()) if len(h) else 0.0
    w = np.convolve(h, np.ones(window) / window, mode="valid")
    above = w >= threshold
    runs = int(np.sum(above[1:] & ~above[:-1]) + above[0])
    return runs, float(w.max())


def sequence_features() -> pd.DataFrame:
    rows = {}
    for g, s in sequences().items():
        n = max(len(s), 1)
        tm, kd_max = tm_helices(s)
        comp = {f"aa_{a}": s.count(a) / n for a in AA}
        rows[g] = {"log_length": np.log(n), "tm_helices": tm, "has_tm": float(tm > 0), "kd_max": kd_max,
                   "frac_charged": sum(s.count(a) for a in "DEKR") / n,
                   "frac_hydrophobic": sum(s.count(a) for a in "AILMFVW") / n, **comp}
    return _zscore(pd.DataFrame.from_dict(rows, orient="index").rename_axis("ensg")).add_prefix("seq:")


def gtex_expression() -> pd.DataFrame:
    """log1p median TPM per GTEx tissue, z-scored per tissue."""
    df = pd.read_csv(ids.RAW / "gtex_median_tpm.gct.gz", sep="\t", skiprows=2)
    df["ensg"] = df.Name.str.split(".").str[0]
    x = df.drop(columns=["Name", "Description"]).groupby("ensg").max()
    return _zscore(np.log1p(x)).add_prefix("gtex:")


def esm_embeddings(model_name: str = "facebook/esm2_t12_35M_UR50D", batch_tokens: int = 8000,
                   max_len: int = 1022) -> pd.DataFrame:
    """Mean-pooled ESM-2 residue embeddings per gene (transformers EsmModel). Sequences longer
    than max_len are split into chunks whose embeddings are length-weighted averaged. Cached in
    $GHELPS_DATA/raw since it is universe-independent and the slowest feature to compute."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    from ghelps import device

    cache = ids.RAW / f"esm_{model_name.split('/')[-1]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    dev = device.get()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(dev).eval()
    seqs = sequences()
    chunks = [(g, s[i:i + max_len]) for g, s in seqs.items() for i in range(0, len(s), max_len)]
    chunks.sort(key=lambda c: len(c[1]))
    sums: dict[str, np.ndarray] = {}
    lens: dict[str, int] = {}
    i = 0
    with torch.no_grad():
        while i < len(chunks):
            j = i + 1  # chunks are sorted by length: grow the batch while its padded size fits
            while j < len(chunks) and (j - i + 1) * (len(chunks[j][1]) + 2) <= batch_tokens:
                j += 1
            batch = chunks[i:j]
            i = j
            enc = tok([s for _, s in batch], return_tensors="pt", padding=True).to(dev)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0  # drop <cls>; <eos> is dropped below
            mask[torch.arange(len(batch)), enc["attention_mask"].sum(1) - 1] = 0
            pooled = (h * mask[..., None]).sum(1).cpu().numpy()
            for (g, s), v in zip(batch, pooled):
                sums[g] = sums.get(g, 0) + v
                lens[g] = lens.get(g, 0) + len(s)
    emb = pd.DataFrame({g: sums[g] / lens[g] for g in sums}).T.astype(np.float32)
    emb.index.name = "ensg"
    emb.columns = [f"esm:{j}" for j in range(emb.shape[1])]
    emb = _zscore(emb)
    emb.to_parquet(cache)
    return emb
