"""Download every raw source into data/raw. Each entry is (path, fetcher); files that
already exist are skipped, so this is safe to re-run and resumes after interruption."""
from __future__ import annotations

import gzip
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

OT_RELEASE = "26.09"
OT = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OT_RELEASE}/output"
STRING = "https://stringdb-downloads.org/download"
MINIKEL = "https://raw.githubusercontent.com/ericminikel/genetic_support/main/data"
INTACT_HUMAN = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/species/human.zip"
HURI = "https://interactome-atlas.org/data/HuRI.tsv"  # the www. host has a mismatched TLS cert


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ghelps"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def url(u: str):
    return lambda dest: _download(u, dest)


def _intact_slim(dest: Path) -> None:
    """IntAct human MITAB -> human-human rows, only the columns the graph builder uses:
    ids A/B, detection method, interaction type, confidence. Streams out of the zip."""
    z = dest.parent / "intact_human.zip"
    if not z.exists():
        _download(INTACT_HUMAN, z)
    tmp = dest.with_suffix(".part")
    with zipfile.ZipFile(z) as zf, zf.open(zf.namelist()[0]) as fh, open(tmp, "w") as out:
        next(fh)
        for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            c = line.rstrip("\n").split("\t")
            if "taxid:9606(" in c[9] and "taxid:9606(" in c[10]:
                out.write("\t".join((c[0], c[1], c[6], c[11], c[14])) + "\n")
    tmp.rename(dest)
    z.unlink()


def _ot_dataset(name: str):
    def fetch(dest: Path) -> None:
        listing = urllib.request.urlopen(f"{OT}/{name}/", timeout=120).read().decode()
        files = sorted({p.split('"')[1] for p in listing.split("href=")[1:] if p.split('"')[1].endswith(".parquet")})
        tmp = dest.with_name(dest.name + ".part")
        tmp.mkdir(parents=True, exist_ok=True)
        for f in files:
            if not (tmp / f).exists():
                _download(f"{OT}/{name}/{f}", tmp / f)
        tmp.rename(dest)
    return fetch


def _pharos(dest: Path) -> None:
    """Target development levels for every Pharos target (GraphQL bulk `download`)."""
    # sqlOnly:false is required, otherwise the API answers result=true with data=null
    q = {"query": '{ download(model:"Targets", fields:["UniProt","Symbol","Target Development Level"], '
                  "sqlOnly:false) { result data errorDetails } }"}
    req = urllib.request.Request("https://pharos-api.ncats.io/graphql", data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))["data"]["download"]
    if not d["result"] or not d["data"]:
        raise RuntimeError(f"Pharos download failed: {d['errorDetails']}")
    df = pd.DataFrame(d["data"])[["UniProt", "Symbol", "Target Development Level"]]
    df.columns = ["uniprot", "sym", "tdl"]
    df.to_csv(dest, sep="\t", index=False)


SOURCES: dict[str, callable] = {
    "9606.protein.links.full.v12.0.txt.gz": url(f"{STRING}/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz"),
    "string_aliases.txt.gz": url(f"{STRING}/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"),
    "intact_human_slim.tsv": _intact_slim,
    "HuRI.tsv": url(HURI),
    "reactome_fi.zip": url("https://reactome.org/download/tools/ReactomeFIs/FIsInGene_04142025_with_annotations.txt.zip"),
    "hgnc.tsv": url("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"),
    "gene2pubmed.gz": url("https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz"),
    "goa_human.gaf.gz": url("https://current.geneontology.org/annotations/goa_human.gaf.gz"),
    "ensembl2reactome.txt": url("https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt"),
    "uniprot_human.tsv.gz": url("https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true"
                                "&fields=accession,gene_primary,xref_pfam,xref_ensembl&format=tsv&compressed=true"),
    "ot/target_tractability": _ot_dataset("target_tractability"),
    "pharos_targets.tsv": _pharos,
    **{f"minikel/{f}": url(f"{MINIKEL}/{f}") for f in ("pp.tsv", "areas.tsv", "indic.tsv", "universe.tsv")},
}


def missing() -> list[str]:
    return [k for k in SOURCES if not (RAW / k).exists()]


def fetch_all() -> None:
    for name in missing():
        dest = RAW / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {name}", flush=True)
        SOURCES[name](dest)
    # sanity: gzip files must be readable
    for name in SOURCES:
        if name.endswith(".gz"):
            with gzip.open(RAW / name) as f:
                f.read(1)
