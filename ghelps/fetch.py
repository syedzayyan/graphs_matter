"""Download every raw source into $GHELPS_DATA/raw (default <repo>/data/raw). Each entry is (path, fetcher); files that
already exist are skipped, so this is safe to re-run and resumes after interruption."""
from __future__ import annotations

import gzip
import http.client
import io
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from ghelps import paths

ROOT = Path(__file__).resolve().parents[1]
RAW = paths.RAW

OT_RELEASE = "26.09"
OT = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OT_RELEASE}/output"
STRING = "https://stringdb-downloads.org/download"
MINIKEL = "https://raw.githubusercontent.com/ericminikel/genetic_support/main/data"
INTACT_HUMAN = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/species/human.zip"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
HURI = "https://interactome-atlas.org/data/HuRI.tsv"  # the www. host has a mismatched TLS cert
BIOGRID_2009 = ("https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-2.0.60/"
                "BIOGRID-ORGANISM-2.0.60.tab.zip")  # released 27 Dec 2009: the dated graph


def _retry(fn, what: str, attempts: int = 5):
    """Call fn() until it succeeds. Streaming endpoints (UniProt REST in particular) sometimes
    drop the connection mid-transfer; client errors (4xx except 429) are not retried."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (http.client.HTTPException, urllib.error.URLError, OSError) as e:
            client_error = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429
            if attempt == attempts or client_error:
                raise
            wait = 30 * 2 ** (attempt - 1)
            print(f"[fetch] {what}: attempt {attempt} failed ({e!r}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def _download(url: str, dest: Path) -> None:
    """Download to a .part file, then rename, so a dropped transfer never leaves a truncated file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ghelps"})

    def once():
        try:
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    _retry(once, dest.name)
    tmp.rename(dest)


def _pmid_years(dest: Path, step: int = 20_000, max_pmid: int = 45_000_000, batch: int = 200) -> None:
    """Anchor table PMID -> year the record entered PubMed (Entrez date; publication date if
    missing), for every `step`-th PMID. PMIDs are assigned on entry, so a monotone fit through
    these anchors dates any PMID to about a year (scripts/analysis.py). Back-indexed old papers
    get their (later) entry year, which can only undercount early study."""
    ids = list(range(step, max_pmid, step))
    rows = []
    for i in range(0, len(ids), batch):
        body = urllib.parse.urlencode({"db": "pubmed", "retmode": "json",
                                       "id": ",".join(map(str, ids[i:i + batch]))}).encode()
        req = urllib.request.Request(f"{EUTILS}/esummary.fcgi", data=body, headers={"User-Agent": "ghelps"})
        res = _retry(lambda: json.load(urllib.request.urlopen(req, timeout=300)), "pubmed esummary")["result"]
        for uid in res.get("uids", []):
            r = res[uid]
            date = next((h["date"] for h in r.get("history", []) if h.get("pubstatus") == "entrez"),
                        r.get("sortpubdate", ""))
            if date[:4].isdigit():
                rows.append((int(uid), int(date[:4])))
        time.sleep(0.4)  # NCBI allows 3 requests/s without an API key
    pd.DataFrame(rows, columns=["pmid", "year"]).to_csv(dest, sep="\t", index=False)


def _chembl_pages(endpoint: str, key: str, **params) -> list[dict]:
    """Every record of a ChEMBL REST list endpoint (follows page_meta.next)."""
    rows, nxt = [], f"/chembl/api/data/{endpoint}.json?" + urllib.parse.urlencode({"limit": 1000, **params})
    while nxt:
        req = urllib.request.Request("https://www.ebi.ac.uk" + nxt, headers={"User-Agent": "ghelps"})
        page = _retry(lambda: json.load(urllib.request.urlopen(req, timeout=300)), f"chembl {endpoint}")
        rows += page[key]
        nxt = page["page_meta"]["next"]
    return rows


def _chembl_approvals(dest: Path) -> None:
    """Approved drug -> human protein target -> UniProt, with the drug's first approval year
    (ChEMBL mechanisms of action). A target's earliest year is when it was first drugged;
    Minikel's year_launch is the launch of a specific indication, often decades later."""
    mech = pd.DataFrame(_chembl_pages("mechanism", "mechanisms", only="molecule_chembl_id,target_chembl_id"))
    mol = pd.DataFrame(_chembl_pages("molecule", "molecules", max_phase=4,
                                     only="molecule_chembl_id,first_approval"))
    mech = mech.dropna().merge(mol.dropna(), on="molecule_chembl_id")
    tids = sorted(mech.target_chembl_id.unique())
    targets = []
    for i in range(0, len(tids), 100):
        targets += _chembl_pages("target", "targets", target_chembl_id__in=",".join(tids[i:i + 100]),
                                 only="target_chembl_id,target_type,organism,target_components")
    # single proteins and complexes (every subunit is drugged); protein families / selectivity
    # groups would credit every member of a family for one drug
    comp = [(t["target_chembl_id"], c["accession"]) for t in targets
            if t["organism"] == "Homo sapiens" and t["target_type"] in ("SINGLE PROTEIN", "PROTEIN COMPLEX")
            for c in t["target_components"] if c.get("accession")]
    out = mech.merge(pd.DataFrame(comp, columns=["target_chembl_id", "uniprot"]), on="target_chembl_id")
    out["first_approval"] = out.first_approval.astype(int)
    out[["uniprot", "molecule_chembl_id", "target_chembl_id", "first_approval"]].drop_duplicates() \
        .to_csv(dest, sep="\t", index=False)


# Human genetic evidence (Open Targets) kept for the genetic label, with the threshold each
# source must pass. GWAS: fine-mapped credible sets whose locus-to-gene score is >= 0.8; gene
# burden: exome-wide significant (p < 2.5e-6); Mendelian sources: their own top confidence
# tiers; ClinVar (EVA): OT score >= 0.8 (pathogenic, reviewed).
GENETIC_SOURCES = {
    "evidence_gwas_credible_sets": lambda d: d.score >= 0.8,
    "evidence_gene_burden": lambda d: d.pValueMantissa * 10.0 ** d.pValueExponent < 2.5e-6,
    "evidence_eva": lambda d: d.score >= 0.8,
    "evidence_gene2phenotype": lambda d: d.confidence.isin(["definitive", "strong"]),
    "evidence_clingen": lambda d: d.confidence.isin(["Definitive", "Strong"]),
    "evidence_orphanet": lambda d: d.score >= 0,
    "evidence_genomics_england": lambda d: d.confidence == "green",
    "evidence_uniprot_variants": lambda d: d.confidence == "high",
    "evidence_uniprot_literature": lambda d: d.confidence == "high",
}


def _genetic_evidence(dest: Path) -> None:
    """Gene x source -> earliest year of qualifying human genetic evidence (publication date,
    else evidence date; NaN if undated). Streams each Open Targets dataset, keeps only the
    rows that pass GENETIC_SOURCES, then deletes the download."""
    rows = []
    for name, keep in GENETIC_SOURCES.items():
        tmp = dest.parent / f"{name}.part"
        shutil.rmtree(tmp, ignore_errors=True)
        _ot_dataset(name)(tmp)
        for f in sorted(tmp.glob("*.parquet")):
            d = pd.read_parquet(f)
            d = d[keep(d)]
            date = d.publicationDate if "publicationDate" in d else d.evidenceDate
            year = pd.to_numeric(date.fillna(d.get("evidenceDate", date)).astype(str).str[:4], errors="coerce")
            rows.append(pd.DataFrame({"ensg": d.targetId.values, "source": name.removeprefix("evidence_"),
                                      "year": year.values}))
        shutil.rmtree(tmp)
    out = pd.concat(rows).groupby(["ensg", "source"], dropna=False).year.min().reset_index()
    out.to_csv(dest, sep="\t", index=False)


# Annotation snapshots from just before the time-split cut-off (2010), for the "dated" feature
# set: what a forecaster in 2009 could have known. GOA human release 80 (2009-12-17); KEGG
# pathways as of 2009-09-16, from the Bioconductor 2.5 KEGG.db package (KEGG's own archive
# is not public).
GOA_2009 = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/HUMAN/gene_association.goa_human.80.gz"
KEGG_2009 = "https://bioconductor.org/packages/2.5/data/annotation/src/contrib/KEGG.db_2.3.5.tar.gz"


def _kegg_2009(dest: Path) -> None:
    """Human KEGG pathway -> Entrez gene pairs from the KEGG.db 2.3.5 SQLite database."""
    import sqlite3
    import tarfile
    tgz = dest.with_suffix(".tar.gz")
    _download(KEGG_2009, tgz)
    db = dest.with_suffix(".sqlite")
    with tarfile.open(tgz) as tf, open(db, "wb") as out:
        shutil.copyfileobj(tf.extractfile("KEGG.db/inst/extdata/KEGG.sqlite"), out)
    with sqlite3.connect(db) as con:
        pairs = pd.read_sql("SELECT pathway_id AS pathway, gene_or_orf_id AS entrez FROM pathway2gene "
                            "WHERE pathway_id LIKE 'hsa%'", con)
    pairs.to_csv(dest, sep="\t", index=False)
    tgz.unlink()
    db.unlink()


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


def _biogrid_2009(dest: Path) -> None:
    """BioGRID 2.0.60 (Dec 2009), human-human rows: Entrez ids A/B, experimental system, PubMed id."""
    z = dest.parent / "biogrid_2.0.60_organism.zip"
    if not z.exists():
        _download(BIOGRID_2009, z)
    tmp = dest.with_suffix(".part")
    with zipfile.ZipFile(z) as zf, zf.open("BIOGRID-ORGANISM-Homo_sapiens-2.0.60.tab.txt") as fh, open(tmp, "w") as out:
        lines = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        for line in lines:  # free-text preamble, then the column header
            if line.startswith("INTERACTOR_A"):
                break
        out.write("entrez_a\tentrez_b\tsystem\tpmid\n")
        for line in lines:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 11 and c[9] == "9606" and c[10] == "9606":
                out.write("\t".join((c[0].removeprefix("ETG"), c[1].removeprefix("ETG"), c[6], c[8])) + "\n")
    tmp.rename(dest)
    z.unlink()


def _drugnomeai_top(dest: Path) -> None:
    """DrugnomeAI's published top-5% gene lists (Raies et al. 2022, Commun Biol,
    Supplementary Data 1; via Europe PMC): one row per gene, list name and rank."""
    z = dest.parent / "drugnomeai_supp.zip"
    if not z.exists():
        _download(f"{EUROPEPMC}/PMC9700683/supplementaryFiles", z)
    with zipfile.ZipFile(z) as zf, zf.open("42003_2022_4245_MOESM4_ESM.xlsx") as fh:
        x = pd.read_excel(io.BytesIO(fh.read()), header=None, skiprows=4)
    names = ["DrugnomeAI-Tclin", "DrugnomeAI-Tier1"]
    rows = [(n, i + 1, str(g).replace("\ufeff", "").strip()) for j, n in enumerate(names)
            for i, g in enumerate(x[j].dropna())]
    pd.DataFrame(rows, columns=["list", "rank", "symbol"]).to_csv(dest, sep="\t", index=False)
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
    "biogrid_2.0.60_human.tsv": _biogrid_2009,  # dated graph (BioGRID as of Dec 2009)
    "reactome_fi.zip": url("https://reactome.org/download/tools/ReactomeFIs/FIsInGene_04142025_with_annotations.txt.zip"),
    "hgnc.tsv": url("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"),
    "gene2pubmed.gz": url("https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz"),
    "pmid_years.tsv": _pmid_years,  # dates gene2pubmed papers (study-before-launch analysis)
    "chembl_approvals.tsv": _chembl_approvals,  # dates when each target was first drugged (time split)
    "genetic_evidence.tsv": _genetic_evidence,  # human genetic evidence per gene (genetic label)
    "goa_human.gaf.gz": url("https://current.geneontology.org/annotations/goa_human.gaf.gz"),
    "goa_human_2009.gaf.gz": url(GOA_2009),  # dated features (GO as of Dec 2009)
    "kegg_2009.tsv": _kegg_2009,               # dated features (KEGG pathways as of Sep 2009)
    "ensembl2reactome.txt": url("https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt"),
    "uniprot_human.tsv.gz": url("https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true"
                                "&fields=accession,gene_primary,xref_pfam,xref_ensembl&format=tsv&compressed=true"),
    "ot/target_tractability": _ot_dataset("target_tractability"),
    # attention-free feature sources: measured / computed uniformly for every gene
    "uniprot_human.fasta.gz": url("https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true"
                                  "&format=fasta&compressed=true"),
    "gtex_median_tpm.gct.gz": url("https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/"
                                  "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz"),
    "pharos_targets.tsv": _pharos,
    "external/drugnomeai_top5pct.tsv": _drugnomeai_top,  # published top lists (report 5.3)
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
