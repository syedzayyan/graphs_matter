#!/bin/sh
# Fetch every raw source not already present under ../../collate_data/data.
set -eu
cd "$(dirname "$0")/../data/raw"
get() { [ -s "$2" ] || curl -fsSL --retry 3 -o "$2" "$1"; echo "ok $2"; }

get https://reactome.org/download/tools/ReactomeFIs/FIsInGene_04142025_with_annotations.txt.zip reactome_fi.zip
get https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt hgnc.tsv
get https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz gene2pubmed.gz
get https://current.geneontology.org/annotations/goa_human.gaf.gz goa_human.gaf.gz
get https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt ensembl2reactome.txt
get "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&fields=accession,gene_primary,xref_pfam,xref_ensembl&format=tsv&compressed=true" uniprot_human.tsv.gz
mkdir -p minikel
for f in pp.tsv areas.tsv indic.tsv universe.tsv drug_phase_summary.tsv; do
  get https://raw.githubusercontent.com/ericminikel/genetic_support/main/data/$f minikel/$f
done
