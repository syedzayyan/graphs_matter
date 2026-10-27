# Time-split drug-target benchmark

A benchmark for gene-level drug-target prioritisation that separates *finding new targets* from *rediscovering the field's portfolio*. It accompanies the paper's report (`summary.md`). The methods are in `methods.md` in the code repository.

## The task

- **Train** on what was known before **2010**: genes first drugged before then (Minikel et al. ≥ Phase III, dated by ChEMBL first approval), or any other label dated before the cut-off. One alternative is supplied: genes with human genetic evidence published before 2010.
- **Test** on genes first drugged from 2010 on (the *future targets*), against the test genes that were never drugged.
- **Score** how many future targets a ranking finds, broken down by how novel they were in 2009. A ranking of the most-studied genes finds many future targets, because most of them were already well studied. The endpoints that matter are the novel strata, compared against that ranker.

## Files (one folder per gene universe: `string`, `reactome_fi`, `huri`)

| file | contents |
|---|---|
| `genes.tsv` | One row per gene. Identifiers; PubMed count before the cut-off and now, plus its percentile; Minikel best phase; `programme_status` (never / success = ≥ Phase III / failure = best Phase I–II with no active programme / censored); first approval year; genetic evidence before the cut-off and at any date; Pharos TDL; number of pre-2010 targets sharing a Pfam domain; hops to the nearest pre-2010 target in the universe's graph. |
| `splits/seed<k>.tsv` | The time split for seeds 0–4: `fold` (train / val / test / excluded), `y_minikel`, `y_genetic`. The test genes and their labels are the same for both training labels. The test negatives are drawn separately for each seed. |
| `future_targets.tsv` | The test positives with their novelty strata. Study before the cut-off (< 0.5 / 0.5–0.9 / ≥ 0.9 percentile); Pfam family already drugged or new; 1 / 2 / 3+ hops from a pre-2010 target; `novel` (all of: not top-decile study, new family, ≥ 2 hops); `novel (2 of 3)` (any two). Also the modality of the first approved drug (small molecule / biologic) and oncology vs not. |
| `edges/<graph>.tsv` | The universe's graph as ENSG pairs. `string/edges/biogrid09.tsv` is BioGRID as of December 2009, a graph without post-2010 knowledge. `string_sub09.tsv` is its density-matched STRING control. |

## Rules for a fair entry

1. **Train only on genes whose fold is `train` or `val`** in that seed's split. Never use the test genes' labels.
2. **Features and graphs must be dated before 2010, or flagged as leaking.** Current literature-derived features (GO, curated pathways, tractability, STRING text mining) know which genes were later drugged. Report results with dated inputs, or say plainly that the inputs leak.
3. **Report the novel strata and the PubMed comparison**, not only overall AUROC.

## Install

```
pip install .            # in the release folder, or in benchmark/ of the code repository
pip install ".[pu]"      # adds the bagging-PU baseline (pulearn)
```

Only numpy, pandas and scipy are needed to score.

## Scoring

```
targetsplit score --release . --universe string --scores my_scores.tsv --out my_report.tsv
```

`my_scores.tsv` has the columns `gene` (or `ensg`) and `score`, and optionally `seed` when the model is retrained per seed. Each novelty metric comes with a 95% bootstrap CI over the future targets (2,000 draws) and a paired difference from the PubMed-before-2010 ranker. The family and closeness rankers are scored alongside. The scorer reports:

- **AUROC and recall in the top 10%** per stratum of `future_targets.tsv`. The printout shows all future targets, `novel (2 of 3)`, `novel` and `new family`; `--out` has every stratum, including first-drug modality and oncology.
- **a top-100 audit:** share that are future targets, that are novel future targets, top-decile studied, already in a programme, in a drugged family, one hop from a pre-2010 target, or with genetic evidence
- **selection and outcome AUROC** among the test genes: tried vs never tried, and succeeded vs failed once tried

`python evaluate.py ...` still works with the same arguments.

From Python:

```python
from targetsplit import Release
rel = Release(".", "string")
for seed in rel.seeds:
    idx, y = rel.train(seed, label="minikel")   # fit only on these genes
    ...                                          # your model -> one score per gene in rel.G
report = rel.score(my_scores)                    # array, Series by gene, or table with seed
```

## A baseline on your features

```
targetsplit baseline --release . --universe string --features X.tsv --out scores.tsv --model pubag
```

`X.tsv` has `gene` and one column per feature. `rf` fits positives against unlabelled genes, as the paper's RF does. `pubag` is bagging PU (Mordelet & Vert 2013, via `pulearn`): each tree sees every positive and a bootstrap of as many unlabelled genes.

## Your own genes and label

```
targetsplit build --genes my_genes.tsv --cutoff 2015 --out my_release --name mine \
                  --domains pfam.tsv --edges biogrid_2014.tsv
targetsplit score --release my_release --universe mine --scores my_scores.tsv
```

`my_genes.tsv` has `gene`, `first_year` (when it first became a positive, empty if never) and `pubmed_before_cutoff`. Optional columns:
- `positive`: undated positives are excluded
- `label_<name>`: another training label, dated before the cut-off
- `programme_status`: never / success / failure / censored
- `flag_<name>`: any 0/1 property to audit in the top 100

`--domains` (gene, domain) defines "drugged family" and `--edges` (a, b) defines hops; without one, that criterion is dropped from the novelty strata. Prefer a graph dated before the cut-off. `targetsplit build --help` has the details.

## Known limits

- **The novel strata are small.** STRING / Reactome FI / HuRI have 44 / 48 / 35 `novel (2 of 3)` targets, so the CIs are wide.
- **The novel strata are partly defined by low 2009 study.** That handicaps the PubMed ranker by construction, so beating it there is a low bar.
- **Future-target labels are development decisions.** A ranking that points away from the portfolio is scored as wrong, whether or not its genes are good targets. The `outcome` AUROC, among tried genes, is the closest thing here to a test of target quality.
- **Approvals without a ChEMBL mechanism can't be dated.** Those genes are `excluded`.

## Sources and licences

| source | release | licence |
|---|---|---|
| Minikel et al. 2024 (*Nature*), `ericminikel/genetic_support` | GitHub data | see repository |
| ChEMBL (first approvals, molecule types) | REST API | CC BY-SA 3.0 |
| Open Targets Platform (genetic evidence) | 26.09 | CC0 1.0 |
| NCBI gene2pubmed, PubMed | current | public domain (NLM terms) |
| UniProt / Pfam | current | CC BY 4.0 |
| Pharos (TDL) | GraphQL | CC BY-SA 4.0 |
| HGNC | complete set | CC0 |
| STRING | v12.0 | CC BY 4.0 |
| Reactome FI | 2025-04-14 | CC BY 4.0 |
| HuRI | interactome-atlas.org | CC BY 4.0 |
| BioGRID | 2.0.60 (27 Dec 2009) | MIT |

Check each licence before redistributing a derived file.
