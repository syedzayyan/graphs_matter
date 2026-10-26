# Methods

This file gives the full details behind `summary.md`. Every value here is taken from the code and configs in this repository: `configs/*.yaml`, `ghelps/` and `scripts/`. When a choice was made to fix a specific failure, the reason is given.

---

## 1. Overview

**Question.** Does a gene–gene graph help a graph neural network (GNN) predict drug targets? Drug targets here are genes that reached Phase III or later in Minikel et al.'s drug-development data. A follow-up question is what such models actually learn when trained on that label.

**Design:**
- **Graphs:** three, each with its own gene universe, positives, splits and tuned hyperparameters. They span literature-derived (STRING), curated (Reactome FI) and systematic, literature-free (HuRI) interaction data.
- **Graph conditions:** every graph model is run with the real graph and with several control graphs (rewired, empty, pathway-only, shuffled edge vectors), so "does the graph help?" is answered within the same model.
- **Experiments:** six. Together they cross graphs, models, losses, feature sets, splits and label sets.

| experiment | runs | what varies |
|---|---|---|
| `main` | 22,440 | 3 graphs × 7 graph conditions × 3 feature sets × 2 splits (random, Pfam) × 16 models × 4 losses × 5 seeds |
| `waves` | 5,580 | therapy-area transfer splits, 2 feature sets, 2 losses |
| `membrane` | 3,180 | positive control: membrane label, Pfam + pathway features |
| `nonsense_labels` | 2,160 | labels shuffled across all genes, or within Pfam families |
| `nonsense_feats` | 780 | node features replaced by Gaussian noise |
| `novelty` | 7,290 | time split; Minikel vs genetic label; 3 feature sets incl. `dated`; 3 losses |
| **total** | **41,430** | |

Not every axis combination exists. For example, feature sets don't apply to feature-free models, and graph-free models have no graph condition (§6.1). The run counts above are the combinations that actually run.

**Pipeline:**
1. `main.py build`: fetch raw sources, then build graphs, structure statistics, edge attributes, features and splits.
2. `tune.py`: Optuna tuning.
3. `main.py run`: experiment grids.
4. `scripts/summarise.py`, `scripts/analysis.py`, `scripts/delong.py`, `scripts/figures.py`, `scripts/report.py`.

On the cluster, `scripts/hpc/submit_all.sh` chains these as Slurm jobs with `afterok` dependencies. Run and tune jobs each use one GPU, 8 CPUs and a 2 h limit.

---

## 2. Data sources

All raw files are downloaded by `ghelps/fetch.py` into `$GHELPS_DATA/raw`, which is on scratch on the cluster.

| source | file / version | used for |
|---|---|---|
| STRING v12.0 | `9606.protein.links.full.v12.0`, protein aliases | STRING graph, STRING-experimental graph, edge vectors |
| IntAct | current human MITAB | edge-vector membership bit |
| Reactome FI | `FIsInGene_04142025_with_annotations` | Reactome FI graph |
| HuRI | `HuRI.tsv` (interactome-atlas.org) | HuRI graph |
| HGNC | complete set | symbol → Ensembl mapping |
| NCBI gene2pubmed | current | PubMed counts per gene (study intensity) |
| PubMed E-utilities | every 20,000th PMID's Entrez date | dating papers (PMID → year) |
| ChEMBL REST | mechanisms, molecules with `max_phase=4`, targets | first approval year per target (time split) |
| Open Targets Platform 26.09 | tractability; 9 genetic-evidence datasets | tractability features; genetic label |
| GO annotations (GOA human) | current `goa_human.gaf.gz` | GO features, membrane label |
| GOA human release 80 | 2009-12-17, GAF 1.0 | `dated` GO features |
| KEGG via Bioconductor 2.5 `KEGG.db` 2.3.5 | KEGG as of 2009-09-16 | `dated` pathway features |
| Reactome | `Ensembl2Reactome_All_Levels` (current) | pathway features, pathway subgraph, pathway negatives |
| UniProt (reviewed human) | TSV (accession, gene, Pfam, Ensembl), FASTA | Pfam features, sequences |
| GTEx v10 | gene median TPM | expression features |
| ESM-2 | `facebook/esm2_t12_35M_UR50D` | protein embeddings |
| Pharos | target development levels (GraphQL) | built but unused in reported experiments |
| Minikel et al. | `pp.tsv`, `indic.tsv`, `areas.tsv`, `universe.tsv` (GitHub `ericminikel/genetic_support`) | drug-target label, phases, indications, therapy areas |

**Download handling.** Downloads go through a `.part` file that is renamed when complete, and are retried with exponential backoff (5 attempts, starting at 30 s). Gzip files are test-read after fetching.

**Identifier mapping** (`ghelps/ids.py`):
- Everything is mapped to Ensembl gene IDs (ENSG) and restricted to protein-coding genes (HGNC).
- UniProt accessions are mapped with isoform suffixes stripped. Entrez IDs and symbols are mapped through HGNC.

---

## 3. Graphs and gene universes

### 3.1 Edge lists (`ghelps/graphs.py`)

All graphs are undirected, deduplicated, without self-loops, and restricted to protein-coding genes on both ends.

| graph | edges kept |
|---|---|
| **STRING** | combined score ≥ 700 ("high confidence"). STRING proteins mapped to ENSG through the `Ensembl_gene` aliases. |
| **STRING-experimental** | experiments channel ≥ 400. Used only for an edge-vector bit. |
| **IntAct** | MI-score ≥ 0.45 and an interaction type in a list of physical/enzymatic types (association, physical association, direct interaction, phosphorylation, ubiquitination, cleavage, …). Used only for an edge-vector bit. |
| **Reactome FI** | pairs annotated only as "predicted" (Reactome's naive-Bayes predictions) dropped; pairs with curated support kept. |
| **HuRI** | all pairs (systematic yeast two-hybrid). |

### 3.2 Universes

Each experiment universe is one graph's node set: the protein-coding genes with at least one edge in that graph. Node indices are fixed by `genes.txt`.

| universe | genes | edges | mean degree | positives (≥ Phase III) | share of the 702 Minikel positives |
|---|---|---|---|---|---|
| STRING | 15,886 | 231,803 | 29.2 | 698 | 99% |
| Reactome FI | 9,749 | 178,420 | 36.6 | 630 | 90% |
| HuRI | 8,156 | 51,335 | 12.6 | 223 | 32% |

### 3.3 Graph conditions

| condition | definition |
|---|---|
| `real` | the graph. |
| `rw0`–`rw2` | degree-preserving rewiring by double-edge swaps (igraph `rewire`, `mode="simple"`), 10 swaps per edge. Seed: `1000·k + crc32(universe/graph) mod 997`. The degree sequence is asserted unchanged. |
| `empty` | no edges. The same model with the graph removed. |
| `pwsub` | pathway-coherent subgraph: only edges whose endpoints share a specific Reactome pathway (5–100 genes). All nodes are kept. |
| `shufattr` | real topology, with edge vectors permuted across edges. Applies only to the edge-aware GAT. |

Each condition gets its own tuned hyperparameters (§7). Reusing the real graph's parameters on the controls collapsed them; for example, GAT on a rewired graph went from 0.83 to 0.54 AUROC. Without per-condition tuning, the real-vs-control comparison would be biased towards the real graph.

### 3.4 Edge vectors (`scripts/build_edge_attr.py`)

Each edge carries a 12-dimensional vector, defined the same way for every graph:
- **7 STRING v12 channel scores ÷ 1000:** neighborhood, fusion, co-occurrence, co-expression, experiments, database, text mining. A pair STRING has no evidence for gets 0.
- **5 membership bits:** whether the pair is an edge in STRING, STRING-experimental, IntAct, Reactome FI and HuRI.

For rewired copies, the real vectors are randomly permuted onto the rewired edges. This keeps their distribution but removes their attachment to the topology.

### 3.5 Structural statistics (`ghelps/structure.py`)

The 13 columns, z-scored, are:
- log degree
- three propagated-ones features Â^k·1 for k = 1–3, where Â = D^−½(A + I)D^−½
- mean, sd, max, min and median neighbour degree
- k-core number, PageRank, local clustering coefficient, normalised betweenness

Neighbour-degree summaries, k-core, PageRank and betweenness are log1p-transformed first. PageRank and betweenness are scaled by 10⁴ before the log.

---

## 4. Labels

**Minikel ≥ Phase III (main label):**
- A gene is positive if its best target–indication pair reached Phase III or later. "Best" uses `ccatnum`, the maximum of the historical and active phase.
- Everything else is unlabelled (PU setting), not negative.

**Label controls:**
- **`minikel_perm`:** Minikel labels permuted across all genes (seed 0).
- **`minikel_famperm`:** labels permuted only within Pfam families. Each gene is assigned to its largest Pfam family, and genes without Pfam form one group. This keeps how many positives each family has, but not which member is positive.
- **`membrane` (positive control):** GO:0016020 (membrane) or GO:0005886 (plasma membrane), any evidence code, excluding NOT-qualified annotations.

**Genetic evidence label (`genetic`, time split only):**
- A gene is positive if it has human genetic evidence for any disease in Open Targets, from at least one of the qualifying sources below.

  | source | threshold |
  |---|---|
  | fine-mapped GWAS credible sets | locus-to-gene ≥ 0.8 |
  | gene burden | p < 2.5 × 10⁻⁶ |
  | ClinVar (EVA) | score ≥ 0.8 |
  | Gene2Phenotype | confidence definitive or strong |
  | ClinGen | Definitive or Strong |
  | Orphanet | any |
  | Genomics England | green |
  | UniProt variants and literature | high confidence |

- Each gene gets the earliest year of qualifying evidence (publication date, else evidence date). Undated evidence is dropped.
- For the time split only evidence from before 2010 counts: 2,965 / 2,212 / 1,216 genes on STRING / Reactome FI / HuRI.

**First drugged year** (`labels.target_first_approval`):
- ChEMBL mechanisms of action are joined to approved molecules (`max_phase = 4`, with `first_approval`) and then to human targets of type SINGLE PROTEIN or PROTEIN COMPLEX. Every subunit of a complex counts as drugged.
- Protein families and selectivity groups are excluded: they would credit every family member for one drug.
- A gene's year is its earliest approval. Minikel's `year_launch` is not used, because it dates one indication's launch, often decades after the target was first drugged.

---

## 5. Node features

### 5.1 Blocks

**Literature-derived blocks** (binary):

| block | content | exclusions and cut-offs |
|---|---|---|
| `go` | GO term membership | Evidence codes IPI and ND, the term "protein binding" (GO:0005515) and NOT-qualified annotations are excluded: they are interaction data and would feed the PPI graph back in. Terms with ≥ 10 genes. |
| `pfam` | Pfam domain membership (UniProt) | Families with ≥ 3 genes. |
| `pathway` | Reactome pathway membership (all levels, human) | Pathways with 5–500 genes. |
| `tract` | Open Targets tractability buckets, per modality and category | Buckets that encode clinical precedence (Approved Drug, Advanced Clinical, Phase 1 Clinical) are excluded, since they are the label. |

**Attention-free blocks** (continuous, z-scored). These are measured or computed the same way for every gene:

| block | content |
|---|---|
| `esm` | ESM-2 (35M) residue embeddings of the canonical reviewed UniProt sequence, mean-pooled without the CLS and EOS tokens. Sequences over 1,022 residues are chunked and length-weighted. If a gene maps to several sequences, the longest is used. |
| `seqfeat` | log length; 20 amino-acid frequencies; charged and hydrophobic fractions; predicted transmembrane helices (runs of 19-residue windows with mean Kyte–Doolittle hydropathy ≥ 1.6), a has-TM flag and the maximum window hydropathy. |
| `gtex` | log1p of the median TPM per GTEx tissue. |

**Dated blocks** (binary). These are as they stood in 2009:

| block | content |
|---|---|
| `go09` | GO from GOA release 80, same filters as `go`. |
| `kegg09` | human KEGG pathways from `KEGG.db` 2.3.5 (Entrez → ENSG), 5–500 genes. |

**Per-universe handling:**
- Genes missing from a binary block get an all-zero row. Genes missing from a continuous block get 0, which is the mean after z-scoring.
- Binary columns with fewer than 2 genes in the universe are dropped, and constant continuous columns are dropped.

### 5.2 Feature sets (`feats` axis)

| set | blocks | used in |
|---|---|---|
| `all` | go, pfam, pathway, tract | main, waves, novelty, label controls |
| `no_tract` | go, pfam, pathway | main |
| `attn_free` | esm, seqfeat, gtex | main, waves, novelty |
| `dated` | esm, seqfeat, gtex, pfam, go09, kegg09 | novelty (time split) |
| `no_loc` | pfam, pathway (no GO or tractability, both of which encode localisation) | membrane |
| `random` | Gaussian noise, N(0, 1), seed 0 | nonsense_feats |
| `all_resid`, `attn_free_resid` | residualised for study, as described below | §4.3 leak analysis only |

**Dimensionality reduction:**
- Each set's blocks are concatenated and reduced with TruncatedSVD to 256 components (seed 0, label-free), then z-scored.
- **`dated`:** two SVDs of 128 components each, one for the continuous blocks and one for the binary blocks, so the continuous blocks' larger variance doesn't swamp the annotations.

**Residualised sets:**
- Each SVD column is replaced by the normal score of its rank among genes in the same PubMed-count bin (50 equal-size bins, ties broken at random with seed 0).
- Each column then has the same distribution at every level of study.

**PubMed count:**
- A gene's number of gene2pubmed papers.
- **Dated counts** (study as known in a given year): papers are dated with an isotonic fit of PubMed entry year on PMID, through the anchor table; then they are counted cumulatively per year.
- Back-indexed old papers get their later entry year, so early study can only be undercounted.

---

## 6. Models

### 6.1 Model list (`ghelps/train.py: LADDER`)

| model | kind | inputs | graph conditions |
|---|---|---|---|
| `deg_lr` (degree LR) | logistic regression, sklearn, balanced, untuned | log degree | real, rw0–2, pwsub |
| `rf_struct` | random forest | structural statistics | real, rw0–2, pwsub |
| `labelprop` | label propagation (PyG) | labels only | real, rw0–2, pwsub |
| `mlp_feat` | MLP | features | none |
| `rf_feat` | random forest | features | none |
| `mlp_feat_struct` | MLP | features + structure | real, rw0–2, pwsub |
| `rf_feat_struct` | random forest | features + structure | real, rw0–2, pwsub |
| `sign_feat` (SIGN) | MLP on [X, ÂX, Â²X, Â³X] (PyG SIGN transform) | features | + empty |
| `cs_feat` (Correct & Smooth) | MLP base, then C&S (PyG, autoscale) | features | + empty |
| `gcn_none` | GCN | constant 1 | real, rw0–2, pwsub |
| `gcn_feat` | GCN | features | + empty |
| `sage_feat` | GraphSAGE (mean aggregation; separate self weight) | features | + empty |
| `gat_none` | GATv2, 4 heads by default | constant 1 | real, rw0–2, pwsub |
| `gat_feat` | GATv2 | features | + empty |
| `gat_edge_feat` | GATv2 with the 12-d edge vector in attention | features | + empty, shufattr |
| `gp` | additive-kernel Gaussian process | per-block features + graph blocks | real, rw0–2, pwsub |

**Notes on specific models:**
- **MLP with 1 layer** is plain logistic regression, and the tuner can choose it.
- **Structural models and the empty graph:** the empty graph and shuffled edge vectors are meaningless for structural models, so they aren't run.
- **Feature sets** are not crossed with feature-free models.

### 6.2 Additive GP (`ghelps/gp.py`)

**Kernel:** k(x, x′) = Σ_b ScaleKernel(RBF on block b). The blocks are:
- each feature block of the set, separately SVD-reduced to 32 components
- z-scored log degree
- 16 Laplacian eigenvectors (PyG `AddLaplacianEigenvectorPE`)
- 2-hop neighbour features: the last block of SIGN(own features, 2 hops), i.e. Â²X, z-scored

**Fitting:**
- Exact GP in float64, Gaussian likelihood on the 0/1 labels, constant mean, Adam (lr 0.05) for 150 iterations on the marginal likelihood.
- **Noise floor:** the noise is floored at a tunable fraction of the label variance. Without it the GP interpolates the labels.
- **Training set:** all training positives plus a seeded sample of training unlabelled genes (tunable size, 500–3,000 under nnPU; under PN the 1:1 training set is used).

**Attribution:**
- Because the kernel is additive, the predictive mean splits exactly into per-block terms k_b(x*, X)·α, where α is the mean cache. This is asserted to sum to the mean.
- A block's **share** is cov(f_b, f) / var(f), so the shares sum to 1.

### 6.3 Losses and training sets

| loss | training set | objective |
|---|---|---|
| `nnpu` | every training-fold gene, positive vs unlabelled | non-negative PU risk (Kiryo et al. 2017), with the class prior tuned |
| `pn` | training positives + an equal-size random draw of training unlabelled genes | binary cross-entropy |
| `pn_pathway` | positives + 1:1 unlabelled genes sharing a specific pathway (≤ 100 genes) with some positive, topped up at random | BCE |
| `pn_pubmed` | positives + for each positive, the unlabelled gene closest in log PubMed count, without replacement | BCE |

**nnPU form:** π·R_p⁺ + max(0, R_u⁻ − π·R_p⁻), with the softplus (logistic) loss. This deviates from the paper's recipe in two ways, both because training is full-batch:
- **Softplus instead of the sigmoid loss.** The sigmoid loss saturated and collapsed models onto the all-negative solution.
- **Clipping instead of the gradient-ascent step.** When the negative risk goes below zero, the paper takes a gradient-ascent step on it. That flipped low-capacity models (e.g. degree-only LR) to the inverted ranking on some seeds.

**Non-torch models:**
- Random forests and degree LR use class-balanced weights.
- Random forests and degree LR use the same training set as the loss specifies. For nnPU that is P vs U, without the PU risk.
- Label propagation uses the training labels directly.

**On the time split,** `pn_pubmed` matches on PubMed counts from before 2010.

### 6.4 Torch training

**Optimisation:**
- Full-batch Adam with tuned learning rate and weight decay.
- Up to 300 epochs, with early stopping (patience 30 epochs, after at least 100 epochs).

**Checkpoint selection:**
- The checkpoint is chosen on the **training objective evaluated on the val fold**. For PN losses, the val set is built the same way as the training set.
- Rank metrics (val AUPRC) stay flat while a weight drifts through zero, which froze low-capacity models at sign-inverted solutions.

**Reproducibility:**
- Torch seeds are set before construction, so initial weights are reproducible per seed.
- If a GPU run runs out of memory (several workers share one GPU, and the exact GP needs about 10 GB), it is rerun on the CPU.

---

## 7. Hyperparameter tuning (`tune.py`, `configs/tune.yaml`)

**Studies:**
- One Optuna study per (universe, model, loss ∈ {nnPU, PN}, tuning condition).
- Tuning conditions are real, rewired (`rw0`, whose parameters are reused for `rw1` and `rw2`), empty, and none (graph-free models).
- `shufattr` and `pwsub` reuse the real graph's parameters.
- `pn_pathway` and `pn_pubmed` reuse PN's parameters.

**Objective:**
- Mean val AUPRC over two dedicated tuning split seeds (5 and 6), which are never used for evaluation (seeds 0–4).
- Test folds are never seen.
- Tuning uses Minikel labels, the random split and `all` features. The parameters are reused for the other feature sets, splits and label sets of the same universe.

**Search:**
- TPE sampler (seed 0), 60 trials per study, a 3,000 s timeout per study.
- Studies are stored in a shared SQLite database, opened under a file lock.
- Trials that raise `RuntimeError` or `ValueError` are caught and skipped.

**Search spaces:**

| model kind | parameters |
|---|---|
| all torch models | lr ∈ [3e-4, 3e-2] (log); weight decay ∈ [1e-6, 5e-2] (log); hidden ∈ {32, 64, 128, 256}; layers 1–3; dropout ∈ [0, 0.8] |
| GAT | + heads ∈ {1, 2, 4, 8} |
| Correct & Smooth | + α ∈ [0.3, 0.99], C&S layers 5–50 |
| random forest | trees ∈ {200, 500, 1000}; min leaf ∈ {1, 2, 5, 10, 20}; max features ∈ {sqrt, 0.1, 0.3} |
| label propagation | α ∈ [0.3, 0.99], layers 2–50 |
| GP | noise floor ∈ [0.05, 0.9]; nnPU only: unlabelled sample ∈ {500, 1000, 2000, 3000} |
| nnPU torch models | + class prior π ∈ [0.005, 0.3] (log) |

**Defaults** apply when no tuned value exists: hidden 64, 2 layers, dropout 0.5, lr 0.01, weight decay 5e-4, π = 0.1, 500 trees, min leaf 5, LP and C&S α 0.85 / 0.8 with 50 layers, GP 2,000 unlabelled genes and noise floor 0.25.

**Known limits:**
- RF on STRING with nnPU hit the study timeout after 12 trials.
- Settings tuned on the random split occasionally fail on other splits; see the training-failure flag in §11.

---

## 8. Splits (`ghelps/splits.py`, `scripts/build_splits.py`)

Seven seeds are built per split: 0–4 for evaluation and 5–6 for tuning. Every split has train / val / test folds; some also have an "excluded" fold.

| split | construction |
|---|---|
| `random` | 70 / 10 / 20, stratified by label. |
| `pfam` | Whole Pfam families held out. Genes are grouped by their largest Pfam family; genes without Pfam are singletons. Groups are assigned in random order to test, then val, until those folds hold about 20% / 10% of the positives. A group is skipped if it would push a fold past 1.5× its gene budget or 1.25× its positive budget. The rest go to train. |
| `wave_<area>` | Therapy-area transfer (Minikel only). Each gene's ≥ Phase III indications are mapped to MeSH areas, ignoring the generic areas signs/symptoms, other and congenital. Test positives are genes whose indications all lie in one area. Training positives never touched that area. Genes in the area and elsewhere are excluded. Waves: oncology (+ hematology); CNS (neurology, psychiatry, ophthalmology); cardiometabolic (cardiovascular, metabolic, endocrine, digestive); inflammation (immune, dermatology, respiratory, musculoskeletal); infection. A wave is skipped on a graph if it has fewer than 30 test genes. HuRI has oncology only, and infection never reaches 30. |
| `time` | Train on positives first drugged before 2010, test on positives first drugged from 2010 on (ChEMBL). Training positives are split train / val in the same 70 : 10 ratio. Unlabelled genes are split 70 / 10 / 20. Positives with no approval date (Phase III only, or no ChEMBL mechanism) are excluded from every fold. All future targets are test positives in every seed; only the unlabelled genes vary. |
| `genetic` / `time` | The Minikel time split's folds and test set, unchanged. Only the training and val labels change to "genetic evidence before 2010". The two labels are scored on identical test genes and negatives. |

**Time-split sizes** (train / test / excluded positives):

| universe | train | test | excluded |
|---|---|---|---|
| STRING | 211 | 135 | 352 |
| Reactome FI | 196 | 123 | 311 |
| HuRI | 57 | 42 | 124 |

Further splits are built but not used in the reported experiments: `community` (Leiden on the graph, resolution 5), `rcnt` (first launch from 2021), and `pharos` (Tchem genes as test positives).

---

## 9. Evaluation

### 9.1 Negative regimes (`splits.negatives`)

Every split seed freezes a table of test negatives per regime.

| regime | negatives for each test positive |
|---|---|
| `random` | every unlabelled test gene, against every test positive. |
| `degree` | 5 drawn from the 20 unlabelled test genes nearest in log degree (on that universe's graph). |
| `pubmed` | the same, matched on log PubMed count. Counts from before 2010 on the time split. |
| `pfam` | 5 sharing a Pfam domain. |
| `pathway` | 5 sharing a specific Reactome pathway (≤ 100 genes). |
| `hop1` / `hop2` | 5 at graph distance 1 / exactly 2. |

**Drawing negatives:**
- Negatives are drawn without replacement across positives, so no negative is reused.
- A positive without a match is dropped from that regime.

**Scoring:**
- Each matched regime is scored as one pooled ranking of the matched positives against their matched negatives. Rank metrics allow this.

**Metrics:**
- AUROC and AUPRC per run and regime.
- Also recorded: val AUPRC (the tuning objective), training-fold AUROC (the fit check), and Spearman(score, degree) on the test fold.

Labels are positive vs unlabelled, so an AUROC measures "ranks known targets above unlabelled genes", not "finds targets".

### 9.2 Training-failure flag

A run's **training-fold AUROC** (positives vs unlabelled in the train fold) is stored with every run. For older runs it is computed after the fact from the saved scores. A model cell (mean over seeds) with training AUROC < 0.6 is marked † and listed. Controls where failing to fit is expected (shuffled labels, noise features) are not listed.

Correct & Smooth and label propagation reset training genes to their labels, so their training AUROC is always about 1. The flag can't detect a failed C&S base model.

### 9.3 Aggregation (`scripts/summarise.py`)

- **Seeds:** for each (regime, universe, graph, labels, feats, split, loss, model, condition), the mean, sd and count over the 5 seeds.
- **Rewired copies** are averaged within a seed first, so each seed counts once.
- **Paired deltas:** real − empty, real − rewired and real − shufattr, paired by seed.

---

## 10. Statistics

### 10.1 Effect sizes (primary; `scripts/delong.py: effects`)

For each contrast, the effect size is the mean AUROC difference, using random negatives.

| contrast | real-graph run compared with |
|---|---|
| `empty` | the same model with no edges |
| `rewired` | the mean over the 3 rewired copies |
| `pwsub` | the same model on the pathway subgraph |
| `shufattr` | the same model with shuffled edge vectors |
| `vs_rf_feat` | the graph model on the real graph vs RF on the same features |
| `vs_minikel_label` | a genetic-trained run vs its Minikel-trained twin |

**Point estimate:** every model is weighted equally. For each cell, deltas are averaged over seeds; for each model, over its cells; then over models. A cell is one (universe, graph, labels, feats, split, loss, model) combination.

**95% CI:** a hierarchical bootstrap with 2,000 draws. Resample models, then cells within each model, then seeds within each cell.

**Breakdowns:** reported overall, and separately by feature set, universe, split, loss and label set. The three universes are too few to resample, so they are reported separately rather than included in the bootstrap.

### 10.2 DeLong counts (supplementary)

- **Per seed:** paired DeLong z for AUROC(real) vs AUROC(control) on the identical test genes of each regime (MLstatkit 0.1.91).
  - The sign of z is taken from the AUROC difference.
  - Identical rankings give z = 0.
  - Other degenerate variances also give z = 0.
- **Combining:** seeds are combined with Stouffer's method (two-sided p). P-values are adjusted with Benjamini–Hochberg within each experiment.
- **Reporting:** cells significant at 5% are counted as real-better or real-worse. Cells share seeds, genes and configurations, so these counts overstate the evidence. They are reported only next to the effect sizes.

### 10.3 Bootstrap over future targets (time split; `scripts/analysis.py: time_novelty`)

**Per-target values.** For each run key (labels, graph condition, feats, model, loss) and each future target, two values are averaged over seeds:
- **win share:** the share of test negatives ranked below the target, with ties counted ½. A stratum's AUROC is the mean of its targets' win shares.
- **top-10% hit:** whether the target is in the top 10% of the full test ranking (positives and unlabelled). Recall in the top 10% is the mean of the hits.

**Bootstrap.** For each stratum, the future targets are resampled (multinomial, 2,000 draws). The test negatives (thousands of genes) are held fixed.
- **Why resample targets:** which targets happened to come up after 2010 is the main uncertainty for strata this small (7–77 genes).
- **Shared draws:** the same draws are used for every run key, so differences are paired.
- **Reported differences:** each model minus the PubMed-before-2010 ranker, and each genetic-trained run minus its Minikel twin.

**Pooled column:**
- Covers the run keys present on all three graphs.
- Genes are the bootstrap clusters, so a gene that appears in several graphs counts once.

**Supporting statistic:** `rho_pubmed_T`, the Spearman correlation of a run's test-gene scores with PubMed counts before 2010, averaged over seeds.

---

## 11. Descriptive analyses (`scripts/analysis.py`)

| output | method |
|---|---|
| `hub_bias.csv` | Degree of positives vs unlabelled genes; degree-only AUROC; share of positives in the top 10% by degree. Degree AUROC within PubMed quintiles. Logistic regression on log PubMed alone vs log PubMed + log degree. |
| `mediation.csv` | Per feature block (32-component SVD), all features, attention-free features and degree: an out-of-fold logistic score (5-fold, balanced). Reports its AUROC overall and within PubMed quintiles, and its Spearman correlation with log PubMed. **Shrinkage** = 1 − b_adjusted / b_alone, the logistic coefficient of the z-scored score logit with and without z-scored log PubMed added. Also the AUROC of features + log PubMed. |
| `study_leak.csv` | Out-of-fold random forest regressor predicting log PubMed from the leading 64 SVD components of each feature set (for `dated`, 32 from each group). 200 trees, max features 0.3, 3-fold, stratified on above/below the median. Reports Spearman(prediction, log PubMed). |
| `prior_study.csv` | For Minikel positives first drugged 1980–2024 (ChEMBL): papers dated up to 1 and 10 years before the first approval, as a percentile among the universe's genes at that date. Also an AUROC against genes with no drug programme in Minikel's data, ranked by their own counts at the same date. |
| `membrane.csv` | Membrane share among positives vs unlabelled genes. |
| `label_bias.csv` | Per label: positives, and the AUROC of PubMed count (now, and before 2010) for separating them from all other genes. |
| `topk_audit.csv` | The top-100 test-fold unlabelled genes per run (main experiment, random split). Reports: median PubMed percentile; share in the top PubMed decile; share in a drug programme (Minikel preclinical–Phase II); share sharing Pfam with a known (train/val) target; share within 1 hop of one (real graph); overlap with a PubMed ranker's top 100; and Spearman(score, PubMed) over all test unlabelled genes. Mean over seeds. |
| `phase_ladder.csv` | Models trained on ≥ Phase III: AUROC of test-fold genes with an earlier programme (preclinical / Phase I / Phase II) against genes with none. Overall, per phase and within PubMed quintiles. |
| `time_novelty.csv` | Novelty strata (below) with the CIs of §10.3. |
| `time_topk_audit.csv` | Time split, nnPU, `all` / `attn_free` / `dated`, real graph or none, both labels: the top-100 test genes per run (each seed's own test set; the test negatives are redrawn per seed). Reports the share that are future targets and novel (2 of 3) future targets, the median PubMed percentile before 2010 and the share in that top decile, the share in a drug programme today, the share sharing Pfam with or within 1 hop of a target drugged before 2010, and the share with genetic evidence before 2010 and at any date. Reference rows: all test genes, and the PubMed-before-2010 ranker. Mean over seeds. |
| `tried_*.csv` | Success given tried (§6.5). Genes are grouped from Minikel's target–indication pairs: never tried (nothing beyond preclinical), tried (≥ Phase I), success (≥ Phase III, i.e. the label), and resolved failure (best Phase I/II with no active programme; active ones are censored). For each predictor it reports AUROC for the label (success vs rest), selection (tried vs never) and outcome (success vs resolved failure), with 95% CIs from 2,000 gene bootstraps. Predictors: PubMed before 2010 and now; genetic evidence before 2010 and any date; out-of-fold scores from the main experiment (each seed's test-set percentile, averaged over the seeds a gene was tested in). Also the odds ratio (Woolf CI, +0.5 correction) of success vs resolved failure for genetic evidence. On the time split, genetic- vs Minikel-trained twins are compared on selection and outcome among the test genes, paired, per graph and pooled with genes as clusters. |
| `time_split_coverage.csv` | Train / test / undated-launched / undated-Phase-III-only / no-programme genes, compared on study before 2010 and now, and on novelty relative to the targets known before 2010. |

**Novelty strata.** Each future target is placed in strata by what was known before 2010:
- **study:** PubMed percentile before 2010 (< 0.5, 0.5–0.9, ≥ 0.9)
- **family:** whether it shares a Pfam domain with a target known before 2010 ("drugged family" vs "new family")
- **distance:** unweighted shortest-path hops to the nearest known target in the universe's graph (1, 2, 3+)

| stratum | rule | n (STRING / Reactome FI / HuRI; pooled) |
|---|---|---|
| **novel** | all three: not top-decile study, new family, ≥ 2 hops | 7 / 17 / 11; 24 |
| **novel (2 of 3)** | any two of the three | 44 / 48 / 35; 77 |

**Reference rankers:**
- PubMed count before 2010
- PubMed count now (leaks the future)
- number of known targets sharing a Pfam domain (ties broken by PubMed count)
- closeness to known targets: negative hop count, capped at 50 (ties broken by PubMed count)

---

## 12. Pre-specified readings

Three sets of interpretations were written into `scripts/report_template.md` before the corresponding results were generated. The git history of that file records when.

**1. Pfam split (§2.2).**
- **Prediction:** if RF with `attn_free` features loses clearly more AUROC from the random to the Pfam split than RF with `all` features, the family shortcut is confirmed. If it loses about as little, the reading of the family-shuffle control has to be weakened.

**2. Genetic label (§6.4).**
- **Primary endpoint:** genetic-trained minus Minikel-trained recall in the top 10% of the novel (2 of 3) future targets, pooled over graphs.
  - The difference is paired and comes with the §10.3 CI.
  - It is computed for RF and the GP under nnPU, with `attn_free` and `dated` features.
- **Supporting measure:** the same difference in Spearman(score, PubMed before 2010).
- **Outcomes:**

  | outcome | condition |
  |---|---|
  | A | CI above 0 for both feature sets |
  | B | CI includes or is below 0, and genetic-trained rankings follow study about as closely |
  | C | CI below 0, and genetic-trained rankings follow study clearly less |

- **Not the endpoint:** overall AUROC on all future targets. The smoke test had already shown the genetic label doing worse there.
**3. Success given tried (§6.5).** Fixed in commit `cea67b8`, before computing:
- **R1:** if outcome AUROC − 0.5 is less than half of selection AUROC − 0.5 (PubMed before 2010; RF with `all` and `attn_free`), the label is mostly selection.
- **R2:** if the odds-ratio CI for pre-2010 genetic evidence is above 1, the genetic label carries outcome signal.
- **R3:** on the time split, genetic − Minikel. An outcome CI that includes or exceeds 0, with the selection CI below 0, would mean a selection deficit. Both below 0 would mean the genetic label, as learned, is weaker.

- **Added after the results (not pre-specified):** the top-100 audit of §6.4 (`time_topk_audit.csv`). It shows the genetic-trained models leaving the drugged families, which is outcome C's mechanism, even though the pre-stated rule gives B.

---

## 13. Known limitations of the methods

**Leakage on the time split:**
- `all` features and the STRING graph are current snapshots, so they leak post-2010 knowledge.
- `dated` uses today's Pfam assignments. They are derived from sequence, but some families were defined after 2009.
- GTEx (in `attn_free` and `dated`) was measured after 2010, though uniformly for every gene.
- No graph is dated.

**Tuning:** done once, on the random split with Minikel labels and `all` features, then reused for every other split, label and feature set.

**Dating:**
- The paper-year fit dates PubMed entry, not publication.
- Positives without a ChEMBL approval are excluded from the time split.

**HuRI** covers 32% of the positives. Its pathway subgraph has 832 edges, so HuRI `pwsub` results are close to empty-graph results.

---

## 14. Software

Python 3.11–3.12, managed with `uv` (`uv.lock` pins every version). Main packages:

| package | version |
|---|---|
| PyTorch | 2.14.1 |
| PyTorch Geometric | 2.8.0.post1 |
| GPyTorch | 1.15.2 |
| scikit-learn | 1.9.1 |
| Optuna | 5.0.0 |
| transformers (ESM-2) | 5.19.0 |
| MLstatkit (DeLong) | 0.1.91 |
| statsmodels | 0.15.0 |
| python-igraph | 1.0.0 |
| leidenalg | 0.12.0 |
| pandas | 3.0.6 |

Runs are deterministic given the seed: split seeds, the torch seed per run, `random_state = seed` for the forests, seeded GP sampling and fixed SVD seeds.
