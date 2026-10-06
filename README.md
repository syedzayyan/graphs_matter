# Does the graph help?

Benchmarks GNNs against degree-, structure- and feature-matched baselines for drug-target
gene prediction, on real gene graphs vs degree-preserving rewired copies.

```bash
uv sync
uv run python tune.py configs/tune.yaml                  # Optuna -> results/tuning/best_params.yaml
uv run python main.py run configs/exp_headline.yaml      # uses best_params.yaml if present
uv run python main.py summarise headline [--metric auprc]

# on the cluster: prepare (fetch + build) -> tune array -> grid, chained with dependencies
bash scripts/hpc/submit_all.sh
```

**Data location.** Raw downloads go in `$GHELPS_DATA/raw` and the built benchmark in `$GHELPS_DATA/processed`. The default is `<repo>/data`. On the cluster, `scripts/hpc/env.sh` sets it to `/data/scratch/bty644/ghelps/data`.

**Data on a fresh checkout.** `tune.py`, `main.py run` and `main.py build` all call `ghelps.ensure.ensure_data()` before doing anything else. It downloads any missing raw source (`ghelps/fetch.py`), then runs any build step whose outputs are missing:
- `graphs`
- `structure`
- `edge_attr`
- `features`
- `splits`

It works under a file lock. On the cluster, `submit_all.sh` also runs a single preparation job first, because shared filesystems don't always honour `flock`.

Other commands:
- `main.py fetch`: download missing raw sources only.
- `main.py build --steps …`: force a rebuild of specific steps.
- `main.py device`: show which device runs will use.

**Outputs**
- `results/<exp>/results.csv`: one row per run × evaluation regime, with the full config and hyperparameters.
- `summary_<metric>.csv`: mean, sd and seed count per model and graph condition.
- `graph_effect_<metric>.csv`: paired-by-seed real − empty, real − rewired and real − shufattr.
- `gp_attribution.csv`: per-block share of the GP's predictive signal.

**Tuning** (`tune.py`) runs one Optuna TPE study per (model, loss, graph condition):
- **Conditions:**
  - `real` (also used for `shufattr` runs)
  - `rewired` (tuned on rw0, used for rw0–rw2)
  - `empty`
  - `none` (graph-free models)
- **Why per condition:** reusing real-graph parameters on the controls collapsed them. For example, GAT on a rewired graph dropped from 0.83 to 0.54 AUROC, which inflated real − rewired.
- **Objective:** mean val AUPRC over the dedicated tuning split seeds 5–6. These seeds are never evaluated; experiments use seeds 0–4, and test folds are never touched.
- **Selection bias:** the stored `_val_auprc` is a max over trials, so it is optimistic. Don't compare it across models.
- **Cluster:** `scripts/hpc/tune.sbatch` runs one array task per (model, condition) pair.
- **Limitations:**
  - Parameters are tuned on `main`/`string` with the full feature set. They are reused for `string_exp`, the HuRI universe and `no_tract`.
  - Tuning uses the random split only.
- **Cluster grids:** large grids run as disjoint shards (`main.py run … --shard i/n`, `scripts/hpc/run.sbatch` array). `submit_all.sh` chains prepare → tune → headline + HuRI → summarise.
- **Result filtering:** `main.py run` writes `manifest.txt`, so `results.csv` only covers the current grid and never mixes in older runs.

## Data decisions

**Gene IDs.** Everything is keyed on Ensembl gene IDs of HGNC-approved protein-coding genes.
Symbols map approved > previous > alias, with ambiguous previous/alias symbols dropped.

**Graphs** (undirected, simple, protein-coding only):

| graph | source | filter |
|---|---|---|
| string | STRING v12 | combined_score ≥ 700 |
| string_exp | STRING v12 | `experiments` channel (not transferred) ≥ 400 |
| intact | IntAct human–human | MI-score ≥ 0.45; association / physical / direct / enzymatic types (no proximity, no colocalisation) |
| reactome_fi | Reactome FI 2025-04-14 | including predicted FIs |
| huri | HuRI | as published |

**STRING overlaps with the features.** STRING's combined score includes two channels that overlap with information the models get elsewhere:
- **Database channel:** imports curated pathways (KEGG, Reactome), so STRING ≥ 700 edges partly duplicate the Reactome pathway features.
- **Text-mining channel:** links genes co-mentioned in papers, which includes drug targets named together.

Both channels are also among the 7 dimensions of the edge vectors. STRING-experimental (`string_exp`, experiments channel only) is the clean comparison and runs alongside STRING in the headline.

**Universes.**
- `main`: the intersection of the four non-HuRI graphs, 9,515 genes.
- `huri`: the five-way intersection, 4,882 genes. It is used for the HuRI study-bias test, with all five graphs on the same genes.

HuRI is a Y2H screen that misses most membrane proteins. Putting it in the main intersection would have cut the Minikel positives from 540 to 195.

**HuRI coverage.** HuRI under-covers the positives. Y2H is weak on membrane proteins (GPCRs, ion channels, transporters), so HuRI contains only 31.8% of the 702 Minikel positives, and 17.4% of those are isolated in the `huri` universe. `results/coverage_table.csv` reports positive coverage, isolated-positive fraction and positive vs unlabelled median degree for every graph and universe.

In the main universe, positives have about twice the median degree of unlabelled genes in STRING (38 vs 17) and Reactome FI (28 vs 15). In STRING-exp (5.5 vs 6) and IntAct (7 vs 8) they don't. That points to curation and text-mining study bias rather than biology.

The HuRI analysis (`configs/exp_huri.yaml`) therefore runs on the HuRI-covered genes, with every other graph randomly subsampled to HuRI's edge count (`<graph>_em`, 18,081 edges, with its own rewired copies). This rules out "fewer edges" as the explanation for a smaller graph effect. It does not fix HuRI missing the membrane targets.

**Rewiring.** 3 copies per graph, made with igraph double-edge swaps (10 × |E| swaps). Degree sequences are asserted equal, and only 2–6% of the original edges survive.

**Labels.** Positives are Minikel et al. 2024 genes whose best target–indication pair reached Phase III or later (`ccat` = max of historical and active phase). Every other gene is unlabelled. Finan Tier 1 was dropped by decision.

**Features.** All feature blocks are fed to every model as a 256-d TruncatedSVD, fit label-free on all genes:
- **GO (GOA human):** excludes `IPI` evidence and GO:0005515 "protein binding", because both are interaction data and would leak the graph into the features.
- **Pfam:** from UniProt reviewed entries.
- **Reactome pathways:** sets of 5–500 genes.
- **Open Targets tractability:** drops the Approved Drug, Advanced Clinical and Phase 1 Clinical buckets for every modality, because they restate clinical phase and so encode the label.
  - **Kept:** non-clinical buckets such as Druggable Family, High-Quality Ligand/Pocket, Structure with Ligand, UniProt/GO/HPA location, Small Molecule Binder and Literature.
  - **Caveat:** these partly reflect past drug programmes. Druggable Family overlaps with how the Finan tiers were built.
  - **Ablation:** every feature model also runs with tractability removed (`feats: no_tract`).

**Splits.** 5 seeds each, 70/10/20 train/val/test:
- `random`: stratified.
- `pfam`: genes grouped by their largest Pfam family, because multi-domain proteins join 4,946 genes into one Pfam component. Whole groups are assigned to folds.
- `community`: Leiden (RBConfiguration, resolution 5, seed 0) run once on the union graph of the universe and reused for every graph and rewired copy.
- `rcnt` *(stand-in)*: test positives are genes first launched in 2021 or later, from Minikel `year_launch` (data runs to 2022). ChEMBL's API was down.
- `pharos` *(stand-in)*: test positives are unlabelled Pharos Tchem genes. Unlabelled Tclin genes are excluded from training and evaluation.
- The real Varformer holdouts are in an unpublished `holdout_genes.xlsx` (sheets `pfam_drgbl`, `rcnt_app_targets`, `chem_targets`).

**Evaluation negatives** come from the test-fold unlabelled genes, with 5 matched negatives per positive drawn without replacement:
- `random`: all test-fold unlabelled genes.
- `degree`: nearest in log-degree on the union graph.
- `pfam`: shares a Pfam family.
- `pathway`: shares a Reactome pathway of ≤ 100 genes.
- `hop1` / `hop2`: union-graph neighbours. **Report these separately; they are rigged against GNNs.**

## Graph conditions

Every graph model runs under each condition it supports:
- `real`
- `rw0`–`rw2`: degree-preserving rewired copies
- `empty`: no edges. The model is unchanged and only the graph is removed, so real − empty is the cleanest "does the graph matter" contrast.
- `shufattr`: real topology with the edge vectors permuted across edges. Used only by the edge-aware model.

**Edge vectors** have the same 12 dimensions for every graph:
- 7 STRING channel scores for the pair (neighbourhood, fusion, co-occurrence, co-expression, experiments, database, text-mining)
- 5 bits for which source graphs contain the edge

On rewired copies the real vectors are permuted onto the new edges.

## Models

- **Library implementations:**
  - PyTorch Geometric: `MLP`, `GCN`, `GraphSAGE` (mean aggregation), `GAT` (GATv2 with `edge_dim` for edge vectors), `LabelPropagation`, `CorrectAndSmooth`, the `SIGN` transform, and `AddLaplacianEigenvectorPE`.
  - scikit-learn: `RandomForestClassifier`.
  - gpytorch: `ExactGP` with an `AdditiveKernel` of `ScaleKernel(RBFKernel(active_dims=block))`. The blocks are own GO, Pfam, pathway and tractability (32-d SVD each), log-degree, topology (16 Laplacian eigenvectors) and 2-hop neighbour features. The per-block share of the predictive mean comes from `prediction_strategy.mean_cache`.
- **Training:**
  - **nnPU:** Kiryo et al. 2017, with a logistic loss and the clipped non-negative risk, prior 0.1. The sigmoid loss and the gradient-ascent step both broke full-batch training.
  - **PN:** 1:1 random unlabelled negatives, as in MORGaN.
  - **Random forests:** fit P vs U with balanced class weights.
  - **GP:** regression on P vs U, with all train positives plus 2,000 sampled unlabelled genes. The noise is floored at 0.25 × the label variance.
  - **Early stopping:** on the training objective evaluated on the val fold (at least 100 epochs, patience 30).
  - **Hyperparameters:** shared defaults in `ghelps/train.py:DEFAULTS`, with no per-model tuning yet.

## Sources

Everything is downloaded by `ghelps/fetch.py` into `data/raw`, and existing files are skipped:
- **STRING v12:** links (all channels) and aliases.
- **IntAct:** the human MITAB archive, stream-filtered to human–human rows. The archive is deleted afterwards.
- **HuRI:** from `interactome-atlas.org`. The `www.` host has a mismatched TLS certificate.
- **Reactome FI** (2025-04-14).
- **HGNC.**
- **gene2pubmed.**
- **GOA human.**
- **Ensembl2Reactome.**
- **UniProt reviewed human, with Pfam.**
- **Open Targets `target_tractability`:** pinned to release 26.09.
- **Pharos TDLs:** via the GraphQL bulk `download` query.
- **Minikel et al. 2024:** from `ericminikel/genetic_support`.

Unpinned: GOA, Ensembl2Reactome, gene2pubmed, UniProt and Pharos all serve their current release. Record the fetch date when freezing the benchmark.
