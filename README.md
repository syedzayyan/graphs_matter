# Does the graph help?

Benchmarks GNNs against degree-, structure- and feature-matched baselines for drug-target
gene prediction, on real gene graphs vs degree-preserving rewired copies.

```bash
uv sync
./scripts/download.sh                          # most raw sources (see "Sources" for the rest)
uv run python main.py build                    # stages 1-2: graphs, stats, edge vectors, features, splits
uv run python main.py device                   # cuda if available, else cpu

uv run python tune.py configs/tune.yaml        # Optuna -> results/tuning/best_params.yaml
uv run python main.py run configs/exp_headline.yaml      # uses best_params.yaml if present
uv run python main.py summarise headline [--metric auprc]

# on the cluster (GPU):
sbatch scripts/hpc/tune.sbatch                 # array job, one task per model
sbatch scripts/hpc/run_headline.sbatch         # then the grid with tuned params
```

**Outputs**
- `results/<exp>/results.csv`: one row per run × evaluation regime, with the full config and hyperparameters.
- `summary_<metric>.csv`: mean, sd and seed count per model and graph condition.
- `graph_effect_<metric>.csv`: paired-by-seed real − empty, real − rewired and real − shufattr.
- `gp_attribution.csv`: per-block share of the GP's predictive signal.

**Tuning**
- One Optuna TPE study per (model, loss), maximising mean val AUPRC over split seeds 0–1.
- It uses the real graph and the train/val folds only.
- The tuned parameters are reused unchanged for the rewired, empty and shufattr conditions, so the graph is the only thing that varies between them.

All choices live in `configs/data.yaml` (data) and `configs/exp_*.yaml` (experiments).
`data/processed/` is the benchmark release.

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

**Universes.**
- `main`: the intersection of the four non-HuRI graphs, 9,515 genes.
- `huri`: the five-way intersection, 4,882 genes. It is used for the HuRI study-bias test, with all five graphs on the same genes.

HuRI is a Y2H screen that misses most membrane proteins. Putting it in the main intersection would have cut the Minikel positives from 540 to 195.

**Rewiring.** 3 copies per graph, made with igraph double-edge swaps (10 × |E| swaps). Degree sequences are asserted equal, and only 2–6% of the original edges survive.

**Labels.** Positives are Minikel et al. 2024 genes whose best target–indication pair reached Phase III or later (`ccat` = max of historical and active phase). Every other gene is unlabelled. Finan Tier 1 was dropped by decision.

**Features.** All feature blocks are fed to every model as a 256-d TruncatedSVD, fit label-free on all genes:
- **GO (GOA human):** excludes `IPI` evidence and GO:0005515 "protein binding", because both are interaction data and would leak the graph into the features.
- **Pfam:** from UniProt reviewed entries.
- **Reactome pathways:** sets of 5–500 genes.
- **Open Targets tractability:** drops the Approved Drug, Advanced Clinical and Phase 1 Clinical buckets, because they encode the label.

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
  - PyTorch Geometric: `MLP`, `GCN`, `GAT` (GATv2 with `edge_dim` for edge vectors), `LabelPropagation`, `CorrectAndSmooth`, the `SIGN` transform, and `AddLaplacianEigenvectorPE`.
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

- **Already local** (`../../collate_data/data`): STRING v12 links, IntAct.
- **`scripts/download.sh`:** Reactome FI, HGNC, gene2pubmed, GOA, UniProt/Pfam, Ensembl2Reactome, Minikel (`ericminikel/genetic_support`).
- **Fetched separately:**
  - STRING aliases.
  - HuRI, from `interactome-atlas.org`; the `www.` host has a bad TLS certificate.
  - Open Targets 26.09 (`target_tractability`, `association_overall_direct`, `drug_*`).
  - Pharos TDLs, via the GraphQL `download` query.
  - Finan Table S1, via the DrugnomeAI repo. It is no longer used.
