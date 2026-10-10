# Does the graph help?

Tests whether gene graphs help graph models predict drug-target genes. Each graph model is run on the real graph, on degree-preserving rewired copies and on an empty graph (the same model with no edges), and compared with feature-only baselines. Nonsense tests and a membrane positive control check that the pipeline itself behaves.

```bash
uv sync
uv run python tune.py configs/tune.yaml                  # Optuna -> results/tuning/best_params.yaml
uv run python main.py run configs/exp_main.yaml          # uses best_params.yaml if present
uv run python main.py summarise main [--metric auprc]
uv run python scripts/analysis.py                        # graph sizes, hub bias, membrane enrichment
uv run python scripts/figures.py --exps main membrane    # figure sets

# on the cluster: prepare (fetch + build) -> tune array -> 4 experiment grids -> summaries,
# analysis and figures, chained with dependencies
bash scripts/hpc/submit_all.sh
# after a code fix: rerun only missing/changed runs, then summaries + figures (no re-tuning)
bash scripts/hpc/resume.sh [experiment ...]
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

## Experiments

| Config | Labels | Features | Question |
|---|---|---|---|
| `exp_main` | Minikel ≥ Phase III | all; no tractability; attention-free | Does the graph matter inside graph models for drug targets? Do literature-derived features beat attention-free ones? |
| `exp_waves` | Minikel, therapeutic-wave splits | all; attention-free | Do models trained on targets from some therapeutic waves recognise targets from another? |
| `exp_membrane` | membrane vs not (GO:0016020 / GO:0005886) | Pfam + pathway (no GO, no tractability) | Positive control: on a learnable, graph-coherent label, does the graph help? If yes here but not on drug targets, the drug-target task is what makes the graph redundant. |
| `exp_nonsense_labels` | Minikel permuted across all genes; Minikel permuted within Pfam families | all | ≈ 0.5 overall. What survives the within-family shuffle was learned from family membership (resemblance to past targets), not from the gene itself. |
| `exp_nonsense_feats` | Minikel | Gaussian noise vectors | Feature models should score ≈ 0.5; graph models keep only what the graph itself gives. |

All experiments:
- **Graphs:** all three, each its own test.
- **Seeds:** 5 evaluation seeds.
- **Models:** the full ladder.
- **Conditions:** every graph condition the experiment's config lists.

## Outputs

**`results/<exp>/`**
- `results.csv`: one row per run × evaluation regime, with the full config and hyperparameters.
- `summary_<metric>.csv`: mean, sd and seed count per model and graph condition.
- `graph_effect_<metric>.csv`: paired-by-seed real − empty, real − rewired and real − shufattr.
- `gp_attribution.csv`: per-block share of the GP's predictive signal.
- `delong.csv`: paired DeLong tests of AUROC (MLstatkit), written by `scripts/delong.py`.
  - **Contrasts:** real vs empty, rewired, shuffled edge vectors and pathway subgraph, plus each graph model vs the no-graph RF.
  - **Pairing:** both runs are scored on the same test genes within a seed. Seeds are combined with Stouffer's method, and p-values are Benjamini–Hochberg-adjusted within each experiment.
  - **Caveat:** labels are positive vs unlabelled, so these compare rankings of known targets against unlabelled genes.

**`results/analysis/`**
- `graph_sizes.csv`
- `hub_bias.csv`: degree of positives, and whether it survives controlling for PubMed count.
- `membrane.csv`: membrane enrichment among the positives.
- `mediation.csv`: for each feature block, its AUROC overall and within PubMed quintiles, its correlation with PubMed count, and how much its logistic effect shrinks once PubMed count is added. This tests whether the features act through study intensity.

**`results/figures/<exp>/<graph>/`** holds figures 1–5, and `results/figures/compare/6_drug_vs_membrane.png` holds the druggability vs membrane comparison. All figures are drawn by the cluster's summarise job, because figure 5 and the analysis need the data the runs used.

## Data decisions

**Gene IDs.** Everything is keyed on Ensembl gene IDs of HGNC-approved protein-coding genes.
Symbols map approved > previous > alias, with ambiguous previous/alias symbols dropped.

**Graphs** (undirected, simple, protein-coding only). Three graphs span the range from literature-heavy to literature-free. Each has its own gene universe (its own node set); labels, splits, negatives and tuned hyperparameters are all per graph.

| graph | source | filter | character |
|---|---|---|---|
| string | STRING v12 | combined_score ≥ 700 | literature + curated + experimental |
| reactome_fi | Reactome FI 2025-04-14 | curated only: pairs annotated purely `predicted` are dropped (79k of 272k rows) | curated |
| huri | HuRI | as published | systematic Y2H, literature-free |

STRING-experimental and IntAct are still built as raw edge lists, because their membership bits are part of the edge vectors, but they're no longer tests of their own.

**Known caveats (state them in the paper):**
- **Database channel:** STRING ≥ 700 includes it, and it imports curated pathways (KEGG, Reactome), so STRING edges partly duplicate the pathway features.
- **Text-mining channel:** STRING ≥ 700 includes it, and it links genes co-mentioned in papers, drug targets among them. HuRI (systematic Y2H) and Reactome FI (curated) are the comparisons without literature co-mention.
- **HuRI coverage:** HuRI is a Y2H screen that misses most membrane proteins. It contains about 32% of the Minikel positives, and about 72% of positives are membrane proteins against about 36% of other genes. HuRI's universe is therefore small and target-poor; `results/analysis/` quantifies this.
- **Hub bias is study bias:** in STRING, positives are hubs, but among equally studied genes degree carries no signal, and PubMed count alone predicts targets at AUROC ≈ 0.85. In experimental graphs, targets are less connected than equally studied genes. `results/analysis/hub_bias.csv` has the numbers.

**Rewiring.** 3 copies per graph, made with igraph double-edge swaps (10 × |E| swaps). Degree sequences are asserted equal, and only 2–6% of the original edges survive.

**Pathway subgraph (`pwsub`).** Keeps the real edges whose two endpoints share a specific Reactome pathway (≤ 100 genes). Modules stay intact; cross-module edges are removed.

**Labels.**
- `minikel`: genes whose best target–indication pair reached Phase III or later (`ccat` = max of historical and active phase). Every other gene is unlabelled. Finan Tier 1 was dropped by decision.
- `minikel_perm`: the same number of positives on randomly chosen genes.
- `minikel_famperm`: Minikel labels shuffled within Pfam families (genes without Pfam form one group).
- `membrane`: the positive-control label.

**Features.** Feature blocks reach the models as a 256-d TruncatedSVD, fit label-free on all genes of the universe:
- **GO (GOA human):** excludes `IPI` evidence and GO:0005515 "protein binding", because both are interaction data.
- **Pfam:** from UniProt reviewed entries.
- **Reactome pathways:** sets of 5–500 genes.
- **Open Targets tractability:** drops the Approved Drug, Advanced Clinical and Phase 1 Clinical buckets for every modality, because they restate clinical phase.
  - **Kept:** non-clinical buckets such as Druggable Family, High-Quality Ligand/Pocket, Structure with Ligand, UniProt/GO/HPA location, Small Molecule Binder and Literature.
  - **Caveat:** these partly reflect past drug programmes. Druggable Family overlaps with how the Finan tiers were built.

The feature sets:
- `all`
- `no_tract`: no tractability.
- `no_loc`: Pfam + pathway only, for the membrane control.
- `random`: Gaussian noise of the same width.
- `attn_free`: attention-free features, computed or measured the same way for every gene, regardless of how well studied it is.
  - **ESM-2:** embeddings of the canonical UniProt sequence. Default `esm2_t12_35M`; set in `configs/data.yaml`.
  - **Sequence-derived:** length, amino-acid composition, and transmembrane helices predicted by Kyte–Doolittle hydropathy.
  - **GTEx v10:** median expression across tissues.
  - **Caveat:** ESM embeddings still encode resemblance to known families, so they test annotation bias, not self-similarity. The within-family shuffle tests self-similarity.
  - **Not included yet:** AlphaFold-derived pocket scores.

**Splits.** 7 seeds each, 70/10/20 train/val/test. Seeds 0–4 are for evaluation; seeds 5–6 are used only for tuning.
- `random`: stratified.
- `pfam`: genes grouped by their largest Pfam family (multi-domain proteins join most genes into one Pfam component). Whole groups are assigned to folds.
- `community`: whole Leiden communities (RBConfiguration, resolution 5, seed 0) assigned to folds.
- `rcnt` *(stand-in, Minikel only)*: test positives are genes first launched in 2021 or later, from Minikel `year_launch`.
- `pharos` *(stand-in, Minikel only)*: test positives are unlabelled Pharos Tchem genes; unlabelled Tclin genes are excluded. The real Varformer holdouts are in an unpublished `holdout_genes.xlsx`.
- `wave_<wave>` *(Minikel only)*: therapeutic-wave transfer, using Minikel's indication areas grouped into waves (`configs/data.yaml`).
  - **Test positives:** genes whose ≥ Phase III indications all fall in that wave.
  - **Training positives:** genes never targeted in that wave.
  - **Excluded:** genes targeted in the wave and elsewhere, from training, testing and negatives.
  - **Generic areas** (signs/symptoms, other, congenital) are ignored.
  - **Waves** with fewer than 30 single-wave genes on a graph aren't built. Infection has only 4 single-wave genes overall, so it only ever contributes training positives.

**Evaluation negatives** come from the test-fold unlabelled genes, with 5 matched negatives per positive drawn without replacement:
- `random`: all test-fold unlabelled genes.
- `degree`: nearest in log-degree.
- `pubmed`: nearest in log PubMed count. This is the study-intensity control.
- `pfam`: shares a Pfam family.
- `pathway`: shares a Reactome pathway of ≤ 100 genes.
- `hop1` / `hop2`: graph neighbours. **Report these separately; they are rigged against GNNs.**

## Graph conditions

Every graph model runs under each condition it supports:
- `real`
- `rw0`–`rw2`: degree-preserving rewired copies
- `empty`: no edges. The model is unchanged and only the graph is removed, so real − empty is the cleanest "does the graph matter" contrast.
- `pwsub`: the pathway-coherent subgraph.
- `shufattr`: real topology with the edge vectors permuted across edges. Edge-aware model only.

**Edge vectors** have the same 12 dimensions for every graph: 7 STRING channel scores for the pair, plus 5 bits for which source graphs contain the edge. On rewired copies, the real vectors are permuted onto the new edges.

## Models and training

**Library implementations:**
- PyTorch Geometric: `MLP`, `GCN`, `GraphSAGE` (mean aggregation), `GAT` (GATv2, with `edge_dim` for edge vectors), `LabelPropagation`, `CorrectAndSmooth`, the `SIGN` transform, and `AddLaplacianEigenvectorPE`.
- scikit-learn: `RandomForestClassifier`, and `LogisticRegression` for `deg_lr`.
- gpytorch: `ExactGP` with an `AdditiveKernel` of `ScaleKernel(RBFKernel(active_dims=block))`. The blocks are each own feature block (32-d SVD), log-degree, topology (16 Laplacian eigenvectors) and 2-hop neighbour features. The per-block share comes from `prediction_strategy.mean_cache`.

**Training regimes** (`loss` axis):
- `nnpu`: Kiryo et al. 2017, with a logistic loss and the clipped non-negative risk. The sigmoid loss and the gradient-ascent step both broke full-batch training.
- `pn`: 1:1 random unlabelled negatives, MORGaN-style.
- `pn_pathway`: two-step-PU-style hard negatives, 1:1. Unlabelled genes that share a specific pathway with a positive, topped up at random.
- `pn_pubmed`: hard negatives matched to each positive on PubMed count, so the model can't win by learning "well studied".
- **Model-specific fitting:**
  - RFs and `deg_lr` fit their training set with balanced class weights.
  - The GP is a regression on P vs U (all train positives plus sampled unlabelled genes), with a noise floor.
- **Early stopping:** on the training objective evaluated on the val fold (at least 100 epochs, patience 30).

**`deg_lr`** is scikit-learn logistic regression on log-degree. It isn't tuned: a one-weight torch model under nnPU converged to the inverted ranking on some seeds.

**Tuning** (`tune.py`) runs one Optuna TPE study per (graph universe, model, loss, graph condition):
- **Conditions:**
  - `real`: also used for `shufattr` and `pwsub`.
  - `rewired`: tuned on rw0, used for rw0–rw2.
  - `empty`
  - `none`: graph-free models.
- **Losses:**
  - `pn_pathway` and `pn_pubmed` reuse the PN parameters.
  - `membrane`, `nonsense` and the other feature sets reuse that universe's Minikel / all-features parameters.
- **Why per condition and per graph:** parameters tuned for one graph collapsed the controls on another. For example, GAT on a rewired graph fell from 0.83 to 0.54 AUROC.
- **Objective:** mean val AUPRC over the tuning split seeds 5–6, which are never evaluated.
- **Selection bias:** the stored `_val_auprc` is a max over trials, so it is optimistic. Don't compare it across models.
- **Cluster jobs:** `scripts/hpc/tune.sbatch` runs one array task per (universe, model, condition), 102 tasks. Grids run as disjoint shards (`main.py run … --shard i/n`).
- **Result filtering:** `manifest.txt` keeps `results.csv` to the current grid.

## Sources

Everything is downloaded by `ghelps/fetch.py` into `$GHELPS_DATA/raw`, and existing files are skipped:
- **STRING v12:** links (all channels) and aliases.
- **IntAct:** the human MITAB archive, stream-filtered to human–human rows.
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
