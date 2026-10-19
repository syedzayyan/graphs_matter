# Does the graph help predict drug targets? Full results

Status as of 19 October 2026 (§6.4, the genetic label, is pending). This covers the complete cluster run: six experiments, three graphs, 5 seeds each, everything tuned per graph. Every table below is generated from `results/` by `scripts/report.py`, so the numbers in tables come straight from the CSVs. Numbers quoted in the prose are hand-written; re-check them after a rerun. Figures are linked from `results/figures/`. Open this file in a Markdown preview (VS Code: ⇧⌘V) to see them inline.

---

## TL;DR: the hypothesis

**Claim.** Target-discovery models trained on "reached Phase III or later" labels don't learn what makes a gene a good target. They learn what the field has already paid attention to. The bias enters through three doors:

1. **The label.** Phase ≥ III status records which genes companies chose to pursue. Those genes were already among the most studied in the genome years before their first drug (about the 90th–96th PubMed percentile 10 years earlier, §4.4).
2. **The features.** GO, pathways and Open Targets tractability are written by the literature, so they encode study intensity:
   - A random forest recovers a gene's PubMed count from them at Spearman ρ ≈ 0.79.
   - Even "attention-free" sequence and expression features give ρ ≈ 0.60, because study is concentrated in particular protein families.
   - No per-feature correction removes this (§4.3).
3. **The graph.** STRING is literature-built: its degree is study intensity (fully mediated by PubMed count, §4.1). Message passing pulls predictions towards neighbours of known targets.

**Consequence: a closed loop.** Models rank highest the genes that are already well studied, in already-drugged families and next to known targets (§5). Their "novel" predictions are mostly genes already in drug programmes. In a time split (train on targets first drugged before 2010, test on those drugged from 2010 on):
- The future targets were themselves mostly more of the same (§6.1): over half were in the top PubMed decile in 2009, and on STRING 77% were one hop from an existing target.
- One-number rankers find them about as well as any model (§6.2): "how studied was it in 2009" and "how many known targets share its Pfam domain".
- With features that don't leak the future, models do *worse* than PubMed count alone.

**For the GNN question.** Inside the same model, removing every edge is as good as the real graph about as often as not (§2). Graph models lose to a plain random forest 442 to 169. The graph helps clearly only on a label that genuinely follows the network (membrane localisation, §8.1), or when the features are uninformative, where it re-injects study intensity (§3.3, §8.3). So the null result is about the task and its label, not the method.

**What this does and doesn't show.** It shows the Phase ≥ III label plus current features reward rediscovery. Whether a different label escapes the loop is tested in §6.4: the same time split, trained on human genetic evidence published before 2010.

---

## How to read this

- **AUROC:** known targets ranked against unlabelled genes (PU setting). An unlabelled gene isn't a true negative, so AUROCs measure "ranks known targets high", not "finds targets".
- **Cells in model tables:** `real / rewired / empty`, meaning the same model with the real graph, degree-preserving rewired copies (mean of 3), and no edges. "(no graph)" marks models that never use a graph. All values are means over 5 seeds.
- **Negative regimes:** each AUROC can use one of these negative sets:
  - **random:** all unlabelled test genes.
  - **degree:** degree-matched.
  - **pubmed:** matched on PubMed count, i.e. as well studied as the target.
  - **pfam:** shares a Pfam domain.
  - **pathway:** shares a specific Reactome pathway.
  - **hop1 / hop2:** 1 or 2 hops from the target in the graph.
- **Splits:**
  - **random:** 70/10/20.
  - **pfam:** whole Pfam families held out.
  - **wave_\*:** therapy area held out.
  - **time:** targets first drugged before 2010 vs from 2010 on.
- **Losses:**
  - **nnPU:** positive–unlabelled.
  - **PN:** 1:1 random negatives.
  - **pn_pathway / pn_pubmed:** 1:1 hard negatives from the same pathway / matched on PubMed count.
- **Feature sets:**
  - **all:** GO, Pfam, pathways, Open Targets tractability.
  - **no_tract:** without tractability.
  - **attn_free:** ESM-2 embeddings, sequence statistics, GTEx expression.
  - **no_loc:** Pfam and pathways (membrane control).
  - **random:** Gaussian noise.
- **DeLong tables:**
  - Each cell is one (graph, model, loss, split, feature set) combination: paired DeLong per seed, combined with Stouffer, Benjamini–Hochberg within each experiment.
  - `real_better / real_worse` count cells significant at 5%; `mean_delta` is the mean AUROC difference.
  - **Contrasts:** `empty` = real vs no edges; `rewired` = real vs degree-preserving rewiring; `pwsub` = real vs pathway-only subgraph; `shufattr` = real vs shuffled edge vectors; `vs_rf_feat` = graph model (real graph) vs RF on features alone.

## Contents

1. [Setup: graphs](#1-setup-graphs)
2. [Does the graph help? (main)](#2-does-the-graph-help-main)
3. [What the graph carries on its own](#3-what-the-graph-carries-on-its-own)
4. [Study bias in label, graph and features](#4-study-bias-in-label-graph-and-features)
5. [What the models "discover"](#5-what-the-models-discover)
6. [Time split: does the label find new targets?](#6-time-split-does-the-label-find-new-targets)
7. [Therapy-area waves](#7-therapy-area-waves)
8. [Controls: membrane, shuffled labels, random features](#8-controls-membrane-shuffled-labels-random-features)
9. [Caveats](#9-caveats)
10. [Next steps](#10-next-steps)

---

## 1. Setup: graphs

**Table 1.1: graph sizes per universe and condition.** Each graph is its own gene universe, with its own positives (Minikel ≥ Phase III), splits and tuned hyperparameters. `positives_covered` is the share of all 702 Minikel ≥ Phase III genes that fall in the universe.
- **HuRI** covers only 32% of targets: Y2H misses membrane proteins (§4.5).
- **Pathway subgraph:** HuRI's has only 832 edges, so HuRI `pwsub` results are near-empty-graph results.

{{graph_sizes}}

---

## 2. Does the graph help? (main)

### 2.1 Headline: AUROC per model, random split

**Table 2.1a: nnPU, all features, random negatives.** `real / rewired / empty` per graph. The useful comparison is real vs empty *within a row*, which is the same model with and without edges. Comparing with a rewired graph overstates the graph's value, because a fake graph actively damages a model.

What it shows:
- **No model beats the GP or RF by much**, and RF uses no graph at all: 0.923 / 0.921 / 0.915.
- **GAT and GCN are worse with the real graph than with no edges** on all three graphs: STRING GAT 0.877 vs 0.920, HuRI GAT 0.699 vs 0.868.
- **Correct & Smooth gains on STRING (0.935 vs 0.912) and HuRI (0.850 vs 0.815); SIGN gains on STRING (0.924 vs 0.900).** Neither gains on Reactome FI.

{{main_compact_random_nnpu}}

**Table 2.1b: the same with PN training (1:1 random negatives).**

{{main_compact_random_pn}}

**Table 2.1c: PN with PubMed-matched training negatives.** Study intensity can't separate the classes during training.

{{main_compact_random_pn_pubmed}}

**Table 2.1d: PN with same-pathway training negatives.**

{{main_compact_random_pn_pathway}}

**Figure 2.1: AUROC per model and graph condition** (random split, all features; left nnPU, right PN). Circles show the real graph, squares the rewired graph, triangles the empty graph. The grey line is RF with no graph. In every panel most real-graph circles sit at or left of the empty-graph triangles, and to the left of the RF line.

![STRING conditions](results/figures/main/string/1_conditions.png)
![Reactome FI conditions](results/figures/main/reactome_fi/1_conditions.png)
![HuRI conditions](results/figures/main/huri/1_conditions.png)

### 2.2 Held-out Pfam families

**Table 2.2a: nnPU, all features, Pfam split** (whole families held out at test time). Performance drops only a little: RF 0.917 / 0.906 / 0.899. The `all` features still carry tractability and GO, which work across families. On HuRI, Correct & Smooth with the empty graph collapses to 0.494, a training failure.

{{main_compact_pfam_nnpu}}

**Table 2.2b: PN, Pfam split.**

{{main_compact_pfam_pn}}

### 2.3 Significance: paired DeLong

**Table 2.3a: all contrasts, main experiment, random negatives.**
- **Real vs empty:** significant either way in 294 of 432 cells (195 better, 99 worse). A modest net gain of +0.022.
- **Real vs rewired:** 478 vs 18 (+0.051).
- **Graph model vs feature-only RF:** the graph model *loses* 442 vs 169 (−0.064).

{{delong_main_contrast}}

**Table 2.3b: by feature set.** The graph's net gain over the empty graph is near zero with `all` features (42 better / 53 worse). With `attn_free` features it is clearly positive (94 / 13). The graph helps when the features don't already carry the information it adds (see §3.3 for what that is). Against RF, graph models lose badly with `all` (23 / 276) and only draw with `attn_free` (98 / 46).

{{delong_main_contrast_feats}}

**Table 2.3c: by graph.** The net gain over the empty graph is almost entirely on STRING (107 / 13). Reactome FI (46 / 42) and HuRI (42 / 44) are a wash. STRING is the literature-derived graph.

{{delong_main_contrast_universe}}

**Table 2.3d: by split.** The graph helps a little more when whole families are held out (111 / 40 vs 84 / 59).

{{delong_main_contrast_split}}

**Table 2.3e: by loss.** With nnPU the empty graph is as good as the real one (34 / 41). The graph gains more under PN losses.

{{delong_main_contrast_loss}}

**Table 2.3f: real vs empty per model and graph**: `better/worse of cells (mean Δ)`, over all feature sets, splits and losses. SIGN and GraphSAGE gain on STRING and HuRI. GCN and GAT lose on Reactome FI and HuRI.

{{delong_main_empty_model}}

**Table 2.3g: every contrast under every negative regime.** Against 1-hop negatives (unlabelled direct neighbours of targets), the real graph is *worse* than the empty one (70 / 200). Message passing raises the neighbours of known targets, which is exactly the "linked to known targets" bias.

{{delong_main_regimes}}

**Figure 2.2: paired graph effect** (real − empty and real − rewired, per seed).

![STRING graph effect](results/figures/main/string/2_graph_effect.png)
![Reactome FI graph effect](results/figures/main/reactome_fi/2_graph_effect.png)
![HuRI graph effect](results/figures/main/huri/2_graph_effect.png)

### 2.4 Under harder negatives

**Table 2.4: AUROC by evaluation-negative regime** (random split, nnPU, all features; feature-only models with no graph, graph models with the real graph).
- Every model drops sharply against PubMed-matched negatives (STRING RF 0.923 → 0.842) and against same-family negatives (→ 0.785).
- 1-hop neighbours of targets are the hardest negatives for graph models (STRING C&S 0.795).

{{main_regimes}}

---

## 3. What the graph carries on its own

### 3.1 Graph-only models

**Table 3.1: models without node features, random split, nnPU, by negative regime and graph condition.**
- **Label propagation on STRING alone reaches 0.914**, close to the feature models. Most of that is study: it falls to 0.789 against PubMed-matched negatives.
- **On HuRI, the literature-free graph,** graph-only models are near chance (label propagation 0.545).
- **Rewiring preserves degree,** so degree LR is unchanged under rewiring. The real-vs-rewired gap for label propagation and RF-on-structure is the part due to *who is connected to whom*.

{{main_graph_only}}

**Figure 3.1: graph-only models across negative regimes, real vs rewired.**

![STRING graph only](results/figures/main/string/3_graph_only.png)
![Reactome FI graph only](results/figures/main/reactome_fi/3_graph_only.png)
![HuRI graph only](results/figures/main/huri/3_graph_only.png)

### 3.2 What the GP relies on

**Table 3.2: additive GP attribution** (random split, nnPU, real graph). `share_b` is block b's share of the predictive variance, computed as cov(f_b, f) / var(f), so the shares sum to 1. The last two columns are the correlations of the GP's predictive uncertainty with degree and PubMed count.
- **With `all` features,** tractability dominates (0.56–0.69) and the graph blocks (degree, topology, neighbour features) contribute 0.03–0.27.
- **With `attn_free` features on STRING,** the GP puts 0.73 of its weight on the graph (degree 0.34 + neighbour features 0.39). When the features are clean, the model leans on the literature graph.

{{gp_attr}}

![STRING GP](results/figures/main/string/4_gp_attribution.png)
![Reactome FI GP](results/figures/main/reactome_fi/4_gp_attribution.png)
![HuRI GP](results/figures/main/huri/4_gp_attribution.png)

### 3.3 Degree dependence

**Figure 3.3: share of positives, and each model's mean score percentile, by degree decile** (test genes pooled over seeds). On STRING, positives concentrate in the top degree deciles, and graph models track degree more steeply than the feature-only RF.

![STRING degree](results/figures/main/string/5_degree.png)
![Reactome FI degree](results/figures/main/reactome_fi/5_degree.png)
![HuRI degree](results/figures/main/huri/5_degree.png)

---

## 4. Study bias in label, graph and features

### 4.1 Hubs vs study

**Table 4.1: do targets sit on hubs, and is that just study?**
- **On STRING,** targets have a median degree of 40 vs 13, and degree alone gives AUROC 0.756.
- **Within PubMed quintiles it falls to 0.565.** Adding degree to PubMed count adds nothing (0.870 → 0.870).
- **On HuRI,** degree is unrelated to study (ρ = −0.03), and targets are *less* connected (AUROC 0.448).

{{hub_bias}}

### 4.2 Are features mediated by study?

**Tables 4.2a–c: per feature block** (out-of-fold logistic score):
- AUROC overall, and within PubMed quintiles.
- The score's correlation with log PubMed count.
- The **shrinkage** of its coefficient once log PubMed is added. 1 means the block's effect is fully explained by study.

On STRING:
- Degree shrinks 98%, pathways 53%, GO 36%, tractability 18%, attention-free features 17%, Pfam 6%.
- Literature-built blocks are substantially study.
- Pfam isn't: its signal is resemblance to previously drugged families.

Reactome FI degree has an unstable near-zero coefficient, so its shrinkage (1.87) is meaningless.

STRING:

{{mediation_string}}

Reactome FI:

{{mediation_reactome_fi}}

HuRI:

{{mediation_huri}}

### 4.3 Can features be made study-free? (no)

**Table 4.3: Spearman ρ between an out-of-fold random forest's prediction of log PubMed count and the truth, from each feature set's leading 64 components.**
- **`*_resid` sets:** every feature column replaced by its rank among genes in the same PubMed-count bin (50 bins), so each column has the same distribution at every level of study.
- **Literature features become *more* revealing of study after this** (STRING 0.79 → 0.89). Their joint annotation pattern *is* study intensity.
- **Attention-free features drop only from 0.60 to 0.53.** Study is concentrated in particular protein families, and ESM embeddings encode family.

So no feature set can be made study-free. The experiments instead remove the *incentive*, with PubMed-matched training negatives (`pn_pubmed`).

{{study_leak}}

### 4.4 Targets were studied before they were drugged

**Table 4.4: study before first approval.** For Minikel ≥ Phase III genes, "first drugged" means the first approval of any drug acting on the gene (ChEMBL mechanisms). Each gene's gene2pubmed papers up to 1 or 10 years before that are ranked:
- **`median_percentile`:** against all genes in the universe at that date.
- **`auroc_vs_no_programme`:** against genes that never had a drug programme.

Results:
- **Targets first drugged 2000–2024** were at the 90th–96th percentile ten years before, with AUROC 0.89–0.98 against never-drugged genes.
- **1980–99 rows at 10 years** reach back before PubMed's gene-linked coverage. Almost every gene has zero papers, so the median is 0.5 and uninformative. Use the 1-year rows for that period.

Paper years come from a monotone fit of PMID to PubMed entry year.

{{prior_study}}

### 4.5 Membrane proteins among targets

**Table 4.5: membrane share of targets vs unlabelled genes.** 66–73% of targets are membrane proteins, against 34–40% of other genes. That is why HuRI (Y2H, weak on membrane proteins) covers so few targets.

{{membrane_enrich}}

---

## 5. What the models "discover"

### 5.1 Top-100 audit

**Tables 5.1a–c: the 100 top-ranked test-fold genes that are not labelled targets**, per model (random split, mean over seeds). Columns:
- **`pubmed_pct_median`:** median PubMed percentile.
- **`frac_top_pubmed_decile`:** share in the top 10% most studied.
- **`frac_in_programme`:** share already in a drug programme (Minikel preclinical–Phase II).
- **`frac_shares_pfam_with_known`:** share sharing a Pfam domain with a training target.
- **`frac_within_1hop_of_known`:** share adjacent to a training target in the real graph.
- **`overlap_with_pubmed_top`:** overlap with a PubMed-count ranker's top 100.
- **`rho_score_pubmed`:** Spearman(score, PubMed count) over all test unlabelled genes.

Reference rows: `all test unlabelled` gives the base rates; `pubmed_only` is a PubMed-count ranker.

**How to read it (STRING):**
- **Base rates:** 8% of genes are in the top study decile, 12% in a programme, 32% share Pfam with a known target, 42% sit within 1 hop of one.
- **All-features models:** their top 100 are 24–42% top-decile, 56–67% already in programmes, 64–72% Pfam-sharing, and 82–96% adjacent to a known target.
- **Switching to `attn_free` (RF):** top-decile share falls to 12%, but the Pfam-sharing share rises to 91%. The bias moves from study to family.
- **Adding the STRING graph to `attn_free`:** top-decile share climbs back to 28–40%, and 92–95% of top picks are adjacent to known targets.

STRING:

{{topk_string}}

Reactome FI:

{{topk_reactome_fi}}

HuRI:

{{topk_huri}}

### 5.2 Phase ladder

**Tables 5.2a–c: trained only on ≥ Phase III, how well do models rank test-fold genes with an earlier programme (preclinical / Phase I / Phase II) above genes with none?**
- On STRING, all-features RF and GP give 0.86 overall and 0.82 within PubMed quintiles. The PubMed-only ranker gives 0.82 overall but only 0.61 within quintiles.
- Phase II genes are ranked highest of all (0.91–0.92).

"Predictions" are largely genes the industry is already pursuing.

STRING:

{{ladder_string}}

Reactome FI:

{{ladder_reactome_fi}}

HuRI:

{{ladder_huri}}

---

## 6. Time split: does the label find new targets?

**Setup:**
- **Training positives:** Minikel ≥ Phase III genes first drugged before 2010 (ChEMBL first approval).
- **Test positives:** genes first drugged in 2010 or later.
- **Excluded:** positives with no approval date (Phase III only, or no ChEMBL mechanism).
- **No future leak in matching:** study-matched training and evaluation negatives use PubMed counts from before 2010.
- **Caveat:** `all` features and the STRING graph are today's snapshots and leak post-2010 knowledge. `attn_free` features and HuRI don't.

### 6.1 What the future targets looked like in 2009

**Table 6.1: the targets first drugged from 2010 on, broken down by what was known in 2009.**
- **Strata:** PubMed percentile before 2010; whether their Pfam family already had a drugged member; hops to the nearest already-drugged target in the graph.
- **"Novel":** not top-decile study, new family, and ≥ 2 hops away.

What it shows:
- **More than half were already in the top 10% most-studied genes** (52–57%), and only 2–4% were below median study.
- **On STRING, 77% were directly linked to an existing target and none were more than 2 hops away.**
- **Truly novel targets are rare:** 5% (STRING), 14% (Reactome FI), 26% (HuRI, whose sparse graph puts most genes far apart).

The targets the field actually found next were mostly more of the same.

{{novelty_composition}}

### 6.2 How well are they found, by novelty?

**Tables 6.2a–c: AUROC of each stratum's future targets against all test-fold unlabelled genes.**
- **Reference rankers (top rows):**
  - PubMed count before 2010
  - PubMed count now, which leaks the future
  - number of known targets sharing a Pfam domain
  - closeness to known targets
- **Models:** `pn_pubmed` rows are the study-matched versions. Strata with n < 10 (e.g. `study<0.5`) are noise.

What it shows:
- **One-number rankers do about as well as any model.** On STRING: PubMed before 2010 0.89, family count 0.91, best model (RF, all features) 0.93.
- **With leak-free `attn_free` features, RF scores 0.81 / 0.80 / 0.79:** below PubMed count alone (0.89 / 0.86 / 0.89).
- **Strong dependence on family:** feature-only models with `attn_free` features score 0.77–0.95 on future targets from already-drugged families, but 0.53–0.70 on new families (all three graphs).
- **On novel targets, models drop to 0.52–0.84**, and graph models with `attn_free` features to about 0.52–0.63 on STRING.
- **The graph only helps for the already-linked:** with `attn_free` features on STRING, real-graph models jump from 0.76 (empty) to 0.84–0.92 overall, but stay at 0.52–0.63 on novel targets.

STRING:

{{novelty_auroc_string}}

Reactome FI:

{{novelty_auroc_reactome_fi}}

HuRI:

{{novelty_auroc_huri}}

**Tables 6.2d–f: recall in the top 10% of the test ranking, per stratum.** This is the operational version: of the future targets in a stratum, how many land in the top 10% of the list?
- **The study, family and closeness rankers recall 0–2% of novel targets.**
- **On STRING, C&S and GCN with `attn_free` features and the real graph recall 0–9% of novel targets** (GP 23–29%), against 37–70% of all future targets.
- **All-features feature-only models recall more novel targets** (STRING RF 69%, MLP 71%), but those features leak post-2010 annotation.

STRING:

{{novelty_recall_top10pct_string}}

Reactome FI:

{{novelty_recall_top10pct_reactome_fi}}

HuRI:

{{novelty_recall_top10pct_huri}}

### 6.3 Time split: graph conditions

**Table 6.3a: time split, nnPU, all features** (`real / rewired / empty`).

{{novelty_head_time_nnpu_all}}

**Table 6.3b: time split, nnPU, attention-free features.** On STRING the real graph adds a lot to graph models (C&S 0.892 vs 0.763 empty), but on HuRI it doesn't (0.708 vs 0.752). The literature graph supplies what the clean features lack, and §6.2 shows that what it supplies is proximity to known targets.

{{novelty_head_time_nnpu_attn_free}}

**Table 6.3c: time split, PN with PubMed-matched negatives, attention-free features.**

{{novelty_head_time_pn_pubmed_attn_free}}

**Table 6.3d: time split, by negative regime** (nnPU; all features, then attention-free).

{{novelty_regimes_time}}

{{novelty_regimes_time_attn}}

**Table 6.3e: DeLong, novelty experiment, by split and by feature set.** The pattern is the same as in main:
- The graph gains over empty only with `attn_free` features (43 better / 8 worse).
- Graph models lose to RF with `all` features (12 / 114).

{{delong_novelty_contrast_split}}

{{delong_novelty_contrast_feats}}

### 6.4 A different label: human genetic evidence

**Setup.** This tests whether a label that doesn't encode drug history finds the *novel* future targets better.
- **Same as the Minikel time split:** folds, test genes (targets first drugged from 2010 on, vs the same unlabelled genes), features and models.
- **Different training label:** genes with human genetic evidence for any disease *published before 2010* (Open Targets):
  - fine-mapped GWAS with locus-to-gene ≥ 0.8
  - exome burden p < 2.5e-6
  - ClinVar pathogenic
  - Gene2Phenotype and ClinGen definitive/strong
  - Orphanet
  - Genomics England green
  - UniProt disease variants and literature
- **Pre-2010 genetic evidence is mostly Mendelian;** GWAS barely existed then.
- **Today's GWAS evidence is useless as a label:** even at L2G ≥ 0.8 it covers 9,141 of 15,886 STRING genes.

Both labels are scored on exactly the same test genes, so they can be compared with paired tests.

**Table 6.4a: how study-biased is each label itself?**
- **Columns:** number of positives; AUROC of PubMed count (today, and before 2010) for separating positives from all other genes; median PubMed percentile of the positives.
- **Phase ≥ III targets** are separated from other genes by PubMed count alone at about 0.84–0.90.
- **Genetic genes before 2010** are less extreme but not unbiased: AUROC 0.70–0.74, median at the 70th–75th percentile, against the 88th–94th for Phase ≥ III targets. Mendelian disease genes are curated from the literature.

{{label_bias}}

**Table 6.4b: time split, nnPU, genetic label** (`real / rewired / empty`). Compare with Tables 6.3a–b, which are trained on the Minikel label.

{{genetic_head_time_nnpu_all}}

{{genetic_head_time_nnpu_attn_free}}

**Tables 6.4c–e: AUROC per novelty stratum, Minikel-trained vs genetic-trained, same models** (nnPU). The question is the `novel` and `new family` columns.

STRING:

{{genetic_auroc_string}}

Reactome FI:

{{genetic_auroc_reactome_fi}}

HuRI:

{{genetic_auroc_huri}}

**Tables 6.4f–h: recall in the top 10%, per stratum, both labels.**

STRING:

{{genetic_recall_top10pct_string}}

Reactome FI:

{{genetic_recall_top10pct_reactome_fi}}

HuRI:

{{genetic_recall_top10pct_huri}}

**Table 6.4i: paired DeLong, genetic-trained vs Minikel-trained twin** (same model, graph condition, features, loss and seed; all future targets vs all test unlabelled). `genetic_better / genetic_worse` count significant cells, first by feature set and loss, then by model and graph.

{{genetic_delong}}

{{genetic_delong_model}}

---

## 7. Therapy-area waves

**Setup:** train on targets never used in a therapy area; test on targets used only in it. Waves with fewer than 30 test genes on a graph are skipped, so HuRI has oncology only.

**Table 7.1a: AUROC per wave, nnPU, all features.**
- RF transfers well across waves (0.85–0.93), better than the graph models.
- GAT and GCN are worse with the real graph than with none in every wave.

{{waves_all}}

**Table 7.1b: the same with attention-free features.** Transfer drops (RF 0.66–0.82), most for oncology. Here the STRING graph helps graph models (C&S 0.80–0.84 vs 0.67–0.76 empty). That's the same re-injection as §6.3.

{{waves_attn_free}}

**Table 7.2: DeLong for waves, overall and by feature set.**

{{delong_waves_contrast}}

{{delong_waves_contrast_feats}}

![STRING waves](results/figures/waves/string/1_conditions.png)

---

## 8. Controls: membrane, shuffled labels, random features

### 8.1 Positive control: membrane localisation

**Setup:** label = GO membrane or plasma membrane; features = Pfam and pathways only (no GO, no tractability, so localisation isn't handed to the model).

**Table 8.1a: AUROC, random split, nnPU** (`real / rewired / empty`). Graph models beat their empty-graph versions on most graphs. On HuRI: SIGN 0.833 vs 0.782, GraphSAGE 0.818 vs 0.770.

{{membrane_head_random_nnpu}}

**Table 8.1b: DeLong.**
- **Real beats empty in 55 of 72 cells (+0.047)** and loses in 7, compared with roughly a draw on drug targets.
- **Per model, every model gains on STRING and HuRI.**

The pipeline can show a graph effect when the label follows the network.

{{delong_membrane_contrast}}

{{delong_membrane_empty_model}}

**Figure 8.1: real − empty on drug targets vs membrane, per model and graph.**

![drug vs membrane](results/figures/compare/6_drug_vs_membrane.png)
![STRING membrane](results/figures/membrane/string/1_conditions.png)

### 8.2 Shuffled labels

**Table 8.2a: labels shuffled across all genes** (nnPU, random split). Everything sits at about 0.50; the harness doesn't invent signal.

{{nl_minikel_perm}}

**Table 8.2b: labels shuffled only within Pfam families.** This keeps how many targets each family has, but not which gene in it.
- **RF still reaches 0.74–0.77 and the GP 0.75–0.78** (MLP 0.57–0.75). Family membership alone carries most of the signal.
- **Graph-only models stay low** (label propagation 0.57–0.69).

{{nl_minikel_famperm}}

**Table 8.2c: DeLong by label set.** No graph gain on either: real vs empty is 0 / 2 and 4 / 10.

{{delong_nonsense_labels_contrast_labels}}

### 8.3 Random features

**Table 8.3a: node features replaced by Gaussian noise.**
- **Without a graph,** models are at chance (0.46–0.51).
- **With the real STRING graph,** graph models reach 0.69–0.89 (C&S 0.886, GAT 0.870), and Reactome FI up to 0.82. On HuRI they stay at 0.47–0.60.

The literature graph *alone* carries the label, through study-correlated topology.

{{nf_random}}

**Table 8.3b: DeLong.** Real beats empty 32 / 0, and beats RF 49 / 0 (RF has only noise).

{{delong_nonsense_feats_contrast}}

---

## 9. Caveats

- **PU labels:** an unlabelled gene can be a well-studied target that never reached Phase III. The phase ladder (§5.2) shows many top-ranked "negatives" are exactly that.
- **Leakage in the time split:**
  - `all` features (GO, pathways, tractability) and the STRING graph are current snapshots, so they leak post-2010 knowledge.
  - `attn_free` features and HuRI are leak-free.
  - Matching uses pre-2010 PubMed counts.
  - The `novel` strata are small (7 / 17 / 11 genes).
- **Dating:**
  - First-drugged year is the first approval of a drug acting on the gene in ChEMBL (single proteins and complexes).
  - 353 of 702 Minikel positives have no ChEMBL approval date and are excluded from the time split.
  - Paper years come from a monotone PMID → entry-year fit. Back-indexed papers get later years, which can only *understate* early study.
- **"Attention-free" is relative:** it removes literature annotation, not study bias (ρ ≈ 0.6 with PubMed, §4.3).
- **Tuning:** done once per graph, model, loss and graph condition, on the random split with Minikel labels and all features, then reused for other feature sets and splits. RF on STRING hit the tuning time limit (11 trials).
- **HuRI** covers only 32% of targets, and its pathway subgraph is nearly empty (832 edges).
- **Pending rerun:** 1,017 `main` runs lost their score files on the cluster. Their AUROCs are in the tables (from the run records), but they're missing from the per-gene analyses (top-k audit, phase ladder, degree figure). The runner now re-runs them; see the note at the end.

## 10. Next steps

1. **More labels on the time split** (§6.4 has genetic evidence): Phase I entries, or genetic evidence restricted to GWAS and burden (hypothesis-free) once enough pre-2010 data exists.
2. **Leak-free literature features:** rebuild GO and pathway features from annotations dated before 2010 (GO annotation dates exist), to make the `all`-features time split fair.
3. **gnomAD constraint and AlphaFold pockets,** as further non-literature features, run through §4.3's leak test.
4. **Write-up:** framed as complementary to Rahimi et al. On essentiality (network-coherent biology) GNNs help; on drug targets, with proper controls and a time split, they mostly reproduce the field's attention.

---

*Generated by `scripts/report.py` from `scripts/report_template.md`, `results/` (CSV tables) and `results/figures/`. Analyses: `scripts/analysis.py` (`results/analysis/*.csv`); significance tests: `scripts/delong.py`; figures: `scripts/figures.py`.*
