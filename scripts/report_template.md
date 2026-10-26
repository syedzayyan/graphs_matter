# Does the graph help predict drug targets? Full results

This covers the complete cluster run: six experiments, three graphs, 5 seeds each, everything tuned per graph. Every table below is generated from `results/` by `scripts/report.py`. Numbers in the TL;DR and in the headline bullets of §2, §6 and §6.4 are filled in from the same CSVs. Other numbers in the prose are hand-written and must be re-checked after each rerun. Figures are linked from `results/figures/`. Open this file in a Markdown preview (VS Code: ⇧⌘V) to see them inline.

---

## TL;DR: the hypothesis

**Claim.** "Reached Phase III or later" records which genes the pharmaceutical industry chose to work on. It does not record what makes a gene a good target. A model that predicts this label well is a good imitation of past portfolio decisions. Trained on it with literature-derived features, target-discovery models learn what the field has already paid attention to, so their "discoveries" go round in a circle: well-studied genes, already-drugged families, neighbours of known targets.

**Why predicting future approvals doesn't rescue the models.** The obvious defence is a time split: train on targets first drugged before 2010, then score the ones first drugged from 2010 on. Those later targets were themselves mostly more of the same (§6.1, numbers as STRING / Reactome FI / HuRI):
- {{n:fut_top_decile}} were already in the top PubMed decile in 2009.
- {{n:fut_1hop}} were one hop from an existing target.

A model that finds them is still imitating portfolio decisions, one decade later. Predicting the label well and finding new targets are different tasks, and a Phase-based benchmark can only score the first. So this report asks *which* future targets get found, by how novel they were in 2009 (§6.2), and whether a label that doesn't record drug history does better (§6.4).

**The bias enters through three doors:**
1. **The label.** Phase ≥ III genes were among the most studied in the genome years before their first drug: the {{n:prior_pct_10y}} PubMed percentile ten years before (§4.4). PubMed count alone separates them from other genes at AUROC {{n:lb_minikel_auroc}} (§6.4).
2. **The features.** GO, pathways and Open Targets tractability are written by the literature, so they encode study intensity:
   - A random forest recovers a gene's PubMed count from them at Spearman ρ {{n:leak_all}}.
   - "Attention-free" sequence and expression features still give {{n:leak_attn_free}}, because study is concentrated in particular protein families.
   - Annotations as they stood in 2009 give {{n:leak_dated}}.
   - No per-feature correction removes this (§4.3).
3. **The graph.** STRING is built from the literature: its degree is study intensity (fully mediated by PubMed count, §4.1). Message passing pulls predictions towards neighbours of known targets.

**What the time split shows** (§6.2; STRING / Reactome FI / HuRI):

| ranker | AUROC, all future targets | recall in top 10%, novel targets |
|---|---|---|
| PubMed count in 2009 | {{n:ref_pubmed_before_T_auroc}} | {{n:ref_pubmed_before_T_recall_novel2}} |
| number of drugged Pfam relatives | {{n:ref_known_family_count_auroc}} | {{n:ref_known_family_count_recall_novel2}} |
| RF, only what was known in 2009 (`dated`) | {{n:rf_time_minikel_dated}} | {{n:rf_time_minikel_dated_recall_novel2}} |
| RF, attention-free features | {{n:rf_time_minikel_attn_free}} | {{n:rf_time_minikel_attn_free_recall_novel2}} |
| RF, today's features (leak post-2010 knowledge) | {{n:rf_time_minikel_all}} | {{n:rf_time_minikel_all_recall_novel2}} |

- **"Novel" here** means at least two of: below the top study decile in 2009, a Pfam family with no drugged member, ≥ 2 hops from a known target. That is {{n:novel2_n}} genes; the strict all-three definition gives only {{n:novel_n}}.
- **Paired differences from the PubMed ranker,** with 95% bootstrap CIs over the future targets, are in Table 6.2h.
- **What it means:**
  - Pooled over graphs, RF with `dated` features puts half the novel future targets in its top 10%, against a quarter for the PubMed ranker (+0.26 [+0.11, +0.41]). So the models are not *only* re-ranking study.
  - This is a low bar: the novel stratum is defined partly by low 2009 study and by new families, which handicaps the two simple rankers by construction.
  - The gain is mostly on STRING and Reactome FI. On HuRI, among leak-free models only RF with `dated` features beats the PubMed ranker.

**For the GNN question.** Effect sizes are mean AUROC differences over models with 95% hierarchical-bootstrap CIs (§2.3):
- **Real graph vs no edges, same model:** {{n:eff_main_empty}} overall.
  - With `all` features: {{n:eff_main_empty_all}}.
  - With `attn_free` features: {{n:eff_main_empty_attn_free}}.
- **Graph model vs a plain random forest on the same features:** {{n:eff_main_vs_rf_feat}}.

The graph clearly helps only in two cases:
- on a label that genuinely follows the network (membrane localisation, §8.1, {{n:eff_membrane_empty}});
- when the features are uninformative, where the graph re-injects study intensity (§3.3, §8.3).

So the null result is about the task and its label, not the method.

**What this does and doesn't show.** It shows that the Phase ≥ III label plus current features reward rediscovery. §6.4 tests whether a different label escapes the loop: the same time split, trained on human genetic evidence published before 2010, with the reading of each outcome fixed in advance.
- **Result: outcome B by the pre-stated rule, but C's mechanism is visible.** The genetic label finds *fewer* novel future targets than the Minikel label on every primary comparison: {{n:primary_rf_feat_dated}} for RF with `dated` features, {{n:primary_gp_attn_free}} for the GP with `attn_free`. Over the whole ranking, its models follow 2009 study as closely as the Minikel ones.
- **Its top picks are different genes.** Of RF's top 100, {{n:tk_rf_genetic_frac_shares_pfam_with_known}} share a Pfam domain with a 2009 target, against {{n:tk_rf_minikel_frac_shares_pfam_with_known}} for Minikel. They are less often already in a programme and less often top-decile studied. Only {{n:tk_rf_genetic_frac_future_target}} became drug targets, against {{n:tk_rf_minikel_frac_future_target}} (Table 6.4l).
- **Reading:** the benchmark counts a hit only when a company later drugged the gene, so a label that leaves the portfolio loses by construction. This is evidence that the benchmark measures imitation, not that genetics finds worse targets. The features still add study bias on top of any label.

---

## How to read this

- **AUROC:** known targets ranked against unlabelled genes (PU setting). An unlabelled gene isn't a true negative, so AUROCs measure "ranks known targets high", not "finds targets".
- **Cells in model tables:** `real / rewired / empty`, meaning the same model with the real graph, degree-preserving rewired copies (mean of 3), and no edges. "(no graph)" marks models that never use a graph. All values are means over 5 seeds.
- **†** marks a model that never fit its own training data (training-fold AUROC < 0.6). That is a training failure, not a result. There are {{n:n_fit_fail}} such cells, all listed in Table 9.1. The flag doesn't apply to Correct & Smooth or label propagation (§9).
- **`x [lo, hi]`:** a point estimate with its 95% bootstrap CI.
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
  - **dated:** only what was known before 2010: attention-free blocks, Pfam, GO as annotated by Dec 2009 (GOA release 80) and KEGG pathways as of Sep 2009. Time split only (§6).
  - **no_loc:** Pfam and pathways (membrane control).
  - **random:** Gaussian noise.
- **Effect sizes (primary):**
  - Mean AUROC difference per contrast, weighting every model equally: the mean over models of the mean over that model's configurations of the mean over seeds.
  - The 95% CI resamples models, then configurations within each model, then seeds.
  - Graphs are too few (3) to resample, so they are reported separately.
- **DeLong tables (supplementary):**
  - Each cell is one (graph, model, loss, split, feature set) combination: paired DeLong per seed, combined with Stouffer, Benjamini–Hochberg within each experiment.
  - `real_better / real_worse` count cells significant at 5%; `mean_delta` is the mean AUROC difference.
  - Cells share seeds, genes and configurations, so counts of significant cells overstate the evidence. Read them alongside the effect sizes.
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
- **No model beats the GP or RF by much**, and RF uses no graph at all: 0.926 / 0.921 / 0.910 (GP 0.941 / 0.919 / 0.928).
- **GAT and GCN are worse with the real graph than with no edges** on all three graphs: STRING GAT 0.882 vs 0.920, HuRI GAT 0.781 vs 0.868.
- **Correct & Smooth gains on STRING (0.935 vs 0.912) and HuRI (0.850 vs 0.815); SIGN gains on STRING (0.916 vs 0.900).** Neither gains on Reactome FI (C&S 0.905 vs 0.919, SIGN 0.916 vs 0.918).

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

**Table 2.2a: random split → Pfam split, per feature set** (nnPU; feature-only models without a graph, graph models with the real graph; cells `random → pfam (Δ)`). The Pfam split holds out whole families at test time.

**Why this matters.** Labels shuffled only within Pfam families keep most of the signal (§8.2), which reads as "models learn resemblance to drugged families". That reading predicts that holding families out should hurt most where nothing else carries signal across families:
- **`all` features** carry tractability and GO, which do transfer across families, so a small drop says little.
- **`attn_free` features** (ESM-2, sequence, expression) transfer mostly through family similarity, so they are the test.

**Interpretation, fixed before the rerun:**
- **If RF with `attn_free` features drops clearly more than with `all`,** the family shortcut is confirmed: without literature annotation, what the model learned was resemblance to drugged families.
- **If it drops about as little,** §8.2 has to be weakened: family membership explains much of the label, but models also find signal that crosses families.

**RF on the Pfam split** (STRING / Reactome FI / HuRI):
- `all`: {{n:rf_pfam_all}} (change {{n:rf_pfam_drop_all}})
- `no_tract`: {{n:rf_pfam_no_tract}} (change {{n:rf_pfam_drop_no_tract}})
- `attn_free`: {{n:rf_pfam_attn_free}} (change {{n:rf_pfam_drop_attn_free}})

**Result: the first reading holds, for feature-only models.**
- RF with `attn_free` features loses 0.07–0.09 when families are held out, against 0.005–0.018 with `all`. Without literature annotation, much of what RF learned was resemblance to drugged families.
- It still reaches 0.72–0.76 on unseen families, so some signal does cross families. §8.2's "family membership alone carries most of the signal" stands, but not "all of it".
- The MLP drops by similar amounts with every feature set, so the contrast is clearest for RF.
- Graph models drop less with `attn_free` features (GP −0.018 / −0.024 / +0.016). With clean features they lean on the graph (§3.2), and the graph links a new family to known targets just as well.

{{pfam_drop}}

**Table 2.2b: nnPU, all features, Pfam split** (`real / rewired / empty`).

{{main_compact_pfam_nnpu}}

**Table 2.2c: PN, Pfam split.**

{{main_compact_pfam_pn}}

### 2.3 Effect sizes and significance

**Table 2.3a: effect sizes, main experiment** (random negatives).
- **Cells:** mean AUROC difference over models with a 95% hierarchical-bootstrap CI.
- **Rows:** overall, then by feature set, graph, split and loss.
- **Columns:** the contrasts.
  - `empty`: real graph vs no edges.
  - `rewired`: real vs degree-preserving rewiring.
  - `pwsub`: real vs pathway-only subgraph.
  - `shufattr`: real vs shuffled edge vectors.
  - `vs_rf_feat`: graph model (real graph) vs RF on the features alone.

What it shows:
- **Real vs empty:** {{n:eff_main_empty}} overall.
  - With `all` features: {{n:eff_main_empty_all}}.
  - With `attn_free`: {{n:eff_main_empty_attn_free}}.
  - By graph (STRING / Reactome FI / HuRI): {{n:eff_main_empty_string}}, {{n:eff_main_empty_reactome_fi}}, {{n:eff_main_empty_huri}}.
- **Real vs rewired:** {{n:eff_main_rewired}}. A fake graph actively damages a model, so this overstates the real graph's value.
- **Graph model vs feature-only RF:** {{n:eff_main_vs_rf_feat}}.
  - With `all` features: {{n:eff_main_vs_rf_feat_all}}.
  - With `attn_free`: {{n:eff_main_vs_rf_feat_attn_free}}.

{{effects_main}}

**Table 2.3b: counts of significant DeLong cells (supplementary).**
- **Real vs empty:** {{n:cnt_main_empty}}.
  - `all` features: {{n:cnt_main_empty_all}}.
  - `attn_free`: {{n:cnt_main_empty_attn_free}}.
- **Graph model vs RF:** {{n:cnt_main_vs_rf_feat}}.

{{delong_main_contrast}}

**Table 2.3c: DeLong counts by feature set.** The graph helps when the features don't already carry the information it adds (see §3.3 for what that is).

{{delong_main_contrast_feats}}

**Table 2.3d: DeLong counts by graph.** STRING is the literature-derived graph.

{{delong_main_contrast_universe}}

**Table 2.3e: DeLong counts by split.**

{{delong_main_contrast_split}}

**Table 2.3f: DeLong counts by loss.**

{{delong_main_contrast_loss}}

**Table 2.3g: real vs empty per model and graph**: `better/worse of cells (mean Δ)`, over all feature sets, splits and losses. Correct & Smooth gains most on STRING (23/0, +0.110) and not elsewhere. SIGN gains on STRING and HuRI, GraphSAGE on all three. GCN loses on Reactome FI and HuRI, GAT on Reactome FI, GAT + edge vec on Reactome FI and HuRI.

{{delong_main_empty_model}}

**Table 2.3h: every contrast under every negative regime.** Against 1-hop negatives (unlabelled direct neighbours of targets), the real graph is *worse* than the empty one (65 better / 202 worse of 432). Message passing raises the neighbours of known targets, which is exactly the "linked to known targets" bias.

{{delong_main_regimes}}

**Figure 2.2: paired graph effect** (real − empty and real − rewired, per seed).

![STRING graph effect](results/figures/main/string/2_graph_effect.png)
![Reactome FI graph effect](results/figures/main/reactome_fi/2_graph_effect.png)
![HuRI graph effect](results/figures/main/huri/2_graph_effect.png)

### 2.4 Under harder negatives

**Table 2.4: AUROC by evaluation-negative regime** (random split, nnPU, all features; feature-only models with no graph, graph models with the real graph).
- Every model drops sharply against PubMed-matched negatives (STRING RF 0.926 → 0.840) and against same-family negatives (→ 0.789).
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
- **`dated` features** (2009 annotations, §6) sit in between: {{n:leak_dated}}. Annotation in 2009 already followed study in 2009, which is fair game for a forecaster at the time.

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
- **All-features models:** their top 100 (nnPU) are 27–43% top-decile, 56–68% already in programmes, 65–75% Pfam-sharing, and 84–97% adjacent to a known target.
- **Switching to `attn_free` (RF):** top-decile share falls to 12%, but the Pfam-sharing share rises to 93%. The bias moves from study to family.
- **Adding the STRING graph to `attn_free`:** top-decile share climbs back to 29–40% (C&S, GCN, GP), and 91–95% of top picks are adjacent to known targets.

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
- **Feature sets and leakage:**
  - **`all`** features and the STRING graph are today's snapshots, so they leak post-2010 knowledge.
  - **`attn_free`** features and HuRI don't leak, but `attn_free` features are weaker.
  - **`dated`** answers "are the clean features just weak?" It is the attention-free blocks plus Pfam, GO as annotated by December 2009 (GOA release 80) and KEGG pathways as of September 2009. That is roughly what a forecaster in 2009 could have used. Tractability can't be dated, so it is left out.

### 6.0 Who the time split leaves out

**Table 6.0: Minikel ≥ Phase III genes by whether they could be dated** (plus never-drugged genes for reference). Study and novelty are measured against the targets known before 2010.

**Why it matters.** {{n:cov_n_excluded}} STRING positives have no ChEMBL approval date and are left out. If they were systematically less studied or further from known targets, the time split would favour rediscovery by construction.

**What it shows (STRING):** the excluded genes look like the test genes.
- Top-decile study in 2009: {{n:cov_excl_top}} (launched / Phase III only) vs {{n:cov_test_top}} in the test set.
- In a drugged family: {{n:cov_excl_fam}} vs {{n:cov_test_fam}}.
- Novel (2 of 3): {{n:cov_excl_novel2}} vs {{n:cov_test_novel2}}.

They are slightly less family-bound and slightly more novel. On Reactome FI the gap is larger: the excluded genes are less studied and more often ≥ 2 hops from a known target. So the exclusion tilts the time split towards rediscovery a little, most on Reactome FI. It does not create the "more of the same" pattern: the never-drugged reference rows show how far all three groups sit from typical genes.

{{time_split_coverage}}

### 6.1 What the future targets looked like in 2009

**Table 6.1: the targets first drugged from 2010 on, broken down by what was known in 2009.**
- **Strata:**
  - PubMed percentile before 2010
  - whether their Pfam family already had a drugged member
  - hops to the nearest already-drugged target in the graph
- **"Novel":** not top-decile study, new family, and ≥ 2 hops away.
- **"Novel (2 of 3)":** any two of those three. It is the larger stratum, used for the main comparisons. On HuRI almost every gene is ≥ 2 hops from everything, so there it mostly means "new family or not top-decile".
- **`pooled`:** the three graphs together, each gene counted once.

What it shows (STRING / Reactome FI / HuRI):
- **Most were already heavily studied:** {{n:fut_top_decile}} were in the top 10% most-studied genes in 2009, and only {{n:fut_below_median}} were below median study.
- **Most were close to known targets:** {{n:fut_1hop}} were directly linked to an existing target, and {{n:fut_3hop}} were 3 or more hops away.
- **Truly novel targets are rare:** {{n:novel_n}} genes ({{n:novel_share}}), {{n:novel_n_pooled}} pooled. Under the 2-of-3 definition: {{n:novel2_n}}, {{n:novel2_n_pooled}} pooled.

The targets the field actually found next were mostly more of the same. That is what the industry did, not what was possible. A model that ranks these targets well has learned the industry's habits.

{{novelty_composition}}

### 6.2 How well are they found, by novelty?

**Reference rankers** (top rows of every table):
- PubMed count before 2010
- PubMed count now, which leaks the future
- number of known targets sharing a Pfam domain
- closeness to known targets

`pn_pubmed` rows are the study-matched versions of each model.

**Tables 6.2a–c: point estimates per stratum** (AUROC of each stratum's future targets against all test-fold unlabelled genes; mean over seeds). Strata with fewer than about 15 genes are too small to read without the CIs in Tables 6.2g–j.

STRING:

{{novelty_auroc_string}}

Reactome FI:

{{novelty_auroc_reactome_fi}}

HuRI:

{{novelty_auroc_huri}}

**Tables 6.2d–f: recall in the top 10% of the test ranking, per stratum.** This is the operational question: of the future targets in a stratum, how many land in the top 10% of the list?

STRING:

{{novelty_recall_top10pct_string}}

Reactome FI:

{{novelty_recall_top10pct_reactome_fi}}

HuRI:

{{novelty_recall_top10pct_huri}}

**Tables 6.2g–j: novel targets with 95% CIs.**
- **The CI** resamples the future targets in the stratum (2000 draws). Which targets happened to come up after 2010 is the main uncertainty for strata this small. The test negatives (thousands of genes) are held fixed.
- **Pooled:** each gene counts once, even when it is in several graphs.
- **Differences from the PubMed-before-2010 ranker** (Tables 6.2h, 6.2j) are paired: the same bootstrap draws for both.

**Table 6.2g: recall in the top 10%, novel (2 of 3).**

{{novelty_ci_recall_top10pct_novel2}}

**Table 6.2h: the same, minus the PubMed-before-2010 ranker.** A model finds more novel targets than study alone only where the CI is above 0.

What it shows (pooled column):
- **`all` features:** every model beats the PubMed ranker except the study-matched C&S and GCN (+0.16 to +0.37 for nnPU). These features leak post-2010 knowledge.
- **`dated` features:** +0.15 to +0.26 for nnPU, with CIs at or above 0 (RF +0.26 [+0.11, +0.41]; GCN's lower bound is +0.00).
- **`attn_free` features:** +0.12 to +0.16, with every CI touching 0.
- **By graph:** the gains come from STRING (all CIs above 0) and mostly Reactome FI. On HuRI, only RF with `dated` (+0.24 [+0.02, +0.46]) and the leaky `all` models other than GCN beat the ranker.
- **Study-matched training (`pn_pubmed`) doesn't help here:** RF `dated` falls from 0.50 to 0.39 recall.
- **Caveat:** "novel (2 of 3)" partly means "not top-decile in 2009", so the PubMed ranker is handicapped by construction. Beating it is a floor, not evidence of discovery.

{{novelty_vs_pubmed_recall_top10pct_novel2}}

**Table 6.2i: AUROC, novel (2 of 3).**

{{novelty_ci_auroc_novel2}}

**Table 6.2j: AUROC minus the PubMed-before-2010 ranker, novel (2 of 3).**

{{novelty_vs_pubmed_auroc_novel2}}

**Tables 6.2k–l: the strict definition (all three criteria; very small n) and new Pfam families only.**

{{novelty_ci_recall_top10pct_novel}}

{{novelty_ci_recall_top10pct_newfam}}

### 6.3 Time split: graph conditions

**Table 6.3a: time split, nnPU, all features** (`real / rewired / empty`).

{{novelty_head_time_nnpu_all}}

**Table 6.3b: time split, nnPU, attention-free features.** On STRING the real graph adds a lot to graph models (C&S 0.892 vs 0.763 empty, GCN 0.867 vs 0.764), but on HuRI it doesn't (C&S 0.708 vs 0.752). The literature graph supplies what the clean features lack, and §6.2 shows that what it supplies is proximity to known targets.

{{novelty_head_time_nnpu_attn_free}}

**Table 6.3c: time split, nnPU, `dated` features** (only what was known in 2009). RF alone reaches 0.904 / 0.891 / 0.896. Only the GP matches it, and only on STRING (0.905). The STRING graph still helps graph models (C&S 0.891 vs 0.834 empty), but less than with `attn_free`. On Reactome FI and HuRI the empty graph is usually better.

{{novelty_head_time_nnpu_dated}}

**Table 6.3d: time split, PN with PubMed-matched negatives, attention-free and `dated` features.**

{{novelty_head_time_pn_pubmed_attn_free}}

{{novelty_head_time_pn_pubmed_dated}}

**Table 6.3e: time split, by negative regime** (nnPU; all, attention-free, then `dated` features).

{{novelty_regimes_time}}

{{novelty_regimes_time_attn}}

{{novelty_regimes_time_dated}}

**Table 6.3f: effect sizes, novelty experiment** (hierarchical bootstrap, as in Table 2.3a). DeLong counts by split and feature set follow as a supplement.
- **Real vs empty on the time split:** -0.004 [-0.029, +0.022]. By feature set: `attn_free` +0.021 [-0.001, +0.041], `dated` -0.013 [-0.041, +0.017].
- **Graph model vs RF:** -0.052 [-0.085, -0.021] with `dated` features.
- **Genetic vs Minikel label, overall AUROC:** -0.127 [-0.157, -0.091]. The genetic-trained twin is worse in 200 of 216 cells and better in none. That matches the smoke test (−0.11 to −0.14), which is why overall AUROC is not the §6.4 endpoint.

{{effects_novelty}}

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
- **Phase ≥ III targets** are separated from other genes by PubMed count alone at {{n:lb_minikel_auroc}}.
- **Genetic genes before 2010** ({{n:lb_genetic_T_n}} genes) are less extreme but not unbiased: {{n:lb_genetic_T_auroc}}. Mendelian disease genes are curated from the literature.

{{label_bias}}

**How the outcome will be read (fixed before the genetic results; see the git history of `scripts/report_template.md`).**

- **Primary endpoint.** Recall in the top 10% of the *novel (2 of 3)* future targets, pooled over the three graphs. It is measured as genetic-trained minus Minikel-trained, paired: same model, features, loss and seeds, with a 95% bootstrap CI over the targets.
  - It is computed for RF and the GP (nnPU) with the two leak-free feature sets, `attn_free` and `dated`.
  - Overall AUROC on all future targets is *not* the endpoint. The genetic label was expected to do worse there (the smoke test gave −0.11 to −0.14), because most future targets are rediscoveries that the Minikel label is built to find.
- **Supporting measure.** How closely each label's ranking follows 2009 study: Spearman(score, PubMed count before 2010) over the test genes (Table 6.4j).

**The outcomes, and what each would mean:**

| outcome | what it would mean |
|---|---|
| **A.** CI above 0 for both leak-free feature sets | A label that doesn't encode drug history finds more of the targets the field later reached in new territory. Label choice, not model choice, is the lever. This is the constructive half of the paper. |
| **B.** CI includes or is below 0, *and* genetic-trained rankings follow study about as closely as Minikel-trained ones | The features carry the bias. Even a less study-biased label (PubMed AUROC {{n:lb_genetic_T_auroc}} vs {{n:lb_minikel_auroc}}) can't escape, because the ranking is driven by what the features know about well-studied genes. The thesis becomes "the loop is in the features as much as the label". |
| **C.** CI below 0, *and* genetic-trained rankings follow study clearly less | The genetic label escapes study bias, but it points at different genes from the ones companies went on to drug: Mendelian disease genes are mostly not drugged. This supports the claim that the label *defines* what counts as a target, so a benchmark built on it measures imitation. It is not evidence that genetics finds better targets. |

**Primary endpoint** (genetic − Minikel, recall in the top 10%, novel (2 of 3), pooled):
- RF, `attn_free`: {{n:primary_rf_feat_attn_free}}
- RF, `dated`: {{n:primary_rf_feat_dated}}
- GP, `attn_free`: {{n:primary_gp_attn_free}}
- GP, `dated`: {{n:primary_gp_dated}}

**Supporting:** genetic − Minikel in Spearman(score, PubMed 2009) ranges from {{n:rho_shift}} across models, feature sets and graphs. For the four primary models, pooled over graphs, it is −0.03 to +0.05:

| model | genetic | Minikel | genetic − Minikel |
|---|---|---|---|
| RF, `attn_free` | +0.23 | +0.21 | +0.01 |
| RF, `dated` | +0.53 | +0.56 | −0.03 |
| GP, `attn_free` | +0.41 | +0.38 | +0.02 |
| GP, `dated` | +0.47 | +0.43 | +0.05 |

The extreme values in the full range come from cells outside the primary endpoint (MLP with `all` features on HuRI, −0.61; GP with `all` features on HuRI, +0.29).

**Result: outcome B by the rule fixed in advance, but the top of the ranking shows outcome C's mechanism, and this benchmark can't separate the two.**
- **All four primary CIs are below 0.** The genetic label puts fewer novel future targets in its top 10% than the Minikel label, by 0.15 to 0.28.
- **Over the whole ranking, the genetic-trained models follow 2009 study as closely as the Minikel-trained ones** (Spearman above), although the label itself is much less study-biased (PubMed AUROC {{n:lb_genetic_T_auroc}} vs {{n:lb_minikel_auroc}}). By the pre-stated rule this is outcome B.
- **At the top of the ranking, the two labels pick different genes** (Table 6.4l). For RF's top 100, genetic vs Minikel, STRING / Reactome FI / HuRI:
  - sharing a Pfam domain with a 2009 target: {{n:tk_rf_genetic_frac_shares_pfam_with_known}} vs {{n:tk_rf_minikel_frac_shares_pfam_with_known}} (base rate {{n:tk_base_frac_shares_pfam_with_known}})
  - in a drug programme today (preclinical–Phase II): {{n:tk_rf_genetic_frac_in_programme}} vs {{n:tk_rf_minikel_frac_in_programme}} (base rate {{n:tk_base_frac_in_programme}})
  - in the top study decile of 2009: {{n:tk_rf_genetic_frac_top_pubmed_decile_T}} vs {{n:tk_rf_minikel_frac_top_pubmed_decile_T}} (base rate {{n:tk_base_frac_top_pubmed_decile_T}})
  - with genetic evidence before 2010: {{n:tk_rf_genetic_frac_genetic_before_T}} vs {{n:tk_rf_minikel_frac_genetic_before_T}} (base rate {{n:tk_base_frac_genetic_before_T}})
  - future targets: {{n:tk_rf_genetic_frac_future_target}} vs {{n:tk_rf_minikel_frac_future_target}} (base rate {{n:tk_base_frac_future_target}})
- **So the genetic label largely leaves the drugged families, and somewhat leaves the studied and already-investigated genes.** Its top picks are disease genes, and companies mostly did not go on to drug them. That is outcome C's mechanism. Spearman is computed over all test genes, so it can't show a change at the top. Its picks are still more studied than the base rate, so the features do pull study bias back in (the part of B that holds).
- **What this means for the question.** A hit here is a gene a company later took to Phase III. A label that points away from the portfolio is wrong by construction on this benchmark, whether or not its genes are good targets. §6.4 therefore can't show that genetics generalises worse to new targets. What it shows is that the benchmark rewards imitating the portfolio, and that every label picks up some study bias from the features. Testing whether genetics finds *better* targets needs an outcome not chosen by drug development, such as success given that a trial started.

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

**Tables 6.4f–h: recall in the top 10%, per stratum, both labels** (point estimates).

STRING:

{{genetic_recall_top10pct_string}}

Reactome FI:

{{genetic_recall_top10pct_reactome_fi}}

HuRI:

{{genetic_recall_top10pct_huri}}

**Table 6.4i: novel (2 of 3) targets, both labels, with 95% CIs**: recall in the top 10%, then the paired difference genetic − Minikel (the primary endpoint is in the `pooled` column), then the same for AUROC.

{{genetic_ci_recall_top10pct_novel2}}

{{genetic_vs_minikel_recall_top10pct_novel2}}

{{genetic_ci_auroc_novel2}}

{{genetic_vs_minikel_auroc_novel2}}

**Table 6.4j: how closely each label's ranking follows study in 2009**: Spearman(score, PubMed count before 2010) over the test genes, mean over seeds.

{{genetic_rho}}

**Table 6.4k: paired DeLong, genetic-trained vs Minikel-trained twin** (same model, graph condition, features, loss and seed; all future targets vs all test unlabelled). `genetic_better / genetic_worse` count significant cells, first by feature set and loss, then by model and graph.

{{genetic_delong}}

{{genetic_delong_model}}

**Table 6.4l: what each label's models rank highest on the time split.** The top 100 test genes (future targets plus never-drugged genes) per run, mean over seeds; nnPU, RF and the GP, leak-free feature sets. Columns: share that are future targets (first drugged from 2010 on), novel (2 of 3) future targets, in the top PubMed decile before 2010, in a drug programme today (Minikel preclinical–Phase II; no dates, so some programmes began after 2010), sharing a Pfam domain with a target drugged before 2010, within 1 hop of one in the real graph, and with genetic evidence published before 2010. A gene with pre-2010 genetic evidence can be a test gene, because the time split's test genes are drawn without regard to the genetic label.

{{genetic_topk}}

---

### 6.5 Success given tried: selection vs outcome

**Why this matters.** The Phase ≥ III label mixes two things: which genes the industry chose to take into trials (*selection*), and which of those then worked (*outcome*). §4–§6 show that models predict the label mostly from study and family resemblance. This section asks whether that predictive power is about selection or about outcome. It also gives the genetic label of §6.4 an outcome that drug development did not choose.

**Groups** (Minikel's target–indication pairs, summarised per gene):
- **never tried:** no programme beyond preclinical
- **tried:** best phase ≥ Phase I
- **success:** best phase ≥ Phase III (exactly the Minikel label)
- **resolved failure:** tried, best phase Phase I or II, and no programme still active. Genes whose best programme is still active at Phase I/II are censored and left out.

**Contrasts** (AUROC, with 95% bootstrap CIs over genes, 2000 draws, paired between predictors):
- **label:** success vs every other gene (what the benchmark scores)
- **selection:** tried vs never tried
- **outcome:** success vs resolved failure

**Predictors:**
- PubMed count before 2010 and now
- genetic evidence before 2010 and at any date (binary)
- out-of-fold model scores from the main experiment (random split, Minikel label, nnPU): each gene's percentile among the test genes, averaged over the seeds in which it was a test gene
- on the time split, both labels' models (§6.4), scored on the test genes: outcome = future targets vs resolved failures, selection = tried vs never-tried test genes

**How the outcome will be read (fixed before these results; see the git history of `scripts/report_template.md`).**
- **R1. Is the label mostly selection?** For PubMed before 2010 and for RF with `all` and `attn_free` features: if outcome AUROC − 0.5 is less than half of selection AUROC − 0.5, the Phase ≥ III label mostly records which genes were tried. A benchmark built on it then mostly measures predicting the industry's choices. If outcome AUROC is close to selection AUROC, the label carries real outcome information and the portfolio critique has to be weakened.
  - *Built-in bias:* successful drugs generate papers, and PubMed counts are not dated to before each trial. This pushes PubMed's outcome AUROC *up*, so a low value is conservative.
- **R2. Does genetic evidence carry outcome signal?** Odds ratio for success vs resolved failure, for genetic evidence before 2010. If its CI is above 1, the genetic label carries outcome signal that drug selection alone doesn't explain. If the CI includes 1, there is no support at the gene level. Gene-level evidence is coarser than Minikel et al.'s indication-matched support, so this is a weak test.
- **R3. Is the genetic label's §6.4 deficit about selection or outcome?** On the time split, paired genetic − Minikel AUROC, pooled over graphs, for RF and the GP with `attn_free` and `dated`:
  - If the outcome CI includes or exceeds 0 while the selection CI is below 0, the deficit is consistent with a selection deficit: the genetic label loses at predicting which genes get tried, not which tried genes succeed.
  - If the outcome CI is below 0 as well, the genetic-trained models are worse at both, and the genetic label is weaker, not only different.
  - The time split has about 100 resolved failures pooled, so a CI that includes 0 is weak evidence by itself. The first branch needs the selection CI below 0 too.

**Group sizes:**

{{tried_groups}}

**Table 6.5a: label, selection and outcome AUROC per predictor** (all genes of each universe; model rows use out-of-fold scores, random split, Minikel label, nnPU).

{{tried_auroc}}

**Table 6.5b: genetic evidence and success given tried**: odds ratio for success vs resolved failure, with 95% CI.

{{tried_genetic_or}}

**Table 6.5c: time split, genetic − Minikel, selection and outcome** (paired, pooled over graphs; per graph below).

{{tried_time}}

## 7. Therapy-area waves

**Setup:** train on targets never used in a therapy area; test on targets used only in it. Waves with fewer than 30 test genes on a graph are skipped, so HuRI has oncology only.

**Table 7.1a: AUROC per wave, nnPU, all features.**
- RF and the GP transfer well across waves (RF 0.85–0.94, GP 0.82–0.95), better than the message-passing models.
- GAT and GCN are worse with the real graph than with none in every wave.

{{waves_all}}

**Table 7.1b: the same with attention-free features.** Transfer drops (RF 0.65–0.82), most for oncology. Here the STRING graph helps graph models (C&S 0.80–0.84 vs 0.67–0.76 empty). That's the same re-injection as §6.3.

{{waves_attn_free}}

**Table 7.2: DeLong for waves, overall and by feature set.**

{{delong_waves_contrast}}

{{delong_waves_contrast_feats}}


---

## 8. Controls: membrane, shuffled labels, random features

### 8.1 Positive control: membrane localisation

**Setup:** label = GO membrane or plasma membrane; features = Pfam and pathways only (no GO, no tractability, so localisation isn't handed to the model).

**Table 8.1a: AUROC, random split, nnPU** (`real / rewired / empty`). Graph models mostly beat their empty-graph versions on STRING and HuRI (HuRI SIGN 0.824 vs 0.782, GraphSAGE 0.817 vs 0.770). On Reactome FI the empty graph is usually as good or better in this cell (C&S 0.826 vs 0.849); Table 8.1b pools all losses and splits.

{{membrane_head_random_nnpu}}

**Table 8.1b: DeLong.**
- **Real beats empty in 50 of 72 cells (+0.043)** and loses in 11, compared with roughly a draw on drug targets.
- **Per model, every model has a positive mean gain on STRING and HuRI.** Reactome FI is mixed (2/2 for most models).

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
- **Whether the models *rely* on it** is tested by the Pfam split per feature set (§2.2).
- **Graph-only models stay low** (label propagation 0.57–0.69).

{{nl_minikel_famperm}}

**Table 8.2c: DeLong by label set.** No graph gain on either: real vs empty is 0 / 2 (shuffled across all genes) and 4 / 11 (within families).

{{delong_nonsense_labels_contrast_labels}}

### 8.3 Random features

**Table 8.3a: node features replaced by Gaussian noise.**
- **Without a graph,** models are at chance (0.47–0.50).
- **With the real STRING graph,** graph models reach 0.53–0.89 (C&S 0.886, GAT 0.869; GCN lowest at 0.533), and on Reactome FI up to 0.81. On HuRI they stay at 0.47–0.59.

The literature graph *alone* carries the label, through study-correlated topology.

{{nf_random}}

**Table 8.3b: DeLong.** Real beats empty 32 / 0 (+0.168), and beats RF 48 / 0 (RF has only noise).

{{delong_nonsense_feats_contrast}}

---

## 9. Caveats

- **PU labels:** an unlabelled gene can be a well-studied target that never reached Phase III. The phase ladder (§5.2) shows many top-ranked "negatives" are exactly that.
- **Leakage in the time split:**
  - `all` features (GO, pathways, tractability) and the STRING graph are current snapshots, so they leak post-2010 knowledge.
  - `attn_free` and `dated` features and HuRI are leak-free. Two exceptions: `dated` uses today's Pfam domain assignments (sequence-derived, but some families were defined after 2009), and GTEx expression (in both sets) was measured after 2010, though uniformly for every gene.
  - Matching uses pre-2010 PubMed counts.
- **Small novel strata:** the strict `novel` stratum has {{n:novel_n}} genes, so every stratified number comes with a bootstrap CI over the targets (Tables 6.2g–l, 6.4i). Conclusions rest on the 2-of-3 definition ({{n:novel2_n}}; {{n:novel2_n_pooled}} pooled).
- **Dating:**
  - First-drugged year is the first approval of a drug acting on the gene in ChEMBL (single proteins and complexes).
  - Minikel positives with no ChEMBL approval date are excluded from the time split. They resemble the test targets in study and novelty (§6.0).
  - Paper years come from a monotone PMID → entry-year fit. Back-indexed papers get later years, which can only *understate* early study.
- **"Attention-free" is relative:** it removes literature annotation, not study bias (ρ ≈ 0.6 with PubMed, §4.3).
- **Tuning:** done once per graph, model, loss and graph condition, on the random split with Minikel labels and all features, then reused for other feature sets and splits. RF on STRING with nnPU hit the tuning time limit (12 trials).
  - Settings chosen on the random split occasionally fail elsewhere. Example: Correct & Smooth with the empty HuRI graph was tuned to a logistic-regression base with nnPU prior 0.018. That base model trains on the random split but never fits on the Pfam split (its training AUROC is 0.46–0.57, checked by hand), so the cell scores 0.494.
  - Such cells are marked † and listed in Table 9.1, not dropped.
  - **The flag can't see this case.** Correct & Smooth and label propagation reset training genes to their true labels, so their final training-fold AUROC is always about 1. A failed C&S base model shows up only as a test AUROC near 0.5, as in HuRI's empty-graph Pfam cells (0.49–0.52 with nnPU).
- **HuRI** covers only 32% of targets, and its pathway subgraph is nearly empty (832 edges).

**Table 9.1: model cells that never fit their training data** (training-fold AUROC < 0.6, random negatives; shuffled-label and random-feature controls excluded, where failing to fit is expected).

{{fit_failures}}

## 10. Next steps

1. **More labels on the time split** (§6.4 has genetic evidence): Phase I entries, or genetic evidence restricted to GWAS and burden (hypothesis-free) once enough pre-2010 data exists.
2. **A dated graph:** STRING 2009 (v8.x) or a 2009 BioGRID release, so graph models on the time split are leak-free too. (Dated features are done: `dated`, §6.)
3. **gnomAD constraint and AlphaFold pockets,** as further non-literature features, run through §4.3's leak test.
4. **Write-up:**
   - Open with the thesis that the label records portfolio decisions, so predicting it well measures imitation. Put it in the introduction, not the discussion: §6.1 is the evidence, and it is also the obvious objection to the time split.
   - Frame the paper as complementary to Rahimi et al.: on essentiality (network-coherent biology) GNNs help; on drug targets, with proper controls and a time split, they mostly reproduce the field's attention.

---

*Generated by `scripts/report.py` from `scripts/report_template.md`, `results/` (CSV tables) and `results/figures/`. Analyses: `scripts/analysis.py` (`results/analysis/*.csv`); significance tests: `scripts/delong.py`; figures: `scripts/figures.py`.*
