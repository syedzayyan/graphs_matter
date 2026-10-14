"""Per-graph descriptive analyses (no model training beyond small logistic regressions).

  uv run python scripts/analysis.py     ->  results/analysis/*.csv

graph_sizes.csv   nodes / edges / degree / isolated nodes per graph and condition (real,
                  rewired, pathway subgraph), plus positives covered
hub_bias.csv      do Minikel positives sit on hubs? degree of positives vs unlabelled,
                  degree-only AUROC, share of positives in the top-10% degree, and whether
                  degree survives controlling for study intensity (PubMed count): degree AUROC
                  within PubMed quintiles, and PubMed-only vs PubMed+degree logistic regression
membrane.csv      fraction of membrane proteins among positives vs unlabelled, per graph
mediation.csv     is the feature -> label association mediated by study intensity? For each
                  feature block: out-of-fold logistic score s, its AUROC overall and within
                  PubMed quintiles, corr(s, log PubMed), and the shrinkage of s's logistic
                  coefficient when log PubMed is added (1 - b_adjusted / b_alone)
prior_study.csv   was study intensity there *before* approval? For genes whose first drug
                  launched 1990-2022 (Minikel), PubMed papers about the gene dated before the
                  launch year (and 10 years before), as a percentile among the universe's genes
                  at that time, and as an AUROC against genes that never got a drug programme
                  (ranked by their own counts at the same date). Papers are dated by PMID via a
                  monotone fit through PubMed entry dates (fetch.py: pmid_years.tsv).
topk_audit.csv    what do the models "discover"? Per run of results/main (minikel, all features,
                  random split; real / empty / no graph), the top-100 ranked test-fold
                  unlabelled genes: PubMed percentile, share in the top PubMed decile, share
                  already in a drug programme (Minikel preclinical-Phase II), share sharing a
                  Pfam domain with a known (train/val) target, share within 1 hop of a known
                  target in the real graph, overlap with the top-100 of a PubMed-count-only
                  ranker, and Spearman(score, PubMed) over all test unlabelled genes. Reference
                  rows: model "pubmed_only" and "all test unlabelled" (the base rates). Mean
                  over seeds.
phase_ladder.csv  trained on >= Phase III only, do models rank test-fold genes with an earlier
                  programme (preclinical / Phase I / Phase II) above genes with none? AUROC
                  overall, per phase, and within PubMed quintiles (does it survive at equal
                  study?), next to the PubMed-only ranker. Mean over seeds.
topk_audit and phase_ladder need results/main (the summarise job runs this after collecting it).

Uses $GHELPS_DATA, so on the cluster it describes exactly the data the runs used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghelps import data, features, labels, paths  # noqa: E402
from ghelps.runner import AXES  # noqa: E402

OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def oof_lr(x, y):
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        s[te] = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return s


FEATURE_BLOCKS = ["go", "pfam", "pathway", "tract"]


def mediation(universe: str, y: np.ndarray, lpm: np.ndarray, quint: np.ndarray, deg: np.ndarray) -> list[dict]:
    import statsmodels.api as sm
    blocks = {b: data._svd(universe, (b,), 32, b) for b in FEATURE_BLOCKS}
    blocks["all features"] = data.features(universe, "all")
    for extra in ("attn_free",):  # attention-free features, when built
        if extra in data.FEATURE_SETS and all((paths.PROCESSED / "features" / universe / f"{b}.parquet").exists()
                                              for b in data.FEATURE_SETS[extra]):
            blocks["attention-free features"] = data.features(universe, extra)
    blocks["degree"] = np.log1p(deg)[:, None]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)  # noqa: E731
    rows = []
    for name, x in blocks.items():
        s = oof_lr(x, y)
        logit_s = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1))
        alone = sm.Logit(y, sm.add_constant(z(logit_s))).fit(disp=0)
        adj = sm.Logit(y, sm.add_constant(np.c_[z(logit_s), z(lpm)])).fit(disp=0)
        within = [auc(y[quint == k], s[quint == k]) for k in range(5)]
        rows.append(dict(block=name, auroc=auc(y, s), auroc_within_pubmed_quintiles=np.nanmean(within),
                         rho_score_log_pubmed=spearmanr(s, lpm).statistic,
                         coef_alone=alone.params[1], coef_adjusted=adj.params[1], coef_pubmed=adj.params[2],
                         shrinkage=1 - adj.params[1] / alone.params[1],
                         auroc_with_pubmed=auc(y, oof_lr(np.c_[x, lpm], y))))
    return rows


# ---------------------------------------------------------------------------------------
# Study before approval

LAUNCH_PERIODS = [(1990, 1999), (2000, 2009), (2010, 2022)]
YEARS = np.arange(1950, 2027)


def gene_year_counts() -> pd.DataFrame:
    """ENSG x year -> cumulative number of gene2pubmed papers entered by the end of that year."""
    anchors = pd.read_csv(paths.RAW / "pmid_years.tsv", sep="\t")
    fit = IsotonicRegression(out_of_bounds="clip").fit(anchors.pmid, anchors.year)
    pairs = features.gene2pubmed_pairs()
    pairs["year"] = np.clip(np.round(fit.predict(pairs.pmid)).astype(int), YEARS[0], YEARS[-1])
    counts = pd.crosstab(pairs.ensg, pairs.year).reindex(columns=YEARS, fill_value=0)
    return counts.cumsum(axis=1)


def prior_study(universe: str, genes: list[str], cum: pd.DataFrame, launch: pd.Series, phase: pd.Series) -> list[dict]:
    c = cum.reindex(genes, fill_value=0)
    launch = launch.reindex(genes).dropna().astype(int)
    never = (phase.reindex(genes).fillna(0) == 0).values  # no drug programme at all
    rows = []
    for lead in (1, 10):  # papers up to the year before launch, and up to 10 years before
        for lo, hi in LAUNCH_PERIODS:
            g = launch[(launch >= lo) & (launch <= hi)]
            pct, vs_never = [], []
            for gene, y in g.items():
                year = min(max(y - lead, YEARS[0]), YEARS[-1])
                col = c[year].values
                v = c.at[gene, year]
                pct.append((col < v).mean() + 0.5 * (col == v).mean())
                ref = col[never]
                vs_never.append((ref < v).mean() + 0.5 * (ref == v).mean())  # = this gene's share of the AUROC
            pct = np.array(pct)
            rows.append(dict(universe=universe, years_before_launch=lead, launch_period=f"{lo}-{hi}",
                             genes=len(g), median_percentile=np.median(pct) if len(g) else np.nan,
                             frac_top_decile=(pct >= 0.9).mean() if len(g) else np.nan,
                             frac_top_quartile=(pct >= 0.75).mean() if len(g) else np.nan,
                             auroc_vs_no_programme=np.mean(vs_never) if len(g) else np.nan))
    return rows


# ---------------------------------------------------------------------------------------
# What the trained models rank highly (needs results/main from the cluster)

AUDIT_EXP, TOP_K = "main", 100


def _run_table(universe: str) -> pd.DataFrame | None:
    f = ROOT / "results" / AUDIT_EXP / "results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=["run_id", *AXES]).drop_duplicates("run_id")
    df["copy"] = df["copy"].fillna("none")
    df["feats"] = df["feats"].fillna("all")
    df["graph"] = df["graph"].fillna("none")
    return df[(df.universe == universe) & (df.labels == "minikel") & (df.feats == "all") & (df.split == "random")
              & df["copy"].isin(["real", "empty", "none"])]


def _top_metrics(top, s_cand, cand, pm, pm_pct, phase, shares_pfam, hops, pm_top) -> dict:
    return dict(pubmed_pct_median=np.median(pm_pct[top]), frac_top_pubmed_decile=(pm_pct[top] >= 0.9).mean(),
                frac_in_programme=((phase[top] >= 1) & (phase[top] <= 3)).mean(),
                frac_shares_pfam_with_known=shares_pfam[top].mean(), frac_within_1hop_of_known=(hops[top] <= 1).mean(),
                overlap_with_pubmed_top=len(np.intersect1d(top, pm_top)) / len(top),
                rho_score_pubmed=spearmanr(s_cand, pm[cand]).statistic if s_cand is not None else np.nan)


def _ladder(s_cand, cand, phase, pm) -> dict:
    ph, lpm = phase[cand], np.log1p(pm[cand])
    q = pd.qcut(pd.Series(lpm).rank(method="first"), 5, labels=False).values
    prog = ph > 0
    out = dict(auroc_programme_vs_none=auc(prog.astype(int), s_cand),
               auroc_within_pubmed_quintiles=np.nanmean([auc(prog[q == k].astype(int), s_cand[q == k])
                                                         for k in range(5)]))
    for k, name in ((1, "preclinical"), (2, "phase1"), (3, "phase2")):
        m = (ph == k) | (ph == 0)
        out[f"auroc_{name}_vs_none"] = auc((ph[m] == k).astype(int), s_cand[m])
    return out


def prediction_audits(universe: str, genes: list[str], graph: str, phase: pd.Series):
    runs = _run_table(universe)
    if runs is None or runs.empty:
        print(f"skip topk_audit / phase_ladder for {universe}: no results/{AUDIT_EXP} runs", flush=True)
        return [], []
    scores_dir = ROOT / "results" / AUDIT_EXP / "scores"
    pm = data.pubmed_counts(universe).astype(float)
    pm_pct = rankdata(pm) / len(pm)
    ph = phase.reindex(genes).fillna(0).values
    pfam = pd.read_parquet(paths.PROCESSED / "features" / universe / "pfam.parquet").reindex(genes, fill_value=0).values > 0
    ei = np.load(paths.PROCESSED / "graphs" / universe / f"{graph}.real.npy")
    adj = coo_matrix((np.ones(len(ei)), (ei[:, 0], ei[:, 1])), shape=(len(genes),) * 2).tocsr()
    audit, ladder = [], []
    for seed, rs in runs.groupby("seed"):
        sp = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / "random" / f"seed{seed}.parquet")
        assert (sp.gene.values == np.array(genes)).all()
        y, fold = sp.y.values, sp.fold.values
        cand = np.flatnonzero((fold == "test") & (y == 0))
        known = (fold != "test") & (y == 1)
        shares_pfam = pfam[:, pfam[known].any(0)].any(1)
        hops = dijkstra(adj, directed=False, unweighted=True, indices=np.flatnonzero(known), min_only=True)
        pm_top = cand[np.argsort(-pm[cand], kind="stable")[:TOP_K]]
        ctx = dict(cand=cand, pm=pm, pm_pct=pm_pct, phase=ph, shares_pfam=shares_pfam, hops=hops, pm_top=pm_top)
        refs = {("pubmed_only", "none", "-"): pm[cand], ("all test unlabelled", "none", "-"): None}
        for (model, copy, loss), s_cand in refs.items():
            top = cand if s_cand is None else pm_top
            audit.append(dict(seed=seed, graph=graph, copy=copy, model=model, loss=loss,
                              **_top_metrics(top, s_cand, **ctx)))
            if s_cand is not None:
                ladder.append(dict(seed=seed, graph=graph, copy=copy, model=model, loss=loss, **_ladder(s_cand, cand, ph, pm)))
        for r in rs.itertuples():
            f = scores_dir / f"{r.run_id}.npy"
            if not f.exists():
                continue
            s = np.load(f).astype(float)
            s_cand = s[cand]
            top = cand[np.argsort(-s_cand, kind="stable")[:TOP_K]]
            key = dict(seed=seed, graph=graph, copy=r.copy, model=r.model, loss=r.loss)
            audit.append(dict(**key, **_top_metrics(top, s_cand, **ctx)))
            ladder.append(dict(**key, **_ladder(s_cand, cand, ph, pm)))
    mean = lambda rows: (pd.DataFrame(rows).groupby(["graph", "copy", "model", "loss"], sort=False)  # noqa: E731
                         .mean(numeric_only=True).drop(columns="seed").reset_index()
                         .assign(universe=universe).to_dict("records"))
    return mean(audit), mean(ladder)


def main():
    pos_all = labels.minikel(CFG["labels"]["minikel_min_phase"])
    membrane = labels.membrane()
    sizes, hubs, mem, med, prior, audit, ladder = [], [], [], [], [], [], []
    phase, launch = labels.minikel_best_phase(), labels.minikel_first_launch()
    cum = gene_year_counts()
    for universe, members in CFG["universes"].items():
        genes = data.genes_of(universe)
        y = np.array([g in pos_all for g in genes], dtype=int)
        is_mem = np.array([g in membrane for g in genes])
        pm = data.pubmed_counts(universe)
        prior += prior_study(universe, genes, cum, launch, phase)
        lpm = np.log1p(pm)
        quint = pd.qcut(pd.Series(lpm).rank(method="first"), 5, labels=False).values
        for graph in members:
            for copy in ("real", "rw0", "pwsub"):
                ei = np.load(paths.PROCESSED / "graphs" / universe / f"{graph}.{copy}.npy")
                deg = np.bincount(ei.ravel(), minlength=len(genes))
                sizes.append(dict(graph=graph, condition=copy, nodes=len(genes), edges=len(ei),
                                  mean_degree=deg.mean(), median_degree=np.median(deg), max_degree=int(deg.max()),
                                  isolated=int((deg == 0).sum()), positives=int(y.sum()),
                                  positives_covered=y.sum() / len(pos_all)))
            d = np.bincount(np.load(paths.PROCESSED / "graphs" / universe / f"{graph}.real.npy").ravel(),
                            minlength=len(genes))
            within = [auc(y[quint == k], d[quint == k]) for k in range(5)]
            hubs.append(dict(
                graph=graph, genes=len(genes), positives=int(y.sum()),
                median_deg_pos=np.median(d[y == 1]), median_deg_unl=np.median(d[y == 0]),
                deg_auroc=auc(y, d), mwu_p=mannwhitneyu(d[y == 1], d[y == 0]).pvalue,
                pos_in_top10pct_deg=y[d >= np.quantile(d, 0.9)].sum() / y.sum(),
                pubmed_auroc=auc(y, pm), rho_deg_pubmed=spearmanr(d, pm).statistic,
                deg_auroc_within_pubmed_quintiles=np.nanmean(within),
                deg_auroc_by_pubmed_quintile=" ".join(f"{w:.2f}" for w in within),
                lr_pubmed_auroc=auc(y, oof_lr(lpm[:, None], y)),
                lr_pubmed_plus_degree_auroc=auc(y, oof_lr(np.c_[lpm, np.log1p(d)], y))))
            med += [dict(graph=graph, pubmed_alone_auroc=auc(y, oof_lr(lpm[:, None], y)), **m)
                    for m in mediation(universe, y, lpm, quint, d)]
            mem.append(dict(graph=graph, membrane_frac_positives=is_mem[y == 1].mean(),
                            membrane_frac_unlabelled=is_mem[y == 0].mean(),
                            membrane_flag_auroc=auc(y, is_mem.astype(float)),
                            membrane_prevalence=is_mem.mean()))
            a, l = prediction_audits(universe, genes, graph, phase)
            audit += a
            ladder += l
    for name, rows in (("graph_sizes", sizes), ("hub_bias", hubs), ("membrane", mem), ("mediation", med),
                       ("prior_study", prior), ("topk_audit", audit), ("phase_ladder", ladder)):
        if not rows:
            continue
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"== {name}\n{t.round(3).to_string(index=False)}\n", flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()
