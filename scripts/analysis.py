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
prior_study.csv   was study intensity there *before* approval? For Minikel >= Phase III genes
                  first drugged 1980-2024 (first approval of any drug acting on them, ChEMBL),
                  PubMed papers about the gene dated before that year (and 10 years before), as
                  a percentile among the universe's genes at that time, and as an AUROC against
                  genes that never got a drug programme (ranked by their own counts at the same
                  date). Papers are dated by PMID via a monotone fit through PubMed entry dates
                  (fetch.py: pmid_years.tsv). Minikel's year_launch is not used: it dates the
                  launch of one indication, often decades after the target was first drugged.
label_bias.csv    how study-biased is each label? Positives, and AUROC of PubMed count (now, and
                  before the time-split cut-off) for separating them from all other genes.
study_leak.csv    how much study intensity does each feature set carry? Out-of-fold RF
                  prediction of log PubMed count from the features: Spearman(prediction, log
                  PubMed). Includes the residualised sets (ghelps/data.py: *_resid).
topk_audit.csv    what do the models "discover"? Per run of results/main (minikel, random split;
                  feature sets all / no_tract / attn_free; real / empty / no graph), the top-100 ranked test-fold
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
time_novelty.csv  time split (results/novelty, split "time"): trained on targets first drugged
                  before 2010, how well are the targets first drugged from 2010 on found, by how
                  novel they were in 2010? Strata of the future targets: study before 2010
                  (PubMed percentile < 0.5 / 0.5-0.9 / >= 0.9), Pfam family already drugged or
                  not, hops to the nearest known target in the real graph (1 / 2 / 3+), and
                  "novel" (not top-decile study, new family, >= 2 hops). Per stratum: AUROC vs
                  all test unlabelled genes, and recall in the top 10% of the test ranking.
                  Reference rankers: PubMed before 2010, PubMed now (leaks the future), number
                  of known targets sharing a Pfam domain, and closeness to known targets. Runs
                  trained on the genetic label (same folds and test genes) appear with labels=genetic.
time_topk_audit.csv  time split: what does each label's model rank highest? Per run of
                  results/novelty (split "time", nnpu; feature sets all / attn_free / dated;
                  real / no graph), the top-100 test genes (future targets + never-drugged
                  genes): share that are future targets, novel (2 of 3) future targets, PubMed
                  percentile before 2010 and share in that top decile, share in a drug programme
                  today (Minikel preclinical-Phase II), share sharing a Pfam domain with / within
                  1 hop of a target drugged before 2010, and share with genetic evidence before
                  2010 / at any date. Reference rows "all test genes" (base rates) and
                  "pubmed_before_T". Mean over seeds.
tried_*.csv       success given tried (report 6.5). Genes by Minikel programme history: never
                  tried (nothing beyond preclinical), tried (>= Phase I), success (>= Phase III,
                  the label), resolved failure (best Phase I/II, no active programme; active
                  ones are censored). AUROC of each predictor for label (success vs rest),
                  selection (tried vs never) and outcome (success vs resolved failure), 95%
                  gene bootstrap CIs. Predictors: PubMed before 2010 / now, genetic evidence
                  before 2010 / any, and out-of-fold scores of results/main (random split,
                  minikel, nnpu; per-seed test percentile averaged over the seeds a gene was
                  tested in). tried_groups: group sizes. tried_genetic_or: odds ratio of
                  success vs resolved failure for genetic evidence. tried_time: time split,
                  genetic- vs Minikel-trained twins on the test genes, selection and outcome
                  AUROC and the paired difference, per graph and pooled (genes as clusters).
time_novelty_tuned.csv, tried_time_tuned.csv  the same as time_novelty / tried_time for
                  results/novelty_tuned (both labels retuned on the time split, tune_time.yaml).
external_audit.csv  published top lists (report 5.3): DrugnomeAI's top-5% genes (Raies et al. 2022,
                  Supplementary Data 1; fetch.py) on the STRING universe, all and "novel" (not
                  Minikel >= Phase III, not Pharos Tclin). Shares: >= Phase III, Tclin, in a
                  programme (preclinical-Phase II), top PubMed decile, sharing Pfam with / within
                  1 hop (STRING) of another >= Phase III gene, genetic evidence, Pharos TDL.
                  Reference rows: all genes, a PubMed ranker and our RF (results/main, random
                  split, out-of-fold) with the same list size.
topk_audit and phase_ladder need results/main, time_novelty and time_topk_audit
results/novelty (the summarise job runs this after collecting them).

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghelps import data, ensure, features, ids, labels, paths  # noqa: E402
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

LAUNCH_PERIODS = [(1980, 1999), (2000, 2009), (2010, 2024)]
YEARS = features.PUBMED_YEARS


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
AUDIT_FEATS = ("all", "no_tract", "attn_free")  # does dropping literature features stop the study-bias cycle?


def _run_table(universe: str) -> pd.DataFrame | None:
    f = ROOT / "results" / AUDIT_EXP / "results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=["run_id", *AXES]).drop_duplicates("run_id")
    df["copy"] = df["copy"].fillna("none")
    df["feats"] = df["feats"].fillna("all")
    df["graph"] = df["graph"].fillna("none")
    return df[(df.universe == universe) & (df.labels == "minikel") & df.feats.isin(AUDIT_FEATS) & (df.split == "random")
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
            key = dict(seed=seed, graph=graph, copy=copy, feats="-", model=model, loss=loss)
            audit.append(dict(**key, **_top_metrics(top, s_cand, **ctx)))
            if s_cand is not None:
                ladder.append(dict(**key, **_ladder(s_cand, cand, ph, pm)))
        for r in rs.itertuples():
            f = scores_dir / f"{r.run_id}.npy"
            if not f.exists():
                continue
            s = np.load(f).astype(float)
            s_cand = s[cand]
            top = cand[np.argsort(-s_cand, kind="stable")[:TOP_K]]
            key = dict(seed=seed, graph=graph, copy=r.copy, feats=r.feats, model=r.model, loss=r.loss)
            audit.append(dict(**key, **_top_metrics(top, s_cand, **ctx)))
            ladder.append(dict(**key, **_ladder(s_cand, cand, ph, pm)))
    mean = lambda rows: (pd.DataFrame(rows).groupby(["graph", "copy", "feats", "model", "loss"], sort=False)  # noqa: E731
                         .mean(numeric_only=True).drop(columns="seed").reset_index()
                         .assign(universe=universe).to_dict("records"))
    return mean(audit), mean(ladder)


# ---------------------------------------------------------------------------------------
# How much study intensity each feature set carries

LEAK_FEATS = ("all", "no_tract", "attn_free", "dated", "all_resid", "attn_free_resid")


def study_leak(universe: str) -> list[dict]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    lpm = np.log1p(data.pubmed_counts(universe))
    rows = []
    for feats in LEAK_FEATS:
        x = data.features(universe, feats)
        k = len(data.SVD_GROUPS.get(feats, ((),)))  # leading SVD components (of each group)
        x = np.hstack([g[:, :64 // k] for g in np.hsplit(x, k)])
        pred = cross_val_predict(RandomForestRegressor(200, max_features=0.3, n_jobs=-1, random_state=0), x, lpm,
                                 cv=StratifiedKFold(3, shuffle=True, random_state=0).split(x, lpm > np.median(lpm)))
        rows.append(dict(universe=universe, feats=feats, rho_rf_pred_log_pubmed=spearmanr(pred, lpm).statistic))
    return rows


# ---------------------------------------------------------------------------------------
# How study-biased is each label itself?

def label_bias(universe: str, genes: list[str], cum: pd.DataFrame, pos_all: set, membrane: set) -> list[dict]:
    T = CFG["splits"]["time_split_year"]
    pm_now = data.pubmed_counts(universe).astype(float)
    pm_T = cum.reindex(genes, fill_value=0)[T - 1].values.astype(float)
    first = labels.target_first_approval()
    sets = {"minikel (>= Phase III)": pos_all,
            f"drugged before {T}": set(first[first < T].index) & pos_all,
            f"genetic evidence before {T}": labels.genetic(before=T),
            "genetic evidence (any date)": labels.genetic(),
            "membrane": membrane}
    rows = []
    for name, s in sets.items():
        y = np.array([g in s for g in genes], dtype=int)
        rows.append(dict(universe=universe, label=name, positives=int(y.sum()), frac=y.mean(),
                         pubmed_auroc_now=auc(y, pm_now), pubmed_auroc_before_T=auc(y, pm_T),
                         median_pubmed_pct=np.median((rankdata(pm_now) / len(pm_now))[y == 1])))
    return rows

# ---------------------------------------------------------------------------------------
# Time split: are future targets found, and which ones?

NOVELTY_EXP, NOVELTY_RECALL_FRAC, NOVELTY_BOOT = "novelty", 0.10, 2000
TUNED_EXP = "novelty_tuned"  # the 6.4 comparison with both labels retuned on the time split
NOVELTY_KEY = ["labels", "copy", "feats", "model", "loss"]


def _novelty_strata(pct: np.ndarray, fam: np.ndarray, hop: np.ndarray) -> dict[str, np.ndarray]:
    """Future targets by how novel they were at T: study then, Pfam family drugged then, hops
    to the targets known then. "novel" needs all three; "novel (2 of 3)" any two (larger n)."""
    crit = np.c_[pct < 0.9, ~fam, hop >= 2]
    return {"all": np.ones(len(pct), bool),
            "study<0.5": pct < 0.5, "study 0.5-0.9": (pct >= 0.5) & (pct < 0.9), "study>=0.9": pct >= 0.9,
            "drugged family": fam, "new family": ~fam,
            "1 hop": hop <= 1, "2 hops": hop == 2, "3+ hops": hop >= 3,
            "novel": crit.all(1), "novel (2 of 3)": crit.sum(1) >= 2}


def _future_context(universe: str, genes: list[str], graph: str, pm_T: np.ndarray):
    """Positions of the future targets (test positives of the minikel time split), their strata,
    and per gene the number of known targets sharing a Pfam domain and the hops to the nearest
    one. Seed-independent: every seed tests all targets first drugged from T on, and the known
    targets (first drugged before T) are the same."""
    pct_T = rankdata(pm_T) / len(pm_T)
    pfam = pd.read_parquet(paths.PROCESSED / "features" / universe / "pfam.parquet").reindex(genes, fill_value=0).values > 0
    ei = np.load(paths.PROCESSED / "graphs" / universe / f"{graph}.real.npy")
    adj = coo_matrix((np.ones(len(ei)), (ei[:, 0], ei[:, 1])), shape=(len(genes),) * 2).tocsr()
    sp = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / "time" / "seed0.parquet")
    assert (sp.gene.values == np.array(genes)).all()
    known = (sp.fold.values != "test") & (sp.fold.values != "excluded") & (sp.y.values == 1)
    fam_known = pfam[:, pfam[known].any(0)].sum(1)
    hops = dijkstra(adj, directed=False, unweighted=True, indices=np.flatnonzero(known), min_only=True)
    fut = np.flatnonzero((sp.fold.values == "test") & (sp.y.values == 1))
    strata = pd.DataFrame(_novelty_strata(pct_T[fut], fam_known[fut] > 0, hops[fut]))
    return fut, strata.assign(gene=np.array(genes)[fut]), fam_known, hops


def _wins(st: np.ndarray, fut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per future target: share of test negatives ranked below it (ties 1/2; its stratum's AUROC
    is the mean of these) and whether it is in the top 10% of the test ranking."""
    neg = np.sort(st[~fut])
    v = st[fut]
    lo, hi = np.searchsorted(neg, v, "left"), np.searchsorted(neg, v, "right")
    return (lo + 0.5 * (hi - lo)) / len(neg), v >= np.quantile(st, 1 - NOVELTY_RECALL_FRAC)


def time_novelty_genes(universe: str, genes: list[str], graph: str, cum: pd.DataFrame,
                       exp: str = NOVELTY_EXP) -> pd.DataFrame:
    """One row per (run key, future target): win share and top-10% hit, averaged over seeds,
    with the target's strata. Reference rankers have labels/copy/feats/loss "-"/"none"."""
    f = ROOT / "results" / exp / "results.csv"
    if not f.exists():
        print(f"skip time_novelty for {universe}: no results/{exp} runs", flush=True)
        return pd.DataFrame()
    runs = pd.read_csv(f, usecols=["run_id", *AXES]).drop_duplicates("run_id")
    runs["copy"] = runs["copy"].fillna("none")
    runs = runs[(runs.universe == universe) & (runs.split == "time") & runs["copy"].isin(["real", "empty", "none"])]
    # strata are defined by drug history (the minikel time split), whichever label a run trained on;
    # every label's time split shares those folds and test genes
    if runs.empty:
        return pd.DataFrame()
    T = CFG["splits"]["time_split_year"]
    pm_T = cum.reindex(genes, fill_value=0)[T - 1].values.astype(float)
    fut_idx, strata, fam_known, hops = _future_context(universe, genes, graph, pm_T)
    refs = {"pubmed_before_T": pm_T, "pubmed_now": data.pubmed_counts(universe).astype(float),
            "known_family_count": fam_known + 1e-3 * rankdata(pm_T),  # PubMed breaks ties
            "closeness_to_known": -np.minimum(hops, 50) + 1e-3 * rankdata(pm_T) / len(pm_T)}
    out = []
    for seed, rs in runs.groupby("seed"):
        sp = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / "time" / f"seed{seed}.parquet")
        test = np.flatnonzero(sp.fold.values == "test")
        fut = sp.y.values[test] == 1
        assert (test[fut] == fut_idx).all(), "future targets differ between seeds"
        scored = [(dict(labels="-", copy="none", feats="-", model=k, loss="-"), v) for k, v in refs.items()]
        for r in rs.itertuples():
            sf = ROOT / "results" / exp / "scores" / f"{r.run_id}.npy"
            if sf.exists():
                scored.append((dict(labels=r.labels, copy=r.copy, feats=r.feats, model=r.model, loss=r.loss),
                               np.load(sf).astype(float)))
        for key, s in scored:
            win, hit = _wins(s[test], fut)
            rho = spearmanr(s[test], pm_T[test]).statistic  # how much the ranking follows study at T
            out.append(pd.DataFrame({**key, "seed": seed, "pos": np.arange(len(fut_idx)), "win": win, "hit": hit,
                                     "rho_pubmed_T": rho}))
    g = (pd.concat(out).groupby(NOVELTY_KEY + ["pos"], sort=False)[["win", "hit", "rho_pubmed_T"]].mean()
         .reset_index())
    return g.merge(strata.reset_index(names="pos"), on="pos").drop(columns="pos").assign(universe=universe, graph=graph)


def time_topk_audit(universe: str, genes: list[str], graph: str, cum: pd.DataFrame, phase: pd.Series) -> list[dict]:
    """Top-100 test genes of each time-split run: are they future targets, previously investigated
    genes, members of drugged families, or genetic-evidence genes? See the module docstring."""
    f = ROOT / "results" / NOVELTY_EXP / "results.csv"
    if not f.exists():
        return []
    runs = pd.read_csv(f, usecols=["run_id", *AXES]).drop_duplicates("run_id")
    runs["copy"] = runs["copy"].fillna("none")
    runs = runs[(runs.universe == universe) & (runs.split == "time") & runs["copy"].isin(["real", "none"])
                & (runs.loss == "nnpu") & runs.feats.isin(["all", "attn_free", "dated"])]
    if runs.empty:
        return []
    T = CFG["splits"]["time_split_year"]
    G = np.array(genes)
    pm_T = cum.reindex(genes, fill_value=0)[T - 1].values.astype(float)
    pct_T = rankdata(pm_T) / len(pm_T)
    fut_idx, strata, fam_known, hops = _future_context(universe, genes, graph, pm_T)
    fut, novel = np.zeros(len(genes), bool), np.zeros(len(genes), bool)
    fut[fut_idx], novel[fut_idx] = True, strata["novel (2 of 3)"].values
    ph = phase.reindex(genes).fillna(0).values
    gen_T, gen_any = np.isin(G, list(labels.genetic(before=T))), np.isin(G, list(labels.genetic()))

    def metrics(top):
        return dict(frac_future_target=fut[top].mean(), frac_novel_future=novel[top].mean(),
                    pubmed_T_pct_median=np.median(pct_T[top]), frac_top_pubmed_decile_T=(pct_T[top] >= 0.9).mean(),
                    frac_in_programme=((ph[top] >= 1) & (ph[top] <= 3)).mean(),
                    frac_shares_pfam_with_known=(fam_known[top] > 0).mean(),
                    frac_within_1hop_of_known=(hops[top] <= 1).mean(),
                    frac_genetic_before_T=gen_T[top].mean(), frac_genetic_any=gen_any[top].mean())
    out = []
    for seed, rs in runs.groupby("seed"):
        sp = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / "time" / f"seed{seed}.parquet")
        test = np.flatnonzero(sp.fold.values == "test")
        ref = dict(seed=seed, labels="-", copy="none", feats="-", loss="-")
        out.append(dict(**ref, model="all test genes", **metrics(test)))
        out.append(dict(**ref, model="pubmed_before_T", **metrics(test[np.argsort(-pm_T[test], kind="stable")[:TOP_K]])))
        for r in rs.itertuples():
            sf = ROOT / "results" / NOVELTY_EXP / "scores" / f"{r.run_id}.npy"
            if sf.exists():
                s = np.load(sf).astype(float)[test]
                out.append(dict(seed=seed, labels=r.labels, copy=r.copy, feats=r.feats, model=r.model, loss=r.loss,
                                **metrics(test[np.argsort(-s, kind="stable")[:TOP_K]])))
    return (pd.DataFrame(out).groupby(NOVELTY_KEY, sort=False).mean(numeric_only=True).drop(columns="seed")
            .reset_index().assign(universe=universe, graph=graph).to_dict("records"))


# ---------------------------------------------------------------------------------------
# Published top lists: what do other tools put at the top?

EXTERNAL_UNIVERSE = "string"


def external_audit(phase: pd.Series, pos_all: set) -> list[dict]:
    f = paths.RAW / "external" / "drugnomeai_top5pct.tsv"
    if not f.exists():
        return []
    u = EXTERNAL_UNIVERSE
    genes = data.genes_of(u)
    G = np.array(genes)
    idx = {g: i for i, g in enumerate(genes)}
    pos = np.isin(G, list(pos_all))
    tdl = labels.pharos_tdl().reindex(genes).fillna("none").values
    ph = phase.reindex(genes).fillna(0).values
    pm = data.pubmed_counts(u).astype(float)
    pct = rankdata(pm) / len(pm)
    gen = np.isin(G, list(labels.genetic()))
    pf = pd.read_parquet(paths.PROCESSED / "features" / u / "pfam.parquet").reindex(genes, fill_value=0).values > 0
    fam_pos = pf.astype(np.int32).T @ pos.astype(np.int32)  # >= Phase III genes per domain
    # shares a domain with another >= Phase III gene (a positive doesn't count itself)
    shares = np.where(pos, (pf & (fam_pos >= 2)).any(1), (pf & (fam_pos >= 1)).any(1))
    ei = np.load(paths.PROCESSED / "graphs" / u / f"{u}.real.npy")
    adj = coo_matrix((np.ones(2 * len(ei)), (np.r_[ei[:, 0], ei[:, 1]], np.r_[ei[:, 1], ei[:, 0]])),
                     shape=(len(genes),) * 2).tocsr()
    near = np.asarray(adj @ pos.astype(float)).ravel() > 0  # a neighbour is a >= Phase III gene
    novel = ~pos & (tdl != "Tclin")

    def metrics(top: np.ndarray) -> dict:
        return dict(n=len(top), frac_phase3=pos[top].mean(), frac_tclin=(tdl[top] == "Tclin").mean(),
                    frac_in_programme=((ph[top] >= 1) & (ph[top] <= 3)).mean(),
                    pubmed_pct_median=np.median(pct[top]), frac_top_pubmed_decile=(pct[top] >= 0.9).mean(),
                    frac_shares_pfam_with_phase3=shares[top].mean(), frac_adjacent_to_phase3=near[top].mean(),
                    frac_genetic=gen[top].mean(),
                    **{f"frac_{t.lower()}": (tdl[top] == t).mean() for t in ("Tchem", "Tbio", "Tdark")})
    ext = pd.read_csv(f, sep="\t")
    ext["ensg"] = ids.map_symbols(ext.symbol).values
    oof = _oof_scores("main", u, "minikel", "random", lambda r: (r.loss == "nnpu") & (r.model == "rf_feat")
                      & r.feats.isin(["all", "attn_free"]))
    rows = []
    for name, d in ext.groupby("list", sort=False):
        top = np.array(sorted({idx[g] for g in d.ensg.dropna() if g in idx}))
        k = len(top)
        refs = {"all genes": np.arange(len(genes)), "PubMed ranker": np.argsort(-pm, kind="stable")[:k]}
        for (m, fe, c), ser in oof.items():
            refs[f"RF ({fe}), ours"] = np.array([idx[g] for g in ser.sort_values(ascending=False).index[:k]])
        for src, t in [(name, top), *refs.items()]:
            for subset, m in (("all", t), ("novel", t[novel[t]])):
                rows.append(dict(list=name, ranker=src, subset=subset, mapped=len(top), listed=len(d),
                                 **metrics(m)))
    return rows


# ---------------------------------------------------------------------------------------
# Success given tried: does what predicts the label predict selection or outcome?

TRIED_BOOT = 2000
TRIED_MAIN = dict(models=("rf_feat", "mlp_feat", "gp", "cs_feat"), feats=("all", "attn_free"))
TRIED_TIME = dict(models=("rf_feat", "gp"), feats=("attn_free", "dated"))


def tried_groups(genes: list[str]) -> pd.DataFrame:
    """Per gene: tried / success / resolved failure / never tried, from Minikel's pairs."""
    pp = labels._minikel_pp()
    g = pp.groupby("ensg").agg(best=("ccatnum", "max"), active=("acat", lambda a: a.notna().any())).reindex(genes)
    best, act = g.best.fillna(0).values, g.active.fillna(False).values.astype(bool)
    tried = best >= 2
    return pd.DataFrame({"tried": tried, "never": ~tried, "success": best >= 4,
                         "failure": tried & (best <= 3) & ~act}, index=genes)


def _boot_auc(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUROC of score s for 0/1 y under each row of bootstrap weights w (B, n); ties count 1/2."""
    _, grp = np.unique(s, return_inverse=True)
    mt = coo_matrix((np.ones(len(s)), (grp, np.arange(len(s))))).tocsr()  # (groups, n), groups in score order
    out = []
    for wc in np.array_split(w, max(1, len(w) // 250)):  # chunks keep (B, n) arrays small
        P, N = (mt @ (wc * y).T).T, (mt @ (wc * (1 - y)).T).T
        below = np.cumsum(N, 1) - N
        out.append((P * (below + 0.5 * N)).sum(1) / np.maximum(P.sum(1) * N.sum(1), 1e-12))
    return np.concatenate(out)


def _cluster_weights(cluster: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap weights per row, resampling clusters (genes) with replacement."""
    _, c = np.unique(cluster, return_inverse=True)
    k = c.max() + 1
    return rng.multinomial(k, np.full(k, 1 / k), size=TRIED_BOOT)[:, c].astype(float)


def _oof_scores(exp: str, universe: str, labels_: str, split: str, keep) -> dict[tuple, pd.Series]:
    """run key -> per-gene mean over seeds of the score's percentile among that seed's test genes."""
    f = ROOT / "results" / exp / "results.csv"
    if not f.exists():
        return {}
    runs = pd.read_csv(f, usecols=["run_id", *AXES]).drop_duplicates("run_id")
    runs["copy"] = runs["copy"].fillna("none")
    runs = runs[(runs.universe == universe) & (runs["labels"] == labels_) & (runs.split == split) & keep(runs)]
    genes = data.genes_of(universe)
    out = {}
    for key, rs in runs.groupby(["model", "feats", "copy"]):
        acc = []
        for r in rs.itertuples():
            sf = ROOT / "results" / exp / "scores" / f"{r.run_id}.npy"
            if not sf.exists():
                continue
            sp = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / split / f"seed{r.seed}.parquet")
            test = np.flatnonzero(sp.fold.values == "test")
            sc = np.load(sf).astype(float)[test]
            acc.append(pd.Series(rankdata(sc) / len(sc), index=np.array(genes)[test]))
        if acc:
            out[key] = pd.concat(acc).groupby(level=0).mean()
    return out


def success_given_tried(universe: str, genes: list[str], cum: pd.DataFrame):
    rng = np.random.default_rng(0)
    T = CFG["splits"]["time_split_year"]
    gr = tried_groups(genes)
    G = np.array(genes)
    pred = {"pubmed_before_T": cum.reindex(genes, fill_value=0)[T - 1].values.astype(float),
            "pubmed_now": data.pubmed_counts(universe).astype(float),
            "genetic_before_T": np.isin(G, list(labels.genetic(before=T))).astype(float),
            "genetic_any": np.isin(G, list(labels.genetic())).astype(float)}
    oof = _oof_scores("main", universe, "minikel", "random", lambda r: (r.loss == "nnpu")
                      & r.model.isin(TRIED_MAIN["models"]) & r.feats.isin(TRIED_MAIN["feats"])
                      & r["copy"].isin(["none", "real"]))
    contrasts = {"label": (gr.success.values, ~gr.success.values), "selection": (gr.tried.values, gr.never.values),
                 "outcome": (gr.success.values, gr.failure.values)}
    groups = [dict(universe=universe, **{k: int(v.sum()) for k, v in gr.items()},
                   censored=int((gr.tried & ~gr.success & ~gr.failure).sum()))]
    rows = []
    for cname, (pos, neg) in contrasts.items():
        sel = pos | neg
        w = _cluster_weights(G[sel], rng)
        y = pos[sel].astype(float)
        scored = [(k, None, v[sel]) for k, v in pred.items()]
        scored += [(f"{m} ({fe}{'' if c == 'none' else ', graph'})", ser, None) for (m, fe, c), ser in oof.items()]
        for name, ser, v in scored:
            if ser is not None:  # out-of-fold: genes never tested are dropped
                v = ser.reindex(G[sel]).values
            ok = ~np.isnan(v)
            b = _boot_auc(v[ok], y[ok], w[:, ok])
            rows.append(dict(universe=universe, predictor=name, contrast=cname, n_pos=int(y[ok].sum()),
                             n_neg=int((1 - y[ok]).sum()), auroc=auc(y[ok], v[ok]),
                             lo=np.quantile(b, 0.025), hi=np.quantile(b, 0.975)))
    ors = []
    for k in ("genetic_before_T", "genetic_any"):
        e = pred[k] > 0
        a, b_, c, d = [int((m1 & m2).sum()) for m1 in (gr.success.values, gr.failure.values) for m2 in (e, ~e)]
        lor, se = np.log((a + .5) * (d + .5) / ((b_ + .5) * (c + .5))), np.sqrt(sum(1 / (x + .5) for x in (a, b_, c, d)))
        ors.append(dict(universe=universe, evidence=k, success_with=a, success_without=b_, failure_with=c,
                        failure_without=d, odds_ratio=np.exp(lor), lo=np.exp(lor - 1.96 * se), hi=np.exp(lor + 1.96 * se)))
    return groups, rows, ors


def _time_tried_rows(universe: str, genes: list[str], exp: str = NOVELTY_EXP) -> pd.DataFrame:
    """Per gene of the time split's test sets: group flags and each twin's out-of-fold percentile."""
    gr = tried_groups(genes)
    sp = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / "time" / "seed0.parquet")
    fut = pd.Series(sp.y.values == 1, index=genes) & (sp.fold.values == "test")
    keep = lambda r: ((r.loss == "nnpu") & r.model.isin(TRIED_TIME["models"]) & r.feats.isin(TRIED_TIME["feats"])  # noqa: E731
                      & r["copy"].isin(["none", "real"]))
    cols = {}
    for lab in ("minikel", "genetic"):
        for (m, fe, c), ser in _oof_scores(exp, universe, lab, "time", keep).items():
            cols[(lab, m, fe)] = ser
    if not cols:
        return pd.DataFrame()
    d = pd.DataFrame(cols)
    d.columns = pd.MultiIndex.from_tuples(d.columns)
    d = d.stack([1, 2], future_stack=True).reset_index(names=["gene", "model", "feats"])
    d["success"] = fut.reindex(d.gene).values  # future targets: >= Phase III, first drugged from T on
    d["failure"] = gr.failure.reindex(d.gene).values
    d["tried"], d["never"] = gr.tried.reindex(d.gene).values, gr.never.reindex(d.gene).values
    return d.assign(universe=universe)


def time_tried(d: pd.DataFrame) -> list[dict]:
    if d.empty:
        return []
    rng = np.random.default_rng(0)
    rows = []
    for uni, du in [*d.groupby("universe", sort=False), ("pooled", d)]:
        for (m, fe), g in du.groupby(["model", "feats"]):
            g = g.dropna(subset=["minikel", "genetic"])
            for cname, pos, neg in (("selection", g.tried, g.never), ("outcome", g.success, g.failure)):
                sel = (pos | neg).values.astype(bool)
                y = pos.values[sel].astype(float)
                w = _cluster_weights(g.gene.values[sel], rng)
                b = {lab: _boot_auc(g[lab].values[sel], y, w) for lab in ("minikel", "genetic")}
                diff = b["genetic"] - b["minikel"]
                rows.append(dict(universe=uni, model=m, feats=fe, contrast=cname, n_pos=int(y.sum()),
                                 n_neg=int((1 - y).sum()),
                                 minikel=auc(y, g.minikel.values[sel]), genetic=auc(y, g.genetic.values[sel]),
                                 genetic_minus_minikel=auc(y, g.genetic.values[sel]) - auc(y, g.minikel.values[sel]),
                                 lo=np.quantile(diff, 0.025), hi=np.quantile(diff, 0.975)))
    return rows


def _boot_means(v: np.ndarray, cluster: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Bootstrap means of v resampling clusters (genes): counts is (B, n_clusters) multinomial."""
    s = np.bincount(cluster, weights=v, minlength=counts.shape[1])
    n = np.bincount(cluster, minlength=counts.shape[1])
    return (counts @ s) / np.maximum(counts @ n, 1)


def time_novelty(gene_rows: pd.DataFrame, pooled: bool = False) -> list[dict]:
    """Per (universe, run key, stratum): AUROC and recall@10% over the stratum's future targets,
    95% bootstrap CIs resampling those targets (2000 draws, shared across run keys so CIs are
    paired), the paired difference from the PubMed-before-T ranker with its CI, and for runs
    trained on another label the paired difference from their Minikel-trained twin (same copy,
    features, model, loss; *_vs_minikel). rho_pubmed_T: Spearman(score, PubMed before T) over
    the test genes, mean over seeds. The CI
    covers which targets happened to come up after T, the dominant uncertainty for strata of
    7-30 genes; negatives (thousands of genes) are held fixed. pooled=True pools the three
    graphs (keys present in all three), resampling genes so one gene in several graphs counts
    as one cluster."""
    if gene_rows.empty:
        return []
    rng = np.random.default_rng(0)
    rows = []
    groups = [("pooled", gene_rows)] if pooled else list(gene_rows.groupby("universe", sort=False))
    for universe, d in groups:
        if pooled:
            n_u = d.groupby(NOVELTY_KEY).universe.transform("nunique")
            d = d[n_u == d.universe.nunique()]
        strata = list(_novelty_strata(np.zeros(1), np.zeros(1, bool), np.zeros(1)))
        ref = d[d.model == "pubmed_before_T"]
        for name in strata:
            ds = d[d[name]]
            genes_s = ds.gene.unique()
            if not len(genes_s):
                continue
            gid = {g: i for i, g in enumerate(genes_s)}
            counts = rng.multinomial(len(genes_s), np.full(len(genes_s), 1 / len(genes_s)), size=NOVELTY_BOOT)
            rs = ref[ref[name]]
            rc = rs.gene.map(gid).values
            ref_b = {m: _boot_means(rs[m].values.astype(float), rc, counts) for m in ("win", "hit")}
            ref_pt = {m: rs[m].mean() for m in ("win", "hit")}
            boots, stratum_rows = {}, []
            for key, k in ds.groupby(NOVELTY_KEY, sort=False):
                c = k.gene.map(gid).values
                row = dict(universe=universe, graph=k.graph.iloc[0] if not pooled else "pooled",
                           **dict(zip(NOVELTY_KEY, key)), stratum=name, n_future=len(genes_s),
                           rho_pubmed_T=k.rho_pubmed_T.mean())
                for m, col in (("win", "auroc"), ("hit", "recall_top10pct")):
                    v = k[m].values.astype(float)
                    b = boots[(key, m)] = _boot_means(v, c, counts)
                    row |= {col: v.mean(), f"{col}_lo": np.quantile(b, 0.025), f"{col}_hi": np.quantile(b, 0.975),
                            f"{col}_vs_pubmed": v.mean() - ref_pt[m],
                            f"{col}_vs_pubmed_lo": np.quantile(b - ref_b[m], 0.025),
                            f"{col}_vs_pubmed_hi": np.quantile(b - ref_b[m], 0.975)}
                stratum_rows.append(row)
            pt = {(tuple(r[k] for k in NOVELTY_KEY), m): r[col] for r in stratum_rows
                  for m, col in (("win", "auroc"), ("hit", "recall_top10pct"))}
            for row in stratum_rows:
                if row["labels"] in ("-", "minikel"):
                    continue
                key = tuple(row[k] for k in NOVELTY_KEY)
                twin = ("minikel", *key[1:])
                for m, col in (("win", "auroc"), ("hit", "recall_top10pct")):
                    if (twin, m) in boots:
                        diff = boots[(key, m)] - boots[(twin, m)]
                        row |= {f"{col}_vs_minikel": row[col] - pt[(twin, m)],
                                f"{col}_vs_minikel_lo": np.quantile(diff, 0.025),
                                f"{col}_vs_minikel_hi": np.quantile(diff, 0.975)}
            rows += stratum_rows
    return rows


def time_split_coverage(universe: str, genes: list[str], graph: str, cum: pd.DataFrame, first_drugged: pd.Series,
                        best_phase: pd.Series, pos_all: set) -> list[dict]:
    """Who does the time split leave out? Minikel >= Phase III positives with no ChEMBL approval
    date are excluded from it. If they differ systematically from the dated ones (less studied,
    further from known targets), the time split is tilted towards rediscovery by construction.
    Per group: study before T and now, novelty relative to the targets known before T."""
    T = CFG["splits"]["time_split_year"]
    pm_T = cum.reindex(genes, fill_value=0)[T - 1].values.astype(float)
    pct_T = rankdata(pm_T) / len(pm_T)
    pct_now = rankdata(data.pubmed_counts(universe)) / len(genes)
    fd = first_drugged.reindex(genes)
    ph = best_phase.reindex(genes).fillna(0).values
    pos = np.array([x in pos_all for x in genes])
    known = pos & (fd < T).values
    pfam = pd.read_parquet(paths.PROCESSED / "features" / universe / "pfam.parquet").reindex(genes, fill_value=0).values > 0
    fam = pfam[:, pfam[known].any(0)].any(1)
    ei = np.load(paths.PROCESSED / "graphs" / universe / f"{graph}.real.npy")
    adj = coo_matrix((np.ones(len(ei)), (ei[:, 0], ei[:, 1])), shape=(len(genes),) * 2).tocsr()
    hops = dijkstra(adj, directed=False, unweighted=True, indices=np.flatnonzero(known), min_only=True)
    groups = {f"first drugged < {T} (train)": known,
              f"first drugged >= {T} (test)": pos & (fd >= T).values,
              "undated, launched (excluded)": pos & fd.isna().values & (ph >= 5),
              "undated, Phase III only (excluded)": pos & fd.isna().values & (ph < 5),
              "no drug programme (reference)": ph == 0}
    rows = []
    for name, m in groups.items():
        nov = _novelty_strata(pct_T[m], fam[m], hops[m])
        rows.append(dict(universe=universe, group=name, genes=int(m.sum()),
                         median_pubmed_pct_before_T=np.median(pct_T[m]), frac_top_decile_before_T=(pct_T[m] >= 0.9).mean(),
                         median_pubmed_pct_now=np.median(pct_now[m]), frac_drugged_family=fam[m].mean(),
                         frac_2plus_hops=(hops[m] >= 2).mean(), frac_novel=nov["novel"].mean(),
                         frac_novel_2of3=nov["novel (2 of 3)"].mean()))
    return rows


def main():
    ensure.ensure_data()  # fetches any raw source added since the data was built (e.g. pmid_years.tsv)
    pos_all = labels.minikel(CFG["labels"]["minikel_min_phase"])
    membrane = labels.membrane()
    sizes, hubs, mem, med, prior, audit, ladder, leak, lbias, cover = [], [], [], [], [], [], [], [], [], []
    novelty_genes, time_topk, tg, tauc, tor, ttime = [], [], [], [], [], []
    tuned_genes, ttime_tuned = [], []  # results/novelty_tuned: both labels retuned on the time split
    phase = labels.minikel_best_phase()
    first_drugged = labels.target_first_approval()
    launch = first_drugged[first_drugged.index.isin(pos_all)]
    cum = features.pubmed_by_year()
    ext_audit = external_audit(phase, pos_all)
    for universe, members in CFG["universes"].items():
        genes = data.genes_of(universe)
        y = np.array([g in pos_all for g in genes], dtype=int)
        is_mem = np.array([g in membrane for g in genes])
        pm = data.pubmed_counts(universe)
        prior += prior_study(universe, genes, cum, launch, phase)
        a, b, c = success_given_tried(universe, genes, cum)
        tg, tauc, tor = tg + a, tauc + b, tor + c
        ttime.append(_time_tried_rows(universe, genes))
        ttime_tuned.append(_time_tried_rows(universe, genes, TUNED_EXP))
        leak += study_leak(universe)
        lbias += label_bias(universe, genes, cum, pos_all, membrane)
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
            novelty_genes.append(time_novelty_genes(universe, genes, graph, cum))
            tuned_genes.append(time_novelty_genes(universe, genes, graph, cum, TUNED_EXP))
            time_topk += time_topk_audit(universe, genes, graph, cum, phase)
            cover += time_split_coverage(universe, genes, graph, cum, first_drugged, phase, pos_all)
    ng = pd.concat(novelty_genes, ignore_index=True) if novelty_genes else pd.DataFrame()
    novelty = time_novelty(ng) + time_novelty(ng, pooled=True)
    ngt = pd.concat([pd.DataFrame(), *tuned_genes], ignore_index=True)
    novelty_tuned = time_novelty(ngt) + time_novelty(ngt, pooled=True)
    if not ng.empty:
        ng.to_csv(OUT / "time_novelty_genes.csv.gz", index=False)
    for name, rows in (("graph_sizes", sizes), ("hub_bias", hubs), ("membrane", mem), ("mediation", med),
                       ("prior_study", prior), ("study_leak", leak), ("label_bias", lbias), ("topk_audit", audit),
                       ("phase_ladder", ladder), ("time_novelty", novelty), ("time_topk_audit", time_topk),
                       ("time_split_coverage", cover), ("tried_groups", tg), ("tried_auroc", tauc),
                       ("tried_genetic_or", tor), ("tried_time", time_tried(pd.concat([pd.DataFrame(), *ttime]))),
                       ("time_novelty_tuned", novelty_tuned), ("external_audit", ext_audit),
                       ("tried_time_tuned", time_tried(pd.concat([pd.DataFrame(), *ttime_tuned])))):
        if not rows:
            continue
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"== {name}\n{t.round(3).to_string(index=False)}\n", flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()
