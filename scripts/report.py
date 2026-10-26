"""Results report: every table built from results/, filled into scripts/report_template.md.

  uv run python scripts/report.py            ->  summary.md (repo root)

The template holds the prose and two kinds of placeholder, both built here from
results/<exp>/*.csv and results/analysis/*.csv:
  {{name}}     a markdown table
  {{n:name}}   a number (or short list of numbers) quoted in the prose
A section whose inputs don't exist yet is skipped; its placeholders render as a note (tables)
or as [name?] (numbers). Figures are linked from results/figures/ (scripts/figures.py). Prose
numbers not written as {{n:...}} are hand-written: re-check them when the results change.
"""
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
A = R / "analysis"
TEMPLATE = ROOT / "scripts" / "report_template.md"
OUT = ROOT / "summary.md"
TABLES: dict[str, str] = {}
NUMS: dict[str, str] = {}
FIT_FAIL = 0.6  # training-fold AUROC below this = the model never fit its training data

U = ["string", "reactome_fi", "huri"]
UN = {"string": "STRING", "reactome_fi": "Reactome FI", "huri": "HuRI"}
MN = {"deg_lr": "degree LR", "rf_struct": "RF struct stats", "labelprop": "label propagation",
      "mlp_feat": "MLP", "rf_feat": "RF", "mlp_feat_struct": "MLP + struct", "rf_feat_struct": "RF + struct",
      "sign_feat": "SIGN", "cs_feat": "Correct & Smooth", "gcn_none": "GCN (no feats)", "gcn_feat": "GCN",
      "sage_feat": "GraphSAGE", "gat_none": "GAT (no feats)", "gat_feat": "GAT", "gat_edge_feat": "GAT + edge vec",
      "gp": "GP (additive)"}
ORDER = list(MN)
REFS = ["pubmed_before_T", "pubmed_now", "known_family_count", "closeness_to_known"]
STRATA = ["all", "study<0.5", "study 0.5-0.9", "study>=0.9", "drugged family", "new family", "1 hop", "2 hops",
          "3+ hops", "novel", "novel (2 of 3)"]
UP = U + ["pooled"]
UNP = UN | {"pooled": "pooled"}
FEATS = ["all", "no_tract", "attn_free", "dated"]
REGIMES = ["random", "degree", "pubmed", "pfam", "pathway", "hop2", "hop1"]


# ------------------------------------------------------------------ helpers

def md(df: pd.DataFrame, fmt="{:.3f}", index=True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = ["–" if isinstance(v, (float, np.floating)) and pd.isna(v)
                 else fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def put(name: str, df: pd.DataFrame, fmt="{:.3f}", index=True) -> None:
    TABLES[name] = md(df, fmt, index)


def num(name: str, v, fmt="{:.2f}") -> None:
    NUMS[name] = v if isinstance(v, str) else fmt.format(v)


def nums3(vals, fmt="{:.2f}") -> str:
    """'a / b / c' over STRING / Reactome FI / HuRI."""
    return " / ".join("–" if pd.isna(v) else fmt.format(v) for v in vals)


def ci(v, lo, hi, fmt="{:+.3f}") -> str:
    return "–" if pd.isna(v) else f"{fmt.format(v)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def summ(exp: str) -> pd.DataFrame:
    s = pd.read_csv(R / exp / "summary_auroc.csv")
    s["feats"] = s["feats"].fillna("all")
    return s


def by_model(df: pd.DataFrame, col="model") -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(ORDER)}
    return df.assign(_o=df[col].map(lambda m: rank.get(m, -1))).sort_values("_o", kind="stable").drop(columns="_o")


def compact(exp, split, loss, feats="all", regime="random", labels="minikel") -> pd.DataFrame:
    """Model x graph: 'real / rewired / empty' for graph models, 'x (no graph)' otherwise."""
    s = summ(exp)
    s = s[(s.split == split) & (s.loss == loss) & (s.feats == feats) & (s.regime == regime) & (s.labels == labels)]
    s = s.assign(c=np.where(s.cond.str.startswith("rw"), "rewired", s.cond))
    s = s[s.c.isin(["real", "rewired", "empty", "none"])]
    if "train_auroc" not in s:
        s = s.assign(train_auroc=np.nan)
    g = s.groupby(["model", "universe", "c"])
    t, fit = g["mean"].mean().unstack("c"), g["train_auroc"].mean().unstack("c")

    def cell(m, u, c):
        v = t.at[(m, u), c] if c in t else np.nan
        return "–" if pd.isna(v) else f"{v:.3f}" + ("†" if fit.at[(m, u), c] < FIT_FAIL else "")
    t["cell"] = [f"{cell(m, u, 'none')} (no graph)" if pd.notna(t.at[(m, u), "none"] if "none" in t else np.nan)
                 else " / ".join(cell(m, u, c) for c in ("real", "rewired", "empty")) for m, u in t.index]
    t = t.cell.unstack("universe").reindex(columns=U)
    t.columns = [UN[u] for u in U]
    t = t.reindex([m for m in ORDER if m in t.index])
    t.index = t.index.map(MN)
    return t


def regimes(exp, models, split="random", loss="nnpu", feats="all", labels="minikel", conds=("real", "none")):
    """Universe x model x graph condition -> AUROC per evaluation-negative regime."""
    s = summ(exp)
    s = s[(s.split == split) & (s.loss == loss) & (s.feats == feats) & (s.labels == labels) & s.model.isin(models)
          & s.cond.isin(conds)]
    t = s.pivot_table(index=["universe", "model", "cond"], columns="regime", values="mean")
    t = t.reindex(columns=[c for c in REGIMES if c in t.columns]).reset_index()
    t["_u"] = t.universe.map(U.index)
    t = by_model(t).sort_values(["_u"], kind="stable").drop(columns="_u")
    t["universe"] = t.universe.map(UN)
    t["model"] = t.model.map(MN)
    return t.rename(columns={"cond": "graph"})


def delong(exp: str) -> pd.DataFrame:
    d = pd.read_csv(R / exp / "delong.csv")
    sig = d.p_bh < 0.05
    return d.assign(better=sig & (d.delta_auroc > 0), worse=sig & (d.delta_auroc < 0))


def counts(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = d.groupby(by).agg(cells=("better", "size"), real_better=("better", "sum"), real_worse=("worse", "sum"),
                          mean_delta=("delta_auroc", "mean"))
    return t.astype({"real_better": int, "real_worse": int})


# ------------------------------------------------------------------ sections

def setup():
    g = pd.read_csv(A / "graph_sizes.csv")
    g["graph"] = g.graph.map(UN)
    put("graph_sizes", g[["graph", "condition", "nodes", "edges", "mean_degree", "median_degree", "max_degree",
                          "isolated", "positives", "positives_covered"]], "{:.2f}", index=False)


def fit_failures():
    """Every (experiment, model, condition) cell whose training-fold AUROC is below FIT_FAIL."""
    rows = []
    for exp in ("main", "waves", "membrane", "nonsense_labels", "nonsense_feats", "novelty"):
        try:
            s = summ(exp)
        except FileNotFoundError:
            continue
        if "train_auroc" not in s:
            continue
        s = s[(s.regime == "random") & ~s.labels.isin(["minikel_perm"]) & ~s.feats.isin(["random"])]
        rows.append(s.assign(exp=exp, n=len(s))[s.train_auroc < FIT_FAIL])
    if not rows:
        return
    f = pd.concat(rows)
    num("n_fit_fail", f"{len(f)}", "{}")
    cols = ["exp", "universe", "labels", "feats", "split", "loss", "model", "cond", "train_auroc", "mean"]
    f = by_model(f[cols].rename(columns={"mean": "test_auroc"}))
    f["model"] = f.model.map(MN)
    put("fit_failures", f if len(f) else pd.DataFrame({"note": ["none"]}), index=False)


def pfam_tables():
    """Random vs Pfam split per feature set: if the signal is resemblance to drugged families,
    holding whole families out should hurt most where nothing else transfers across them."""
    s = summ("main")
    s = s[(s.regime == "random") & (s.loss == "nnpu") & (s.labels == "minikel")
          & ((s.model.isin(["rf_feat", "mlp_feat"]) & (s.cond == "none"))
             | (s.model.isin(["gp", "cs_feat", "sign_feat", "gcn_feat"]) & (s.cond == "real")))]
    t = s.pivot_table(index=["model", "feats", "universe"], columns="split", values="mean")
    t["drop"] = t.pfam - t.random
    rf = t.loc["rf_feat"]
    for fe in ("all", "no_tract", "attn_free"):
        num(f"rf_pfam_{fe}", nums3([rf.at[(fe, u), "pfam"] if (fe, u) in rf.index else np.nan for u in U], "{:.3f}"))
        num(f"rf_pfam_drop_{fe}", nums3([rf.at[(fe, u), "drop"] if (fe, u) in rf.index else np.nan for u in U], "{:+.3f}"))
    t["cell"] = [f"{r.random:.3f} → {r.pfam:.3f} ({r.drop:+.3f})" for r in t.itertuples()]
    w = t.cell.unstack("universe").reindex(columns=U)
    w.columns = [UN[u] for u in U]
    w = w.reset_index()
    w["_f"] = w.feats.map(FEATS.index)
    w = by_model(w.sort_values("_f")).drop(columns="_f")
    w["model"] = w.model.map(MN)
    put("pfam_drop", w, index=False)


def main_tables():
    for sp in ("random", "pfam"):
        for lo in ("nnpu", "pn", "pn_pathway", "pn_pubmed"):
            put(f"main_compact_{sp}_{lo}", compact("main", sp, lo))
    put("main_regimes", regimes("main", ["rf_feat", "mlp_feat", "gp", "cs_feat", "gcn_feat", "gat_feat"]), index=False)
    s = summ("main")
    s = s[(s.split == "random") & (s.loss == "nnpu") & (s.labels == "minikel") & (s.feats == "all")
          & s.model.isin(["labelprop", "deg_lr", "rf_struct", "gcn_none", "gat_none"])]
    s = s.assign(graph=np.where(s.cond.str.startswith("rw"), "rewired", s.cond))
    t = s.groupby(["universe", "model", "graph", "regime"])["mean"].mean().unstack("regime")
    t = t.reindex(columns=REGIMES).reset_index()
    t["_u"] = t.universe.map(U.index)
    t = by_model(t).sort_values(["_u"], kind="stable").drop(columns="_u")
    t["universe"], t["model"] = t.universe.map(UN), t.model.map(MN)
    put("main_graph_only", t, index=False)


def delong_tables():
    for exp in ("main", "waves", "membrane", "nonsense_labels", "nonsense_feats", "novelty"):
        if not (R / exp / "delong.csv").exists():
            continue
        d = delong(exp)
        dr = d[d.regime == "random"]
        put(f"delong_{exp}_contrast", counts(dr, ["contrast"]), "{:+.3f}")
        for axis in ("feats", "universe", "labels", "split", "loss"):
            if dr[axis].nunique() > 1:
                put(f"delong_{exp}_contrast_{axis}", counts(dr, ["contrast", axis]), "{:+.3f}")
        e = dr[dr.contrast == "empty"]
        if len(e):
            t = e.groupby(["model", "universe"]).agg(b=("better", "sum"), w=("worse", "sum"), n=("better", "size"),
                                                     dm=("delta_auroc", "mean"))
            t["cell"] = [f"{int(r.b)}/{int(r.w)} of {int(r.n)} ({r.dm:+.3f})" for r in t.itertuples()]
            t = t.cell.unstack("universe").reindex(columns=U)
            t.columns = [UN[u] for u in U]
            t = t.reindex([m for m in ORDER if m in t.index])
            t.index = t.index.map(MN)
            put(f"delong_{exp}_empty_model", t)
        t = d.groupby(["contrast", "regime"]).agg(b=("better", "sum"), w=("worse", "sum"), n=("better", "size"))
        t["cell"] = [f"{int(r.b)}/{int(r.w)} of {int(r.n)}" for r in t.itertuples()]
        put(f"delong_{exp}_regimes", t.cell.unstack("regime"))


def effect_tables():
    """Hierarchical-bootstrap effect sizes (scripts/delong.py: effects.csv)."""
    for exp in ("main", "waves", "membrane", "nonsense_labels", "nonsense_feats", "novelty"):
        f = R / exp / "effects.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f)
        e["cell"] = [ci(r.mean_delta, r.ci_lo, r.ci_hi) for r in e.itertuples()]
        w = e.pivot_table(index=["by", "level"], columns="contrast", values="cell", aggfunc="first")
        n = e[e.by == "-"].set_index("contrast")
        w.columns = [f"{c} ({int(n.at[c, 'n_models'])} models)" if c in n.index else c for c in w.columns]
        put(f"effects_{exp}", w.fillna("–"))
        for r in e.itertuples():
            name = f"eff_{exp}_{r.contrast}" + ("" if r.by == "-" else f"_{r.level}")
            num(name, ci(r.mean_delta, r.ci_lo, r.ci_hi))
    if not (R / "main" / "delong.csv").exists():
        return
    d = delong("main")
    d = d[d.regime == "random"]
    for c in ("empty", "rewired", "vs_rf_feat"):
        x = d[d.contrast == c]
        num(f"cnt_main_{c}", f"{int(x.better.sum())} better / {int(x.worse.sum())} worse of {len(x)}")
        for fe in ("all", "attn_free"):
            y = x[x.feats == fe]
            num(f"cnt_main_{c}_{fe}", f"{int(y.better.sum())} / {int(y.worse.sum())} of {len(y)}")


def feature_tables():
    s = summ("main")
    fs = s[s.split.isin(["random", "pfam"]) & (s.loss == "nnpu") & (s.labels == "minikel")
           & ((s.model.isin(["rf_feat", "mlp_feat"]) & (s.cond == "none")) | (s.model.isin(["gp", "cs_feat"])
                                                                                & (s.cond == "real")))]
    for sp in ("random", "pfam"):
        t = fs[fs.split == sp].pivot_table(index=["universe", "model", "feats"], columns="regime", values="mean")
        t = t.reindex(columns=["random", "pubmed", "degree", "pfam", "pathway", "hop1"]).reset_index()
        t["_u"], t["_f"] = t.universe.map(U.index), t.feats.map(FEATS.index)
        t = t.sort_values(["_u", "model", "_f"]).drop(columns=["_u", "_f"])
        t["universe"], t["model"] = t.universe.map(UN), t.model.map(MN)
        put(f"feats_{sp}", t, index=False)
    r = fs[(fs.split == "random") & (fs.model == "rf_feat") & fs.regime.isin(["random", "pubmed"])].pivot_table(
        index=["universe", "regime"], columns="feats", values="mean")
    gap = (r["all"] - r["attn_free"]).unstack("regime")
    gap["share_removed_by_matching"] = 1 - gap.pubmed / gap.random
    gap = gap.reindex(U)
    gap.index = gap.index.map(UN)
    put("feats_gap", gap)


def analysis_tables():
    h = pd.read_csv(A / "hub_bias.csv").set_index("graph").reindex(U)
    h = h.drop(columns=["mwu_p"]).T
    h.columns = [UN[u] for u in h.columns]
    put("hub_bias", h)
    m = pd.read_csv(A / "mediation.csv")
    cols = {"auroc": "AUROC", "auroc_within_pubmed_quintiles": "AUROC within PubMed quintiles",
            "rho_score_log_pubmed": "rho(score, log PubMed)", "shrinkage": "shrinkage",
            "auroc_with_pubmed": "AUROC with PubMed"}
    for u in U:
        t = m[m.graph == u].set_index("block")[list(cols)].rename(columns=cols)
        t.loc["(PubMed alone)", "AUROC"] = m[m.graph == u].pubmed_alone_auroc.iloc[0]
        put(f"mediation_{u}", t)
    mem = pd.read_csv(A / "membrane.csv")
    mem["graph"] = mem.graph.map(UN)
    put("membrane_enrich", mem, index=False)
    p = pd.read_csv(A / "prior_study.csv")
    p["universe"] = p.universe.map(UN)
    put("prior_study", p, index=False)
    sl = pd.read_csv(A / "study_leak.csv").pivot_table(index="feats", columns="universe", values="rho_rf_pred_log_pubmed")
    sl = sl.reindex(["all", "no_tract", "attn_free", "dated", "all_resid", "attn_free_resid"]).reindex(columns=U)
    for fe in ("all", "attn_free", "dated", "all_resid", "attn_free_resid"):
        if fe in sl.index:
            num(f"leak_{fe}", nums3(sl.loc[fe].values))
    sl.columns = [UN[u] for u in U]
    put("study_leak", sl)
    pr = pd.read_csv(A / "prior_study.csv")
    pr = pr[(pr.years_before_launch == 10) & pr.launch_period.isin(["2000-2009", "2010-2024"])]
    num("prior_pct_10y", f"{pr.median_percentile.min() * 100:.0f}th–{pr.median_percentile.max() * 100:.0f}th")
    num("prior_auroc_10y", f"{pr.auroc_vs_no_programme.min():.2f}–{pr.auroc_vs_no_programme.max():.2f}")


def label_bias():
    lb = pd.read_csv(A / "label_bias.csv")
    lb["universe"] = lb.universe.map(UN)
    put("label_bias", lb, index=False)
    for key, name in (("minikel", "minikel (>= Phase III)"), ("genetic_T", "genetic evidence before 2010")):
        x = lb[lb.label == name].set_index("universe").reindex([UN[u] for u in U])
        num(f"lb_{key}_auroc", nums3(x.pubmed_auroc_now))
        num(f"lb_{key}_auroc_T", nums3(x.pubmed_auroc_before_T))
        num(f"lb_{key}_n", nums3(x.positives, "{:.0f}"))


def coverage_table():
    c = pd.read_csv(A / "time_split_coverage.csv")
    c["universe"] = c.universe.map(UN)
    put("time_split_coverage", c, index=False)
    s = c[c.universe == "STRING"].set_index("group")
    ex = [g for g in s.index if "excluded" in g]
    te = [g for g in s.index if "(test)" in g][0]
    num("cov_n_excluded", f"{int(s.loc[ex, 'genes'].sum())}", "{}")
    num("cov_test_top", f"{s.at[te, 'frac_top_decile_before_T']:.0%}")
    num("cov_excl_top", " / ".join(f"{s.at[g, 'frac_top_decile_before_T']:.0%}" for g in ex))
    num("cov_test_novel2", f"{s.at[te, 'frac_novel_2of3']:.0%}")
    num("cov_excl_novel2", " / ".join(f"{s.at[g, 'frac_novel_2of3']:.0%}" for g in ex))
    num("cov_test_fam", f"{s.at[te, 'frac_drugged_family']:.0%}")
    num("cov_excl_fam", " / ".join(f"{s.at[g, 'frac_drugged_family']:.0%}" for g in ex))


def discovery_tables():
    ta = pd.read_csv(A / "topk_audit.csv")
    ta = ta[ta.loss.isin(["-", "nnpu", "pn_pubmed"]) & ta.model.isin(
        ["pubmed_only", "all test unlabelled", "rf_feat", "mlp_feat", "gp", "cs_feat", "gcn_feat"])]
    ta = ta[(ta.model != "gcn_feat") | (ta["copy"] == "real")]
    cols = ["copy", "feats", "model", "loss", "pubmed_pct_median", "frac_top_pubmed_decile", "frac_in_programme",
            "frac_shares_pfam_with_known", "frac_within_1hop_of_known", "overlap_with_pubmed_top", "rho_score_pubmed"]
    for u in U:
        t = ta[ta.universe == u][cols].sort_values(["copy", "model", "feats", "loss"]).copy()
        t["model"] = t.model.map(lambda x: MN.get(x, x))
        put(f"topk_{u}", t, "{:.2f}", index=False)
    pl = pd.read_csv(A / "phase_ladder.csv")
    pl = pl[pl.loss.isin(["-", "nnpu", "pn_pubmed"]) & pl.model.isin(["pubmed_only", "rf_feat", "mlp_feat", "gp", "cs_feat"])]
    cols = ["copy", "feats", "model", "loss", "auroc_programme_vs_none", "auroc_within_pubmed_quintiles",
            "auroc_preclinical_vs_none", "auroc_phase1_vs_none", "auroc_phase2_vs_none"]
    for u in U:
        t = pl[pl.universe == u][cols].sort_values(["copy", "model", "feats", "loss"]).copy()
        t["model"] = t.model.map(lambda x: MN.get(x, x))
        put(f"ladder_{u}", t, index=False)


def gp_table():
    gp = pd.read_csv(R / "main" / "gp_attribution.csv")
    gp["feats"] = gp.feats.fillna("all")
    gp = gp[(gp.split == "random") & (gp.loss == "nnpu") & (gp["cond"] == "real")]
    sh = [c for c in gp.columns if c.startswith("gp_share_")]
    t = gp.set_index(["universe", "feats"])[sh + ["gp_sd_vs_degree", "gp_sd_vs_pubmed"]]
    t.columns = [c.replace("gp_share_", "") for c in t.columns]
    t = t.dropna(axis=1, how="all").reset_index()
    t["universe"] = t.universe.map(UN)
    put("gp_attr", t, "{:.2f}", index=False)


def _strata(nv, u, metric, label_rows, models=("rf_feat", "mlp_feat", "gp", "cs_feat", "gcn_feat", "labelprop"),
            losses=("nnpu", "pn_pubmed"), refs=True):
    x = nv[nv.universe == u]
    keep = x[(x.model.isin(REFS) & refs) | (x.model.isin(models) & x.loss.isin(losses) & x["copy"].isin(["real", "none"])
                                             & x["labels"].isin(label_rows))]
    t = keep.pivot_table(index=["labels", "model", "copy", "feats", "loss"], columns="stratum", values=metric)
    t = t.reindex(columns=[c for c in STRATA if c in t.columns]).reset_index()
    t["_r"] = ~t.model.isin(REFS)
    t = by_model(t).sort_values(["_r", "feats", "loss", "labels"], kind="stable").drop(columns="_r")
    t["model"] = t.model.map(lambda v: MN.get(v, v))
    return t


CI_MODELS = ("rf_feat", "mlp_feat", "gp", "cs_feat", "gcn_feat")


def _ci_table(nv, metric, stratum, labels=("minikel",), models=CI_MODELS, losses=("nnpu", "pn_pubmed"),
              refs=True, diff=None) -> pd.DataFrame:
    """Rows = reference rankers and models, columns = graphs + pooled; cells 'value [95% CI]'
    for one stratum. diff="pubmed": the paired difference from PubMed-before-T; diff="minikel":
    from the run's Minikel-trained twin (genetic rows only)."""
    x = nv[nv.stratum == stratum]
    keep = x[(x.model.isin(REFS) & refs) | (x.model.isin(models) & x.loss.isin(losses) & x["copy"].isin(["real", "none"])
                                             & x["labels"].isin(labels))]
    if diff == "pubmed":
        keep = keep[keep.model != "pubmed_before_T"]
    c = f"{metric}_vs_{diff}" if diff else metric
    if c not in keep:
        raise KeyError(c)
    keep = keep.dropna(subset=[c])
    keep = keep.assign(cell=[ci(getattr(r, c), getattr(r, f"{c}_lo"), getattr(r, f"{c}_hi"),
                                "{:+.2f}" if diff else "{:.2f}") for r in keep.itertuples()])
    w = keep.pivot_table(index=["labels", "model", "copy", "feats", "loss"], columns="universe", values="cell",
                         aggfunc="first").reindex(columns=[u for u in UP if u in keep.universe.unique()])
    n = x[x.model == "pubmed_before_T"].set_index("universe").n_future
    w.columns = [f"{UNP[u]} (n={int(n[u])})" if u in n.index else UNP[u] for u in w.columns]
    w = w.reset_index()
    w["_r"], w["_f"] = ~w.model.isin(REFS), w.feats.map(lambda f: (["-"] + FEATS).index(f))
    w = by_model(w).sort_values(["_r", "_f", "loss", "labels"], kind="stable").drop(columns=["_r", "_f"])
    w["model"] = w.model.map(lambda v: MN.get(v, v))
    return w.fillna("–")


def novelty_tables():
    nv = pd.read_csv(A / "time_novelty.csv")
    if "labels" not in nv:
        nv["labels"] = np.where(nv.model.isin(REFS), "-", "minikel")
    comp = nv[nv.model == "pubmed_before_T"].pivot_table(index="stratum", columns="universe", values="n_future")
    comp = comp.reindex(STRATA).reindex(columns=[u for u in UP if u in comp.columns])
    comp = comp.apply(lambda c: [f"{int(v)} ({v / c['all']:.0%})" if pd.notna(v) else "0" for v in c])
    comp.columns = [UNP[u] for u in comp.columns]
    put("novelty_composition", comp)
    n = nv[nv.model == "pubmed_before_T"].pivot_table(index="universe", columns="stratum", values="n_future")
    n = n.reindex(UP).fillna(0)
    share = lambda s: nums3([n.at[u, s] / n.at[u, "all"] for u in U], "{:.0%}")  # noqa: E731
    num("fut_n", nums3(n.loc[U, "all"], "{:.0f}"))
    num("fut_top_decile", share("study>=0.9"))
    num("fut_below_median", share("study<0.5"))
    num("fut_1hop", share("1 hop"))
    num("fut_3hop", share("3+ hops"))
    num("fut_new_family", share("new family"))
    num("novel_n", nums3(n.loc[U, "novel"], "{:.0f}"))
    num("novel_share", share("novel"))
    num("novel2_n", nums3(n.loc[U, "novel (2 of 3)"], "{:.0f}"))
    num("novel_n_pooled", f"{n.at['pooled', 'novel']:.0f}" if "pooled" in n.index else "–")
    num("novel2_n_pooled", f"{n.at['pooled', 'novel (2 of 3)']:.0f}" if "pooled" in n.index else "–")

    def pick(model, stratum="all", metric="auroc", feats="-", loss="-", labels="-", copy="none", us=U):
        x = nv[(nv.model == model) & (nv.stratum == stratum) & (nv.feats == feats) & (nv.loss == loss)
               & (nv["labels"] == labels) & (nv["copy"] == copy)].set_index("universe")
        return [x.at[u, metric] if u in x.index else np.nan for u in us]
    for ref in REFS:
        num(f"ref_{ref}_auroc", nums3(pick(ref)))
        num(f"ref_{ref}_recall_novel", nums3(pick(ref, "novel", "recall_top10pct"), "{:.0%}"))
        num(f"ref_{ref}_recall_novel2", nums3(pick(ref, "novel (2 of 3)", "recall_top10pct"), "{:.0%}"))
    for fe in FEATS:
        for lab in ("minikel", "genetic"):
            k = dict(feats=fe, loss="nnpu", labels=lab)
            num(f"rf_time_{lab}_{fe}", nums3(pick("rf_feat", **k)))
            num(f"rf_time_{lab}_{fe}_novel2", nums3(pick("rf_feat", "novel (2 of 3)", **k)))
            num(f"rf_time_{lab}_{fe}_recall_novel2", nums3(pick("rf_feat", "novel (2 of 3)", "recall_top10pct", **k),
                                                       "{:.0%}"))
    for metric in ("auroc", "recall_top10pct"):
        for u in U:
            put(f"novelty_{metric}_{u}", _strata(nv, u, metric, ["minikel"]), "{:.2f}", index=False)
            if (nv.labels == "genetic").any():  # label comparison: same models, both labels side by side
                put(f"genetic_{metric}_{u}", _strata(nv, u, metric, ["minikel", "genetic"],
                                                     models=("rf_feat", "mlp_feat", "gp", "cs_feat"),
                                                     losses=("nnpu",), refs=False), "{:.2f}", index=False)
        for st, tag in (("novel", "novel"), ("novel (2 of 3)", "novel2"), ("new family", "newfam")):
            put(f"novelty_ci_{metric}_{tag}", _ci_table(nv, metric, st), index=False)
            put(f"novelty_vs_pubmed_{metric}_{tag}", _ci_table(nv, metric, st, diff="pubmed"), index=False)
            if (nv.labels == "genetic").any():
                put(f"genetic_ci_{metric}_{tag}", _ci_table(nv, metric, st, labels=("minikel", "genetic"),
                                                            models=("rf_feat", "mlp_feat", "gp", "cs_feat"),
                                                            losses=("nnpu",), refs=False), index=False)
                put(f"genetic_vs_minikel_{metric}_{tag}", _ci_table(nv, metric, st, labels=("genetic",),
                                                                    models=("rf_feat", "mlp_feat", "gp", "cs_feat"),
                                                                    losses=("nnpu",), refs=False, diff="minikel"),
                    index=False)
    if (nv.labels == "genetic").any():
        genetic_endpoints(nv)
    for lo in ("nnpu", "pn", "pn_pubmed"):
        for fe in ("all", "attn_free", "dated"):
            put(f"novelty_head_time_{lo}_{fe}", compact("novelty", "time", lo, feats=fe))
            if (summ("novelty").labels == "genetic").any():
                put(f"genetic_head_time_{lo}_{fe}", compact("novelty", "time", lo, feats=fe, labels="genetic"))
    for fe, name in (("all", "novelty_regimes_time"), ("attn_free", "novelty_regimes_time_attn"),
                     ("dated", "novelty_regimes_time_dated")):
        put(name, regimes("novelty", ["rf_feat", "mlp_feat", "gp", "cs_feat", "gcn_feat"], split="time", feats=fe),
            index=False)


PRIMARY = dict(stratum="novel (2 of 3)", universe="pooled", loss="nnpu", models=("rf_feat", "gp"),
               feats=("attn_free", "dated"))


def genetic_endpoints(nv):
    """§6.4's pre-stated endpoint: recall@10% of the novel (2 of 3) future targets, pooled over
    graphs, genetic-trained minus Minikel-trained (paired bootstrap CI), for RF and the GP with
    the leak-free feature sets. Plus how closely each label's rankings follow study at T."""
    x = nv[(nv.stratum == PRIMARY["stratum"]) & (nv.universe == PRIMARY["universe"]) & (nv.loss == PRIMARY["loss"])
           & (nv["labels"] == "genetic") & nv.model.isin(PRIMARY["models"]) & nv.feats.isin(PRIMARY["feats"])
           & nv["copy"].isin(["real", "none"])]
    for r in x.itertuples():
        num(f"primary_{r.model}_{r.feats}", ci(r.recall_top10pct_vs_minikel, r.recall_top10pct_vs_minikel_lo,
                                               r.recall_top10pct_vs_minikel_hi, "{:+.2f}"))
    rho = nv[(nv.stratum == "all") & nv["labels"].isin(["minikel", "genetic"]) & (nv.loss == "nnpu")
             & nv.model.isin(["rf_feat", "mlp_feat", "gp", "cs_feat"]) & nv["copy"].isin(["real", "none"])]
    w = rho.pivot_table(index=["model", "copy", "feats", "universe"], columns="labels", values="rho_pubmed_T")
    w["genetic − minikel"] = w.genetic - w.minikel
    w = w.reset_index()
    w["_u"] = w.universe.map(UP.index)
    w = by_model(w.sort_values(["_u", "feats"])).drop(columns="_u")
    w["universe"], w["model"] = w.universe.map(UNP), w.model.map(MN)
    put("genetic_rho", w, "{:+.2f}", index=False)
    num("rho_shift", f"{w['genetic − minikel'].min():+.2f} to {w['genetic − minikel'].max():+.2f}")


TK_COLS = {"frac_future_target": "future target", "frac_novel_future": "novel future target",
           "frac_top_pubmed_decile_T": "top study decile 2009", "frac_in_programme": "in a programme today",
           "frac_shares_pfam_with_known": "Pfam with a 2009 target",
           "frac_within_1hop_of_known": "1 hop from a 2009 target",
           "frac_genetic_before_T": "genetic evidence pre-2010"}


def genetic_topk():
    """§6.4: the top-100 test genes of each label's primary models on the time split."""
    t = pd.read_csv(A / "time_topk_audit.csv")
    ref = t[t["labels"] == "-"].copy()
    ref["model"] = ref.model.map({"all test genes": "all test genes (base rate)", "pubmed_before_T": "PubMed 2009"})
    run = t[t.model.isin(PRIMARY["models"]) & t.feats.isin(PRIMARY["feats"]) & t["labels"].isin(["minikel", "genetic"])
            & t["copy"].isin(["real", "none"])].copy()
    run["_o"] = [(PRIMARY["models"].index(r.model), r.feats, r.labels != "minikel") for r in run.itertuples()]
    run = run.sort_values("_o")
    run["model"] = [f"{MN[r.model]} `{r.feats}`, {'Minikel' if r.labels == 'minikel' else 'genetic'}"
                    for r in run.itertuples()]
    rows = []
    for u in U:
        x = pd.concat([ref[ref.universe == u], run[run.universe == u]])
        rows.append(x.assign(universe=UN[u])[["universe", "model", *TK_COLS]])
    tab = pd.concat(rows).rename(columns=TK_COLS)
    put("genetic_topk", tab.map(lambda v: f"{v:.0%}" if isinstance(v, float) else v), index=False)
    for c in TK_COLS:
        num(f"tk_base_{c}", nums3([ref[(ref.universe == u) & (ref.model.str.startswith("all"))][c].iloc[0] for u in U],
                                  "{:.0%}"))
        for lab in ("minikel", "genetic"):
            v = [run[(run.universe == u) & (run["labels"] == lab) & (run.model.str.startswith("RF"))][c] for u in U]
            span = lambda x: "–".join(dict.fromkeys(f"{w:.0%}" for w in (x.min(), x.max())))  # noqa: E731
            num(f"tk_rf_{lab}_{c}", " / ".join(span(x) for x in v))


TRIED_PRED = {"pubmed_before_T": "PubMed before 2010", "pubmed_now": "PubMed now",
              "genetic_before_T": "genetic evidence before 2010", "genetic_any": "genetic evidence (any date)"}


def tried_tables():
    """§6.5: success given tried (scripts/analysis.py: tried_*.csv)."""
    g = pd.read_csv(A / "tried_groups.csv").set_index("universe").reindex(U)
    g.index = pd.Index([UN[u] for u in U], name="universe")
    put("tried_groups", g, "{:.0f}")
    a = pd.read_csv(A / "tried_auroc.csv")
    a["cell"] = [f"{r.auroc:.2f} [{r.lo:.2f}, {r.hi:.2f}]" for r in a.itertuples()]
    w = a.pivot_table(index=["universe", "predictor"], columns="contrast", values="auroc")
    a = a.pivot_table(index=["universe", "predictor"], columns="contrast", values="cell", aggfunc="first")
    a["outcome / selection excess"] = ((w.outcome - 0.5) / (w.selection - 0.5)).map("{:.2f}".format)
    a = a[["label", "selection", "outcome", "outcome / selection excess"]].reset_index()
    a["_u"], a["_p"] = a.universe.map(U.index), a.predictor.map(lambda x: (x not in TRIED_PRED, x))
    a = a.sort_values(["_u", "_p"]).drop(columns=["_u", "_p"])
    a["universe"] = a.universe.map(UN)
    a["predictor"] = a.predictor.map(lambda x: TRIED_PRED.get(x) or MN.get(x.split(" (")[0], x)
                                     + (" (" + x.split(" (", 1)[1] if " (" in x else ""))
    put("tried_auroc", a, index=False)
    ratio = lambda p: nums3([(w.at[(u, p), "outcome"] - .5) / (w.at[(u, p), "selection"] - .5) for u in U])  # noqa
    for p_, k in (("pubmed_before_T", "pubmed"), ("genetic_before_T", "genetic"), ("rf_feat (all)", "rf_all"),
                  ("rf_feat (attn_free)", "rf_attn_free")):
        num(f"tried_ratio_{k}", ratio(p_))
        for c in ("selection", "outcome"):
            num(f"tried_{c}_{k}", nums3([w.at[(u, p_), c] for u in U]))
    o = pd.read_csv(A / "tried_genetic_or.csv")
    o["odds ratio"] = [f"{r.odds_ratio:.2f} [{r.lo:.2f}, {r.hi:.2f}]" for r in o.itertuples()]
    o["universe"], o["evidence"] = o.universe.map(UN), o.evidence.map(TRIED_PRED)
    put("tried_genetic_or", o.drop(columns=["odds_ratio", "lo", "hi"]), "{:.0f}", index=False)
    x = o[o.evidence == TRIED_PRED["genetic_before_T"]]
    num("tried_or_genetic", " / ".join(x["odds ratio"]))
    t = pd.read_csv(A / "tried_time.csv")
    t["genetic − Minikel"] = [ci(r.genetic_minus_minikel, r.lo, r.hi) for r in t.itertuples()]
    t["_u"] = t.universe.map(UP.index)
    t = by_model(t.sort_values("feats")).sort_values(["_u", "contrast"], ascending=[True, False], kind="stable")
    for r in t[t.universe == "pooled"].itertuples():
        num(f"tried_time_{r.contrast}_{r.model}_{r.feats}", t.at[r.Index, "genetic − Minikel"])
    t["universe"], t["model"] = t.universe.map(UNP), t.model.map(MN)
    put("tried_time", t[["universe", "contrast", "model", "feats", "n_pos", "n_neg", "minikel", "genetic",
                         "genetic − Minikel"]], index=False)


def genetic_delong():
    d = delong("novelty")
    d = d[(d.contrast == "vs_minikel_label") & (d.regime == "random")]
    if d.empty:
        return
    t = counts(d, ["feats", "loss"]).rename(columns={"real_better": "genetic_better", "real_worse": "genetic_worse"})
    put("genetic_delong", t, "{:+.3f}")
    t = d.groupby(["model", "universe", "feats"]).agg(b=("better", "sum"), w=("worse", "sum"), n=("better", "size"),
                                                      dm=("delta_auroc", "mean"))
    t["cell"] = [f"{int(r.b)}/{int(r.w)} of {int(r.n)} ({r.dm:+.3f})" for r in t.itertuples()]
    t = t.cell.unstack("universe").reindex(columns=U)
    t.columns = [UN[u] for u in U]
    t = by_model(t.reset_index())
    t["model"] = t.model.map(MN)
    put("genetic_delong_model", t, index=False)


def waves_tables():
    w = summ("waves")
    w = w[(w.regime == "random") & (w.loss == "nnpu") & w.cond.isin(["real", "empty", "none"])
          & w.model.isin(["rf_feat", "mlp_feat", "gp", "cs_feat", "gcn_feat", "gat_feat", "labelprop"])]
    for fe in ("all", "attn_free"):
        t = w[w.feats == fe].pivot_table(index=["universe", "split"], columns=["model", "cond"], values="mean")
        t.columns = [f"{MN[m]} {c}" if c != "none" else MN[m] for m, c in t.columns]
        t = t.reset_index()
        t["universe"] = t.universe.map(UN)
        put(f"waves_{fe}", t, index=False)


def control_tables():
    put("membrane_head_random_nnpu", compact("membrane", "random", "nnpu", feats="no_loc", labels="membrane"))
    put("membrane_head_pfam_nnpu", compact("membrane", "pfam", "nnpu", feats="no_loc", labels="membrane"))
    for lab in ("minikel_perm", "minikel_famperm"):
        put(f"nl_{lab}", compact("nonsense_labels", "random", "nnpu", labels=lab))
    put("nf_random", compact("nonsense_feats", "random", "nnpu", feats="random"))


SECTIONS = [setup, fit_failures, main_tables, pfam_tables, delong_tables, effect_tables, feature_tables,
            analysis_tables, label_bias, coverage_table, discovery_tables, gp_table, novelty_tables, genetic_topk,
            tried_tables, genetic_delong, waves_tables, control_tables]


def main():
    for section in SECTIONS:
        try:
            section()
        except (FileNotFoundError, KeyError) as e:
            print(f"skip {section.__name__}: {type(e).__name__}: {e}", flush=True)
        except Exception:
            print(f"error in {section.__name__}:\n{traceback.format_exc()}", flush=True)
    missing = []

    def fill(m):
        if m.group(1) in TABLES:
            return TABLES[m.group(1)]
        missing.append(m.group(1))
        return f"*(table `{m.group(1)}` not available: its results are missing)*"

    def fill_num(m):
        if m.group(1) in NUMS:
            return NUMS[m.group(1)]
        missing.append("n:" + m.group(1))
        return f"[{m.group(1)}?]"

    text = re.sub(r"\{\{n:(\w+)\}\}", fill_num, TEMPLATE.read_text())
    text = re.sub(r"\{\{(\w+)\}\}", fill, text)
    OUT.write_text(text)
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    gone = [i for i in images if not (ROOT / i).exists()]
    print(f"wrote {OUT}: {len(TABLES)} tables, {len(NUMS)} numbers, {len(images)} figures"
          + (f"; missing tables: {missing}" if missing else "") + (f"; missing figures: {gone}" if gone else ""))


if __name__ == "__main__":
    main()
