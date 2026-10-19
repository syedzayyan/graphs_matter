"""Results report: every table built from results/, filled into scripts/report_template.md.

  uv run python scripts/report.py            ->  summary.md (repo root)

The template holds the prose and {{name}} placeholders; each placeholder is a markdown table
built here from results/<exp>/*.csv and results/analysis/*.csv. A section whose inputs don't
exist yet is skipped and its placeholders render as a note. Figures are linked from
results/figures/ (scripts/figures.py). Numbers quoted in the template's prose are written by
hand: re-check them when the results change.
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
          "3+ hops", "novel"]
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
    t = s.groupby(["model", "universe", "c"])["mean"].mean().unstack("c")
    t["cell"] = t.apply(lambda r: f"{r['none']:.3f} (no graph)" if pd.notna(r.get("none")) else " / ".join(
        "–" if pd.isna(r.get(c)) else f"{r.get(c):.3f}" for c in ("real", "rewired", "empty")), axis=1)
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


def feature_tables():
    s = summ("main")
    fs = s[s.split.isin(["random", "pfam"]) & (s.loss == "nnpu") & (s.labels == "minikel")
           & ((s.model.isin(["rf_feat", "mlp_feat"]) & (s.cond == "none")) | (s.model.isin(["gp", "cs_feat"])
                                                                                & (s.cond == "real")))]
    for sp in ("random", "pfam"):
        t = fs[fs.split == sp].pivot_table(index=["universe", "model", "feats"], columns="regime", values="mean")
        t = t.reindex(columns=["random", "pubmed", "degree", "pfam", "pathway", "hop1"]).reset_index()
        t["_u"], t["_f"] = t.universe.map(U.index), t.feats.map(["all", "no_tract", "attn_free"].index)
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
    sl = sl.reindex(["all", "no_tract", "attn_free", "all_resid", "attn_free_resid"]).reindex(columns=U)
    sl.columns = [UN[u] for u in U]
    put("study_leak", sl)


def label_bias():
    lb = pd.read_csv(A / "label_bias.csv")
    lb["universe"] = lb.universe.map(UN)
    put("label_bias", lb, index=False)


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


def novelty_tables():
    nv = pd.read_csv(A / "time_novelty.csv")
    if "labels" not in nv:
        nv["labels"] = np.where(nv.model.isin(REFS), "-", "minikel")
    comp = nv[nv.model == "pubmed_before_T"].pivot_table(index="stratum", columns="universe", values="n_future")
    comp = comp.reindex(STRATA).reindex(columns=U)
    comp = comp.apply(lambda c: [f"{int(v)} ({v / c['all']:.0%})" if pd.notna(v) else "0" for v in c])
    comp.columns = [UN[u] for u in U]
    put("novelty_composition", comp)
    for metric in ("auroc", "recall_top10pct"):
        for u in U:
            put(f"novelty_{metric}_{u}", _strata(nv, u, metric, ["minikel"]), "{:.2f}", index=False)
            if (nv.labels == "genetic").any():  # label comparison: same models, both labels side by side
                put(f"genetic_{metric}_{u}", _strata(nv, u, metric, ["minikel", "genetic"],
                                                     models=("rf_feat", "mlp_feat", "gp", "cs_feat"),
                                                     losses=("nnpu",), refs=False), "{:.2f}", index=False)
    for lo in ("nnpu", "pn", "pn_pubmed"):
        for fe in ("all", "attn_free"):
            put(f"novelty_head_time_{lo}_{fe}", compact("novelty", "time", lo, feats=fe))
            if (summ("novelty").labels == "genetic").any():
                put(f"genetic_head_time_{lo}_{fe}", compact("novelty", "time", lo, feats=fe, labels="genetic"))
    for fe, name in (("all", "novelty_regimes_time"), ("attn_free", "novelty_regimes_time_attn")):
        put(name, regimes("novelty", ["rf_feat", "mlp_feat", "gp", "cs_feat", "gcn_feat"], split="time", feats=fe),
            index=False)


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


SECTIONS = [setup, main_tables, delong_tables, feature_tables, analysis_tables, label_bias, discovery_tables,
            gp_table, novelty_tables, genetic_delong, waves_tables, control_tables]


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

    text = re.sub(r"\{\{(\w+)\}\}", fill, TEMPLATE.read_text())
    OUT.write_text(text)
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    gone = [i for i in images if not (ROOT / i).exists()]
    print(f"wrote {OUT}: {len(TABLES)} tables, {len(images)} figures"
          + (f"; missing tables: {missing}" if missing else "") + (f"; missing figures: {gone}" if gone else ""))


if __name__ == "__main__":
    main()
