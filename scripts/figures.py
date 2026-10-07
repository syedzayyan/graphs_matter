"""Figures: five sets per graph and experiment.   uv run python scripts/figures.py [--exps main membrane ...]

results/figures/<experiment>/<graph>/
  1_conditions.png    AUROC per model under real / rewired / empty graph vs no-graph baselines
  2_graph_effect.png  paired-by-seed real−empty and real−rewired deltas
  3_graph_only.png    graph-only models across evaluation-negative regimes, real vs rewired
  4_gp_attribution.png  GP share of the predictive signal per kernel block
  5_degree.png        positive rate and model score percentile by degree decile
results/figures/compare/6_drug_vs_membrane.png   real−empty on druggability vs membrane

Defaults shown: random split, nnPU, all features (unless a panel says otherwise).
Colours: validated categorical palette (dataviz skill reference instance); condition
identity is also carried by marker shape, never colour alone.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghelps import data, paths  # noqa: E402

RES = ROOT / "results"
OUT = RES / "figures"
FEATS = "all"  # primary feature set of the experiment being drawn (set in main)

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COND = {  # colour, marker, label
    "real": ("#2a78d6", "o", "real graph"),
    "rewired": ("#eb6834", "s", "rewired (degree-preserving)"),
    "empty": ("#1baf7a", "^", "empty graph (same model)"),
    "shufattr": (MUTED, "D", "real graph, shuffled edge vectors"),
}
BLOCKS = {  # GP kernel blocks, fixed categorical order
    "go": ("#2a78d6", "GO"), "pfam": ("#eb6834", "Pfam"), "pathway": ("#1baf7a", "pathway"),
    "tract": ("#eda100", "tractability"), "degree": ("#e87ba4", "degree"),
    "topology": ("#008300", "topology"), "neighbour_feat": ("#4a3aa7", "neighbour features"),
}
FEATURE_GRAPH_MODELS = ["sign_feat", "cs_feat", "gcn_feat", "sage_feat", "gat_feat", "gat_edge_feat"]
NICE = {"sign_feat": "SIGN", "cs_feat": "Correct & Smooth", "gcn_feat": "GCN", "sage_feat": "GraphSAGE",
        "gat_feat": "GAT", "gat_edge_feat": "GAT + edge vectors", "gp": "GP (additive)",
        "mlp_feat_struct": "MLP + struct stats", "rf_feat_struct": "RF + struct stats",
        "labelprop": "label propagation", "rf_struct": "RF on struct stats", "gcn_none": "GCN, no features",
        "gat_none": "GAT, no features", "deg_lr": "degree only (LR)", "mlp_feat": "MLP (no graph)",
        "rf_feat": "RF (no graph)"}
FEATS_NICE = {"all": "all features", "no_tract": "no tractability", "no_loc": "Pfam + pathway",
              "random": "random vectors"}
GRAPH_NICE = {"string": "STRING ≥700", "string_exp": "STRING experimental", "intact": "IntAct",
              "reactome_fi": "Reactome FI (curated)", "huri": "HuRI"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 2,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def load(exp: str) -> pd.DataFrame:
    df = pd.read_csv(RES / exp / "results.csv")
    df["copy"] = df["copy"].fillna("none")
    df["graph"] = df["graph"].fillna("-")
    df["feats"] = df["feats"].fillna("all")
    df["cond"] = df["copy"].where(~df["copy"].str.startswith("rw"), "rewired")
    df["exp"] = exp
    return df


def per_seed(df: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """One number per (cell, seed, condition); rewired copies averaged within a seed."""
    keys = ["graph", "feats", "split", "loss", "regime", "model", "cond", "seed"]
    return df.groupby(keys)[metric].mean().reset_index()


def sel(d, **kw):
    for k, v in kw.items():
        d = d[d[k].isin(v) if isinstance(v, (list, tuple, set)) else d[k] == v]
    return d


def header(fig, title, sub):
    """Title + subtitle stacked above the plot area at fixed point offsets, so they never
    collide whatever the figure height (savefig bbox=tight grows the canvas to fit)."""
    from matplotlib.transforms import ScaledTranslation
    top = max(ax.get_position().y1 for ax in fig.axes)
    for text, dy, kw in ((sub, 22, dict(fontsize=8.5, color=INK2)), (title, 36, dict(fontsize=11.5, color=INK))):
        fig.text(0.0, top, text, ha="left", va="bottom",
                 transform=fig.transFigure + ScaledTranslation(0, dy / 72, fig.dpi_scale_trans), **kw)


# ------------------------------------------------------------------ 1. conditions

def fig_conditions(ps: pd.DataFrame, graph: str, out: Path):
    models = [m for m in FEATURE_GRAPH_MODELS + ["gp", "mlp_feat_struct", "rf_feat_struct", "labelprop", "rf_struct"]
              if m in set(ps.model)]
    losses = [l for l in ("nnpu", "pn") if l in set(ps.loss)]
    fig, axes = plt.subplots(1, len(losses), figsize=(5.2 * len(losses), 0.38 * len(models) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, loss in zip(axes, losses):
        d = sel(ps, split="random", regime="random", feats=FEATS, loss=loss)
        base = sel(d, graph="-")
        for name, ls in (("rf_feat", "-"), ("mlp_feat", ":")):
            v = base[base.model == name].auroc
            if len(v):
                ax.axvline(v.mean(), color=MUTED, ls=ls, lw=1.2, zorder=1)
        g = sel(d, graph=graph)
        offs = {"real": -0.22, "rewired": 0.0, "empty": 0.22, "shufattr": 0.33}
        for i, m in enumerate(models):
            for c, (col, mk, _) in COND.items():
                v = g[(g.model == m) & (g.cond == c)].auroc
                if len(v) == 0:
                    continue
                ax.errorbar(v.mean(), i + offs[c], xerr=v.std(), fmt=mk, ms=6, color=col,
                            mfc=col if c != "shufattr" else SURFACE, mec=col, mew=1.5, elinewidth=1.2,
                            capsize=0, zorder=3)
        ax.set_yticks(range(len(models)), [NICE[m] for m in models])
        ax.invert_yaxis()
        ax.set_xlabel("test AUROC (mean ± sd over 5 seeds)")
        ax.set_title(f"{loss.upper() if loss == 'pn' else 'nnPU'} training")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], ls="", marker=mk, color=col, mfc=col if c != "shufattr" else SURFACE, ms=6,
                          label=lab) for c, (col, mk, lab) in COND.items() if c in set(sel(ps, graph=graph).cond)]
    handles += [plt.Line2D([], [], color=MUTED, ls=ls, lw=1.2, label=lab)
                for ls, lab in (("-", "RF, no graph"), (":", "MLP, no graph"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8)
    header(fig, f"{GRAPH_NICE.get(graph, graph)}: does the graph help inside the model?",
           "Random split, random unlabelled negatives, all features. Grey lines: feature-only baselines (no graph, per loss).")
    fig.savefig(out / "1_conditions.png")
    plt.close(fig)


# ------------------------------------------------------------------ 2. graph effect

def fig_graph_effect(ps: pd.DataFrame, graph: str, out: Path):
    models = [m for m in FEATURE_GRAPH_MODELS + ["gp"] if m in set(sel(ps, graph=graph).model)]
    panels = [(s, l) for s in ("random", "pfam") for l in ("nnpu", "pn")
              if len(sel(ps, graph=graph, split=s, loss=l))]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 0.42 * len(models) + 1.5), sharey=True, sharex=True,
                             squeeze=False)
    for ax, (split, loss) in zip(axes[0], panels):
        d = sel(ps, graph=graph, split=split, loss=loss, regime="random", feats=FEATS)
        w = d.pivot_table(index=["model", "seed"], columns="cond", values="auroc")
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        for i, m in enumerate(models):
            if m not in w.index.get_level_values(0):
                continue
            x = w.loc[m]
            for j, (other, off) in enumerate((("empty", -0.15), ("rewired", 0.15))):
                if other not in x or x[other].isna().all():
                    continue
                delta = (x["real"] - x[other]).dropna()
                col, mk, _ = COND[other]
                ax.errorbar(delta.mean(), i + off, xerr=delta.std(), fmt=mk, color=col, ms=6, elinewidth=1.2, zorder=3)
                ax.scatter(delta, np.full(len(delta), i + off), s=8, color=col, alpha=0.35, lw=0, zorder=2)
        ax.set_yticks(range(len(models)), [NICE[m] for m in models])
        ax.invert_yaxis()
        ax.set_title(f"{split} split · {'nnPU' if loss == 'nnpu' else 'PN'}")
        ax.set_xlabel("Δ AUROC  (real − control)")
        ax.grid(axis="y", visible=False)
        lo, hi = ax.get_xlim()
        ax.text(hi, len(models) - 0.45, "real better →", ha="right", va="top", fontsize=7.5, color=INK2,
                fontfamily="DejaVu Sans")
        ax.text(lo, len(models) - 0.45, "← control better", ha="left", va="top", fontsize=7.5, color=INK2,
                fontfamily="DejaVu Sans")
    handles = [plt.Line2D([], [], ls="", marker=COND[c][1], color=COND[c][0], ms=6, label=lab)
               for c, lab in (("empty", "real − empty graph"), ("rewired", "real − rewired graph"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8)
    header(fig, f"{GRAPH_NICE.get(graph, graph)}: paired graph effect",
           "Large marker: mean ± sd of the per-seed difference; small dots: individual seeds. All features.")
    fig.savefig(out / "2_graph_effect.png")
    plt.close(fig)


# ------------------------------------------------------------------ 3. graph-only signal

def fig_graph_only(ps: pd.DataFrame, graph: str, out: Path):
    models = [m for m in ("labelprop", "rf_struct", "gcn_none", "gat_none", "deg_lr") if m in set(sel(ps, graph=graph).model)]
    regimes = [r for r in ("random", "degree", "pfam", "pathway", "hop2", "hop1") if r in set(ps.regime)]
    fig, axes = plt.subplots(1, len(models), figsize=(2.5 * len(models), 3.0), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        d = sel(ps, graph=graph, model=m, split="random", loss="nnpu", feats=FEATS)
        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        for c, off in (("real", -0.12), ("rewired", 0.12)):
            col, mk, _ = COND[c]
            g = d[d.cond == c].groupby("regime").auroc.agg(["mean", "std"]).reindex(regimes)
            ax.errorbar(np.arange(len(regimes)) + off, g["mean"], yerr=g["std"], fmt=mk, color=col, ms=5,
                        elinewidth=1.1, zorder=3)
        ax.set_xticks(range(len(regimes)), regimes, rotation=45, ha="right")
        ax.set_title(NICE[m])
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("test AUROC")
    axes[0][0].text(len(regimes) - 0.5, 0.505, "chance", fontsize=7, color=MUTED, ha="right", va="bottom")
    handles = [plt.Line2D([], [], ls="", marker=COND[c][1], color=COND[c][0], ms=6, label=COND[c][2])
               for c in ("real", "rewired")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    header(fig, f"{GRAPH_NICE.get(graph, graph)}: how much label signal is in the graph alone?",
           "Models without node features. x: evaluation negatives (random unlabelled, degree-matched, same Pfam, "
                  "same pathway, 2-hop, 1-hop). Random split, nnPU.")
    fig.savefig(out / "3_graph_only.png")
    plt.close(fig)


# ------------------------------------------------------------------ 4. GP attribution

def fig_gp(df: pd.DataFrame, graph: str, out: Path):
    cols = [f"gp_share_{b}" for b in BLOCKS]
    g = sel(df, graph=graph, model="gp", split="random", loss="nnpu", regime="random")
    if g.empty or cols[0] not in g:
        return False
    g = g.drop_duplicates("run_id")
    rows = [(f, c) for f in sorted(g.feats.unique()) for c in ("real", "rewired") if len(g[(g.feats == f) & (g.cond == c)])]
    if not rows:
        return False
    shares = pd.DataFrame([g[(g.feats == f) & (g.cond == c)][cols].mean() for f, c in rows]).fillna(0)
    shares = shares.clip(lower=0)  # tiny negative covariance shares (|x| < 0.01) can't be stacked
    shares = shares.div(shares.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.4))
    for i, (_, r) in enumerate(shares.iterrows()):
        left = 0
        for b, (col, lab) in BLOCKS.items():
            v = r[f"gp_share_{b}"]
            if v <= 0:
                continue
            ax.barh(i, v - 0.004, left=left + 0.002, color=col, height=0.62, edgecolor=SURFACE, lw=0)
            if v >= 0.07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if b in ("go", "pfam", "topology", "neighbour_feat", "pathway") else INK)
            left += v
    ax.set_yticks(range(len(rows)), [f"{FEATS_NICE.get(f, f)} · {c} graph" for f, c in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of the GP's predictive signal on test genes")
    ax.grid(axis="y", visible=False)
    handles = [matplotlib.patches.Patch(color=col, label=lab) for col, lab in BLOCKS.values()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
    header(fig, f"{GRAPH_NICE.get(graph, graph)}: what does the additive GP rely on?",
           "share_b = cov(f_b, f) / var(f), mean over 5 seeds (random split, nnPU). Negative shares (<1%) set to 0.")
    fig.savefig(out / "4_gp_attribution.png")
    plt.close(fig)
    return True


# ------------------------------------------------------------------ 5. degree dependence

def fig_degree(df: pd.DataFrame, exp: str, universe: str, graph: str, out: Path):
    models = [m for m in ("rf_feat", "gp", "gcn_feat", "sage_feat", "gat_feat", "labelprop") if m in set(df.model)]
    colors = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    deg = data._struct(universe, graph, "real")[1]
    probe = sel(df, model=models[0]).run_id.iloc[0]
    n_scores = len(np.load(RES / exp / "scores" / f"{probe}.npy"))
    if n_scores != len(deg):
        print(f"  skip 5_degree for {graph}: scores cover {n_scores} genes but the local {universe} universe has "
              f"{len(deg)} -- point GHELPS_DATA at the processed data the runs used", flush=True)
        return False
    runs = sel(df, split="random", loss="nnpu", feats=FEATS, regime="random").drop_duplicates("run_id")
    pos_rows, score_rows = [], []
    for seed in range(5):
        s = pd.read_parquet(paths.PROCESSED / "splits" / universe / "minikel" / "random" / f"seed{seed}.parquet")
        te = (s.fold == "test").values
        dec = pd.qcut(pd.Series(deg[te]).rank(method="first"), 10, labels=False).values
        pos_rows.append(pd.DataFrame({"decile": dec, "y": s.y.values[te]}))
        for m in models:
            r = runs[(runs.model == m) & (runs.seed == seed) & (runs.graph.isin([graph, "-"]))
                     & (runs["copy"].isin(["real", "none"]))]
            if r.empty:
                continue
            sc = np.load(RES / exp / "scores" / f"{r.run_id.iloc[0]}.npy")[te]
            pct = pd.Series(sc).rank(pct=True).values
            score_rows.append(pd.DataFrame({"decile": dec, "pct": pct, "model": m}))
    pos = pd.concat(pos_rows).groupby("decile").y.mean()
    sc = pd.concat(score_rows).groupby(["model", "decile"]).pct.mean().unstack(0)
    edges = np.percentile(deg[np.isfinite(deg)], np.linspace(0, 100, 11))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    a1.bar(pos.index, pos.values, color="#2a78d6", width=0.8, edgecolor=SURFACE, lw=0)
    for x, v in pos.items():
        a1.text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=INK2)
    a1.set_ylabel("positives among\ntest genes")
    a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.grid(axis="x", visible=False)
    for m, col in zip(models, colors):
        if m not in sc:
            continue
        a2.plot(sc.index, sc[m], color=col, marker="o", ms=4, lw=2, label=NICE[m])
        a2.text(9.2, sc[m].iloc[-1], NICE[m], color=INK2, fontsize=7.5, va="center")
    a2.axhline(0.5, color=MUTED, ls=":", lw=1)
    a2.set_ylabel("mean score percentile\n(within test genes)")
    a2.set_xlabel(f"degree decile in the real graph (1 = lowest; degree {edges[0]:.0f}–{edges[-1]:.0f})")
    a2.set_xticks(range(10), [str(i + 1) for i in range(10)])
    a2.set_xlim(-0.5, 11.2)
    a2.grid(axis="x", visible=False)
    a2.legend(loc="upper left", fontsize=7.5, ncol=2)
    header(fig, f"{GRAPH_NICE.get(graph, graph)}: who gets ranked high, by degree?",
           "Test genes pooled over 5 seeds, random split, nnPU, all features. A model that tracks degree "
                  "rises from left to right regardless of the top panel.")
    fig.savefig(out / "5_degree.png")
    plt.close(fig)
    return True


def fig_compare(effects: dict[str, pd.DataFrame], out: Path):
    """Real − empty AUROC per graph model: druggability (main) vs the membrane positive control.
    If the graph helps on membrane but not on druggability, the task -- not the models or
    graphs -- is what makes the graph redundant."""
    models = [m for m in FEATURE_GRAPH_MODELS if any(m in set(e.model) for e in effects.values())]
    graphs = [g for g in GRAPH_NICE if any(g in set(e.graph) for e in effects.values())]
    tasks = [("main", "drug target (Minikel)", "#2a78d6", "o"), ("membrane", "membrane (positive control)", "#eb6834", "s")]
    fig, axes = plt.subplots(1, len(graphs), figsize=(2.9 * len(graphs), 0.42 * len(models) + 1.6), sharey=True,
                             sharex=True, squeeze=False)
    for ax, g in zip(axes[0], graphs):
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        for k, (exp, lab, col, mk) in enumerate(tasks):
            e = effects.get(exp)
            if e is None:
                continue
            d = sel(e, graph=g, split="random", loss="nnpu", regime="random")
            w = d.pivot_table(index=["model", "seed"], columns="cond", values="auroc")
            for i, m in enumerate(models):
                if m not in w.index.get_level_values(0) or "empty" not in w:
                    continue
                delta = (w.loc[m]["real"] - w.loc[m]["empty"]).dropna()
                ax.errorbar(delta.mean(), i + (k - 0.5) * 0.3, xerr=delta.std(), fmt=mk, color=col, ms=6,
                            elinewidth=1.2, zorder=3)
        ax.set_yticks(range(len(models)), [NICE[m] for m in models])
        ax.invert_yaxis()
        ax.set_title(GRAPH_NICE.get(g, g))
        ax.set_xlabel("Δ AUROC (real − empty)")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], ls="", marker=mk, color=col, ms=6, label=lab) for _, lab, col, mk in tasks]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)
    header(fig, "Does the graph help on a graph-friendly label but not on druggability?",
           "Paired real − empty graph AUROC (same model, mean ± sd over 5 seeds). Random split, nnPU. "
           "Druggability: all features; membrane: Pfam + pathway features.")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "6_drug_vs_membrane.png")
    plt.close(fig)


def main():
    global FEATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["main", "membrane", "nonsense_labels", "nonsense_feats"])
    a = ap.parse_args()
    style()
    effects = {}
    for exp in a.exps:
        if not (RES / exp / "results.csv").exists():
            print(f"skip {exp}: no results.csv", flush=True)
            continue
        df = load(exp)
        feats = set(df.feats)
        FEATS = "all" if "all" in feats else sorted(feats)[0]
        effects[exp] = per_seed(df[df.feats == FEATS])
        for universe in sorted(df.universe.unique()):
            du = df[df.universe == universe]
            ps = per_seed(du)
            for graph in [g for g in du.graph.unique() if g != "-"]:
                out = OUT / exp / graph
                out.mkdir(parents=True, exist_ok=True)
                fig_conditions(ps, graph, out)
                fig_graph_effect(ps, graph, out)
                fig_graph_only(ps, graph, out)
                gp_ok = fig_gp(du, graph, out)
                deg_ok = fig_degree(du, exp, universe, graph, out)
                print(f"{exp}/{graph}: {3 + bool(gp_ok) + bool(deg_ok)} figures -> {out}", flush=True)
    if {"main", "membrane"} <= set(effects):
        fig_compare(effects, OUT / "compare")
        print(f"compare: 6_drug_vs_membrane.png -> {OUT / 'compare'}", flush=True)


if __name__ == "__main__":
    main()
