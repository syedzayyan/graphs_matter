"""Aggregate an experiment's runs (called by `main.py summarise <exp> [metric]`).

Reads results/<exp>/results.csv (rebuilt from the run JSONs) and writes:
  summary_<metric>.csv   per (regime, split, loss, model, condition): mean, sd, n seeds
  graph_effect_<metric>.csv   paired-by-seed deltas real−empty, real−rewired, real−shufattr
                              (mean, sd, #seeds where real wins) -- "does the graph matter?"
  gp_attribution.csv     GP per-block share of the predictive signal, by condition
and prints the graph-effect table for the random and degree-matched regimes.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghelps import runner  # noqa: E402

exp = sys.argv[1]
metric = sys.argv[2] if len(sys.argv) > 2 else "auroc"
out = ROOT / "results" / exp
pd.set_option("display.width", 250)

df = runner.collect(out)
df["copy"] = df["copy"].fillna("none")
df["feats"] = df["feats"].fillna("all") if "feats" in df else "all"
df["graph"] = df["graph"].fillna("-")  # graph-free models
df["cond"] = df["copy"].where(~df["copy"].str.startswith("rw"), "rewired")
keys = ["regime", "universe", "graph", "feats", "split", "loss", "model"]

# rewired copies are averaged within a seed first, so every seed counts once per condition
per_seed = df.groupby(keys + ["seed", "cond"])[metric].mean().reset_index()
summary = per_seed.groupby(keys + ["cond"])[metric].agg(["mean", "std", "count"]).reset_index()
summary.to_csv(out / f"summary_{metric}.csv", index=False)

wide = per_seed.pivot_table(index=keys + ["seed"], columns="cond", values=metric)
effects = []
for other in ["empty", "rewired", "shufattr"]:
    if other not in wide or "real" not in wide:
        continue
    d = (wide["real"] - wide[other]).dropna().rename("delta").reset_index()
    g = d.groupby(keys).delta.agg(mean="mean", sd="std", n="count",
                                 real_wins=lambda s: int((s > 0).sum())).reset_index()
    effects.append(g.assign(contrast=f"real−{other}"))
effect = pd.concat(effects, ignore_index=True)
effect.to_csv(out / f"graph_effect_{metric}.csv", index=False)

gp_cols = [c for c in df.columns if c.startswith("gp_share_")] + ["gp_sd_vs_degree", "gp_sd_vs_pubmed"]
if gp_cols[0] in df:
    gp = (df[(df.model == "gp") & (df.regime == "random")]
          .groupby(["universe", "graph", "feats", "split", "loss", "cond"])[[c for c in gp_cols if c in df]].mean().reset_index())
    gp.to_csv(out / "gp_attribution.csv", index=False)

for regime in ["random", "degree"]:
    t = effect[effect.regime == regime].copy()
    t["cell"] = t.apply(lambda r: f"{r['mean']:+.3f}±{r['sd']:.3f} ({r.real_wins}/{r.n})", axis=1)
    print(f"=== does the graph matter? {metric}, regime={regime} (paired by seed) ===")
    print(t.pivot_table(index=["universe", "graph", "feats", "split", "loss", "model"], columns="contrast", values="cell", aggfunc="first")
          .fillna("").to_string())
    print()
print(f"wrote summary_{metric}.csv, graph_effect_{metric}.csv" + (", gp_attribution.csv" if gp_cols[0] in df else "")
      + f" to {out}")
