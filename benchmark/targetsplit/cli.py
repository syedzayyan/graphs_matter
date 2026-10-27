"""targetsplit <command>: score | build | baseline (README.md)."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

SCORE_HELP = """Score a gene ranking on a release.

  targetsplit score --release R --universe U --scores my_scores.tsv [--out report.tsv]

my_scores.tsv: columns gene (or ensg) and score (higher = more target-like), and optionally seed
when the model was retrained per seed on that seed's training genes. Without seed, the same
scores are evaluated on every seed's test set.

Reported for your ranking and the reference rankers (PubMed before the cut-off, known targets
in the gene's domain families, closeness to known targets):
  novelty  AUROC and recall in the top 10% per stratum of future targets, 95% bootstrap CI over
           the stratum's targets, and the paired difference from the PubMed ranker
  top100   shares of the top-100 test genes that are future targets, well studied, in a drugged
           family, next to a known target, ...
  tried    among test genes: selection (tried vs never) and outcome (succeeded vs failed) AUROC
"""


def score(argv) -> None:
    from targetsplit.core import PRIMARY_STRATA
    from targetsplit.release import Release
    ap = argparse.ArgumentParser(prog="targetsplit score", description=SCORE_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=".")
    ap.add_argument("--universe", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--strata", nargs="*", help="future_targets.tsv columns (default: all)")
    ap.add_argument("--no-references", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rel = Release(a.release, a.universe)
    res = rel.score(pd.read_csv(a.scores, sep="\t"), a.name, a.strata, not a.no_references)
    shown = res[(res.section != "novelty") | res.stratum.isin(a.strata or PRIMARY_STRATA)]
    pd.set_option("display.width", 200)
    print(shown.drop(columns="section").round(3).fillna("").to_string(index=False))
    if a.out:
        res.to_csv(a.out, sep="\t", index=False)


def main() -> None:
    cmds = {"score": score,
            "build": lambda argv: __import__("targetsplit.build", fromlist=["main"]).main(argv),
            "baseline": lambda argv: __import__("targetsplit.baselines", fromlist=["main"]).main(argv)}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__.strip(), "\n\n" + "\n".join(f"  targetsplit {c} --help" for c in cmds))
        raise SystemExit(2)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
