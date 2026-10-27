"""Kept for releases made before the targetsplit package: same as `targetsplit score`.

  python evaluate.py --release . --universe string --scores my_scores.tsv [--out report.tsv]
"""
import sys

from targetsplit.cli import score

if __name__ == "__main__":
    score(sys.argv[1:])
