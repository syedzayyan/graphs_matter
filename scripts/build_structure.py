"""Stage 1b: structural stats for every (universe, graph, real/rewired copy).

Output: data/processed/structure/<universe>/<graph>.<copy>.parquet, rows aligned to genes.txt
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghelps import paths, structure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G = paths.PROCESSED / "graphs"
OUT = paths.PROCESSED / "structure"


def job(path: Path) -> str:
    uni = path.parent.name
    dest = OUT / uni / path.name.replace(".npy", ".parquet")
    if dest.exists():
        return f"skip {dest.name}"
    n = len((path.parent / "genes.txt").read_text().split())
    df = structure.stats(np.load(path), n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest)
    return f"done {uni}/{dest.name}"


if __name__ == "__main__":
    paths = sorted(G.glob("*/*.npy"))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for msg in ex.map(job, paths):
            print(msg, flush=True)
