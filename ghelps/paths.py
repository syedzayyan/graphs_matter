"""Where data lives. Set GHELPS_DATA to keep it outside the repo (e.g. HPC scratch);
defaults to <repo>/data. Raw downloads go in $GHELPS_DATA/raw, the built benchmark in
$GHELPS_DATA/processed."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GHELPS_DATA", ROOT / "data")).expanduser().resolve()
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
