"""Compute device: CUDA when available, otherwise CPU. Override with GHELPS_DEVICE."""
from __future__ import annotations

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def get() -> torch.device:
    want = os.environ.get("GHELPS_DEVICE", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GHELPS_DEVICE=cuda but CUDA is not available")
    return torch.device(want)
