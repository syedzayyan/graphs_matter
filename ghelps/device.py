"""Compute device: CUDA when available, otherwise CPU. Override with GHELPS_DEVICE."""
from __future__ import annotations

import os
from contextlib import contextmanager
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


@contextmanager
def forced(name: str):
    """Temporarily run on another device (e.g. CPU after a CUDA out-of-memory error)."""
    old = os.environ.get("GHELPS_DEVICE")
    os.environ["GHELPS_DEVICE"] = name
    get.cache_clear()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GHELPS_DEVICE")
        else:
            os.environ["GHELPS_DEVICE"] = old
        get.cache_clear()


def is_oom(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
