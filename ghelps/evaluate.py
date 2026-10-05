"""Metrics on the frozen evaluation regimes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _auc(pos_scores, neg_scores) -> dict:
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    s = np.r_[pos_scores, neg_scores]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {"auroc": np.nan, "auprc": np.nan, "n_pos": len(pos_scores), "n_neg": len(neg_scores)}
    return {"auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s),
            "n_pos": len(pos_scores), "n_neg": len(neg_scores)}


def by_regime(scores: np.ndarray, negatives: pd.DataFrame) -> dict[str, dict]:
    """Matched regimes are scored as one pooled ranking over the positives that found a
    match and their matched negatives (rank metrics, so pooling is legitimate)."""
    out = {}
    for regime, g in negatives.groupby("regime"):
        pos = g.pos[g.pos >= 0].unique()
        neg = g.neg[g.neg >= 0].unique()
        out[regime] = _auc(scores[pos], scores[neg])
    return out


def val_auprc(scores: np.ndarray, y: np.ndarray, fold: np.ndarray) -> float:
    m = fold == "val"
    if y[m].sum() == 0:
        return np.nan
    return average_precision_score(y[m], scores[m])


def degree_dependence(scores: np.ndarray, degree: np.ndarray | None, mask: np.ndarray) -> dict:
    if degree is None:
        return {}
    rho = spearmanr(scores[mask], degree[mask]).statistic
    return {"spearman_degree_test": float(rho)}
