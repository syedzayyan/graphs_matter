"""Training objectives: non-negative PU (Kiryo et al. 2017) and plain PN."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nnpu(logits: torch.Tensor, is_pos: torch.Tensor, prior: float) -> torch.Tensor:
    """Non-negative PU risk (Kiryo et al. 2017) with a logistic (softplus) loss, in the
    clipped form  pi*R_p+ + max(0, R_u- - pi*R_p-).

    Two deviations from the paper's recipe, both because training here is full-batch:
    - softplus instead of the sigmoid loss, which saturated and collapsed models onto the
      all-negative solution (risk == prior);
    - clipping instead of a gradient-ascent step on -R_neg when it goes negative: that
      step pushes positive logits down and flipped low-capacity models (e.g. degree-only
      logistic regression) to the inverted ranking on some seeds.
    """
    lp, lu = logits[is_pos], logits[~is_pos]
    r_p_pos = F.softplus(-lp).mean()
    r_p_neg = F.softplus(lp).mean()
    r_u_neg = F.softplus(lu).mean()
    return prior * r_p_pos + torch.clamp(r_u_neg - prior * r_p_neg, min=0)


def pn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.float())
