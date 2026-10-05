"""Additive-kernel GP (gpytorch) for attributing predictions to information sources.

    k(x, x') = sum_b  ScaleKernel(RBFKernel(active_dims = block b))

Blocks: own GO / Pfam / pathway / tractability features, log-degree, graph topology
(Laplacian eigenvectors) and neighbour features (Â²X). Because the kernel is additive, the
predictive mean splits exactly into per-block terms  k_b(x*, X) @ mean_cache.
"""
from __future__ import annotations

import gpytorch
import numpy as np
import torch

TRAIN_ITERS = 150


class AdditiveGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, blocks: dict[str, list[int]]):
        super().__init__(x, y, likelihood)
        self.names = list(blocks)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.AdditiveKernel(*[
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=idx))
            for idx in blocks.values()])

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def fit_predict(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, blocks: dict[str, list[int]],
                noise_floor_frac: float = 0.25, seed: int = 0, device: torch.device | str = "cpu"):
    """Gaussian likelihood on 0/1 P-vs-U labels, noise floored at a fraction of the label
    variance (otherwise the GP interpolates the labels). Returns the predictive mean for
    every gene, its per-block decomposition, the latent sd and the fitted hyperparameters."""
    torch.manual_seed(seed)
    X = torch.as_tensor(x, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y[train_idx], dtype=torch.float64, device=device)
    floor = noise_floor_frac * yt.var().item()
    lik = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(floor))
    model = AdditiveGP(X[train_idx], yt, lik, blocks).double().to(device)
    lik.double().to(device)

    model.train(), lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(TRAIN_ITERS):
        opt.zero_grad()
        loss = -mll(model(X[train_idx]), yt)
        loss.backward()
        opt.step()

    model.eval(), lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model(X)
        mean, sd = post.mean.cpu().numpy(), post.variance.clamp_min(0).sqrt().cpu().numpy()
        alpha = model.prediction_strategy.mean_cache
        parts = {n: (k(X, X[train_idx]) @ alpha).cpu().numpy()
                 for n, k in zip(model.names, model.covar_module.kernels)}
    const = model.mean_module.constant.item()
    assert np.allclose(sum(parts.values()) + const, mean, atol=1e-5), "additive decomposition mismatch"
    hyper = {
        "outputscale": {n: k.outputscale.item() for n, k in zip(model.names, model.covar_module.kernels)},
        "lengthscale": {n: k.base_kernel.lengthscale.item() for n, k in zip(model.names, model.covar_module.kernels)},
        "noise": lik.noise.item(), "noise_floor": floor, "neg_mll": loss.item(),
    }
    return mean, parts, sd, hyper
