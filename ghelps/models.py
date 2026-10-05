"""Model constructors and propagation baselines, all PyTorch Geometric built-ins."""
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.nn import CorrectAndSmooth, LabelPropagation
from torch_geometric.nn.models import GAT, GCN, MLP
from torch_geometric.transforms import SIGN


def mlp(d_in: int, hidden: int, layers: int, dropout: float) -> MLP:
    if layers == 1:  # plain logistic regression
        return MLP([d_in, 1], norm=None)
    return MLP(in_channels=d_in, hidden_channels=hidden, out_channels=1, num_layers=layers,
               dropout=dropout, norm=None)


def gcn(d_in: int, hidden: int, layers: int, dropout: float) -> GCN:
    return GCN(d_in, hidden, layers, out_channels=1, dropout=dropout)


def gat(d_in: int, hidden: int, layers: int, dropout: float, heads: int = 4,
        edge_dim: int | None = None) -> GAT:
    # GATv2 so that edge features enter the attention scores when edge_dim is set
    return GAT(d_in, hidden, layers, out_channels=1, dropout=dropout, heads=heads, v2=True,
               edge_dim=edge_dim)


def sign_features(x: torch.Tensor, edge_index: torch.Tensor, hops: int = 3) -> torch.Tensor:
    """[X, ÂX, ..., Â^K X] via the SIGN transform (GCN-normalised adjacency)."""
    d = SIGN(hops)(Data(x=x, edge_index=edge_index, num_nodes=x.size(0)))
    return torch.cat([d.x] + [d[f"x{k}"] for k in range(1, hops + 1)], dim=1)


def label_propagation(y: torch.Tensor, edge_index: torch.Tensor, train_mask: torch.Tensor,
                      layers: int = 50, alpha: float = 0.85) -> torch.Tensor:
    out = LabelPropagation(layers, alpha)(y, edge_index, mask=train_mask)
    return out[:, 1]


def correct_and_smooth(base_logit: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor,
                       train_mask: torch.Tensor, layers: int = 50, alpha: float = 0.8) -> torch.Tensor:
    p = torch.sigmoid(base_logit)
    y_soft = torch.stack([1 - p, p], dim=1)
    cs = CorrectAndSmooth(layers, alpha, layers, alpha, autoscale=True)
    y_soft = cs.correct(y_soft, y[train_mask], train_mask, edge_index)
    y_soft = cs.smooth(y_soft, y[train_mask], train_mask, edge_index)
    return y_soft[:, 1]
