"""Functional-covariate SoftComp model for Case III."""

from __future__ import annotations

import torch
import torch.nn as nn

from .softcomp import BaseSoftCompNet, ResidualBlock


class FunctionalSoftCompNet(BaseSoftCompNet):
    """SoftComp variant for covariates observed on a function grid."""

    def __init__(
        self,
        num_covariates: int,
        n_grid: int,
        embed_dim: int = 4,
        num_causes: int = 2,
        hidden_dim: int = 32,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_covariates = int(num_covariates)
        self.n_grid = int(n_grid)
        self.embed_dim = int(embed_dim)
        self.num_causes = int(num_causes)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.embeddings = nn.ModuleList(
            [nn.Linear(self.n_grid, self.embed_dim) for _ in range(self.num_covariates)]
        )
        layers = [nn.Linear(self.num_covariates * self.embed_dim + 1, self.hidden_dim), nn.ReLU()]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        self.input_proj = nn.Sequential(*layers)
        backbone = []
        for _ in range(self.num_blocks):
            backbone.append(ResidualBlock(self.hidden_dim))
            backbone.append(nn.ReLU())
            if self.dropout > 0:
                backbone.append(nn.Dropout(self.dropout))
        self.backbone = nn.Sequential(*backbone)
        self.output_layer = nn.Linear(self.hidden_dim, self.num_causes)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.shape[1] == self.num_covariates * self.n_grid:
            return x.reshape(x.shape[0], self.num_covariates, self.n_grid)
        if x.dim() != 3:
            raise ValueError("FunctionalSoftCompNet expects X with shape (n, p, G)")
        if x.shape[1] != self.num_covariates or x.shape[2] != self.n_grid:
            raise ValueError(
                f"Expected X shape (n, {self.num_covariates}, {self.n_grid}); got {tuple(x.shape)}"
            )
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(1)
        pieces = [layer(x[:, j, :]) for j, layer in enumerate(self.embeddings)]
        z = torch.cat(pieces + [t.to(dtype=x.dtype, device=x.device)], dim=1)
        h = self.input_proj(z)
        h = self.backbone(h)
        return self.output_layer(h)
