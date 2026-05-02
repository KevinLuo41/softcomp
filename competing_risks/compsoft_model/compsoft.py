"""CompSoft scalar-covariate model, losses, fitting, and prediction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device=device, dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.long, device=device)


def _resolve_device(device: Optional[Any]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def competing_risks_loss(
    logits: torch.Tensor,
    delta: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log likelihood with class 0 reserved for survival/censoring."""
    batch = logits.shape[0]
    zeros = torch.zeros(batch, 1, device=logits.device, dtype=logits.dtype)
    full_logits = torch.cat([zeros, logits], dim=1)
    log_probs = F.log_softmax(full_logits, dim=1)
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    return F.nll_loss(log_probs, delta.long(), weight=class_weights)


class ResidualBlock(nn.Module):
    """Residual MLP block used by CompSoft."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BaseCompSoftNet(nn.Module):
    """Common training and prediction utilities for CompSoft variants."""

    num_causes: int

    def predict_cif(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cause-specific CIFs and survival probability at time ``t``."""
        logits = self.forward(x, t)
        batch = logits.shape[0]
        zeros = torch.zeros(batch, 1, device=logits.device, dtype=logits.dtype)
        full_logits = torch.cat([zeros, logits], dim=1)
        probs = F.softmax(full_logits, dim=1)
        return probs[:, 1:], probs[:, 0]

    @torch.no_grad()
    def predict_cif_grid(
        self,
        x: Any,
        t_grid: Any,
        batch_size: int = 4096,
        device: Optional[Any] = None,
        as_numpy: bool = True,
    ) -> Tuple[Any, Any]:
        """Predict CIFs on a time grid.

        Returns ``cif`` with shape ``(n, K, T)`` and ``survival`` with shape
        ``(n, T)``.
        """
        was_training = self.training
        self.eval()
        dev = _resolve_device(device)
        self.to(dev)
        x_tensor = _as_float_tensor(x, dev)
        t_tensor = _as_float_tensor(t_grid, dev).flatten()
        n = x_tensor.shape[0]
        n_times = t_tensor.numel()
        cif_chunks = []
        surv_chunks = []
        for start in range(0, n, batch_size):
            xb = x_tensor[start : start + batch_size]
            batch_cifs = []
            batch_surv = []
            for tj in t_tensor:
                tt = tj.expand(xb.shape[0])
                fj, sj = self.predict_cif(xb, tt)
                batch_cifs.append(fj.unsqueeze(-1))
                batch_surv.append(sj.unsqueeze(-1))
            cif_chunks.append(torch.cat(batch_cifs, dim=-1).cpu())
            surv_chunks.append(torch.cat(batch_surv, dim=-1).cpu())
        cif = torch.cat(cif_chunks, dim=0)
        survival = torch.cat(surv_chunks, dim=0)
        if was_training:
            self.train()
        if as_numpy:
            return cif.numpy(), survival.numpy()
        return cif, survival

    def _brier_loss(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        d_batch: torch.Tensor,
        brier_n_times: int,
        brier_t_max: float,
        device: torch.device,
    ) -> torch.Tensor:
        bn = x_batch.shape[0]
        t_brier = torch.rand(bn, brier_n_times, device=device) * brier_t_max
        brier_loss = torch.zeros((), device=device)
        for j in range(brier_n_times):
            tj = t_brier[:, j]
            f_pred, _ = self.predict_cif(x_batch, tj)
            target = torch.zeros_like(f_pred)
            for k in range(self.num_causes):
                target[:, k] = ((y_batch <= tj) & (d_batch == k + 1)).float()
            brier_loss = brier_loss + ((f_pred - target) ** 2).mean()
        return brier_loss / brier_n_times

    def fit(
        self,
        X_train: Any,
        Y_train: Any,
        Delta_train: Any,
        X_val: Optional[Any] = None,
        Y_val: Optional[Any] = None,
        Delta_val: Optional[Any] = None,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_aug: int = 2,
        aug_weight: float = 0.5,
        class_weights: Optional[Any] = None,
        patience: int = 0,
        brier_lambda: float = 0.0,
        brier_n_times: int = 5,
        brier_t_max: Optional[float] = None,
        verbose: bool = True,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Fit CompSoft using NLL, time augmentation, and optional Brier loss."""
        dev = _resolve_device(device)
        self.to(dev)
        x_train = _as_float_tensor(X_train, dev)
        y_train = _as_float_tensor(Y_train, dev).flatten()
        d_train = _as_long_tensor(Delta_train, dev).flatten()
        if x_train.shape[0] != y_train.shape[0] or y_train.shape[0] != d_train.shape[0]:
            raise ValueError("X_train, Y_train, and Delta_train must have matching length")

        x_val = y_val = d_val = None
        if X_val is not None and Y_val is not None and Delta_val is not None:
            x_val = _as_float_tensor(X_val, dev)
            y_val = _as_float_tensor(Y_val, dev).flatten()
            d_val = _as_long_tensor(Delta_val, dev).flatten()

        weights = None
        if class_weights is not None:
            weights = _as_float_tensor(class_weights, dev)
            if weights.numel() != self.num_causes + 1:
                raise ValueError("class_weights must have length K + 1")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        brier_t_max_value = float(brier_t_max if brier_t_max is not None else y_train.max().item())
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": None,
            "best_val_loss": None,
        }
        best_state = None
        best_val = float("inf")
        stale_epochs = 0
        n = x_train.shape[0]

        for epoch in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(n, device=dev)
            total_loss = 0.0
            total_seen = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = x_train[idx]
                yb = y_train[idx]
                db = d_train[idx]
                batch = xb.shape[0]
                optimizer.zero_grad()
                logits = self(xb, yb)
                loss = competing_risks_loss(logits, db, weights)
                if n_aug > 0:
                    u = torch.rand(batch, n_aug, device=dev)
                    t_aug = u * yb.unsqueeze(1)
                    d_zeros = torch.zeros(batch, device=dev, dtype=torch.long)
                    for j in range(n_aug):
                        logits_aug = self(xb, t_aug[:, j])
                        loss = loss + competing_risks_loss(logits_aug, d_zeros, weights) * aug_weight
                    loss = loss / (1.0 + n_aug * aug_weight)
                if brier_lambda > 0:
                    loss = loss + brier_lambda * self._brier_loss(
                        xb, yb, db, brier_n_times, brier_t_max_value, dev
                    )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * batch
                total_seen += batch
            scheduler.step()

            train_loss = total_loss / max(1, total_seen)
            history["train_loss"].append(train_loss)
            history["lr"].append(float(scheduler.get_last_lr()[0]))
            val_loss = None
            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    val_logits = self(x_val, y_val)
                    val_loss = float(competing_risks_loss(val_logits, d_val, weights).cpu())
                history["val_loss"].append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = deepcopy(self.state_dict())
                    history["best_epoch"] = epoch
                    history["best_val_loss"] = val_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if patience > 0 and stale_epochs >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}; best epoch {history['best_epoch']}")
                        break
            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
                msg = f"epoch={epoch:04d} train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.6f}"
                print(msg)

        if best_state is not None:
            self.load_state_dict(best_state)
        self.history_ = history
        return history


class CompSoftNet(BaseCompSoftNet):
    """Scalar-input CompSoft network."""

    def __init__(
        self,
        input_dim: int,
        num_causes: int = 2,
        hidden_dim: int = 16,
        num_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_causes = int(num_causes)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        layers = [nn.Linear(self.input_dim + 1, self.hidden_dim), nn.ReLU()]
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(1)
        xt = torch.cat([x, t.to(dtype=x.dtype, device=x.device)], dim=1)
        h = self.input_proj(xt)
        h = self.backbone(h)
        return self.output_layer(h)
