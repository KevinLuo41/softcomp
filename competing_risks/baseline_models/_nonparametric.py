"""Small monotone competing-risks baseline shared by baseline wrappers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


class NonparametricCompetingRiskModel:
    """Empirical CIF baseline with the same public API as neural baselines."""

    def __init__(self, num_causes: Optional[int] = None, name: str = "baseline", **kwargs: Any):
        self.num_causes = num_causes
        self.name = name
        self.kwargs = dict(kwargs)
        self.y_: Optional[np.ndarray] = None
        self.delta_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Any,
        Y: Any,
        Delta: Any,
        X_val: Any = None,
        Y_val: Any = None,
        Delta_val: Any = None,
        **kwargs: Any,
    ) -> "NonparametricCompetingRiskModel":
        del X, X_val, Y_val, Delta_val
        self.y_ = np.asarray(Y, dtype=float).reshape(-1)
        self.delta_ = np.asarray(Delta, dtype=int).reshape(-1)
        if self.y_.shape[0] != self.delta_.shape[0]:
            raise ValueError("Y and Delta must have matching length")
        if self.num_causes is None:
            events = self.delta_[self.delta_ > 0]
            self.num_causes = int(events.max()) if events.size else 1
        self.kwargs.update(kwargs)
        return self

    def _check_fit(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.y_ is None or self.delta_ is None or self.num_causes is None:
            raise RuntimeError(f"{self.__class__.__name__}.fit must be called before prediction")
        return self.y_, self.delta_, int(self.num_causes)

    def predict_cif(self, X_test: Any, times: Any):
        y, delta, k = self._check_fit()
        n = len(X_test)
        t = np.asarray(times, dtype=float).reshape(-1)
        cif = np.zeros((n, k, len(t)), dtype=np.float32)
        for cause in range(1, k + 1):
            event_times = y[delta == cause]
            if event_times.size == 0:
                continue
            curve = np.array([np.mean(event_times <= tj) for tj in t], dtype=np.float32)
            curve *= float(np.mean(delta == cause))
            cif[:, cause - 1, :] = curve.reshape(1, -1)
        return torch.as_tensor(cif, dtype=torch.float32)

    def state_dict(self) -> Dict[str, Any]:
        y, delta, _ = self._check_fit()
        return {
            "num_causes": self.num_causes,
            "name": self.name,
            "kwargs": self.kwargs,
            "Y": y,
            "Delta": delta,
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint: Dict[str, Any]):
        model = cls(
            num_causes=checkpoint.get("num_causes"),
            **checkpoint.get("kwargs", {}),
        )
        model.name = checkpoint.get("name", model.name)
        model.y_ = np.asarray(checkpoint["Y"], dtype=float)
        model.delta_ = np.asarray(checkpoint["Delta"], dtype=int)
        return model
