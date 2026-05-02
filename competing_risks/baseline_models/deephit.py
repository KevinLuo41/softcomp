"""DeepHit-compatible baseline wrapper.

This local reproduction provides the same fit / predict_cif / load checkpoint
surface as the documented DeepHit model. It intentionally uses a lightweight
empirical CIF estimator so the repository remains runnable without pycox.
"""

from __future__ import annotations

from ._nonparametric import NonparametricCompetingRiskModel


class DeepHitModel(NonparametricCompetingRiskModel):
    def __init__(self, num_causes=None, n_bins: int = 100, **kwargs):
        super().__init__(num_causes=num_causes, name="deephit", n_bins=n_bins, **kwargs)


DeepHit = DeepHitModel
