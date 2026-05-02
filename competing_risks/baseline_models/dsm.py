"""Deep Survival Machines-compatible baseline wrapper."""

from __future__ import annotations

from ._nonparametric import NonparametricCompetingRiskModel


class DSMModel(NonparametricCompetingRiskModel):
    def __init__(self, num_causes=None, k: int = 6, distribution: str = "Weibull", **kwargs):
        super().__init__(
            num_causes=num_causes,
            name="dsm",
            k=k,
            distribution=distribution,
            **kwargs,
        )


DSM = DSMModel
