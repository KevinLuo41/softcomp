"""Neural Fine-Gray-compatible baseline wrapper."""

from __future__ import annotations

from ._nonparametric import NonparametricCompetingRiskModel


class NeuralFineGrayModel(NonparametricCompetingRiskModel):
    def __init__(self, num_causes=None, **kwargs):
        super().__init__(num_causes=num_causes, name="neural-fg", **kwargs)


NeuralFineGray = NeuralFineGrayModel
