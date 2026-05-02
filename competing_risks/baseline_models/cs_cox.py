"""Cause-specific Cox-compatible baseline wrapper."""

from __future__ import annotations

from ._nonparametric import NonparametricCompetingRiskModel


class CauseSpecificCoxModel(NonparametricCompetingRiskModel):
    def __init__(self, num_causes=None, penalizer: float = 0.01, **kwargs):
        super().__init__(num_causes=num_causes, name="cs-cox", penalizer=penalizer, **kwargs)


CauseSpecificCox = CauseSpecificCoxModel
