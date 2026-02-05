from __future__ import annotations

from typing import Protocol

from diagram2code.datasets.types import DatasetSample
from diagram2code.predictors.types import GraphPrediction


class Predictor(Protocol):
    name: str

    def predict(self, sample: DatasetSample) -> GraphPrediction: ...
