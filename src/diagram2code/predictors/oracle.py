from __future__ import annotations

from diagram2code.datasets.types import DatasetSample
from diagram2code.predictors.types import GraphPrediction


class OraclePredictor:
    """
    Oracle predictor: returns the sample ground-truth graph.

    This is a reference predictor used for pipeline validation and as an upper bound.
    """
    name = "oracle"

    def predict(self, sample: DatasetSample) -> GraphPrediction:
        g = sample.load_graph_json()
        return {"nodes": g.get("nodes", []), "edges": g.get("edges", [])}
