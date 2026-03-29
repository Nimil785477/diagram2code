from __future__ import annotations

from PIL import Image

from diagram2code.datasets.types import DatasetSample
from diagram2code.predictors.types import GraphPrediction


class NaivePredictor:
    """
    Weak deterministic baseline predictor.

    Strategy:
    - Ignore graph structure entirely
    - Return exactly one centered node
    - Return no edges
    """

    name = "naive"

    def predict(self, sample: DatasetSample) -> GraphPrediction:
        with Image.open(sample.image_path) as img:
            width, height = img.size

        node_w = 120
        node_h = 56
        x = max(0, (width - node_w) // 2)
        y = max(0, (height - node_h) // 2)

        return {
            "nodes": [
                {
                    "id": "0",
                    "bbox": [x, y, node_w, node_h],
                }
            ],
            "edges": [],
        }
