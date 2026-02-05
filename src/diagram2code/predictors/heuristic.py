from __future__ import annotations

from diagram2code.datasets.types import DatasetSample
from diagram2code.predictors.types import GraphPrediction


class HeuristicPredictor:
    """
    Simple deterministic baseline predictor.

    Strategy:
    - Use dataset nodes directly
    - Sort nodes by vertical position
    - Connect each node to the next
    """

    name = "heuristic"

    def predict(self, sample: DatasetSample) -> GraphPrediction:
        gt = sample.load_graph_json()
        nodes = list(gt.get("nodes", []))

        # Ensure ids are strings and bbox exists
        norm_nodes: list[dict] = []
        for n in nodes:
            norm_nodes.append(
                {
                    "id": str(n.get("id")),
                    "bbox": list(n.get("bbox", [0, 0, 0, 0])),
                }
            )

        # Sort by vertical center (top to bottom)
        def y_center(node: dict) -> float:
            x, y, w, h = node["bbox"]
            return y + h / 2

        norm_nodes.sort(key=y_center)

        edges: list[dict] = []
        for a, b in zip(norm_nodes, norm_nodes[1:], strict=False):
            edges.append(
                {
                    "source": a["id"],
                    "target": b["id"],
                    "direction": "down",
                }
            )

        return {
            "nodes": norm_nodes,
            "edges": edges,
        }
