from __future__ import annotations

from diagram2code.datasets.types import DatasetSample
from diagram2code.predictors.types import GraphPrediction


class HeuristicPredictor:
    """
    Intermediate deterministic baseline predictor.

    Strategy:
    - Use dataset nodes directly
    - Sort nodes by geometric center: top-to-bottom, then left-to-right
    - Connect each node to the next as a simple chain
    """

    name = "heuristic"

    def predict(self, sample: DatasetSample) -> GraphPrediction:
        gt = sample.load_graph_json()
        nodes = list(gt.get("nodes", []))

        norm_nodes: list[dict] = []
        for n in nodes:
            bbox = list(n.get("bbox", [0, 0, 0, 0]))
            if len(bbox) != 4:
                bbox = [0, 0, 0, 0]

            norm_nodes.append(
                {
                    "id": str(n.get("id")),
                    "bbox": bbox,
                }
            )

        def center_key(node: dict) -> tuple[float, float]:
            x, y, w, h = node["bbox"]
            return (y + h / 2, x + w / 2)

        norm_nodes.sort(key=center_key)

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
