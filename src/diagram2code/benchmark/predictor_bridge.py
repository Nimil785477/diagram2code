from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diagram2code.datasets.types import DatasetSample
from diagram2code.predictors.base import Predictor as Phase4Predictor


@dataclass(frozen=True)
class PredGraph:
    nodes: list[dict]
    edges: list[dict]


class SamplePredictorAdapter:
    """
    Bridge: adapts a Phase-4 Predictor.predict(sample)->GraphPrediction
    to the legacy benchmark interface __call__(image_path)->PredGraph.

    Legacy benchmark uses canonical edge keys: {"from": ..., "to": ...}.
    """

    def __init__(
        self,
        predictor: Phase4Predictor,
        sample_by_image: dict[Path, DatasetSample],
    ) -> None:
        self._predictor = predictor
        self._sample_by_image = sample_by_image

    def __call__(self, image_path: Path) -> PredGraph:
        sample = self._sample_by_image[image_path]
        pred = self._predictor.predict(sample)

        nodes: list[dict] = []
        for n in pred["nodes"]:
            nid = str(n.get("id"))
            bbox = list(n.get("bbox", [0, 0, 0, 0]))
            nodes.append({"id": nid, "bbox": bbox})

        edges: list[dict] = []
        for e in pred["edges"]:
            src = e.get("source", e.get("from"))
            dst = e.get("target", e.get("to"))
            if src is None or dst is None:
                raise ValueError(f"Predicted edge missing endpoints: {e}")
            edges.append({"from": str(src), "to": str(dst)})

        return PredGraph(nodes=nodes, edges=edges)
