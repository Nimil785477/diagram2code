from __future__ import annotations

from pathlib import Path

from diagram2code.benchmark.predictor import PredGraph, Predictor
from diagram2code.datasets import DatasetError, load_dataset


def make_oracle_predictor(dataset_path: Path) -> Predictor:
    """
    Oracle predictor for Phase 3 datasets:
    - loads dataset.json + graphs/*.json
    - returns PredGraph exactly matching GT (canonical edge keys: from/to)
    """
    ds = load_dataset(dataset_path)

    by_image: dict[Path, PredGraph] = {}

    for split in ds.splits():
        for sample in ds.samples(split):
            if sample.image_path in by_image:
                continue

            g = sample.load_graph_json()
            nodes_out: list[dict] = []
            for n in g.get("nodes", []):
                # node ids must be strings in Phase 3
                nid = str(n.get("id"))
                bbox = n.get("bbox", [0, 0, 0, 0])
                nodes_out.append({"id": nid, "bbox": list(bbox)})

            edges_out: list[dict] = []
            for e in g.get("edges", []):
                # accept both source/target and from/to if present
                src = e.get("source", e.get("from"))
                dst = e.get("target", e.get("to"))
                edges_out.append({"from": str(src), "to": str(dst)})

            by_image[sample.image_path] = PredGraph(nodes=nodes_out, edges=edges_out)

    def predictor(image_path: Path) -> PredGraph:
        try:
            return by_image[image_path]
        except KeyError as exc:
            raise DatasetError(
                f"Oracle predictor: image not found in dataset: {image_path}"
            ) from exc

    return predictor


def make_predictor(name: str, dataset_path: Path, out_dir: Path | None) -> Predictor:
    """
    Factory for benchmark predictors.
    (No new predictors added in Phase 3.)
    """
    if name == "oracle":
        return make_oracle_predictor(dataset_path)

    if name == "vision":
        # Existing vision backend (unchanged)
        from diagram2code.benchmark.predictor_vision import make_vision_predictor

        return make_vision_predictor(out_dir=out_dir)

    raise ValueError(f"Unknown predictor backend: {name}")
