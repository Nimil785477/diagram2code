from __future__ import annotations

from pathlib import Path

from diagram2code.benchmark.dataset import load_dataset
from diagram2code.benchmark.predictor import PredGraph, Predictor
from diagram2code.benchmark.predictors_vision import VisionPredictor


def make_oracle_predictor(dataset_path: Path) -> Predictor:
    """
    Oracle predictor: returns exact GT but with string pred ids.
    Deterministic + perfect. Used for CLI tests and sanity checks.
    """
    ds = load_dataset(dataset_path)
    gt_by_name = {item.image_path.name: item.gt for item in ds.items}

    def _predict(image_path: Path) -> PredGraph:
        gt = gt_by_name[image_path.name]
        nodes = [{"id": f"p{n.id}", "bbox": list(n.bbox)} for n in gt.nodes]
        edges = [{"from": f"p{e.from_id}", "to": f"p{e.to_id}"} for e in gt.edges]
        return PredGraph(nodes=nodes, edges=edges)

    return _predict


def make_predictor(kind: str, *, dataset_path: Path, out_dir: str | Path | None) -> Predictor:
    if kind == "oracle":
        return make_oracle_predictor(dataset_path)
    if kind == "vision":
        return VisionPredictor(out_dir=out_dir)
    raise ValueError(f"Unknown predictor kind: {kind!r}")
