from __future__ import annotations

from pathlib import Path

from diagram2code.benchmark.predictor import Predictor
from diagram2code.benchmark.predictor_bridge import SamplePredictorAdapter
from diagram2code.datasets import load_dataset
from diagram2code.predictors.registry import get_predictor


def _make_phase4_predictor(name: str, dataset_path: Path) -> Predictor:
    """
    Wrap a Phase-4 predictor (predict(sample)->GraphPrediction) into the legacy
    benchmark Predictor callable (image_path->PredGraph) using SamplePredictorAdapter.
    """
    ds = load_dataset(dataset_path)

    # Map image_path -> DatasetSample once, reused for all predictions
    sample_by_image: dict[Path, object] = {}
    for split in ds.splits():
        for sample in ds.samples(split):
            sample_by_image.setdefault(sample.image_path, sample)

    predictor_cls = get_predictor(name)
    predictor = predictor_cls()  # type: ignore[call-arg]

    return SamplePredictorAdapter(predictor, sample_by_image)


def make_predictor(name: str, dataset_path: Path, out_dir: Path | None) -> Predictor:
    """
    Factory for benchmark predictors.

    - vision: legacy CV backend (image->PredGraph)
    - otherwise: Phase-4 predictor registry (sample->GraphPrediction) bridged to legacy
    """
    if name == "vision":
        from diagram2code.benchmark.predictors_vision import VisionPredictor
        return VisionPredictor(out_dir=out_dir)

    # Phase-4 predictor registry (includes "oracle")
    return _make_phase4_predictor(name, dataset_path)
