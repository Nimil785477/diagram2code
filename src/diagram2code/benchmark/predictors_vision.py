from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from diagram2code.benchmark.predictor import PredGraph
from diagram2code.vision.detect_arrows import detect_arrow_edges
from diagram2code.vision.detect_shapes import detect_rectangles
from diagram2code.vision.preprocess import preprocess_image


def _extract_binary(pre) -> object:
    """
    preprocess_image() returns a result object; we need the numpy binary image for OpenCV.
    Try a small set of known/likely attribute names.
    """
    for attr in (
        "image_bin",
        "binary_img",
        "binary",
        "binary_image",
        "thresh",
        "thresh_img",
        "mask",
        "mask_img",
        "img",  # last resort if preprocess returns directly named image
    ):
        if hasattr(pre, attr):
            return getattr(pre, attr)

    # Helpful error for fast diagnosis
    cand = [a for a in dir(pre) if "bin" in a.lower() or "thresh" in a.lower() or a.endswith("img")]
    raise AttributeError(
        "PreprocessResult has no known binary image attribute. "
        f"Tried common names; candidates on object: {sorted(set(cand))}"
    )


@dataclass(frozen=True)
class VisionPredictor:
    """
    Adapter: image -> PredGraph using current CV pipeline.

    preprocess_image() requires out_dir and returns a result object.
    We extract the numpy binary image and feed it to downstream detectors.
    """

    out_dir: str | Path | None = None

    def __call__(self, image_path: Path) -> PredGraph:
        tmp_dir: Path | None = None
        out_dir = Path(self.out_dir) if self.out_dir is not None else None

        try:
            if out_dir is None:
                tmp_dir = Path(tempfile.mkdtemp(prefix="diagram2code_pre_"))
                out_dir = tmp_dir

            pre = preprocess_image(image_path, out_dir)
            binary = _extract_binary(pre)

            nodes = detect_rectangles(binary)
            edges = detect_arrow_edges(binary, nodes)

            pred_nodes = [{"id": n.id, "bbox": list(n.bbox)} for n in nodes]
            pred_edges = [{"from": u, "to": v} for (u, v) in edges]
            return PredGraph(nodes=pred_nodes, edges=pred_edges)

        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
