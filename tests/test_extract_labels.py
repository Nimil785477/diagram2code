from pathlib import Path

import cv2

from diagram2code.vision.detect_shapes import detect_rectangles
from diagram2code.vision.extract_labels import extract_node_labels
from diagram2code.vision.preprocess import preprocess_image


def test_extract_labels_returns_dict(tmp_path: Path):
    img_path = Path("tests/fixtures/simple.png")

    # preprocess to get nodes from the same pipeline
    result = preprocess_image(img_path, tmp_path)
    nodes = detect_rectangles(result.image_bin)

    # OCR runs on original BGR image (not the binary image)
    bgr = cv2.imread(str(img_path))
    assert bgr is not None

    labels = extract_node_labels(bgr, nodes)

    assert isinstance(labels, dict)
    # keys should be node ids
    for k in labels.keys():
        assert isinstance(k, int)
    # values should be strings (possibly empty depending on OCR quality)
    for v in labels.values():
        assert isinstance(v, str)
