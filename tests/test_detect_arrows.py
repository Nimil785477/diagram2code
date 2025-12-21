from pathlib import Path

from diagram2code.vision.preprocess import preprocess_image
from diagram2code.vision.detect_shapes import detect_rectangles
from diagram2code.vision.detect_arrows import detect_arrow_edges


def test_detect_arrow_between_two_nodes_direction(tmp_path: Path):
    img = Path("tests/fixtures/simple.png")
    res = preprocess_image(img, tmp_path)

    nodes = detect_rectangles(res.image_bin)
    edges = detect_arrow_edges(res.image_bin, nodes)

    assert edges == [(0, 1)]
