from pathlib import Path

from diagram2code.vision.preprocess import preprocess_image
from diagram2code.vision.detect_shapes import detect_rectangles


def test_end_to_end_branching(tmp_path: Path):
    img_path = Path("tests/fixtures/branching.png")
    out_dir = tmp_path

    result = preprocess_image(img_path, out_dir)

    nodes = detect_rectangles(result.image_bin)

    assert len(nodes) == 4
