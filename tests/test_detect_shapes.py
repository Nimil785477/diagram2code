from pathlib import Path

from diagram2code.vision.detect_shapes import detect_rectangles
from diagram2code.vision.preprocess import preprocess_image


def test_detect_rectangles_on_fixture(tmp_path: Path):
    fixture = Path("tests/fixtures/simple.png")
    assert fixture.exists()

    res = preprocess_image(fixture, tmp_path)
    nodes = detect_rectangles(res.image_bin)

    # Your fixture has 2 squares; we expect 2 nodes
    assert len(nodes) == 2
