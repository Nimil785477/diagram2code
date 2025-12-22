from pathlib import Path

from diagram2code.vision.preprocess import preprocess_image
from diagram2code.vision.detect_shapes import detect_rectangles
from diagram2code.vision.detect_arrows import detect_arrow_edges


def test_end_to_end_branching(tmp_path: Path):
    img_path = Path("tests/fixtures/branching.png")
    out_dir = tmp_path

    result = preprocess_image(img_path, out_dir)

    nodes = detect_rectangles(
        result.image_bin,
        debug_path=out_dir / "debug_nodes.png",
    )
    assert len(nodes) == 4

    edges = detect_arrow_edges(
        result.image_bin,
        nodes,
        debug_path=out_dir / "debug_arrows.png",
    )
    assert len(edges) >= 3
