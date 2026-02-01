from pathlib import Path

from diagram2code.vision.detect_arrows import detect_arrow_edges
from diagram2code.vision.detect_shapes import detect_rectangles
from diagram2code.vision.preprocess import preprocess_image


def test_end_to_end_branching_arrows(tmp_path: Path):
    img_path = Path("tests/fixtures/branching_arrows.png")
    out_dir = tmp_path

    result = preprocess_image(img_path, out_dir)

    nodes = detect_rectangles(result.image_bin)
    edges = detect_arrow_edges(result.image_bin, nodes)

    assert len(nodes) == 4

    # relaxed first (we tighten later)
    assert len(edges) >= 3

    # edges should reference valid node ids
    node_ids = {n.id for n in nodes}
    for a, b in edges:
        assert a in node_ids
        assert b in node_ids
