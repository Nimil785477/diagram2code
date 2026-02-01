import cv2
import numpy as np

from diagram2code.schema import Node
from diagram2code.vision.detect_arrows import detect_arrow_edges


def _bbox_center(b):
    x, y, w, h = b
    return (x + w // 2, y + h // 2)


def _bbox_corner_towards(b_from, b_to, margin=8):
    """
    Pick a point on the border of b_from in the direction of b_to.
    This ensures arrow endpoints are OUTSIDE node interiors.
    """
    fx, fy = _bbox_center(b_from)
    tx, ty = _bbox_center(b_to)

    x, y, w, h = b_from
    left, right = x, x + w
    top, bottom = y, y + h

    # Direction vector
    dx = tx - fx
    dy = ty - fy

    # Choose border point biased by direction
    px = right if dx >= 0 else left
    py = bottom if dy >= 0 else top

    # Move a bit outside the node
    px = px + (margin if dx >= 0 else -margin)
    py = py + (margin if dy >= 0 else -margin)
    return (int(px), int(py))


def _make_binary_diagram_diagonal(direction: str) -> tuple[np.ndarray, list[Node]]:
    """
    Binary image: white strokes (255) on black background (0), like preprocess output.
    Two filled nodes + a diagonal arrow whose arrowhead remains OUTSIDE target node.
    """
    img = np.zeros((220, 300), dtype=np.uint8)

    # Node 1 (top-left)
    n1_bbox = (30, 40, 70, 45)  # x,y,w,h
    cv2.rectangle(
        img,
        (n1_bbox[0], n1_bbox[1]),
        (n1_bbox[0] + n1_bbox[2], n1_bbox[1] + n1_bbox[3]),
        255,
        -1,
    )

    # Node 2 (bottom-right)
    n2_bbox = (190, 140, 70, 45)
    cv2.rectangle(
        img,
        (n2_bbox[0], n2_bbox[1]),
        (n2_bbox[0] + n2_bbox[2], n2_bbox[1] + n2_bbox[3]),
        255,
        -1,
    )

    if direction == "1to2":
        start = _bbox_corner_towards(n1_bbox, n2_bbox, margin=10)
        end = _bbox_corner_towards(
            n2_bbox, n1_bbox, margin=10
        )  # outside n2 towards n1 (arrowhead outside)
        cv2.arrowedLine(img, start, end, 255, 6, tipLength=0.25)
        expected = (1, 2)
    else:
        start = _bbox_corner_towards(n2_bbox, n1_bbox, margin=10)
        end = _bbox_corner_towards(n1_bbox, n2_bbox, margin=10)
        cv2.arrowedLine(img, start, end, 255, 6, tipLength=0.25)
        expected = (2, 1)

    nodes = [Node(id=1, bbox=n1_bbox), Node(id=2, bbox=n2_bbox)]
    return img, nodes, expected


def test_detect_arrows_diagonal_1_to_2():
    binary, nodes, expected = _make_binary_diagram_diagonal("1to2")
    edges = detect_arrow_edges(binary, nodes, min_area=50)
    assert expected in edges


def test_detect_arrows_diagonal_2_to_1():
    binary, nodes, expected = _make_binary_diagram_diagonal("2to1")
    edges = detect_arrow_edges(binary, nodes, min_area=50)
    assert expected in edges
