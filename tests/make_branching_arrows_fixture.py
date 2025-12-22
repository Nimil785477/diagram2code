from pathlib import Path
from PIL import Image, ImageDraw
import math

SIMPLE = Path("tests/fixtures/simple.png")
OUT = Path("tests/fixtures/branching_arrows.png")

# BBoxes from your earlier graph output (simple.png)
NODE0 = (50, 124, 50 + 76, 124 + 76)   # red square
NODE1 = (199, 124, 199 + 76, 124 + 76) # blue square

def center(x: int, y: int, s: int) -> tuple[int, int]:
    return (x + s // 2, y + s // 2)

def draw_arrow(draw: ImageDraw.ImageDraw, p1, p2, width=6, head_len=18, head_w=12):
    # line
    draw.line([p1, p2], fill="black", width=width)

    # arrowhead (triangle) pointing toward p2
    x1, y1 = p1
    x2, y2 = p2
    ang = math.atan2(y2 - y1, x2 - x1)

    hx = x2
    hy = y2

    back_x = hx - head_len * math.cos(ang)
    back_y = hy - head_len * math.sin(ang)

    left_x = back_x + head_w * math.cos(ang + math.pi / 2)
    left_y = back_y + head_w * math.sin(ang + math.pi / 2)

    right_x = back_x + head_w * math.cos(ang - math.pi / 2)
    right_y = back_y + head_w * math.sin(ang - math.pi / 2)

    draw.polygon([(hx, hy), (left_x, left_y), (right_x, right_y)], fill="black")

def main():
    base = Image.open(SIMPLE).convert("RGBA")
    n0 = base.crop(NODE0)
    n1 = base.crop(NODE1)

    # Canvas
    W, H = 520, 320
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    s = 76  # patch size

    # DAG layout:
    # 0 (left) -> 1 (upper mid)
    # 0 (left) -> 2 (lower mid)
    # 1 -> 3 (right)
    # 2 -> 3 (right)
    # Use the same patches so detector sees familiar nodes.
    pos = {
        0: (40, 120),   # left
        1: (210, 40),   # upper mid
        2: (210, 200),  # lower mid
        3: (400, 120),  # right
    }
    patches = {0: n0, 1: n1, 2: n0, 3: n1}

    for nid, (x, y) in pos.items():
        canvas.paste(patches[nid], (x, y), patches[nid])

    d = ImageDraw.Draw(canvas)

    # Arrow endpoints: from node center to node center
    def node_center(nid: int):
        x, y = pos[nid]
        return center(x, y, s)

    # Draw arrows (slightly shorten so heads don't sit inside node)
    def shorten(p1, p2, trim=45):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = (x2 - x1, y2 - y1)
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        return (int(x1 + ux * trim), int(y1 + uy * trim)), (int(x2 - ux * trim), int(y2 - uy * trim))

    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for a, b in edges:
        p1, p2 = shorten(node_center(a), node_center(b), trim=48)
        draw_arrow(d, p1, p2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT)
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
