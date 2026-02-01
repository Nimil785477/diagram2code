from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("tests/fixtures/branching.png")


def main() -> None:
    W, H = 600, 400
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # Node boxes (x, y, size)
    nodes = {
        0: (80, 60, 90),  # top-left
        1: (60, 240, 90),  # bottom-left
        2: (260, 240, 90),  # bottom-mid
        3: (430, 150, 90),  # right-mid
    }

    def rect_center(x: int, y: int, s: int) -> tuple[int, int]:
        return (x + s // 2, y + s // 2)

    # Draw rectangles (white fill, thick black outline)
    for _nid, (x, y, s) in nodes.items():
        d.rectangle([x, y, x + s, y + s], outline="black", width=6, fill="white")

    # Simple arrows (lines + triangular head)
    def arrow(a: int, b: int) -> None:
        ax, ay, asz = nodes[a]
        bx, by, bsz = nodes[b]
        x1, y1 = rect_center(ax, ay, asz)
        x2, y2 = rect_center(bx, by, bsz)

        # line
        d.line([x1, y1, x2, y2], fill="black", width=6)

        # arrowhead near end (very simple)
        # small triangle aligned roughly towards the end
        hx, hy = x2, y2
        d.polygon([(hx, hy), (hx - 14, hy - 10), (hx - 14, hy + 10)], fill="black")

    # Branching DAG:
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    arrow(0, 1)
    arrow(0, 2)
    arrow(1, 3)
    arrow(2, 3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
