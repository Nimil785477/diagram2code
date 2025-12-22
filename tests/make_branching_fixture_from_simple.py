from pathlib import Path
from PIL import Image

SIMPLE = Path("tests/fixtures/simple.png")
OUT = Path("tests/fixtures/branching.png")
#we reuse crops from simple.png to ensure detector compatibility
# These match your graph.json bbox format: [x, y, w, h]
# From your earlier output:
# node0 bbox: [50, 124, 76, 76]
# node1 bbox: [199, 124, 76, 76]
NODE0 = (50, 124, 50 + 76, 124 + 76)   # (left, top, right, bottom)
NODE1 = (199, 124, 199 + 76, 124 + 76)

def main():
    base = Image.open(SIMPLE).convert("RGBA")
    n0 = base.crop(NODE0)
    n1 = base.crop(NODE1)

    # New canvas (white)
    W, H = 400, 300
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # Paste 4 nodes (2 red-style, 2 blue-style)
    positions = [
        (40, 40),   # node 0-like
        (40, 160),  # node 1-like
        (220, 160), # node 0-like
        (220, 40),  # node 1-like
    ]
    patches = [n0, n1, n0, n1]

    for patch, (x, y) in zip(patches, positions):
        canvas.paste(patch, (x, y), patch)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT)
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
