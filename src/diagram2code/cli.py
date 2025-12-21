import argparse
from pathlib import Path

import cv2


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="diagram2code",
        description="Convert simple diagram images into runnable code.",
    )
    parser.add_argument("input", nargs="?", help="Path to input image")
    parser.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--version", action="store_true", help="Print version")

    args = parser.parse_args(argv)

    if args.version:
        print("diagram2code 0.0.1")
        return 0

    if not args.input:
        parser.print_help()
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diagram2code.vision.preprocess import preprocess_image
    from diagram2code.vision.detect_shapes import detect_rectangles, draw_nodes_on_image

    # Step 1: preprocess
    result = preprocess_image(args.input, out_dir)
    print(f"✅ Wrote: {result.output_path}")

    # Step 2: detect rectangles
    nodes = detect_rectangles(result.image_bin)

    # draw nodes on original image
    bgr = cv2.imread(str(args.input))
    debug = draw_nodes_on_image(bgr, nodes)
    debug_path = out_dir / "debug_nodes.png"
    cv2.imwrite(str(debug_path), debug)

    print(f"✅ Detected nodes: {len(nodes)}")
    print(f"✅ Wrote: {debug_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
