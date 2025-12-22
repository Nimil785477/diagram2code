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
    parser.add_argument("--labels", default=None, help="Path to labels JSON (optional)")

    args = parser.parse_args(argv)

    if args.version:
        print("diagram2code 0.0.1")
        return 0

    if not args.input:
        parser.print_help()
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- imports for pipeline ---
    from diagram2code.vision.preprocess import preprocess_image
    from diagram2code.vision.detect_shapes import (
        detect_rectangles,
        draw_nodes_on_image,
    )
    from diagram2code.vision.detect_arrows import detect_arrow_edges
    from diagram2code.export_graph import save_graph_json

    # ============================
    # Step 1: Preprocess
    # ============================
    result = preprocess_image(args.input, out_dir)
    print(f"✅ Wrote: {result.output_path}")

    # ============================
    # Step 2: Detect nodes
    # ============================
    nodes = detect_rectangles(result.image_bin)

    bgr = cv2.imread(str(args.input))
    debug_nodes = draw_nodes_on_image(bgr, nodes)
    debug_nodes_path = out_dir / "debug_nodes.png"
    cv2.imwrite(str(debug_nodes_path), debug_nodes)
    print(f"✅ Detected nodes: {len(nodes)}")
    print(f"✅ Wrote: {debug_nodes_path}")

    # ============================
    # Step 3: Detect arrows (edges)
    # ============================
    edges = detect_arrow_edges(result.image_bin, nodes)
    print(f"✅ Detected edges: {edges}")

    # ============================
    # Step 4: Export graph.json  👈 THIS IS THE NEW PART
    # ============================
    graph_path = save_graph_json(nodes, edges, out_dir / "graph.json")
    print(f"✅ Wrote: {graph_path}")

    from diagram2code.export_matplotlib import generate_from_graph_json

    script_path = generate_from_graph_json(out_dir / "graph.json", out_dir / "render_graph.py")
    print(f"✅ Wrote: {script_path}")

    from diagram2code.labels import load_labels
    from diagram2code.export_program import generate_from_graph_json as gen_program

    labels_path = args.labels if args.labels else (out_dir / "labels.json")
    labels = load_labels(labels_path)
    program_path = gen_program(out_dir / "graph.json", out_dir / "generated_program.py", labels=labels)
    print(f"✅ Wrote: {program_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
