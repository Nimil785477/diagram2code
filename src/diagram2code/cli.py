from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def safe_print(msg: str) -> None:
    """Avoid UnicodeEncodeError on Windows CI/console encodings."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", errors="replace").decode("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="diagram2code",
        description="Convert simple diagram images into runnable code.",
    )
    parser.add_argument("input", nargs="?", help="Path to input image")
    parser.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument("--labels", default=None, help="Path to labels JSON (optional)")
    parser.add_argument("--export", type=str, default=None, help="Export runnable bundle to directory")

    args = parser.parse_args(argv)

    if args.version:
        safe_print("diagram2code 0.0.1")
        return 0

    if not args.input:
        parser.print_help()
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- imports for pipeline ---
    from diagram2code.vision.preprocess import preprocess_image
    from diagram2code.vision.detect_shapes import detect_rectangles, draw_nodes_on_image
    from diagram2code.vision.detect_arrows import detect_arrow_edges
    from diagram2code.export_graph import save_graph_json
    from diagram2code.export_matplotlib import generate_from_graph_json as gen_render
    from diagram2code.export_program import generate_from_graph_json as gen_program

    # ============================
    # Step 1: Preprocess
    # ============================
    result = preprocess_image(args.input, out_dir)
    safe_print(f"Wrote: {result.output_path}")

    # Load original BGR for overlays + OCR
    bgr = cv2.imread(str(args.input))
    if bgr is None:
        safe_print(f"ERROR: Failed to read image: {args.input}")
        return 1

    # ============================
    # Step 2: Detect nodes
    # ============================
    nodes = detect_rectangles(result.image_bin)

    debug_nodes = draw_nodes_on_image(bgr, nodes)
    debug_nodes_path = out_dir / "debug_nodes.png"
    cv2.imwrite(str(debug_nodes_path), debug_nodes)

    safe_print(f"Detected nodes: {len(nodes)}")
    safe_print(f"Wrote: {debug_nodes_path}")

    # ============================
    # Step 3: Detect arrows (edges)
    # ============================
    debug_arrows_path = out_dir / "debug_arrows.png"
    edges = detect_arrow_edges(result.image_bin, nodes, debug_path=debug_arrows_path)
    safe_print(f"Detected edges: {edges}")
    safe_print(f"Wrote: {debug_arrows_path}")

    # ============================
    # Step 4: Export graph.json
    # ============================
    graph_path = save_graph_json(nodes, edges, out_dir / "graph.json")
    safe_print(f"Wrote: {graph_path}")

    # ============================
    # Step 5: Export render_graph.py
    # ============================
    render_script_path = gen_render(out_dir / "graph.json", out_dir / "render_graph.py")
    safe_print(f"Wrote: {render_script_path}")

    # ============================
    # Step 6: Labels (manual OR OCR)
    # ============================
    labels: dict[int, str] = {}

    if args.labels:
        from diagram2code.labels import load_labels

        labels = load_labels(args.labels)
        safe_print(f"Loaded labels from: {args.labels}")
    else:
        # OCR labels and write to out_dir/labels.json
        from diagram2code.vision.extract_labels import extract_node_labels

        labels = extract_node_labels(bgr, nodes)

        labels_path = out_dir / "labels.json"
        labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        safe_print(f"Wrote: {labels_path}")

    # ============================
    # Step 7: Export generated_program.py
    # ============================
    program_path = gen_program(out_dir / "graph.json", out_dir / "generated_program.py", labels=labels)
    safe_print(f"Wrote: {program_path}")

    # ============================
    # Optional: Export bundle
    # ============================
    export_dir = Path(args.export) if args.export else None
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

        # Always copy core artifacts
        shutil.copy2(out_dir / "graph.json", export_dir / "graph.json")
        shutil.copy2(out_dir / "generated_program.py", export_dir / "generated_program.py")

        # Copy optional artifacts if present
        for name in [
            "labels.json",
            "render_graph.py",
            "render_graph.png",
            "debug_nodes.png",
            "debug_arrows.png",
            "preprocessed.png",
        ]:
            p = out_dir / name
            if p.exists():
                shutil.copy2(p, export_dir / name)

        # Run scripts
        (export_dir / "run.ps1").write_text("python .\\generated_program.py\n", encoding="utf-8")
        (export_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\nset -e\npython3 generated_program.py\n",
            encoding="utf-8",
        )

        # Export README
        (export_dir / "README_EXPORT.md").write_text(
            "# diagram2code export\n\n"
            "This folder contains exported artifacts produced by `diagram2code`.\n\n"
            "## Run\n\n"
            "### Windows (PowerShell)\n"
            "```powershell\n"
            ".\\run.ps1\n"
            "```\n\n"
            "### macOS / Linux\n"
            "```bash\n"
            "bash run.sh\n"
            "```\n",
            encoding="utf-8",
        )

        safe_print(f"Exported bundle to: {export_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
