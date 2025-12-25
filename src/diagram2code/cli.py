from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def safe_print(msg: str) -> None:
    # Avoid UnicodeEncodeError on Windows CI/console encodings
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

    # labels:
    parser.add_argument("--labels", default=None, help="Path to labels JSON (optional)")
    parser.add_argument(
        "--extract-labels",
        action="store_true",
        help="Extract labels via OCR and write labels.json into --out (optional; requires pytesseract + tesseract).",
    )

    # export:
    parser.add_argument("--export", type=str, default=None, help="Export runnable bundle to directory")

    args = parser.parse_args(argv)

    if args.version:
        try:
            from importlib.metadata import version
            safe_print(f"diagram2code {version('diagram2code')}")
        except Exception:
            safe_print("diagram2code (unknown version)")
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
    from diagram2code.export_matplotlib import generate_from_graph_json as gen_render_script
    from diagram2code.export_program import generate_from_graph_json as gen_program
    from diagram2code.labels import load_labels

    # ============================
    # Step 1: Preprocess
    # ============================
    result = preprocess_image(args.input, out_dir)
    safe_print(f"Wrote: {result.output_path}")

    # ============================
    # Step 2: Detect nodes
    # ============================
    nodes = detect_rectangles(result.image_bin)

    bgr = cv2.imread(str(args.input))
    if bgr is None:
        safe_print(f"Error: Could not read image: {args.input}")
        return 1

    debug_nodes = draw_nodes_on_image(bgr, nodes)
    debug_nodes_path = out_dir / "debug_nodes.png"
    cv2.imwrite(str(debug_nodes_path), debug_nodes)
    safe_print(f"Detected nodes: {len(nodes)}")
    safe_print(f"Wrote: {debug_nodes_path}")

    # ============================
    # Step 3: Detect arrows (edges)
    # ============================
    edges = detect_arrow_edges(result.image_bin, nodes, debug_path=out_dir / "debug_arrows.png")
    safe_print(f"Wrote: {out_dir / 'debug_arrows.png'}")

    # ============================
    # Step 4: graph.json
    # ============================
    graph_path = save_graph_json(nodes, edges, out_dir / "graph.json")
    safe_print(f"Wrote: {graph_path}")

    # ============================
    # Step 5: render_graph.py
    # ============================
    script_path = gen_render_script(out_dir / "graph.json", out_dir / "render_graph.py")
    safe_print(f"Wrote: {script_path}")

    # ============================
    # Step 6: labels (either load from --labels, or OCR extract to out_dir/labels.json, or empty)
    # ============================
    labels_dict = {}

    if args.labels:
        labels_dict = load_labels(args.labels)

    elif args.extract_labels:
        # OCR is optional: only run when user asks for it
        try:
            from diagram2code.vision.extract_labels import extract_node_labels

            labels_dict = extract_node_labels(bgr, nodes) or {}
        except Exception as e:
            # IMPORTANT: do NOT fail the whole CLI; just continue without labels
            safe_print(f"Warning: OCR label extraction failed/unavailable: {type(e).__name__}: {e}")
            labels_dict = {}

        labels_out = out_dir / "labels.json"
        labels_out.write_text(
            json.dumps({str(k): v for k, v in labels_dict.items()}, indent=2),
            encoding="utf-8",
        )
        safe_print(f"Wrote: {labels_out}")

    # ============================
    # Step 7: generated_program.py
    # ============================
    program_path = gen_program(out_dir / "graph.json", out_dir / "generated_program.py", labels=labels_dict)
    safe_print(f"Wrote: {program_path}")

    # ============================
    # Step 8: optional export bundle
    # ============================
    export_dir = Path(args.export) if args.export else None
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        # Copy required artifacts
        shutil.copy2(out_dir / "graph.json", export_dir / "graph.json")
        shutil.copy2(out_dir / "generated_program.py", export_dir / "generated_program.py")

        # Optional artifacts if they exist
        for name in [
            "labels.json",
            "debug_nodes.png",
            "debug_arrows.png",
            "preprocessed.png",
            "render_graph.py",
            "render_graph.png",
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

        # README
        (export_dir / "README_EXPORT.md").write_text(
            "# diagram2code export\n\n"
            "This folder contains an exported runnable bundle.\n\n"
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
