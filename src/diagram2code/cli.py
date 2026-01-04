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


def _edge_to_pair(e):
    """
    Convert various edge representations to (src, dst).
    Supports:
      - tuple/list: (src, dst)
      - objects with .src/.dst
      - objects with .from_id/.to_id
      - objects with .u/.v
    """
    if isinstance(e, (tuple, list)) and len(e) >= 2:
        return int(e[0]), int(e[1])
    for a, b in [("src", "dst"), ("from_id", "to_id"), ("u", "v")]:
        if hasattr(e, a) and hasattr(e, b):
            return int(getattr(e, a)), int(getattr(e, b))
    return None, None


def _resolve_labels(
    *,
    args,
    nodes,
    bgr,
    export_dir: Path | None,
    write_labels_json: bool,
    out_dir: Path,
) -> tuple[dict[int, str], str]:
    """
    Resolve labels using priority:
      1) --labels <file>
      2) auto-detect export_dir/labels.json (if --export provided)
      3) --extract-labels (OCR)
      4) none

    Returns: (labels_dict, labels_source_string)
    """
    from diagram2code.labels import load_labels

    labels_dict: dict[int, str] = {}
    source = "none"

    # 1) explicit --labels
    labels_path = Path(args.labels) if args.labels else None
    if labels_path is not None:
        labels_dict = load_labels(labels_path)
        return labels_dict, f"--labels ({labels_path})"

    # 2) auto-detect from export folder
    if export_dir is not None:
        auto = export_dir / "labels.json"
        if auto.exists():
            labels_dict = load_labels(auto)
            return labels_dict, f"auto ({auto})"

    # 3) OCR
    if args.extract_labels:
        try:
            from diagram2code.vision.extract_labels import extract_node_labels
        except ImportError:
            safe_print(
                "OCR requested but pytesseract is not installed.\n"
                "Install OCR extra:\n"
                "  pip install \"diagram2code[ocr]\"\n"
                "Then install the system Tesseract binary (see README)."
            )
            return {}, "ocr (unavailable: missing pytesseract)"

        labels_dict = extract_node_labels(bgr, nodes)

        if not labels_dict:
            safe_print(
                "OCR ran but returned no labels.\n"
                "If you expected labels, ensure Tesseract is installed and available on PATH.\n"
                " - Windows: choco install tesseract\n"
                " - macOS: brew install tesseract\n"
                " - Ubuntu/Debian: sudo apt-get install -y tesseract-ocr"
            )

        if write_labels_json:
            labels_out = out_dir / "labels.json"
            labels_out.write_text(
                json.dumps({str(k): v for k, v in labels_dict.items()}, indent=2),
                encoding="utf-8",
            )
            safe_print(f"Wrote: {labels_out}")

        return labels_dict, "ocr"

    return {}, source


def _print_graph_summary(nodes, edges, labels_dict: dict[int, str], labels_source: str) -> None:
    safe_print("\nGraph summary")
    safe_print(f"Labels source: {labels_source}")
    safe_print(f"Nodes: {len(nodes)}")
    for n in nodes:
        label = labels_dict.get(int(n.id), "")
        # Node bbox in your project is (x, y, w, h)
        x, y, w, h = n.bbox
        safe_print(f"  - id={n.id} bbox=({x}, {y}, {w}, {h}) label='{label}'")

    safe_print(f"Edges: {len(edges)}")
    for e in edges:
        src, dst = _edge_to_pair(e)
        if src is None:
            safe_print(f"  - {e}")
        else:
            safe_print(f"  - {src} -> {dst}")


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
    parser.add_argument(
        "--labels-template",
        action="store_true",
        help="Write a labels.template.json (node_id -> empty string) into --out, based on detected nodes.",
    )

    # export:
    parser.add_argument("--export", type=str, default=None, help="Export runnable bundle to directory")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection and print what would be generated, without writing any files.",
    )

    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Print a human-readable summary of detected nodes/edges (and labels if available).",
    )

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

    # Friendly error for missing input path
    in_path = Path(args.input)
    if not in_path.exists():
        safe_print(f"Error: input image not found: {in_path}")
        return 2

    # pipeline imports
    from diagram2code.vision.detect_arrows import detect_arrow_edges
    from diagram2code.vision.detect_shapes import detect_rectangles, draw_nodes_on_image

    # Resolve export dir (for auto-label detection and dry-run reporting)
    export_dir = Path(args.export) if args.export else None

    # ============================
    # DRY RUN
    # ============================
    if args.dry_run:
        safe_print("Dry run: no files will be written.")

        from diagram2code.vision.preprocess import preprocess_bgr_to_bin

        bgr = cv2.imread(str(in_path))
        if bgr is None:
            safe_print(f"Error: Could not read image: {in_path}")
            return 1

        _, image_bin = preprocess_bgr_to_bin(bgr)

        nodes = detect_rectangles(image_bin)
        edges = detect_arrow_edges(image_bin, nodes, debug_path=None)

        labels_dict, labels_source = _resolve_labels(
            args=args,
            nodes=nodes,
            bgr=bgr,
            export_dir=export_dir,
            write_labels_json=False,         # IMPORTANT: dry-run writes nothing
            out_dir=Path(args.out),          # not used when write_labels_json=False
        )

        safe_print(f"Would detect nodes: {len(nodes)}")
        safe_print(f"Would detect edges: {len(edges)}")
        safe_print(f"Would write outputs to: {Path(args.out)}")

        if args.labels_template:
            safe_print("Would write: labels.template.json")

        if args.labels:
            safe_print(f"Would load labels from: {args.labels}")
        elif args.export:
            safe_print("Would auto-detect labels.json inside export folder (if present)")
        elif args.extract_labels:
            safe_print("Would run OCR (requires diagram2code[ocr] + system tesseract)")

        safe_print("Would write: debug_nodes.png, debug_arrows.png, graph.json, render_graph.py, generated_program.py")
        if args.export:
            safe_print(f"Would export bundle to: {Path(args.export)}")

        if args.print_graph:
            _print_graph_summary(nodes, edges, labels_dict, labels_source)

        return 0

    # ============================
    # NORMAL RUN (writes artifacts)
    # ============================
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diagram2code.vision.preprocess import preprocess_image
    from diagram2code.export_graph import save_graph_json
    from diagram2code.export_matplotlib import generate_from_graph_json as gen_render_script
    from diagram2code.export_program import generate_from_graph_json as gen_program

    # Step 1: preprocess
    result = preprocess_image(str(in_path), out_dir)
    safe_print(f"Wrote: {result.output_path}")

    # Step 2: nodes
    nodes = detect_rectangles(result.image_bin)

    bgr = cv2.imread(str(in_path))
    if bgr is None:
        safe_print(f"Error: Could not read image: {in_path}")
        return 1

    debug_nodes = draw_nodes_on_image(bgr, nodes)
    debug_nodes_path = out_dir / "debug_nodes.png"
    cv2.imwrite(str(debug_nodes_path), debug_nodes)
    safe_print(f"Detected nodes: {len(nodes)}")
    safe_print(f"Wrote: {debug_nodes_path}")

    # labels template
    if args.labels_template:
        template_path = out_dir / "labels.template.json"
        template = {str(n.id): "" for n in nodes}
        template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
        safe_print(f"Wrote: {template_path}")

    # Step 3: edges
    debug_arrows_path = out_dir / "debug_arrows.png"
    edges = detect_arrow_edges(result.image_bin, nodes, debug_path=debug_arrows_path)
    safe_print(f"Wrote: {debug_arrows_path}")

    # Step 4: graph.json
    graph_path = save_graph_json(nodes, edges, out_dir / "graph.json")
    safe_print(f"Wrote: {graph_path}")

    # Step 5: render script
    script_path = gen_render_script(out_dir / "graph.json", out_dir / "render_graph.py")
    safe_print(f"Wrote: {script_path}")

    # Ensure export dir exists if requested (needed for label auto-detect + export)
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

    # Step 6: labels (ONE place only)
    labels_dict, labels_source = _resolve_labels(
        args=args,
        nodes=nodes,
        bgr=bgr,
        export_dir=export_dir,
        write_labels_json=True,     # normal run may write labels.json for OCR
        out_dir=out_dir,
    )

    # print graph after labels resolved
    if args.print_graph:
        _print_graph_summary(nodes, edges, labels_dict, labels_source)

    # Step 7: program
    program_path = gen_program(out_dir / "graph.json", out_dir / "generated_program.py", labels=labels_dict)
    safe_print(f"Wrote: {program_path}")

    # Step 8: export bundle
    if export_dir:
        import shutil

        shutil.copy2(out_dir / "graph.json", export_dir / "graph.json")
        shutil.copy2(out_dir / "generated_program.py", export_dir / "generated_program.py")

        for name in [
            "labels.json",
            "labels.template.json",
            "debug_nodes.png",
            "debug_arrows.png",
            "preprocessed.png",
            "render_graph.py",
            "render_graph.png",
        ]:
            p = out_dir / name
            if p.exists():
                shutil.copy2(p, export_dir / name)

        # Run scripts (work from any directory)
        (export_dir / "run.ps1").write_text(
            '$ErrorActionPreference = "Stop"\n'
            'python "$PSScriptRoot\\generated_program.py"\n',
            encoding="utf-8",
        )
        (export_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'python3 "$DIR/generated_program.py"\n',
            encoding="utf-8",
        )

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
        safe_print("\nExport complete.\n")
        safe_print("Next steps:")
        safe_print("  Windows (PowerShell):")
        safe_print(f"    cd {export_dir}")
        safe_print("    .\\run.ps1\n")
        safe_print("  Linux / macOS:")
        safe_print(f"    cd {export_dir}")
        safe_print("    bash run.sh\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
