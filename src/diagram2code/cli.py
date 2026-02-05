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
                '  pip install "diagram2code[ocr]"\n'
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


def _print_predictors_list() -> None:
    from diagram2code.benchmark.predictor_backends import (
        available_predictors,
        predictor_descriptions,
    )

    preds = available_predictors()
    desc = predictor_descriptions()

    safe_print("Available predictors:")
    for p in preds:
        extra = f" - {desc[p]}" if p in desc else ""
        safe_print(f"  - {p}{extra}")


def cmd_benchmark(args) -> int:
    """
    Phase 3 dataset-first benchmarking entry.
    Step 3: support dataset references via DatasetRegistry (e.g. example:minimal_v1).
    Step 4: add split/limit selection (passed through to runner when supported).
    Phase 5.1: CLI usability (list predictors, friendly errors).
    """
    # Phase 5.1: discovery without needing dataset
    if getattr(args, "list_predictors", False):
        _print_predictors_list()
        return 0

    if not args.dataset:
        safe_print("Error: --dataset is required (or use --list-predictors).")
        return 2

    from diagram2code.benchmark.predictor_backends import available_predictors, make_predictor
    from diagram2code.benchmark.runner import run_benchmark
    from diagram2code.benchmark.serialize import write_benchmark_json
    from diagram2code.datasets import DatasetRegistry, load_dataset

    # Validate predictor name (even though argparse choices usually enforces this)
    preds = available_predictors()
    if args.predictor not in preds:
        safe_print(f"Error: Unknown predictor '{args.predictor}'.")
        safe_print(f"Available predictors: {', '.join(preds)}")
        return 2

    # Step 3: dataset ref -> resolved filesystem root
    dataset_ref = args.dataset
    dataset_root = DatasetRegistry().resolve_root(dataset_ref)

    # Step 5.1: validate split early for friendlier error
    if args.split is not None:
        ds = load_dataset(dataset_root)
        splits = ds.splits()
        if args.split not in splits:
            safe_print(f"Error: Unknown split '{args.split}'.")
            safe_print(f"Available splits: {', '.join(splits)}")
            return 2

    predictor = make_predictor(
        args.predictor,
        dataset_path=dataset_root,
        out_dir=args.predictor_out,
    )

    # Step 4: split/limit (runner may or may not support these kwargs yet).
    # Keep backward compatibility to avoid breaking if runner signature is older.
    try:
        result = run_benchmark(
            dataset_path=dataset_root,
            predictor=predictor,
            alpha=args.alpha,
            split=args.split,
            limit=args.limit,
        )
    except TypeError:
        if args.split is not None or args.limit is not None:
            safe_print(
                "Note: --split/--limit were provided, but the current benchmark runner "
                "does not accept them yet. Running full dataset without split/limit."
            )
        result = run_benchmark(dataset_path=dataset_root, predictor=predictor, alpha=args.alpha)

    # stdout summary (stable + easy to grep)
    agg = result.aggregate
    print(f"node: p={agg.node.precision:.3f} r={agg.node.recall:.3f} f1={agg.node.f1:.3f}")
    print(f"edge: p={agg.edge.precision:.3f} r={agg.edge.recall:.3f} f1={agg.edge.f1:.3f}")
    print(f"direction_accuracy={agg.direction_accuracy:.3f}")
    print(f"exact_match_rate={agg.exact_match_rate:.3f}")
    print(f"runtime_mean_s={agg.runtime_mean_s}")

    if args.json is not None:
        write_benchmark_json(result, args.json)
        print(f"Wrote JSON: {args.json}")

    return 0


def _build_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diagram2code",
        description="Convert simple diagram images into runnable code.",
    )

    # Default mode positional MUST exist here (no subparsers in this parser)
    parser.add_argument("input", nargs="?", default=None, help="Path to input image")

    parser.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--version", action="store_true", help="Print version")

    # labels:
    parser.add_argument("--labels", default=None, help="Path to labels JSON (optional)")
    parser.add_argument(
        "--extract-labels",
        action="store_true",
        help=(
            "Extract labels via OCR and write labels.json into --out (optional; "
            "requires pytesseract + tesseract)."
        ),
    )
    parser.add_argument(
        "--labels-template",
        action="store_true",
        help=(
            "Write a labels.template.json (node_id -> empty string) into --out, "
            "based on detected nodes."
        ),
    )

    # export:
    parser.add_argument(
        "--export", type=str, default=None, help="Export runnable bundle to directory"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection and print what would be generated, without writing any files.",
    )

    parser.add_argument(
        "--print-graph",
        action="store_true",
        help=("Print a human-readable summary of detected nodes/edges (and labels if available)."),
    )

    parser.add_argument(
        "--render-graph",
        action="store_true",
        help="Render graph.json as a visual graph image (requires matplotlib + networkx).",
    )

    parser.add_argument(
        "--render-format",
        choices=["png", "svg"],
        default="png",
        help="Output format for --render-graph (default: png).",
    )

    parser.add_argument(
        "--render-layout",
        choices=["auto", "topdown", "spring"],
        default="auto",
        help="Layout for --render-graph: auto (DAG->topdown), topdown, or spring.",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not write debug artifacts (preprocessed/debug images and render_graph.py).",
    )

    return parser


def _build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diagram2code benchmark",
        description="Run benchmarking on an explicit dataset (Phase 3 dataset-first).",
    )

    # Phase 5.1: discovery mode
    parser.add_argument(
        "--list-predictors",
        action="store_true",
        help="List available predictors and exit.",
    )

    # Step 3: accept dataset references, not only Paths.
    # NOTE: not required because --list-predictors should work standalone.
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        help=(
            "Dataset reference or root path. Examples: "
            "example:minimal_v1, /path/to/dataset, my-registered-name"
        ),
    )

    # Step 4: split/limit support
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate (e.g., test, train, all). Default is runner-defined.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples (deterministic).",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Node match threshold (center distance rule)",
    )

    # Dynamic predictor choices (Phase 5.1)
    from diagram2code.benchmark.predictor_backends import (
        available_predictors,
        predictor_descriptions,
    )

    predictor_choices = available_predictors()
    desc = predictor_descriptions()
    default_pred = "vision" if "vision" in predictor_choices else predictor_choices[0]

    parser.add_argument(
        "--predictor",
        choices=predictor_choices,
        default=default_pred,
        help=(
            "Prediction backend. "
            f"Available: {', '.join(predictor_choices)}. "
            f"Defaults to '{default_pred}'."
        ),
    )

    parser.add_argument(
        "--predictor-out",
        type=Path,
        default=None,
        help="Output dir for predictor artifacts (used by vision predictor).",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write results to JSON file")

    # Attach descriptions (nice-to-have) via epilog
    lines = ["", "Predictors:"]
    for p in predictor_choices:
        if p in desc:
            lines.append(f"  - {p}: {desc[p]}")
        else:
            lines.append(f"  - {p}")
    parser.epilog = "\n".join(lines)

    return parser


def main(argv=None) -> int:
    if argv is None:
        import sys

        argv = sys.argv[1:]

    # Dispatch: if first token is "benchmark", parse benchmark args only.
    if argv[:1] == ["benchmark"]:
        bench_parser = _build_benchmark_parser()
        args = bench_parser.parse_args(argv[1:])
        return cmd_benchmark(args)

    parser = _build_default_parser()
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
            write_labels_json=False,  # IMPORTANT: dry-run writes nothing
            out_dir=Path(args.out),  # not used when write_labels_json=False
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

        if args.no_debug:
            safe_print("Would write: graph.json, generated_program.py")
        else:
            safe_print(
                "Would write: preprocessed.png, debug_nodes.png, debug_arrows.png, "
                "graph.json, render_graph.py, generated_program.py"
            )

        if args.export:
            safe_print(f"Would export bundle to: {Path(args.export)}")

        if args.print_graph:
            _print_graph_summary(nodes, edges, labels_dict, labels_source)

        if args.render_graph:
            safe_print(f"Would render: graph.{args.render_format} (skipped in dry-run)")

        return 0

    # ============================
    # NORMAL RUN (writes artifacts)
    # ============================
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diagram2code.export_graph import save_graph_json
    from diagram2code.export_matplotlib import generate_from_graph_json as gen_render_script
    from diagram2code.export_program import generate_from_graph_json as gen_program
    from diagram2code.vision.preprocess import preprocess_image

    # Step 1: preprocess (gated preprocessed.png write)
    result = preprocess_image(str(in_path), out_dir, write_debug=not args.no_debug)
    if not args.no_debug:
        safe_print(f"Wrote: {result.output_path}")

    # Step 2: nodes
    nodes = detect_rectangles(result.image_bin)

    bgr = cv2.imread(str(in_path))
    if bgr is None:
        safe_print(f"Error: Could not read image: {in_path}")
        return 1

    safe_print(f"Detected nodes: {len(nodes)}")

    # debug_nodes.png (gated)
    if not args.no_debug:
        debug_nodes = draw_nodes_on_image(bgr, nodes)
        debug_nodes_path = out_dir / "debug_nodes.png"
        cv2.imwrite(str(debug_nodes_path), debug_nodes)
        safe_print(f"Wrote: {debug_nodes_path}")

    # labels template (keep writing if requested)
    if args.labels_template:
        template_path = out_dir / "labels.template.json"
        template = {str(n.id): "" for n in nodes}
        template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
        safe_print(f"Wrote: {template_path}")

    # Step 3: edges (gated debug_arrows.png write via debug_path=None)
    debug_arrows_path = None if args.no_debug else (out_dir / "debug_arrows.png")
    edges = detect_arrow_edges(result.image_bin, nodes, debug_path=debug_arrows_path)
    if debug_arrows_path is not None:
        safe_print(f"Wrote: {debug_arrows_path}")

    # Step 4: graph.json
    graph_path = save_graph_json(nodes, edges, out_dir / "graph.json")
    safe_print(f"Wrote: {graph_path}")

    # Step 5: render script (gated)
    if not args.no_debug:
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
        write_labels_json=True,  # normal run may write labels.json for OCR
        out_dir=out_dir,
    )

    # print graph after labels resolved
    if args.print_graph:
        _print_graph_summary(nodes, edges, labels_dict, labels_source)

    # Step 7: program
    program_path = gen_program(
        out_dir / "graph.json", out_dir / "generated_program.py", labels=labels_dict
    )
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
            '$ErrorActionPreference = "Stop"\npython "$PSScriptRoot\\generated_program.py"\n',
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

    # Step 9: render graph image (after graph.json exists; after export copy)
    if args.render_graph:
        target_dir = export_dir if export_dir else out_dir
        target_graph = target_dir / "graph.json"
        target_labels = target_dir / "labels.json"
        target_img = target_dir / f"graph.{args.render_format}"

        if not target_graph.exists():
            raise RuntimeError("--render-graph requested but graph.json was not generated")

        try:
            from diagram2code.render_graph import RenderOptions, render_graph
        except ImportError:
            safe_print(
                "Graph rendering requested but required dependencies are missing.\n"
                "Ensure 'matplotlib' and 'networkx' are installed."
            )
            raise

        opts = RenderOptions(
            output_path=target_img,
            layout=args.render_layout,
        )

        render_graph(
            target_graph,
            target_img,
            labels_json_path=target_labels if target_labels.exists() else None,
            options=opts,
        )
        safe_print(f"Wrote: {target_img}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
