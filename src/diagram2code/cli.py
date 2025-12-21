import argparse
from pathlib import Path


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

    from diagram2code.vision.preprocess import preprocess_image

    result = preprocess_image(args.input, Path(args.out))
    print(f"✅ Wrote: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
