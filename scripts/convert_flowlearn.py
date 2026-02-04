from __future__ import annotations

import argparse
from pathlib import Path

from diagram2code.datasets.adapters.flowlearn import convert_flowlearn


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="convert_flowlearn",
        description="Convert FlowLearn subsets into diagram2code Phase-3 dataset format.",
    )

    p.add_argument("--flowlearn-root", type=Path, required=True)
    p.add_argument(
        "--subset",
        choices=[
            "SimFlowchart/char",
            "SimFlowchart/word",
        ],
        default="SimFlowchart/char",
    )
    p.add_argument("--split", choices=["train", "test", "all"], default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--strict", action="store_true", help="Fail fast on first bad record")

    args = p.parse_args(argv)

    convert_flowlearn(
        flowlearn_root=args.flowlearn_root,
        subset=args.subset,
        split=args.split,
        out=args.out,
        limit=args.limit,
        strict=args.strict,
    )
    print(f"Wrote Phase-3 dataset to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
