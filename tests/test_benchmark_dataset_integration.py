from __future__ import annotations

from pathlib import Path

from diagram2code.benchmark.predictor_backends import make_predictor
from diagram2code.benchmark.runner import run_benchmark
from diagram2code.datasets.synthetic_basic import generate_synthetic_basic


def test_benchmark_on_phase3_dataset_oracle_is_perfect(tmp_path: Path) -> None:
    ds_root = tmp_path / "synthetic_phase3"
    generate_synthetic_basic(ds_root, n=6, seed=0, split="test")

    predictor = make_predictor("oracle", dataset_path=ds_root, out_dir=None)

    result = run_benchmark(
        dataset_path=ds_root,
        predictor=predictor,
        alpha=0.35,
        split="test",
        limit=6,
    )

    assert len(result.samples) == 6

    # Oracle must be perfect
    assert result.aggregate.node.f1 == 1.0
    assert result.aggregate.edge.f1 == 1.0
    assert result.aggregate.direction_accuracy == 1.0
    assert result.aggregate.exact_match_rate == 1.0
