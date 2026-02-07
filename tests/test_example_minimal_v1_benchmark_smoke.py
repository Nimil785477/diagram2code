from diagram2code.benchmark.predictor_backends import make_predictor
from diagram2code.benchmark.runner import run_benchmark
from diagram2code.datasets import DatasetRegistry


def test_example_minimal_v1_oracle_smoke():
    root = DatasetRegistry().resolve_root("example:minimal_v1")
    predictor = make_predictor("oracle", dataset_path=root, out_dir=None)

    # Keep compatible with runner signature (your CLI already passes split/limit)
    result = run_benchmark(
        dataset_path=root,
        predictor=predictor,
        alpha=0.35,
        split="test",
        limit=1,
    )

    assert result.aggregate.exact_match_rate == 1.0
