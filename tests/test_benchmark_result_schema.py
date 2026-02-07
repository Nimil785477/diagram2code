import json

from diagram2code.benchmark.result_schema import SCHEMA_VERSION, BenchmarkResult


def test_benchmark_result_schema_roundtrip(tmp_path):
    out = tmp_path / "result.json"

    r = BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        dataset="flowlearn:dummy",
        split="test",
        predictor="oracle",
        num_samples=3,
        metrics={"exact_match_rate": 1.0, "edge_f1": 1.0},
        run={"diagram2code_version": "0.0.0"},
    )
    r.validate()

    out.write_text(
        json.dumps(r.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = json.loads(out.read_text(encoding="utf-8"))

    expected_keys = {
        "schema_version",
        "dataset",
        "split",
        "predictor",
        "num_samples",
        "metrics",
        "run",
    }
    assert set(loaded.keys()) == expected_keys
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert isinstance(loaded["metrics"]["exact_match_rate"], float)
