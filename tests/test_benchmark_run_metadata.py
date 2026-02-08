import json
from pathlib import Path

from diagram2code.benchmark.result_schema import SCHEMA_VERSION
from diagram2code.benchmark.serialize import write_benchmark_json


class _Agg:
    class _PR:
        def __init__(self) -> None:
            self.precision = 1.0
            self.recall = 1.0
            self.f1 = 1.0

    def __init__(self) -> None:
        self.node = self._PR()
        self.edge = self._PR()
        self.direction_accuracy = 1.0
        self.exact_match_rate = 1.0
        self.runtime_mean_s = None


class _Result:
    def __init__(self) -> None:
        self.aggregate = _Agg()
        self.dataset = "example:minimal_v1"
        self.split = "test"
        self.predictor = "oracle"
        self.num_samples = 1


def test_write_benchmark_json_includes_run_metadata(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    write_benchmark_json(_Result(), out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION

    run = data["run"]
    assert isinstance(run["timestamp_utc"], str) and run["timestamp_utc"].endswith("Z")
    assert isinstance(run["python"], str) and run["python"]
    assert isinstance(run["platform"], str) and run["platform"]

    # New fields (git_sha may be empty in CI or non-git contexts)
    assert "diagram2code_version" in run
    assert isinstance(run["diagram2code_version"], str)

    assert "git_sha" in run
    assert isinstance(run["git_sha"], str)
