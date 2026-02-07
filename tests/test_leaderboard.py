import json
from pathlib import Path

from diagram2code.benchmark.leaderboard import CSV_COLUMNS, build_rows, write_csv
from diagram2code.benchmark.result_schema import SCHEMA_VERSION, BenchmarkResult


def _write_result(path: Path, *, predictor: str, exact: float) -> None:
    r = BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        dataset="example:minimal_v1",
        split="test",
        predictor=predictor,
        num_samples=1,
        metrics={
            "exact_match_rate": exact,
            "edge_f1": exact,
            "node_f1": exact,
            "direction_accuracy": exact,
        },
        run={"timestamp_utc": "2026-02-07T12:00:00Z", "python": "3.x", "platform": "test"},
    )
    r.validate()
    path.write_text(json.dumps(r.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_leaderboard_build_and_csv(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_result(a, predictor="oracle", exact=1.0)
    _write_result(b, predictor="heuristic", exact=0.5)

    rows = build_rows([a, b])
    assert len(rows) == 2

    out = tmp_path / "leaderboard.csv"
    write_csv(rows, out)

    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == ",".join(CSV_COLUMNS)
    assert len(text) == 3  # header + 2 rows
