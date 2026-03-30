import json
from pathlib import Path

from diagram2code.benchmark.leaderboard import CSV_COLUMNS, build_rows, write_csv
from diagram2code.benchmark.result_schema import SCHEMA_VERSION, BenchmarkResult


def _write_result(
    path: Path,
    *,
    predictor: str,
    exact: float,
    edge_f1: float | None = None,
    node_f1: float | None = None,
    direction_accuracy: float | None = None,
    node_count_error: float | None = None,
    edge_count_error: float | None = None,
) -> None:
    r = BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        dataset="example:minimal_v1",
        split="test",
        predictor=predictor,
        num_samples=1,
        metrics={
            "exact_match_rate": exact,
            "edge_f1": exact if edge_f1 is None else edge_f1,
            "node_f1": exact if node_f1 is None else node_f1,
            "direction_accuracy": exact if direction_accuracy is None else direction_accuracy,
            "node_count_error": 0.0 if node_count_error is None else node_count_error,
            "edge_count_error": 0.0 if edge_count_error is None else edge_count_error,
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


def test_leaderboard_sorts_best_rows_first(tmp_path: Path) -> None:
    naive = tmp_path / "naive.json"
    heuristic = tmp_path / "heuristic.json"
    oracle = tmp_path / "oracle.json"

    _write_result(
        naive,
        predictor="naive",
        exact=0.0,
        edge_f1=0.0,
        node_f1=0.2,
        direction_accuracy=0.0,
        node_count_error=2.0,
        edge_count_error=1.0,
    )
    _write_result(
        heuristic,
        predictor="heuristic",
        exact=0.5,
        edge_f1=0.6,
        node_f1=0.8,
        direction_accuracy=0.5,
        node_count_error=1.0,
        edge_count_error=0.5,
    )
    _write_result(
        oracle,
        predictor="oracle",
        exact=1.0,
        edge_f1=1.0,
        node_f1=1.0,
        direction_accuracy=1.0,
        node_count_error=0.0,
        edge_count_error=0.0,
    )

    rows = build_rows([heuristic, naive, oracle])
    predictors = [row.as_dict()["predictor"] for row in rows]

    assert predictors == ["oracle", "heuristic", "naive"]
