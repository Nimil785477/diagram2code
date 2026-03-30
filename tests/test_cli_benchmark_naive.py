from __future__ import annotations

import json

from diagram2code.cli import main
from diagram2code.datasets.synthflow import build_synthflow_dataset


def test_cli_benchmark_naive_on_synthflow(tmp_path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    out_json = tmp_path / "result.json"

    build_synthflow_dataset(out=dataset_dir, split="test", num_samples=3, seed=0)

    rc = main(
        [
            "benchmark",
            "--dataset",
            str(dataset_dir),
            "--split",
            "test",
            "--predictor",
            "naive",
            "--limit",
            "3",
            "--json",
            str(out_json),
        ]
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert out_json.exists()
    assert "node_count_error=" in captured.out
    assert "edge_count_error=" in captured.out

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["predictor"] == "naive"
    assert "node_count_error" in payload["metrics"]
    assert "edge_count_error" in payload["metrics"]


def test_cli_benchmark_heuristic_outperforms_naive_on_synthflow(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    naive_json = tmp_path / "naive.json"
    heuristic_json = tmp_path / "heuristic.json"

    build_synthflow_dataset(out=dataset_dir, split="test", num_samples=3, seed=0)

    rc_naive = main(
        [
            "benchmark",
            "--dataset",
            str(dataset_dir),
            "--split",
            "test",
            "--predictor",
            "naive",
            "--limit",
            "3",
            "--json",
            str(naive_json),
        ]
    )

    rc_heuristic = main(
        [
            "benchmark",
            "--dataset",
            str(dataset_dir),
            "--split",
            "test",
            "--predictor",
            "heuristic",
            "--limit",
            "3",
            "--json",
            str(heuristic_json),
        ]
    )

    assert rc_naive == 0
    assert rc_heuristic == 0
    assert naive_json.exists()
    assert heuristic_json.exists()

    naive_payload = json.loads(naive_json.read_text(encoding="utf-8"))
    heuristic_payload = json.loads(heuristic_json.read_text(encoding="utf-8"))

    assert naive_payload["predictor"] == "naive"
    assert heuristic_payload["predictor"] == "heuristic"

    assert heuristic_payload["metrics"]["node_f1"] >= naive_payload["metrics"]["node_f1"]
    assert heuristic_payload["metrics"]["edge_f1"] >= naive_payload["metrics"]["edge_f1"]
    assert (
        heuristic_payload["metrics"]["exact_match_rate"]
        >= naive_payload["metrics"]["exact_match_rate"]
    )


def test_cli_benchmark_oracle_outperforms_naive_on_synthflow(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    naive_json = tmp_path / "naive.json"
    oracle_json = tmp_path / "oracle.json"

    build_synthflow_dataset(out=dataset_dir, split="test", num_samples=3, seed=0)

    rc_naive = main(
        [
            "benchmark",
            "--dataset",
            str(dataset_dir),
            "--split",
            "test",
            "--predictor",
            "naive",
            "--limit",
            "3",
            "--json",
            str(naive_json),
        ]
    )

    rc_oracle = main(
        [
            "benchmark",
            "--dataset",
            str(dataset_dir),
            "--split",
            "test",
            "--predictor",
            "oracle",
            "--limit",
            "3",
            "--json",
            str(oracle_json),
        ]
    )

    assert rc_naive == 0
    assert rc_oracle == 0
    assert naive_json.exists()
    assert oracle_json.exists()

    naive_payload = json.loads(naive_json.read_text(encoding="utf-8"))
    oracle_payload = json.loads(oracle_json.read_text(encoding="utf-8"))

    assert naive_payload["predictor"] == "naive"
    assert oracle_payload["predictor"] == "oracle"

    assert oracle_payload["metrics"]["node_f1"] > naive_payload["metrics"]["node_f1"]
    assert oracle_payload["metrics"]["edge_f1"] > naive_payload["metrics"]["edge_f1"]
    assert (
        oracle_payload["metrics"]["exact_match_rate"] > naive_payload["metrics"]["exact_match_rate"]
    )
