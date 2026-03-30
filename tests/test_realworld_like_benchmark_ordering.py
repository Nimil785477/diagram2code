from __future__ import annotations

import json

from diagram2code.cli import main


def test_realworld_like_benchmark_ordering(tmp_path) -> None:
    dataset_dir = tmp_path / "realworld_like_ds"
    naive_json = tmp_path / "naive.json"
    heuristic_json = tmp_path / "heuristic.json"
    oracle_json = tmp_path / "oracle.json"

    rc_build = main(
        [
            "dataset",
            "build",
            "realworld-like",
            "--out",
            str(dataset_dir),
            "--split",
            "test",
            "--num-samples",
            "6",
            "--seed",
            "0",
        ]
    )
    assert rc_build == 0

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
            "6",
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
            "6",
            "--json",
            str(heuristic_json),
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
            "6",
            "--json",
            str(oracle_json),
        ]
    )

    assert rc_naive == 0
    assert rc_heuristic == 0
    assert rc_oracle == 0

    naive_payload = json.loads(naive_json.read_text(encoding="utf-8"))
    heuristic_payload = json.loads(heuristic_json.read_text(encoding="utf-8"))
    oracle_payload = json.loads(oracle_json.read_text(encoding="utf-8"))

    assert oracle_payload["predictor"] == "oracle"
    assert heuristic_payload["predictor"] == "heuristic"
    assert naive_payload["predictor"] == "naive"

    assert (
        oracle_payload["metrics"]["exact_match_rate"]
        > heuristic_payload["metrics"]["exact_match_rate"]
    )
    assert heuristic_payload["metrics"]["edge_f1"] > naive_payload["metrics"]["edge_f1"]
    assert heuristic_payload["metrics"]["node_f1"] >= naive_payload["metrics"]["node_f1"]
    assert oracle_payload["metrics"]["edge_f1"] == 1.0
