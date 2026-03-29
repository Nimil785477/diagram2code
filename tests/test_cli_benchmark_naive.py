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
