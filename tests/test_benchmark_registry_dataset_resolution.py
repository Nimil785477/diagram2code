from __future__ import annotations

import json
from pathlib import Path

from diagram2code.cli import main


def test_benchmark_resolves_registry_dataset_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DIAGRAM2CODE_BENCHMARK_TIMESTAMP_UTC", "1970-01-01T00:00:00Z")
    monkeypatch.setenv("DIAGRAM2CODE_SEED", "0")

    fixture_root = Path("tests/fixtures/flowlearn_smoke")
    assert (fixture_root / "dataset.json").exists()

    import diagram2code.cli as cli_mod

    def _fake_resolve(name: str, *, fetch_missing: bool, assume_yes: bool) -> Path:
        assert name == "flowlearn"
        return fixture_root

    monkeypatch.setattr(cli_mod, "_resolve_registry_dataset_root", _fake_resolve)

    out = tmp_path / "r.json"
    rc = main(
        [
            "benchmark",
            "--dataset",
            "flowlearn",
            "--predictor",
            "oracle",
            "--json",
            str(out),
        ]
    )
    assert rc == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.1"
    assert data["predictor"] == "oracle"
