from __future__ import annotations

import json
import re
from pathlib import Path

from diagram2code.cli import main

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _find_meta(obj: dict) -> dict | None:
    """
    Be tolerant to schema evolution.

    We look for metadata in:
      - top-level "meta"
      - top-level "run"
      - nested "meta" under "run"
    """
    if isinstance(obj.get("meta"), dict):
        return obj["meta"]
    if isinstance(obj.get("run"), dict):
        run = obj["run"]
        if isinstance(run.get("meta"), dict):
            return run["meta"]
        return run
    return None


def test_cli_benchmark_result_metadata(tmp_path: Path) -> None:
    out_json = tmp_path / "result.json"

    rc = main(
        [
            "benchmark",
            "--dataset",
            "example:minimal_v1",
            "--predictor",
            "oracle",
            "--json",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    meta = _find_meta(data)
    assert meta is not None, (
        f"Could not find metadata block in result json keys={list(data.keys())}"
    )

    # Required trace fields
    assert meta.get("dataset_ref") in {"example:minimal_v1", "example:minimal_v1/"} or (
        isinstance(meta.get("dataset_ref"), str)
        and meta["dataset_ref"].startswith("example:minimal_v1")
    )
    assert meta.get("predictor") == "oracle"

    cli = meta.get("cli")
    assert isinstance(cli, str) and "diagram2code" in cli and "benchmark" in cli

    assert data["dataset"] == "example:minimal_v1"
    assert data["predictor"] == "oracle"
    assert data["split"] in {"unknown", "test"}  # depending on how the test runs
    # Optional: if present, must look like a sha256
    msha = meta.get("dataset_manifest_sha256")
    if msha is not None:
        assert isinstance(msha, str) and _SHA256_RE.match(msha), f"Invalid sha256: {msha!r}"
