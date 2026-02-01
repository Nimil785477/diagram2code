from pathlib import Path

from diagram2code.cli import main


def test_cli_dry_run_writes_nothing(tmp_path: Path):
    rc = main(["tests/fixtures/branching.png", "--out", str(tmp_path), "--dry-run"])
    assert rc == 0
    # should NOT create artifacts
    assert not (tmp_path / "graph.json").exists()
    assert not (tmp_path / "generated_program.py").exists()
    assert not (tmp_path / "debug_nodes.png").exists()
