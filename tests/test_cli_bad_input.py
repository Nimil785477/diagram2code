from pathlib import Path

from diagram2code.cli import main


def test_cli_bad_input_path_returns_2(tmp_path: Path):
    rc = main(["does_not_exist.png", "--out", str(tmp_path)])
    assert rc == 2
