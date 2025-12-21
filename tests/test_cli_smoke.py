from diagram2code.cli import main


def test_help_smoke(capsys):
    code = main([])
    out = capsys.readouterr().out.lower()
    assert code == 0
    assert "diagram2code" in out


def test_version():
    assert main(["--version"]) == 0
