from __future__ import annotations

from pathlib import Path

from diagram2code.cli import main


def test_benchmark_passes_resolved_dataset_path_to_predictor(monkeypatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "resolved"
    fake_root.mkdir()

    # Force resolver to return our fake root
    import diagram2code.cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_resolve_registry_dataset_root",
        lambda *_args, **_kwargs: fake_root,
    )

    # Intercept make_predictor call and assert it receives resolved path
    import diagram2code.benchmark.predictor_backends as pb

    def _fake_make_predictor(name: str, dataset_path: str, out_dir=None):
        assert name == "oracle"
        assert dataset_path == str(fake_root)
        # Return anything; benchmark will proceed until it tries to load dataset.json,
        # but we can stop early by raising SystemExit(0).
        raise SystemExit(0)

    monkeypatch.setattr(pb, "make_predictor", _fake_make_predictor)

    rc = main(["benchmark", "--dataset", "flowlearn", "--predictor", "oracle"])
    assert rc == 0
