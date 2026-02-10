from __future__ import annotations

import sys
from types import SimpleNamespace

from diagram2code.datasets.fetching.fetcher import _hf_snapshot_download


def test_hf_snapshot_reports_files_and_total(monkeypatch, tmp_path, capsys):
    dest = tmp_path / "hf"
    repo_id = "org/repo"
    rev = "abc123"

    # Fake huggingface_hub module API
    class _Sibling:
        def __init__(self, size: int):
            self.size = size

    class _FakeHfApi:
        def repo_info(self, repo_id, repo_type, revision, files_metadata):
            assert repo_id == "org/repo"
            assert repo_type == "dataset"
            assert revision == "abc123"
            assert files_metadata is True
            return SimpleNamespace(siblings=[_Sibling(100), _Sibling(924)])

    calls = {}

    def _fake_snapshot_download(**kwargs):
        calls["kwargs"] = kwargs
        # simulate download side effects
        dest.mkdir(parents=True, exist_ok=True)

    fake_mod = SimpleNamespace(HfApi=_FakeHfApi, snapshot_download=_fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    _hf_snapshot_download(repo_id, rev, dest)
    out = capsys.readouterr().out

    assert "Fetching HF snapshot: org/repo@abc123" in out
    assert "Files: 2" in out
    assert "Total:" in out

    kwargs = calls["kwargs"]
    assert kwargs["repo_id"] == repo_id
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["revision"] == rev
    assert kwargs["local_dir"] == str(dest)
