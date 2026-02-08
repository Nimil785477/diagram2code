from __future__ import annotations

from pathlib import Path

from diagram2code.datasets.fetching.fetcher import DefaultDownloader


def test_default_downloader_supports_file_url(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")

    # file:// URL (absolute)
    url = src.resolve().as_uri()

    dest = tmp_path / "out.bin"
    DefaultDownloader().download_to_path(url, dest)

    assert dest.read_bytes() == b"hello"
