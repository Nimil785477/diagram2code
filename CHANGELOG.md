# Changelog

## 0.1.5
- Freeze benchmark result JSON schema v1 and add docs.
- Add leaderboard aggregation command (CSV/Markdown) and format docs.
- Add reproducibility checklist docs.
- Auto-fill benchmark run metadata (`diagram2code_version`, `git_sha`, `platform`, `python`, `timestamp_utc`).
- Fix `example:minimal_v1` graph to include bbox and add benchmark smoke coverage.

## 0.1.4
- Add graph rendering options (`--render-graph`, `--render-layout`, `--render-format`).
- Add `--no-debug` to suppress debug artifacts.
- Improve arrow direction detection (incl. diagonal support) and add tests.
- Documentation updates for new CLI flags and examples.

## 0.1.0
- End-to-end pipeline: preprocess → node detection → arrow detection → graph export → program export
- Optional OCR label extraction via `--extract-labels`
- Optional export bundle via `--export`
- Debug overlays: `debug_nodes.png`, `debug_arrows.png`
- CI installs Tesseract for OCR tests (Ubuntu)
- Windows-safe CLI printing
