# Changelog

## 0.1.8

### Added
- `diagram2code dataset build synthflow` local synthetic dataset builder.
- SynthFlow v2 generation with:
  - layout variation
  - decision-node rendering
  - deterministic bbox jitter
- `naive` predictor as a weak deterministic baseline.
- Benchmark robustness metrics:
  - `node_count_error`
  - `edge_count_error`
- CLI summary output for `dataset build synthflow`:
  - generator
  - split
  - sample count
  - seed

### Changed
- Benchmark result JSON now includes `node_count_error` and `edge_count_error`.
- Benchmark CLI output now prints count-error metrics alongside existing aggregate metrics.
- Predictor discovery now includes `naive`.
- SynthFlow benchmark generation is now more meaningful while preserving deterministic seeded output and Phase-3 dataset compatibility.

### Quality
- Added SynthFlow v2 generation and determinism coverage.
- Added naive predictor benchmark coverage.
- Test suite expanded; all tests passing (127 tests).
- Package build verified with `python -m build` and `python -m twine check dist/*`.

## 0.1.6 - 0.1.7

### Added
- `diagram2code benchmark info` command for inspecting result JSON files.
- `--fail-on-missing-manifest` strict reproducibility mode.
- Manifest summary fields in `dataset info`.
- Improved provenance handling in benchmark JSON (schema v1.1).

### Changed
- Result schema bumped to 1.1.
- Top-level dataset/split/predictor fields now reliably populated.

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
