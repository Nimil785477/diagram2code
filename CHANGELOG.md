# Changelog

## 0.1.4
- Add graph rendering options (--render-graph, --render-layout, --render-format).
- Add --no-debug to suppress debug artifacts.
- Improve arrow direction detection (incl. diagonal support) and add tests.
- Documentation updates for new CLI flags and examples.

## 0.1.0
- End-to-end pipeline: preprocess → node detection → arrow detection → graph export → program export
- Optional OCR label extraction via `--extract-labels`
- Optional export bundle via `--export`
- Debug overlays: `debug_nodes.png`, `debug_arrows.png`
- CI installs Tesseract for OCR tests (Ubuntu)
- Windows-safe CLI printing
