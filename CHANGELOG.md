# Changelog

## 0.1.0
- End-to-end pipeline: preprocess → node detection → arrow detection → graph export → program export
- Optional OCR label extraction via `--extract-labels`
- Optional export bundle via `--export`
- Debug overlays: `debug_nodes.png`, `debug_arrows.png`
- CI installs Tesseract for OCR tests (Ubuntu)
- Windows-safe CLI printing
