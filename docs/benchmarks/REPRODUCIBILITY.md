# Benchmark Reproducibility Checklist

This checklist standardizes how `diagram2code benchmark` results are produced and shared so that
runs are comparable across machines and over time.

This document complements:
- `docs/benchmarks/RESULT_SCHEMA.md` (result JSON schema v1)
- `docs/benchmarks/LEADERBOARD_FORMAT.md` (leaderboard row format v1)

## 1. What you must record (minimum)

### Command used
Record the exact command line:

```bash
python -m diagram2code.cli benchmark --dataset <REF> --split <SPLIT> --predictor <PRED> --json <FILE>
```

Include any extra flags:

- `--alpha`
- `--limit`
- predictor-specific flags such as `--predictor-out` (if used)

### Dataset identity
Record:

- Dataset reference string used (e.g. `example:minimal_v1`, `flowlearn:simflowchart_legacy`)
- The split evaluated (`test`, `train`, etc.)
- Number of evaluated samples (`num_samples` in JSON)

If the dataset is local and not publicly available, also record:
- dataset root path (for your own records)
-  how it was obtained / generated

### Predictor identity
Record:
- Predictor name (e.g. `oracle`, `heuristic`, `vision`)
- Any predictor configuration (if applicable)

### Result JSON
Store the result JSON produced by `--json` unchanged.
This is the source of truth for metrics and leaderboard aggregation.

## 2. Strongly recommended metadata (for cross-machine comparison)

To compare results across machines, also record:
- `diagram2code_version` (installed package version)
- `git_sha` (if running from a git checkout)
- `python` version
- OS / platform identifier

These should be placed under `run.*` in the result JSON when available.

## 3. Vision predictor notes
If using `predictor=vision`, also record:
- OpenCV version
- Any OCR dependencies used (pytesseract + system tesseract version) if OCR is enabled
- Any non-default preprocessing parameters (if applicable)

## 4. Determinism
Prefer deterministic predictors (`oracle`, `heuristic`) for baseline comparisons.
- If a predictor uses randomness, record:
    - random seed
    - any nondeterministic hardware flags

## 5. Recommended file layout
A suggested structure for keeping results organized:
```text
results/
  <dataset>/
    <split>/
      <predictor>/
        run-<timestamp>/
          result.json
          artifacts/   (optional)
```
Example:
```text
results/
  example_minimal_v1/
    test/
      oracle/
        run-2026-02-08T12-00-00Z/
          result.json
```    

## 6. Quick verification steps
Before sharing results, run:
```bash
ruff check .
pytest -q
python -m diagram2code.cli benchmark --dataset <REF> --predictor <PRED> --json outputs/result.json
python -m diagram2code.cli leaderboard --input outputs/result.json --out outputs/leaderboard.csv
```

## 7. Checklist (copy/paste)
- [x] Command recorded (full CLI)
- [ ] Dataset reference recorded
- [ ] Split recorded
- [ ] Predictor recorded + config recorded
- [ ] Result JSON saved (schema v1)
- [ ] diagram2code_version recorded (recommended)
- [ ] git_sha recorded (recommended)
- [ ] Platform + Python recorded (recommended)
- [ ] If vision: OpenCV/OCR versions recorded (recommended)

## 8. Strict Reproducibility Mode

To ensure benchmarking is performed only on properly installed datasets
(with a verified `manifest.json`), use:

```bash
diagram2code benchmark \
  --dataset flowlearn \
  --predictor oracle \
  --fail-on-missing-manifest
```
If the dataset root does not contain a `manifest.json`,
the command will exit with error code 2.

This prevents benchmarking on ad-hoc folders and improves
reproducibility guarantees.