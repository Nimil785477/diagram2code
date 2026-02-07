

# Heuristic Predictor

`heuristic` is a deterministic baseline predictor intended for fast smoke tests and comparisons.

## What it does

- Produces graph predictions without ML inference.
- Intended as a baseline, not as a high-accuracy model.

## Determinism

- Deterministic by design (no randomness).
- Output should be stable across runs for the same input.

## CLI usage

```bash
python -m diagram2code.cli benchmark \
  --dataset flowlearn:simflowchart_legacy \
  --split test \
  --predictor heuristic \
  --json outputs/bench_heuristic.json
```