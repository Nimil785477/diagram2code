

# Oracle Predictor

`oracle` is a perfect predictor used to validate the benchmark pipeline and metric implementation.

## What it does

- Returns the dataset ground-truth graph as the prediction.
- Expected to achieve perfect scores (e.g., exact match = 1.0) when the pipeline is correct.

## CLI usage

```bash
python -m diagram2code.cli benchmark \
  --dataset flowlearn:simflowchart_legacy \
  --split test \
  --predictor oracle \
  --json outputs/bench_oracle.json
```