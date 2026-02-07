# Benchmark Result JSON Schema (v1)

This document freezes the JSON format written by `diagram2code benchmark --json ...`.

## Versioning

- `schema_version` is a string.
- This document describes schema version `"1"`.
- Within schema `"1"`, existing fields will not be removed or renamed.
- New fields may only be added in a backwards-compatible way (prefer `run`).

## Top-level fields

| Field | Type | Required | Notes |
|---|---:|:---:|---|
| `schema_version` | string | ✅ | Always `"1"` for this schema |
| `dataset` | string | ✅ | Dataset reference or resolved root identifier |
| `split` | string | ✅ | Evaluated split (e.g. `test`, `train`, `all`) |
| `predictor` | string | ✅ | Predictor name (e.g. `oracle`, `heuristic`, `vision`) |
| `num_samples` | int | ✅ | Number of evaluated samples |
| `metrics` | object | ✅ | Flat mapping `metric_name -> float` |
| `run` | object | ❌ | Optional metadata for reproducibility/debug |

## Metrics naming

Metrics are a flat map of floats. Current keys (schema v1):

- `node_precision`, `node_recall`, `node_f1`
- `edge_precision`, `edge_recall`, `edge_f1`
- `direction_accuracy`
- `exact_match_rate`
- `runtime_mean_s`

Additional metric keys may be added later (still within v1).

## Example

```json
{
  "schema_version": "1",
  "dataset": "flowlearn:simflowchart_legacy",
  "split": "test",
  "predictor": "heuristic",
  "num_samples": 120,
  "metrics": {
    "direction_accuracy": 0.83,
    "edge_f1": 0.71,
    "edge_precision": 0.74,
    "edge_recall": 0.69,
    "exact_match_rate": 0.42,
    "node_f1": 0.88,
    "node_precision": 0.90,
    "node_recall": 0.86,
    "runtime_mean_s": 0.031
  },
  "run": {
    "platform": "Windows-10-10.0.22631-SP0",
    "python": "3.11.7",
    "timestamp_utc": "2026-02-05T13:40:12Z"
  }
}
```