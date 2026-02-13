# Benchmark Result JSON Schema (v1)

This document freezes the JSON format written by `diagram2code benchmark --json ...`.
## Schema Version

Current schema version: `1.1`

Version history:

- `1`   — Initial frozen benchmark result contract.
- `1.1` — Added improved CLI provenance handling and stricter reproducibility support
          (top-level dataset/split/predictor now reliably populated from CLI metadata).

Backward compatibility:
- The CLI can read and summarize both `1` and `1.1` result files.
- Writers always emit the current `SCHEMA_VERSION`.

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
  "schema_version": "1.1",
  "dataset": "example:minimal_v1",
  "split": "test",
  "predictor": "oracle",
  "num_samples": 3,
  "metrics": {
    "node_precision": 1.0,
    "node_recall": 1.0,
    "node_f1": 1.0,
    "edge_precision": 1.0,
    "edge_recall": 1.0,
    "edge_f1": 1.0,
    "direction_accuracy": 1.0,
    "exact_match_rate": 1.0
  },
  "run": {
    "timestamp_utc": "2026-02-13T18:35:33Z",
    "diagram2code_version": "0.1.6",
    "git_sha": "abc1234",
    "python": "3.11",
    "platform": "Windows-10-10.0.19045",
    "cli": "diagram2code benchmark",
    "dataset_ref": "example:minimal_v1",
    "dataset_root": "/abs/path/to/cache",
    "predictor_out": null,
    "dataset_manifest_sha256": "..."
  }
}
```

## Inspecting Result Files

Use the CLI to inspect benchmark JSON files:

```bash
diagram2code benchmark info outputs/result.json
```
Example Output:
```yaml
schema_version: 1.1
dataset: example:minimal_v1
split: test
predictor: oracle
num_samples: 3

metrics:
  node_f1: 1.0
  edge_f1: 1.0
  direction_accuracy: 1.0
  exact_match_rate: 1.0
```