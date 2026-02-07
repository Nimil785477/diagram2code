# Leaderboard Format (v1)

This document defines a stable, comparable format for aggregating `diagram2code benchmark` result JSONs
(schema v1) into a leaderboard table.

One benchmark run = one leaderboard row.

## Inputs

- Benchmark result JSONs written by:
  - `diagram2code benchmark --json <path>`
- Must conform to schema v1 documented in `RESULT_SCHEMA.md`.

## Row schema (v1)

Columns are flat, machine-friendly, and map directly from result JSON fields.

### Identity
- `timestamp_utc` (string) — from `run.timestamp_utc` if present, else empty
- `dataset` (string) — from `dataset`
- `split` (string) — from `split`
- `predictor` (string) — from `predictor`
- `schema_version` (string) — from `schema_version`
- `num_samples` (int) — from `num_samples`

### Primary metrics
- `exact_match_rate` (float) — from `metrics.exact_match_rate`
- `edge_f1` (float) — from `metrics.edge_f1`
- `node_f1` (float) — from `metrics.node_f1`
- `direction_accuracy` (float) — from `metrics.direction_accuracy`

### Runtime (optional)
- `runtime_mean_s` (float) — from `metrics.runtime_mean_s` if present, else empty

### Reproducibility
- `diagram2code_version` (string) — from `run.diagram2code_version` if present, else empty
- `git_sha` (string) — from `run.git_sha` if present, else empty
- `platform` (string) — from `run.platform` if present, else empty
- `python` (string) — from `run.python` if present, else empty

## CSV header (recommended order)

```text
timestamp_utc,dataset,split,predictor,schema_version,num_samples,exact_match_rate,edge_f1,node_f1,direction_accuracy,runtime_mean_s,diagram2code_version,git_sha,platform,python
```

### Missing values
- If a metric or run field is missing, output an empty cell.
- Do not invent values.

### Example Row
```text
2026-02-07T12:34:56Z,example:minimal_v1,test,heuristic,1,120,0.42,0.71,0.88,0.83,0.031,0.1.6,f550d76,Windows-10-10.0.22631-SP0,3.12.1
```

### Stability guarantees
Within v1:
- Column names will not be removed or renamed.
- Additional columns may be appended to the end of the row schema if needed for future work.

