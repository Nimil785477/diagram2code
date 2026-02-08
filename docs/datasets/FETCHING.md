# Dataset Fetching

This document describes how diagram2code fetches external datasets into a local cache.

## Cache layout

`{cache_root}/{name}/{version}/`

- `manifest.json` — reproducibility record (schema v1)
- `raw/` — downloaded artifacts
- `prepared/` — normalized dataset in diagram2code Dataset format (future step)

## Environment variables

- `DIAGRAM2CODE_CACHE_DIR`: override the cache base directory. The datasets cache root becomes `{DIAGRAM2CODE_CACHE_DIR}/datasets`.

## Remote dataset registry

Built-in remote datasets are pinned and versioned in code. Updating the "latest" dataset requires a new diagram2code release.
