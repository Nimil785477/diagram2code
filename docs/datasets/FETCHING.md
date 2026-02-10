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

## Fetch a dataset
```bash
diagram2code dataset fetch <name>
```
Some datasets may be large (e.g., Hugging Face snapshots). For these, you must confirm:
```bash
diagram2code dataset fetch <name> --yes
```
## Verify a dataset
```bash
diagram2code dataset verify <name>
```
Verification checks that a manifest exists and matches the expected dataset identity.

## Locate installed datasets
```bash
diagram2code dataset path <name>
```
## Cache override
Set `DIAGRAM2CODE_CACHE_DIR` to override the cache root.

Windows PowerShell:
```powershell
$env:DIAGRAM2CODE_CACHE_DIR="D:\diagram2code_cache"
diagram2code dataset fetch tiny_remote_v1
diagram2code dataset verify tiny_remote_v1
```
macOS/Linux:
```bash
export DIAGRAM2CODE_CACHE_DIR="$HOME/diagram2code_cache"
diagram2code dataset fetch tiny_remote_v1
diagram2code dataset verify tiny_remote_v1
```
## Common issues
**“Manifest not found or unreadable”**
- The dataset was not fetched yet, or
- You are verifying under a different cache directory than where you fetched.

Fix:

- Ensure `DIAGRAM2CODE_CACHE_DIR` is set in the current shell
- Re-run `dataset fetch`, then `dataset verify`

**Hugging Face downloads are slow**

HF snapshot downloads are network + file-count heavy by nature.
Use `--yes` to confirm and allow the fetch to proceed.