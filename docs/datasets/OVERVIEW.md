# Datasets Overview

diagram2code supports **installed datasets** in a local cache, plus **remote datasets** that can be fetched on demand.

## Quick start

List available remote datasets:

```bash
diagram2code dataset list
```
Inspect a dataset descriptor:
```bash
diagram2code dataset info <name>
```
Fetch into local cache:
```bash
diagram2code dataset fetch <name> [--yes]
```
Verify installation:
```bash
diagram2code dataset verify <name>
```
Print local install path:
```bash
diagram2code dataset path <name>
```
## Cache location
By default, datasets are installed under the platform cache directory:

- `{user_cache_dir}/diagram2code/datasets/<name>/<version>/`

You can override the cache root:
```bash
# Windows PowerShell
$env:DIAGRAM2CODE_CACHE_DIR="D:\diagram2code_cache"
diagram2code dataset fetch tiny_remote_v1

# macOS/Linux
export DIAGRAM2CODE_CACHE_DIR="$HOME/diagram2code_cache"
diagram2code dataset fetch tiny_remote_v1
```
## Benchmark integration
You can benchmark using:
- `example:*` built-in datasets
- explicit filesystem dataset paths
- remote dataset names (installed or fetched)

Examples:
```bash
# Built-in example dataset
diagram2code benchmark --dataset example:minimal_v1 --predictor oracle

# Remote dataset already installed
diagram2code benchmark --dataset tiny_remote_v1 --predictor oracle

# Fetch remote dataset automatically if missing
diagram2code benchmark --dataset tiny_remote_v1 --predictor oracle --fetch-missing --yes
```
## Safety
Datasets that download from Hugging Face snapshots may be large.
Fetching them requires explicit confirmation:
```bash
diagram2code dataset fetch flowlearn --yes
# or:
diagram2code benchmark --dataset flowlearn --fetch-missing --yes
```
