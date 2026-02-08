from __future__ import annotations

from .descriptors import DatasetDescriptor


def built_in_descriptors() -> dict[str, DatasetDescriptor]:
    # Step 6.1: first external dataset comes here (pinned)
    return {}
