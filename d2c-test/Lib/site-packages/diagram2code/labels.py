from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict


def load_labels(path: str | Path) -> Dict[int, str]:
    """
    Load labels.json of form: {"0": "Step_1_Load_Data", "1": "Step_2_Train_Model"}
    Returns {0: "...", 1: "..."}.
    """
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except ValueError:
            continue
    return out


def to_valid_identifier(label: str, fallback: str) -> str:
    """
    Convert Style-2 labels into valid Python identifiers.
    - keeps underscores
    - replaces spaces/dashes with underscores
    - removes other invalid chars
    - ensures doesn't start with digit
    """
    s = label.strip()
    if not s:
        return fallback

    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^0-9a-zA-Z_]", "", s)

    s = re.sub(r"_+", "_", s)
    
    if not s:
        return fallback

    if s[0].isdigit():
        s = "_" + s

    return s
