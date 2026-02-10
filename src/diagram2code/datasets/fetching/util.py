from __future__ import annotations


def format_bytes(n: int | None) -> str:
    if n is None or n < 0:
        return "unknown size"

    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    i = 0
    while f >= 1024.0 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"
