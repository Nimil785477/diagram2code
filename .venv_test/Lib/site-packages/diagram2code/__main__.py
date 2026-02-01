from __future__ import annotations

def main() -> None:
    # Import here to avoid side-effects at import time
    from .cli import main as cli_main
    cli_main()

if __name__ == "__main__":
    main()
