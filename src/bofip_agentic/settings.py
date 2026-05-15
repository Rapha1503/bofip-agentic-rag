from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
REPORTS_DIR = DATA_DIR / "reports"

# Where the BOFIP corpus lives (raw_docs JSONL, chunks JSONL, .npy caches, models)
# Default: project's own data/ directory. Override with BOFIP_DATA_ROOT env var.
_DATA_ROOT = os.environ.get("BOFIP_DATA_ROOT", "")
if _DATA_ROOT:
    BOFIP_DATA_ROOT = Path(_DATA_ROOT).expanduser().resolve()
else:
    BOFIP_DATA_ROOT = PROJECT_ROOT


def get_raw_bofip_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        env_value = os.environ.get("RAW_BOFIP_ROOT", "").strip()
        if not env_value:
            raise RuntimeError(
                "RAW_BOFIP_ROOT is not configured. "
                "Set the environment variable or pass --raw-root explicitly."
            )
        path = Path(env_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RAW_BOFIP_ROOT does not exist: {path}")
    return path


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, INTERIM_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
