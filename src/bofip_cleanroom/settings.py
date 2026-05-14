from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
REPORTS_DIR = DATA_DIR / "reports"


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
