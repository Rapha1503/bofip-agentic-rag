from __future__ import annotations

import os
from pathlib import Path

from .settings import PROJECT_ROOT


def load_env_file(path: Path | None = None) -> dict[str, str]:
    env_path = (path or (PROJECT_ROOT / ".env")).resolve()
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def load_default_env_files() -> dict[str, str]:
    loaded: dict[str, str] = {}
    for candidate in (PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env"):
        loaded.update(load_env_file(candidate))
    return loaded
