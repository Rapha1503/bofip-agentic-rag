from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.artifact_download import (  # noqa: E402
    download_missing_runtime_artifacts,
    validate_runtime_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate BOFiP Agentic RAG runtime artifacts.")
    parser.add_argument("--base-url", help="Artifact base URL. Defaults to the GitHub release URL.")
    args = parser.parse_args()

    downloaded = download_missing_runtime_artifacts(PROJECT_ROOT, base_url=args.base_url)
    for path in downloaded:
        print(f"downloaded {path.relative_to(PROJECT_ROOT).as_posix()}")

    errors = validate_runtime_artifacts(PROJECT_ROOT)
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 1
    print("runtime artifacts ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

