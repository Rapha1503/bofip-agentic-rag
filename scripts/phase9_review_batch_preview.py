from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import write_json
from bofip_cleanroom.llm_preview import review_batch_preview_payload
from bofip_cleanroom.settings import REPORTS_DIR


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _default_output_path(input_path: Path) -> Path:
    filename = input_path.name
    if "preview_eval" in filename:
        return input_path.with_name(filename.replace("preview_eval", "review"))
    return input_path.with_name(input_path.stem + "__review.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Review a phase9 batch preview report with the local structured-answer validator.")
    parser.add_argument("--input", type=str, default=str(REPORTS_DIR / "phase9_batch_preview_eval_gemini_v1.json"))
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    batch_payload = _load_json(input_path)
    review_payload = review_batch_preview_payload(
        {
            **batch_payload,
            "source_report": str(input_path),
        }
    )
    output_path = Path(args.output).resolve() if args.output else _default_output_path(input_path)
    write_json(output_path, review_payload)
    print(f"Phase9 batch review written to: {output_path}")
    print(f"case_count = {review_payload['case_count']}")
    print(f"format_valid_count = {review_payload['format_valid_count']}")
    print(f"format_invalid_count = {review_payload.get('format_invalid_count', 0)}")
    print(f"provider_rate_limit_count = {review_payload.get('provider_rate_limit_count', 0)}")
    print(f"provider_timeout_count = {review_payload.get('provider_timeout_count', 0)}")
    print(f"provider_internal_error_count = {review_payload.get('provider_internal_error_count', 0)}")
    print(f"missing_api_key_count = {review_payload.get('missing_api_key_count', 0)}")
    print(f"runtime_error_count = {review_payload.get('runtime_error_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
