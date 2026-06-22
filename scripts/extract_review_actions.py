from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.eval_artifacts import write_json
from bofip_agentic.eval_schema import ReviewAction, as_jsonable


ACTION_RE = re.compile(r"^-\s*(?:\[(?P<severity>[^\]]+)\])?(?:\[(?P<area>[^\]]+)\])?\s*(?P<body>.+)$")


def extract_actions(review_text: str) -> list[ReviewAction]:
    if "END_OF_RESPONSE" not in review_text:
        raise ValueError("Review is incomplete: missing END_OF_RESPONSE")

    actions: list[ReviewAction] = []
    in_section = False

    for raw_line in review_text.splitlines():
        line = raw_line.strip()
        heading = line.lower().lstrip("# ").strip()

        if heading == "recommended next fixes":
            in_section = True
            continue
        if in_section and line.startswith("##"):
            break
        if not in_section:
            continue

        match = ACTION_RE.match(line)
        if not match:
            continue

        body = match.group("body").strip()
        actions.append(
            ReviewAction(
                severity=(match.group("severity") or "medium").strip().lower(),
                area=(match.group("area") or "general").strip().lower(),
                title=body[:80],
                recommendation=body,
            )
        )

    return actions


def write_actions_markdown(path: str | Path, actions: list[ReviewAction]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Review Actions", ""]
    lines.extend(f"- [{action.severity}][{action.area}] {action.recommendation}" for action in actions)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract action items from a ChatGPT review.")
    parser.add_argument("review_path")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args(argv)

    review_path = Path(args.review_path)
    actions = extract_actions(review_path.read_text(encoding="utf-8"))

    output_json = Path(args.output_json) if args.output_json else review_path.with_name("review_actions.json")
    output_md = Path(args.output_md) if args.output_md else review_path.with_name("review_actions.md")

    write_json(output_json, [as_jsonable(action) for action in actions])
    write_actions_markdown(output_md, actions)
    print(f"Actions: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
