from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_agentic.eval_artifacts import assert_no_secrets


def select_evidence_cards(run_dir: Path, *, max_cards: int = 8, preferred_ids: list[str] | None = None) -> list[Path]:
    cards_dir = run_dir / "evidence_cards"
    cards = sorted(cards_dir.glob("*.md"))
    preferred: list[Path] = []
    if preferred_ids:
        by_stem = {path.stem: path for path in cards}
        preferred = [by_stem[qid] for qid in preferred_ids if qid in by_stem]
    remaining = [path for path in cards if path not in preferred]
    return (preferred + remaining)[:max_cards]


def build_review_context(run_id: str, summary: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    lines = [
        "# BOFiP Agentic RAG Review Context",
        "",
        "Project identity: BOFiP Agentic RAG, a retrieval-augmented assistant for BOFiP fiscal questions.",
        f"Run id: {run_id}",
        f"Total queries: {summary.get('total_queries', 0)}",
    ]
    if config:
        lines.extend(["", "## Config"])
        for key in sorted(config):
            value = config[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Constraints",
            "- ChatGPT reviewer-only: review the packet; do not act as the runtime or claim live BOFiP lookup.",
            "- Do not assume gold labels were shown to the runtime.",
            "- distinguish retrieval failures from generation failures.",
            "- avoid fiscal hardcoding in proposed fixes.",
            "- mark uncertainty when evidence is incomplete or ambiguous.",
        ]
    )
    return "\n".join(lines)


def build_review_prompt(cards: list[str], *, run_id: str) -> str:
    joined_cards = "\n\n---\n\n".join(cards)
    return f"""Review BOFiP Agentic RAG run {run_id}.

Required sections exactly:
- Verdict
- Remaining blockers
- Recommended next fixes
- Minimal validation set
- Overfit and leakage risks

Review instructions:
- separate retrieval failures from generation failures.
- flag overfit and leakage risks.
- avoid fiscal hardcoding in any proposed fixes.
- propose validation cases from different BOFiP families.
- mark uncertain claims clearly.

Evidence cards:

{joined_cards}

End your answer with exactly:
END_OF_RESPONSE
"""


def sanitize_card_for_chatgpt(card: str) -> str:
    """Keep source metadata for review while omitting raw BOFiP snippets."""
    lines: list[str] = []
    in_sources = False
    snippet_omitted_for_source = False

    for line in card.splitlines():
        if line.startswith("## "):
            in_sources = line.strip().lower() == "## sources retenues"
            snippet_omitted_for_source = False
            lines.append(line)
            continue

        if not in_sources:
            lines.append(line)
            continue

        stripped = line.strip()
        if line.startswith("### "):
            snippet_omitted_for_source = False
            lines.append(line)
        elif not stripped or stripped.startswith("- "):
            lines.append(line)
        elif not snippet_omitted_for_source:
            lines.append("[Snippet omitted from ChatGPT review packet.]")
            snippet_omitted_for_source = True

    return "\n".join(lines)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return _load_json_object(run_dir / "summary.json")


def _load_config(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    config = _load_json_object(run_dir / "config.json")
    if config:
        return config
    embedded = summary.get("config")
    if isinstance(embedded, dict):
        return embedded
    return {}


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return payload
    return {}


def _build_packet_parts(
    run_dir: Path,
    *,
    max_cards: int,
    preferred_ids: list[str] | None,
) -> tuple[str, str]:
    summary = _load_summary(run_dir)
    config = _load_config(run_dir, summary)
    run_id = str(summary.get("run_id") or run_dir.name)
    context = build_review_context(run_id=run_id, summary=summary, config=config)
    card_paths = select_evidence_cards(run_dir, max_cards=max_cards, preferred_ids=preferred_ids)
    cards = [sanitize_card_for_chatgpt(path.read_text(encoding="utf-8")) for path in card_paths]
    prompt = build_review_prompt(cards, run_id=run_id)
    assert_no_secrets(context)
    assert_no_secrets(prompt)
    return context, prompt


def _write_review_packet(run_dir: Path, context: str, prompt: str) -> tuple[Path, Path]:
    review_dir = run_dir / "chatgpt-review"
    context_path = review_dir / "context.md"
    prompts_path = review_dir / "prompts.md"
    review_dir.mkdir(parents=True, exist_ok=True)
    context_path.write_text(context + "\n", encoding="utf-8")
    prompts_path.write_text(prompt + "\n", encoding="utf-8")
    return context_path, prompts_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a ChatGPT review prompt packet from a BOFiP eval run.")
    parser.add_argument(
        "run_dir_pos",
        nargs="?",
        type=Path,
        help="Evaluation run directory containing summary.json and evidence_cards/.",
    )
    parser.add_argument("--run-dir", type=Path, help="Evaluation run directory containing summary.json and evidence_cards/.")
    parser.add_argument("--max-cards", type=int, default=8, help="Maximum number of evidence cards to include.")
    parser.add_argument("--preferred-id", action="append", default=[], help="Evidence card stem to prioritize; repeatable.")
    parser.add_argument("--output", type=Path, help="Compatibility output file for the combined packet.")
    args = parser.parse_args()

    run_dir = args.run_dir or args.run_dir_pos
    if run_dir is None:
        parser.error("run_dir is required; pass --run-dir RUN_DIR")
    context, prompt = _build_packet_parts(run_dir, max_cards=args.max_cards, preferred_ids=args.preferred_id)
    context_path, prompts_path = _write_review_packet(run_dir, context, prompt)
    if args.output:
        packet = f"{context}\n\n---\n\n{prompt}"
        assert_no_secrets(packet)
        args.output.write_text(packet, encoding="utf-8")
    print(context_path)
    print(prompts_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
