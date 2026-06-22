from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_evidence_cards(run_dir: Path, *, max_cards: int = 8, preferred_ids: list[str] | None = None) -> list[Path]:
    cards_dir = run_dir / "evidence_cards"
    cards = sorted(cards_dir.glob("*.md"))
    preferred: list[Path] = []
    if preferred_ids:
        by_stem = {path.stem: path for path in cards}
        preferred = [by_stem[qid] for qid in preferred_ids if qid in by_stem]
    remaining = [path for path in cards if path not in preferred]
    return (preferred + remaining)[:max_cards]


def build_review_context(run_id: str, summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BOFiP Agentic RAG Review Context",
            "",
            f"Run id: {run_id}",
            f"Total queries: {summary.get('total_queries', 0)}",
            "Do not assume gold labels were shown to the runtime.",
            "Review retrieval, source selection, answer grounding, and overfit risk.",
        ]
    )


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


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_packet(
    run_dir: Path,
    *,
    max_cards: int,
    preferred_ids: list[str] | None,
) -> str:
    summary = _load_summary(run_dir)
    run_id = str(summary.get("run_id") or run_dir.name)
    context = build_review_context(run_id=run_id, summary=summary)
    card_paths = select_evidence_cards(run_dir, max_cards=max_cards, preferred_ids=preferred_ids)
    cards = [path.read_text(encoding="utf-8") for path in card_paths]
    prompt = build_review_prompt(cards, run_id=run_id)
    return f"{context}\n\n---\n\n{prompt}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a ChatGPT review prompt packet from a BOFiP eval run.")
    parser.add_argument("run_dir", type=Path, help="Evaluation run directory containing summary.json and evidence_cards/.")
    parser.add_argument("--max-cards", type=int, default=8, help="Maximum number of evidence cards to include.")
    parser.add_argument("--preferred-id", action="append", default=[], help="Evidence card stem to prioritize; repeatable.")
    parser.add_argument("--output", type=Path, help="Output prompt file. Defaults to RUN_DIR/review_prompt.md.")
    args = parser.parse_args()

    output = args.output or args.run_dir / "review_prompt.md"
    packet = _build_packet(args.run_dir, max_cards=args.max_cards, preferred_ids=args.preferred_id)
    output.write_text(packet, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
