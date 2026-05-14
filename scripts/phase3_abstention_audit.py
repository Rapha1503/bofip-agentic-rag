from __future__ import annotations

import argparse
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.lexical_retrieval import DocumentLexicalIndex, get_document_search_text_fn, tokenize
from bofip_cleanroom.models import chunk_node_from_dict, raw_document_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs
from bofip_cleanroom.two_stage_retrieval import TwoStageLexicalRetriever


def _query_tokens(query: str) -> set[str]:
    return set(tokenize(query))


def _overlap_metrics(query: str, text: str) -> tuple[int, float]:
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return 0, 0.0
    t_tokens = set(tokenize(text))
    overlap = len(q_tokens & t_tokens)
    return overlap, overlap / len(q_tokens)


def _uncovered_ratio(query: str, text: str) -> tuple[int, float]:
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return 0, 0.0
    t_tokens = set(tokenize(text))
    uncovered = len([token for token in q_tokens if token not in t_tokens])
    return uncovered, uncovered / len(q_tokens)


def _expected_behavior(payload: dict) -> str:
    return payload.get("expected_behavior") or ("answer" if payload.get("expected_boi") else "abstain")


def _safe_margin(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return 0.0
    return float(first) - float(second)


def _decision(features: dict, *, title_overlap_min: int, chunk_overlap_min: int, doc_margin_min: float) -> str:
    if features["title_overlap_count"] >= title_overlap_min:
        return "answer"
    if features["chunk_overlap_count"] >= chunk_overlap_min and features["doc_margin"] >= doc_margin_min:
        return "answer"
    return "abstain"


def _decision_uncovered_ratio(features: dict, *, uncovered_ratio_min: float) -> str:
    return "abstain" if features["combined_uncovered_ratio"] >= uncovered_ratio_min else "answer"


def _decision_uncovered_ratio_docpluschunks(features: dict, *, uncovered_ratio_min: float) -> str:
    return "abstain" if features["docplus_top2_uncovered_ratio"] >= uncovered_ratio_min else "answer"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a transparent abstention gate over the clean-room benchmark.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--document-mode", type=str, default="sections_firstpara", choices=["base", "sections", "sections_firstpara"])
    parser.add_argument("--local-strategy", type=str, default="chunk", choices=["chunk", "section_then_chunk"])
    parser.add_argument("--chunk-mode", type=str, default="body", choices=["full", "leaf", "body"])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    queries = read_jsonl(Path(args.queries))

    doc_search_text_fn = get_document_search_text_fn(args.document_mode)
    doc_index = DocumentLexicalIndex(documents, search_text_fn=doc_search_text_fn)
    two_stage = TwoStageLexicalRetriever(
        documents,
        chunks,
        document_mode=args.document_mode,
        local_chunk_mode=args.chunk_mode,
        local_strategy=args.local_strategy,
    )
    doc_map = {document.boi_reference: document for document in documents}

    rows: list[dict] = []
    for payload in queries:
        query = payload["query"]
        expected_behavior = _expected_behavior(payload)
        doc_hits = doc_index.search_documents(query, top_k=3)
        result = two_stage.search(query, top_docs=3, chunks_per_doc=3, max_chunks=6)
        top_doc = doc_hits[0] if doc_hits else None
        second_doc = doc_hits[1] if len(doc_hits) > 1 else None
        top_chunk = result.chunk_hits[0] if result.chunk_hits else None

        same_doc_local_hits = [hit for hit in result.chunk_hits if top_doc and hit.boi_reference == top_doc.boi_reference]
        second_local = same_doc_local_hits[1] if len(same_doc_local_hits) > 1 else None

        top_doc_repr = doc_search_text_fn(doc_map[top_doc.boi_reference]) if top_doc else ""
        title_overlap_count, title_overlap_ratio = _overlap_metrics(query, top_doc.best_chunk.text if top_doc else "")
        doc_repr_overlap_count, doc_repr_overlap_ratio = _overlap_metrics(query, top_doc_repr)
        chunk_overlap_count, chunk_overlap_ratio = _overlap_metrics(query, top_chunk.chunk.text if top_chunk else "")
        combined_text = " ".join(
            part
            for part in [
                top_doc.best_chunk.text if top_doc else "",
                top_chunk.chunk.text if top_chunk else "",
            ]
            if part
        )
        combined_uncovered_count, combined_uncovered_ratio = _uncovered_ratio(query, combined_text)
        docplus_top2_text = " ".join(
            part
            for part in [
                top_doc_repr,
                *[hit.chunk.text for hit in same_doc_local_hits[:2]],
            ]
            if part
        )
        docplus_top2_uncovered_count, docplus_top2_uncovered_ratio = _uncovered_ratio(query, docplus_top2_text)

        rows.append(
            {
                "id": payload["id"],
                "pattern": payload.get("pattern"),
                "query": query,
                "expected_boi": payload.get("expected_boi"),
                "expected_behavior": expected_behavior,
                "top_doc_boi": top_doc.boi_reference if top_doc else None,
                "top_doc_score": round(top_doc.score, 4) if top_doc else None,
                "second_doc_boi": second_doc.boi_reference if second_doc else None,
                "second_doc_score": round(second_doc.score, 4) if second_doc else None,
                "doc_margin": round(_safe_margin(top_doc.score if top_doc else None, second_doc.score if second_doc else None), 4),
                "title_overlap_count": title_overlap_count,
                "title_overlap_ratio": round(title_overlap_ratio, 4),
                "doc_repr_overlap_count": doc_repr_overlap_count,
                "doc_repr_overlap_ratio": round(doc_repr_overlap_ratio, 4),
                "top_chunk_id": top_chunk.chunk.chunk_id if top_chunk else None,
                "top_chunk_boi": top_chunk.boi_reference if top_chunk else None,
                "top_chunk_local_score": round(top_chunk.local_score, 4) if top_chunk else None,
                "second_local_score_same_doc": round(second_local.local_score, 4) if second_local else None,
                "local_margin_same_doc": round(_safe_margin(top_chunk.local_score if top_chunk else None, second_local.local_score if second_local else None), 4),
                "chunk_overlap_count": chunk_overlap_count,
                "chunk_overlap_ratio": round(chunk_overlap_ratio, 4),
                "combined_uncovered_count": combined_uncovered_count,
                "combined_uncovered_ratio": round(combined_uncovered_ratio, 4),
                "docplus_top2_uncovered_count": docplus_top2_uncovered_count,
                "docplus_top2_uncovered_ratio": round(docplus_top2_uncovered_ratio, 4),
                "top_chunk_text": top_chunk.chunk.text[:320] if top_chunk else None,
            }
        )

    candidate_rules: list[dict] = []
    for title_overlap_min, chunk_overlap_min, doc_margin_min in product([1, 2, 3], [1, 2, 3], [0.0, 0.5, 1.0, 2.0, 4.0]):
        decisions = [
            _decision(
                row,
                title_overlap_min=title_overlap_min,
                chunk_overlap_min=chunk_overlap_min,
                doc_margin_min=doc_margin_min,
            )
            for row in rows
        ]
        tp = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "abstain" and decision == "abstain")
        tn = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "answer" and decision == "answer")
        fp = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "answer" and decision == "abstain")
        fn = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "abstain" and decision == "answer")
        abstain_precision = tp / (tp + fp) if (tp + fp) else 0.0
        abstain_recall = tp / (tp + fn) if (tp + fn) else 0.0
        answer_recall = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = (tp + tn) / len(rows) if rows else 0.0
        candidate_rules.append(
            {
                "title_overlap_min": title_overlap_min,
                "chunk_overlap_min": chunk_overlap_min,
                "doc_margin_min": doc_margin_min,
                "accuracy": round(accuracy, 4),
                "abstain_precision": round(abstain_precision, 4),
                "abstain_recall": round(abstain_recall, 4),
                "answer_recall": round(answer_recall, 4),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    for uncovered_ratio_min in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        decisions = [_decision_uncovered_ratio(row, uncovered_ratio_min=uncovered_ratio_min) for row in rows]
        tp = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "abstain" and decision == "abstain")
        tn = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "answer" and decision == "answer")
        fp = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "answer" and decision == "abstain")
        fn = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "abstain" and decision == "answer")
        abstain_precision = tp / (tp + fp) if (tp + fp) else 0.0
        abstain_recall = tp / (tp + fn) if (tp + fn) else 0.0
        answer_recall = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = (tp + tn) / len(rows) if rows else 0.0
        candidate_rules.append(
            {
                "rule_family": "combined_uncovered_ratio",
                "uncovered_ratio_min": uncovered_ratio_min,
                "accuracy": round(accuracy, 4),
                "abstain_precision": round(abstain_precision, 4),
                "abstain_recall": round(abstain_recall, 4),
                "answer_recall": round(answer_recall, 4),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    for uncovered_ratio_min in [0.2, 0.3, 0.4, 0.5, 0.6]:
        decisions = [_decision_uncovered_ratio_docpluschunks(row, uncovered_ratio_min=uncovered_ratio_min) for row in rows]
        tp = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "abstain" and decision == "abstain")
        tn = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "answer" and decision == "answer")
        fp = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "answer" and decision == "abstain")
        fn = sum(1 for row, decision in zip(rows, decisions) if row["expected_behavior"] == "abstain" and decision == "answer")
        abstain_precision = tp / (tp + fp) if (tp + fp) else 0.0
        abstain_recall = tp / (tp + fn) if (tp + fn) else 0.0
        answer_recall = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = (tp + tn) / len(rows) if rows else 0.0
        candidate_rules.append(
            {
                "rule_family": "docplus_top2_uncovered_ratio",
                "uncovered_ratio_min": uncovered_ratio_min,
                "accuracy": round(accuracy, 4),
                "abstain_precision": round(abstain_precision, 4),
                "abstain_recall": round(abstain_recall, 4),
                "answer_recall": round(answer_recall, 4),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    candidate_rules = sorted(
        candidate_rules,
        key=lambda row: (row["accuracy"], row["abstain_precision"], row["abstain_recall"], row["answer_recall"]),
        reverse=True,
    )
    best_rule = candidate_rules[0]

    labeled_rows = []
    for row in rows:
        if best_rule.get("rule_family") == "combined_uncovered_ratio":
            decision = _decision_uncovered_ratio(row, uncovered_ratio_min=best_rule["uncovered_ratio_min"])
        elif best_rule.get("rule_family") == "docplus_top2_uncovered_ratio":
            decision = _decision_uncovered_ratio_docpluschunks(row, uncovered_ratio_min=best_rule["uncovered_ratio_min"])
        else:
            decision = _decision(
                row,
                title_overlap_min=best_rule["title_overlap_min"],
                chunk_overlap_min=best_rule["chunk_overlap_min"],
                doc_margin_min=best_rule["doc_margin_min"],
            )
        enriched = dict(row)
        enriched["predicted_behavior"] = decision
        enriched["decision_correct"] = decision == row["expected_behavior"]
        labeled_rows.append(enriched)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "chunks_path": str(Path(args.chunks).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "document_mode": args.document_mode,
        "local_strategy": args.local_strategy,
        "chunk_mode": args.chunk_mode,
        "query_count": len(rows),
        "behavior_counts": {
            "answer": sum(1 for row in rows if row["expected_behavior"] == "answer"),
            "abstain": sum(1 for row in rows if row["expected_behavior"] == "abstain"),
        },
        "best_rule": best_rule,
        "top_candidate_rules": candidate_rules[:12],
        "rows": labeled_rows,
    }

    output_path = Path(args.output).resolve() if args.output else REPORTS_DIR / f"phase3_abstention_audit_{Path(args.queries).stem}.json"
    write_json(output_path, report)
    print(f"Abstention audit written to: {output_path}")
    print(f"Best rule: {best_rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
