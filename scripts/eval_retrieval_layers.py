"""Layer-by-layer retrieval evaluation for BOFiP Agentic RAG.

This script does not call an LLM. It diagnoses where gold evidence disappears:
document retrieval, chunk candidates, or final selected chunks.

Examples:
    $env:PYTHONPATH="src"
    py -3.11 scripts/eval_retrieval_layers.py --limit 10
    py -3.11 scripts/eval_retrieval_layers.py --limit 10 --reranker
    py -3.11 scripts/eval_retrieval_layers.py --mode hybrid --limit 10 --device cpu
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.eval_harness import QueryGold, RetrievalLayer, evaluate_layers
from bofip_agentic.rag_runtime import RagRuntime


DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "passage_gold_v3.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "reports" / "eval_retrieval_layers.json"
BOFIP_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")
REQUIRED_CHUNK_FIELDS = ("document_id", "boi_reference", "section_path", "paragraph_range", "text")


@dataclass
class EvalQuery:
    query_id: str
    query: str
    category: str
    gold_doc_refs: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    expected_section_terms: list[str] = field(default_factory=list)
    expected_text_terms: list[str] = field(default_factory=list)
    must_not_match_refs: list[str] = field(default_factory=list)
    note: str = ""

    def to_query_gold(self) -> QueryGold:
        return QueryGold(
            query_id=self.query_id,
            query=self.query,
            category=self.category,
            gold_doc_refs=list(self.gold_doc_refs),
            gold_chunk_ids=list(self.gold_chunk_ids),
            note=self.note,
        )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _coerce_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def normalize_doc_ref(value: str) -> str:
    return BOFIP_DATE_SUFFIX_RE.sub("", str(value).strip().upper())


def normalize_doc_refs_in_queries(queries: list[QueryGold]) -> list[QueryGold]:
    return [
        QueryGold(
            query_id=query.query_id,
            query=query.query,
            category=query.category,
            gold_doc_refs=[normalize_doc_ref(ref) for ref in query.gold_doc_refs],
            gold_chunk_ids=list(query.gold_chunk_ids),
            note=query.note,
        )
        for query in queries
    ]


def normalize_doc_refs_in_eval_queries(queries: list[EvalQuery]) -> list[EvalQuery]:
    return [
        EvalQuery(
            query_id=query.query_id,
            query=query.query,
            category=query.category,
            gold_doc_refs=[normalize_doc_ref(ref) for ref in query.gold_doc_refs],
            gold_chunk_ids=list(query.gold_chunk_ids),
            expected_section_terms=list(query.expected_section_terms),
            expected_text_terms=list(query.expected_text_terms),
            must_not_match_refs=[normalize_doc_ref(ref) for ref in query.must_not_match_refs],
            note=query.note,
        )
        for query in queries
    ]


def build_gold_compatibility_report(
    queries: list[QueryGold],
    *,
    active_doc_refs: set[str],
    active_chunk_ids: set[str],
) -> dict[str, int]:
    active_doc_refs_normalized = {normalize_doc_ref(ref) for ref in active_doc_refs}
    gold_doc_refs = [ref for query in queries for ref in query.gold_doc_refs]
    gold_chunk_ids = [chunk_id for query in queries for chunk_id in query.gold_chunk_ids]
    return {
        "gold_doc_refs": len(gold_doc_refs),
        "gold_doc_refs_exact_matches": sum(1 for ref in gold_doc_refs if ref in active_doc_refs),
        "gold_doc_refs_normalized_matches": sum(
            1 for ref in gold_doc_refs
            if normalize_doc_ref(ref) in active_doc_refs_normalized
        ),
        "gold_chunk_ids": len(gold_chunk_ids),
        "gold_chunk_ids_exact_matches": sum(1 for chunk_id in gold_chunk_ids if chunk_id in active_chunk_ids),
    }


def load_eval_queries(path: Path, *, limit: int = 0) -> list[EvalQuery]:
    rows = _read_jsonl(path)
    if limit > 0:
        rows = rows[:limit]

    queries: list[EvalQuery] = []
    for index, row in enumerate(rows, start=1):
        query = row.get("query") or row.get("question")
        if not query:
            continue
        queries.append(
            EvalQuery(
                query_id=str(row.get("query_id") or row.get("id") or f"q{index:04d}"),
                query=str(query),
                category=str(row.get("category") or row.get("theme") or "unknown"),
                gold_doc_refs=_coerce_list(
                    row.get("expected_doc_refs")
                    or row.get("gold_doc_refs")
                    or row.get("required_docs")
                ),
                gold_chunk_ids=_coerce_list(row.get("expected_chunk_ids") or row.get("gold_chunk_ids")),
                expected_section_terms=_coerce_list(row.get("expected_section_terms")),
                expected_text_terms=_coerce_list(row.get("expected_text_terms")),
                must_not_match_refs=_coerce_list(row.get("must_not_match_refs")),
                note=str(row.get("note") or ""),
            )
        )
    return queries


def load_query_golds(path: Path, *, limit: int = 0) -> list[QueryGold]:
    return [query.to_query_gold() for query in load_eval_queries(path, limit=limit)]


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_accents.casefold()).strip()


def term_coverage_rank(items: list[str], terms: list[str]) -> int | None:
    normalized_terms = [_normalize_text(term) for term in terms if _normalize_text(term)]
    if not normalized_terms:
        return None

    cumulative = ""
    for index, item in enumerate(items, start=1):
        cumulative = f"{cumulative}\n{_normalize_text(item)}"
        if all(term in cumulative for term in normalized_terms):
            return index
    return None


def _family_from_ref(boi_reference: str) -> str:
    parts = str(boi_reference).upper().split("-")
    if len(parts) >= 2 and parts[0] == "BOI":
        return parts[1]
    return "UNKNOWN"


def _field_is_missing(item, field_name: str) -> bool:
    if isinstance(item, dict):
        value = item.get(field_name)
    else:
        value = getattr(item, field_name, None)
    return value in (None, "", [])


def build_corpus_sanity_report(documents, chunks) -> dict:
    chunk_doc_ids = {getattr(chunk, "document_id", "") for chunk in chunks}
    documents_without_chunks = [
        document for document in documents
        if getattr(document, "document_id", "") not in chunk_doc_ids
    ]
    missing_chunk_fields = []
    for chunk in chunks:
        missing = [field_name for field_name in REQUIRED_CHUNK_FIELDS if _field_is_missing(chunk, field_name)]
        if missing:
            missing_chunk_fields.append(
                {
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "boi_reference": getattr(chunk, "boi_reference", ""),
                    "missing_fields": missing,
                }
            )

    return {
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "documents_without_chunks_count": len(documents_without_chunks),
        "documents_without_chunks_by_type": dict(
            Counter(getattr(document, "document_type", "UNKNOWN") or "UNKNOWN" for document in documents_without_chunks)
        ),
        "documents_without_chunks_by_family": dict(
            Counter(_family_from_ref(getattr(document, "boi_reference", "")) for document in documents_without_chunks)
        ),
        "documents_without_chunks_sample": [
            {
                "document_id": getattr(document, "document_id", ""),
                "boi_reference": getattr(document, "boi_reference", ""),
                "document_type": getattr(document, "document_type", ""),
                "family": _family_from_ref(getattr(document, "boi_reference", "")),
            }
            for document in documents_without_chunks[:50]
        ],
        "family_distribution": dict(
            Counter(_family_from_ref(getattr(document, "boi_reference", "")) for document in documents)
        ),
        "chunk_required_field_missing_count": len(missing_chunk_fields),
        "chunk_required_field_missing_sample": missing_chunk_fields[:50],
    }


def _coerce_rank(value) -> int | None:
    if value in (None, "", 0):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rank_sort_value(rank: int | None) -> int:
    return 1_000_000 if rank is None else rank


def build_regression_report(baseline: dict, current: dict, *, layer_name: str) -> dict:
    baseline_by_id = {
        item["query_id"]: _coerce_rank(item.get("layers", {}).get(layer_name, {}).get("rank"))
        for item in baseline.get("metrics", {}).get("per_query", [])
    }
    current_by_id = {
        item["query_id"]: _coerce_rank(item.get("layers", {}).get(layer_name, {}).get("rank"))
        for item in current.get("metrics", {}).get("per_query", [])
    }
    improved = []
    worsened = []
    unchanged = []
    missing_in_baseline = []
    for query_id, current_rank in current_by_id.items():
        if query_id not in baseline_by_id:
            missing_in_baseline.append(query_id)
            continue
        baseline_rank = baseline_by_id[query_id]
        row = {"query_id": query_id, "baseline_rank": baseline_rank, "current_rank": current_rank}
        if _rank_sort_value(current_rank) < _rank_sort_value(baseline_rank):
            improved.append(row)
        elif _rank_sort_value(current_rank) > _rank_sort_value(baseline_rank):
            worsened.append(row)
        else:
            unchanged.append(row)
    return {
        "layer": layer_name,
        "improved": improved,
        "worsened": worsened,
        "unchanged_count": len(unchanged),
        "missing_in_baseline": missing_in_baseline,
    }


def _join_section_path(value) -> str:
    if isinstance(value, list):
        return " > ".join(str(part) for part in value if part)
    return str(value or "")


def _chunk_text(value) -> str:
    return str(getattr(value, "text", "") or "")


def _runtime_layers(
    runtime: RagRuntime,
    query: str,
    *,
    mode: str,
    use_reranker: bool,
    normalize_doc_refs: bool,
    top_docs: int,
    chunks_per_doc: int,
    max_chunks: int,
    use_anchor_filter: bool = True,
) -> dict[str, list[str]]:
    use_dense = mode == "hybrid"
    result = runtime.retrieve(
        query,
        top_docs=top_docs,
        chunks_per_doc=chunks_per_doc,
        max_chunks=max_chunks,
        use_dense=use_dense,
        use_chunk_dense=use_dense,
        use_reranker=use_reranker,
        use_anchor_filter=use_anchor_filter,
    )
    log = result.pipeline_log
    doc_transform = normalize_doc_ref if normalize_doc_refs else str
    chunk_by_id = getattr(runtime, "_eval_chunk_by_id", None)
    if chunk_by_id is None:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in getattr(runtime, "chunks", [])}
        try:
            setattr(runtime, "_eval_chunk_by_id", chunk_by_id)
        except Exception:
            pass
    candidate_chunks = [
        chunk_by_id[chunk_id]
        for chunk_id in log.get("stage2_candidate_chunk_ids", [])
        if chunk_id in chunk_by_id
    ]
    return {
        "stage1_docs": [doc_transform(hit.boi_reference) for hit in result.stage1_hits],
        "stage2_candidate_docs": [doc_transform(ref) for ref in log.get("stage2_candidate_doc_refs", [])],
        "stage2_candidate_chunks": list(log.get("stage2_candidate_chunk_ids", [])),
        "stage2_candidate_sections": [_join_section_path(chunk.section_path) for chunk in candidate_chunks],
        "stage2_candidate_texts": [_chunk_text(chunk) for chunk in candidate_chunks],
        "final_docs": [doc_transform(hit.boi_reference) for hit in result.stage2_chunks],
        "final_chunks": [hit.chunk_id for hit in result.stage2_chunks],
        "final_sections": [_join_section_path(hit.section_path) for hit in result.stage2_chunks],
        "final_texts": [_chunk_text(hit) for hit in result.stage2_chunks],
    }


def build_term_metrics(
    queries: list[EvalQuery],
    *,
    retrieve_layers,
    k_values: list[int],
) -> dict:
    specs = [
        ("section", "expected_section_terms", ["stage2_candidate_sections", "final_sections"]),
        ("text", "expected_text_terms", ["stage2_candidate_texts", "final_texts"]),
    ]
    report = {"per_query": []}
    max_k = max(k_values)
    for metric_name, attr_name, layer_names in specs:
        eligible = [query for query in queries if getattr(query, attr_name)]
        report[f"{metric_name}_eligible_queries"] = len(eligible)
        report[f"{metric_name}_hit_at"] = {layer: {} for layer in layer_names}
        report[f"{metric_name}_mrr"] = {layer: 0.0 for layer in layer_names}
        for layer in layer_names:
            ranks: list[int | None] = []
            for query in eligible:
                layers = retrieve_layers(query.query)
                rank = term_coverage_rank(list(layers.get(layer, []))[:max_k], getattr(query, attr_name))
                ranks.append(rank)
            for k in k_values:
                hits = sum(1 for rank in ranks if rank is not None and rank <= k)
                report[f"{metric_name}_hit_at"][layer][k] = hits / len(eligible) if eligible else 0.0
            reciprocal = sum((1.0 / rank) for rank in ranks if rank)
            report[f"{metric_name}_mrr"][layer] = reciprocal / len(eligible) if eligible else 0.0

    for query in queries:
        layers = retrieve_layers(query.query)
        report["per_query"].append(
            {
                "query_id": query.query_id,
                "section_terms": query.expected_section_terms,
                "text_terms": query.expected_text_terms,
                "section_ranks": {
                    layer: term_coverage_rank(list(layers.get(layer, []))[:max_k], query.expected_section_terms)
                    for layer in ("stage2_candidate_sections", "final_sections")
                },
                "text_ranks": {
                    layer: term_coverage_rank(list(layers.get(layer, []))[:max_k], query.expected_text_terms)
                    for layer in ("stage2_candidate_texts", "final_texts")
                },
            }
        )
    return report


def build_must_not_report(queries: list[EvalQuery], *, retrieve_layers) -> list[dict]:
    violations = []
    doc_layers = ("stage1_docs", "stage2_candidate_docs", "final_docs")
    for query in queries:
        forbidden = set(query.must_not_match_refs)
        if not forbidden:
            continue
        layers = retrieve_layers(query.query)
        for layer in doc_layers:
            matched = [ref for ref in layers.get(layer, []) if ref in forbidden]
            if matched:
                violations.append({"query_id": query.query_id, "layer": layer, "matched_refs": matched})
    return violations


def _metric_at(values: dict, k: int) -> float:
    return float(values.get(k, values.get(str(k), 0.0)))


def format_summary_markdown(report: dict) -> str:
    config = report.get("config", {})
    metrics = report.get("metrics", {})
    layer_hit_at = metrics.get("layer_hit_at", {})
    layer_mrr = metrics.get("layer_mrr", {})
    k_values = config.get("k") or [1, 3, 5, 8]
    max_k = max(int(k) for k in k_values)
    lines = [
        "# Retrieval Audit",
        "",
        f"- Mode: `{config.get('mode', '')}`",
        f"- Corpus docs/chunks: `{report.get('corpus_sanity', {}).get('documents_count', 0)}` / `{report.get('corpus_sanity', {}).get('chunks_count', 0)}`",
        f"- Queries: `{metrics.get('queries_count', 0)}`",
        f"- Elapsed: `{report.get('elapsed_s', 0)}s`",
        "",
        "## Layer Metrics",
        "",
        "| Layer | Hit@maxK | MRR |",
        "| --- | ---: | ---: |",
    ]
    for layer in ("stage1_docs", "stage2_candidate_docs", "stage2_candidate_chunks", "final_docs", "final_chunks"):
        lines.append(
            f"| `{layer}` | {_metric_at(layer_hit_at.get(layer, {}), max_k):.1%} | "
            f"{float(layer_mrr.get(layer, 0.0)):.3f} |"
        )

    term_metrics = report.get("term_metrics", {})
    if term_metrics:
        lines.extend(
            [
                "",
                "## Section/Text Terms",
                "",
                f"- Section eligible queries: `{term_metrics.get('section_eligible_queries', 0)}`",
                f"- Text eligible queries: `{term_metrics.get('text_eligible_queries', 0)}`",
                f"- Final section term Hit@{max_k}: `{_metric_at(term_metrics.get('section_hit_at', {}).get('final_sections', {}), max_k):.1%}`",
                f"- Final text term Hit@{max_k}: `{_metric_at(term_metrics.get('text_hit_at', {}).get('final_texts', {}), max_k):.1%}`",
            ]
        )

    comparison = report.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                f"- Layer: `{comparison.get('layer', '')}`",
                f"- Improved: `{len(comparison.get('improved', []))}`",
                f"- Worsened: `{len(comparison.get('worsened', []))}`",
                f"- Unchanged: `{comparison.get('unchanged_count', 0)}`",
            ]
        )
        for item in comparison.get("worsened", [])[:10]:
            lines.append(
                f"- Worsened `{item['query_id']}`: {item['baseline_rank']} -> {item['current_rank']}"
            )

    if report.get("must_not_violations"):
        lines.extend(["", "## Must-Not Violations", ""])
        for item in report["must_not_violations"][:10]:
            lines.append(f"- `{item['query_id']}` `{item['layer']}`: {', '.join(item['matched_refs'])}")
    return "\n".join(lines) + "\n"


def _metrics_to_json(metrics) -> dict:
    return {
        "queries_count": metrics.queries_count,
        "categories_count": metrics.categories_count,
        "layer_hit_at": metrics.layer_hit_at,
        "layer_mrr": metrics.layer_mrr,
        "layer_ndcg_at": metrics.layer_ndcg_at,
        "transition_misses": metrics.transition_misses,
        "first_miss_counts": metrics.first_miss_counts,
        "per_query": [
            {
                "query_id": result.query_id,
                "category": result.category,
                "first_miss_layer": result.first_miss_layer,
                "layers": {
                    name: asdict(layer_result)
                    for name, layer_result in result.layers.items()
                },
            }
            for result in metrics.per_query
        ],
    }


def _parse_k_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one k value is required")
    return sorted(set(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate BOFiP retrieval layers without LLM calls")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--raw-docs-path", type=Path, default=None)
    parser.add_argument("--chunks-path", type=Path, default=None)
    parser.add_argument("--doc-dense-path", type=Path, default=None)
    parser.add_argument("--chunk-dense-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", choices=["lexical", "hybrid"], default="lexical")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-docs", type=int, default=8)
    parser.add_argument("--chunks-per-doc", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--k", type=_parse_k_values, default=[1, 3, 5, 8])
    parser.add_argument(
        "--disable-anchor-filter",
        action="store_true",
        help="Benchmark hybrid without dense-anchor filtering. Useful for diagnosing dense-anchor regressions.",
    )
    parser.add_argument(
        "--normalize-doc-refs",
        action="store_true",
        help="Strip trailing BOFiP date suffixes from document refs before scoring doc layers",
    )
    args = parser.parse_args()

    eval_queries = load_eval_queries(args.input, limit=args.limit)
    if not eval_queries:
        print(f"No query loaded from {args.input}")
        return 1
    raw_queries = [query.to_query_gold() for query in eval_queries]
    if args.normalize_doc_refs:
        eval_queries = normalize_doc_refs_in_eval_queries(eval_queries)
    queries = [query.to_query_gold() for query in eval_queries]

    started = time.time()
    runtime = RagRuntime.from_local_corpus(
        corpus="commentary",
        raw_docs_path=args.raw_docs_path,
        chunks_path=args.chunks_path,
        doc_dense_path=args.doc_dense_path,
        chunk_dense_path=args.chunk_dense_path,
        device=args.device,
        load_dense=args.mode == "hybrid",
        load_reranker=args.reranker,
    )
    compatibility = build_gold_compatibility_report(
        raw_queries,
        active_doc_refs={document.boi_reference for document in runtime.documents},
        active_chunk_ids={chunk.chunk_id for chunk in runtime.chunks},
    )
    corpus_sanity = build_corpus_sanity_report(runtime.documents, runtime.chunks)

    layers = [
        RetrievalLayer("stage1_docs", "doc"),
        RetrievalLayer("stage2_candidate_docs", "doc"),
        RetrievalLayer("stage2_candidate_chunks", "chunk"),
        RetrievalLayer("final_docs", "doc"),
        RetrievalLayer("final_chunks", "chunk"),
    ]
    cached_layers: dict[str, dict[str, list[str]]] = {}

    def retrieve_cached(query: str) -> dict[str, list[str]]:
        if query not in cached_layers:
            cached_layers[query] = _runtime_layers(
                runtime,
                query,
                mode=args.mode,
                use_reranker=args.reranker,
                normalize_doc_refs=args.normalize_doc_refs,
                top_docs=args.top_docs,
                chunks_per_doc=args.chunks_per_doc,
                max_chunks=args.max_chunks,
                use_anchor_filter=not args.disable_anchor_filter,
            )
        return cached_layers[query]

    metrics = evaluate_layers(
        queries,
        retrieve_layers=retrieve_cached,
        layers=layers,
        k_values=args.k,
    )
    term_metrics = build_term_metrics(eval_queries, retrieve_layers=retrieve_cached, k_values=args.k)
    must_not_violations = build_must_not_report(eval_queries, retrieve_layers=retrieve_cached)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - started, 2),
        "config": {
            "input": str(args.input),
            "raw_docs_path": str(args.raw_docs_path) if args.raw_docs_path else "",
            "chunks_path": str(args.chunks_path) if args.chunks_path else "",
            "doc_dense_path": str(args.doc_dense_path) if args.doc_dense_path else "",
            "chunk_dense_path": str(args.chunk_dense_path) if args.chunk_dense_path else "",
            "mode": args.mode,
            "reranker": args.reranker,
            "device": args.device,
            "top_docs": args.top_docs,
            "chunks_per_doc": args.chunks_per_doc,
            "max_chunks": args.max_chunks,
            "k": args.k,
            "normalize_doc_refs": args.normalize_doc_refs,
            "disable_anchor_filter": args.disable_anchor_filter,
        },
        "corpus_sanity": corpus_sanity,
        "gold_compatibility": compatibility,
        "metrics": _metrics_to_json(metrics),
        "term_metrics": term_metrics,
        "must_not_violations": must_not_violations,
    }
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        report["comparison"] = build_regression_report(baseline, report, layer_name="final_docs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(format_summary_markdown(report), encoding="utf-8")

    print(f"Layer retrieval eval: {metrics.queries_count} queries in {report['elapsed_s']}s")
    print(
        "  gold compatibility: "
        f"doc exact {compatibility['gold_doc_refs_exact_matches']}/{compatibility['gold_doc_refs']}, "
        f"doc normalized {compatibility['gold_doc_refs_normalized_matches']}/{compatibility['gold_doc_refs']}, "
        f"chunk exact {compatibility['gold_chunk_ids_exact_matches']}/{compatibility['gold_chunk_ids']}"
    )
    for layer in layers:
        hit_at = metrics.layer_hit_at[layer.name]
        printable = " ".join(f"@{k}={hit_at[k]:.1%}" for k in args.k)
        print(f"  {layer.name:<24} {printable} MRR={metrics.layer_mrr[layer.name]:.3f}")
    if metrics.transition_misses:
        print(f"  transition_misses: {metrics.transition_misses}")
    if report.get("comparison"):
        comparison = report["comparison"]
        print(
            "  comparison(final_docs): "
            f"improved={len(comparison['improved'])} "
            f"worsened={len(comparison['worsened'])} "
            f"unchanged={comparison['unchanged_count']}"
        )
    if must_not_violations:
        print(f"  must_not_violations: {len(must_not_violations)}")
    print(f"Report: {args.output}")
    if args.summary_output:
        print(f"Summary: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
