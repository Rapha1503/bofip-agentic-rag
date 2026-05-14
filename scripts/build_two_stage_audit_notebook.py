from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def _md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def _code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}

    nb.cells = [
        _md(
            """
            # Two-Stage Retrieval Audit

            Notebook de transparence sur le retrieval BOFIP en **deux étages**:
            1. retrieval documentaire BOI
            2. retrieval de passages dans les documents retenus

            Portée:
            - pas de LLM
            - pas d'abstention encore
            - focus sur la lisibilité du pipeline et le chunk ordering
            """
        ),
        _code(
            """
            from pathlib import Path
            import json
            import sys

            PROJECT_ROOT = Path.cwd().resolve().parent
            SRC_ROOT = PROJECT_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from bofip_cleanroom.jsonio import read_jsonl
            from bofip_cleanroom.models import chunk_node_from_dict, raw_document_from_dict

            def load_json(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))

            raw_docs = [raw_document_from_dict(item) for item in read_jsonl(PROJECT_ROOT / "data" / "interim" / "raw_docs_sample_1000.jsonl")]
            chunks = [chunk_node_from_dict(item) for item in read_jsonl(PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_1000.jsonl")]
            queries = read_jsonl(PROJECT_ROOT / "data" / "interim" / "retrieval_queries_sample_1000_v1.jsonl")

            doc_eval = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_1000.json")
            chunk_eval = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_batch_eval_chunks_section_window_sample_1000.json")
            two_stage_full = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_two_stage_probe_sample_1000_full.json")
            two_stage_body = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_two_stage_probe_sample_1000_body.json")
            chunk_order_audit = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_chunk_order_audit_sample_1000.json")

            query_map = {row["id"]: row for row in queries}
            doc_eval_map = {row["id"]: row for row in doc_eval["results"]}
            chunk_eval_map = {row["id"]: row for row in chunk_eval["results"]}
            two_stage_full_map = {row["id"]: row for row in two_stage_full["rows"]}
            two_stage_body_map = {row["id"]: row for row in two_stage_body["rows"]}
            raw_doc_map = {doc.boi_reference: doc for doc in raw_docs}

            print("Loaded two-stage audit artifacts.")
            """
        ),
        _md(
            """
            ## 1. Gate actuel

            Le point important à lire correctement:
            - le gate `80%` a été franchi **au niveau retrieval documentaire**
            - pas encore au niveau génération
            - le prochain problème est la sélection du bon passage **dans** le bon document
            """
        ),
        _code(
            """
            {
                "doc_lexical_metrics": doc_eval["metrics"],
                "flat_chunk_lexical_metrics": chunk_eval["metrics"],
                "chunk_order_audit_summary": chunk_order_audit["summary"],
            }
            """
        ),
        _md(
            """
            ## 2. Ce qui change entre stage 2 `full` et stage 2 `body`

            Le stage 1 garde la structure BOI.
            Le stage 2 `body` retire le bruit structurel inutile et cherche le meilleur passage sur le texte du chunk lui-même.
            """
        ),
        _code(
            """
            chunk_order_audit["examples_where_body_differs_from_full"][:8]
            """
        ),
        _code(
            """
            def document_preview(boi_reference, section_limit=10):
                doc = raw_doc_map[boi_reference]
                return {
                    "boi_reference": doc.boi_reference,
                    "title": doc.title,
                    "publication_date": doc.publication_date,
                    "sections_preview": [
                        {
                            "level": section.level,
                            "title": section.title,
                            "path": section.path,
                        }
                        for section in doc.sections[:section_limit]
                    ],
                }

            document_preview("BOI-BIC-RICI-10-10-20-25-20250813")
            """
        ),
        _md(
            """
            ## 3. Helpers d'affichage
            """
        ),
        _code(
            """
            def top_chunks(trace_row, limit=6):
                return [
                    {
                        "global_rank": hit["global_rank"],
                        "document_rank": hit["document_rank"],
                        "document_score": hit["document_score"],
                        "local_rank": hit["local_rank"],
                        "local_score": hit["local_score"],
                        "boi_reference": hit["boi_reference"],
                        "chunk_kind": hit["chunk_kind"],
                        "section_path": hit["section_path"],
                        "text": hit["text"],
                    }
                    for hit in trace_row["chunk_hits"][:limit]
                ]

            def top_documents(trace_row, limit=3):
                return trace_row["document_hits"][:limit]

            def compare_trace(query_id):
                query = query_map[query_id]
                doc_row = doc_eval_map[query_id]
                return {
                    "query_id": query_id,
                    "pattern": query.get("pattern"),
                    "query": query["query"],
                    "expected_boi": query.get("expected_boi"),
                    "stage1_top_docs": top_documents(two_stage_body_map[query_id]),
                    "stage2_full_top_chunks": top_chunks(two_stage_full_map[query_id]),
                    "stage2_body_top_chunks": top_chunks(two_stage_body_map[query_id]),
                    "stage1_doc_eval": {
                        "returned_boi": doc_row["returned_boi"],
                        "supported_query": doc_row.get("supported_query"),
                        "hit@1": doc_row.get("hit@1"),
                        "hit@3": doc_row.get("hit@3"),
                        "hit@5": doc_row.get("hit@5"),
                    },
                }
            """
        ),
        _md(
            """
            ## 4. Exemples réels, variés

            Chaque bloc montre:
            - la question
            - le stage 1: top documents
            - le stage 2 actuel `full`
            - le stage 2 passage `body`
            """
        ),
        _code("""compare_trace("q02")"""),
        _code("""compare_trace("q04")"""),
        _code("""compare_trace("q08")"""),
        _code("""compare_trace("q11")"""),
        _code("""compare_trace("q12")"""),
        _code("""compare_trace("q14")"""),
        _code("""compare_trace("q29")"""),
        _code("""compare_trace("q30")"""),
        _code("""compare_trace("u04")"""),
        _code("""compare_trace("u05")"""),
        _md(
            """
            ## 5. Lecture d'ingénieur

            À lire de façon stricte:
            - si le stage 1 se trompe, le stage 2 ne peut pas sauver le bon document
            - quand le stage 1 est correct, `body` donne souvent un passage plus substantiel que `full`
            - les questions hors corpus / à fausse prémisse retournent encore des matchs de proximité: l'abstention n'est pas encore branchée
            """
        ),
        _code(
            """
            {
                "doc_stage_misses": [
                    row for row in doc_eval["results"]
                    if row["supported_query"] and not row["hit@1"]
                ],
                "unsupported_examples": [
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "returned_boi": row["returned_boi"][:3],
                    }
                    for row in doc_eval["results"]
                    if not row["supported_query"]
                ][:5],
            }
            """
        ),
        _md(
            """
            ## 6. Conclusion actuelle

            Le double étage montre enfin quelque chose de propre:
            - retrieval documentaire BOI: oui, crédible
            - retrieval de passages: encore à stabiliser, mais maintenant inspectable
            - prochaine étape: améliorer l'ordering local et préparer la gate d'abstention
            """
        ),
    ]

    output_path = NOTEBOOKS_DIR / "04_two_stage_retrieval_audit.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
