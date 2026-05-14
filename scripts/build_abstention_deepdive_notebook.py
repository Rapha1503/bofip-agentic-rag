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
            # Document Retrieval, Deep Dive, Abstention

            Ce notebook répond à trois questions, dans cet ordre:
            1. retrouve-t-on le bon document BOFIP ?
            2. extrait-on un bon passage dans ce document ?
            3. sait-on déjà quand il faut s'abstenir ?

            La priorité reste volontairement:
            - `#1` bon document
            - `#2` bon extrait
            - `#3` abstention
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

            def load_json(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))

            queries_v2 = read_jsonl(PROJECT_ROOT / "data" / "interim" / "retrieval_queries_sample_1000_v2.jsonl")
            doc_eval_v2 = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v2.json")
            two_stage_body_v2 = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_two_stage_probe_sample_1000_v2_body.json")
            deep_dive_v2 = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_deep_dive_probe_retrieval_queries_sample_1000_v2_body.json")
            abstention_v2 = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_abstention_audit_retrieval_queries_sample_1000_v2.json")

            query_map = {row["id"]: row for row in queries_v2}
            doc_eval_map = {row["id"]: row for row in doc_eval_v2["results"]}
            two_stage_map = {row["id"]: row for row in two_stage_body_v2["rows"]}
            deep_dive_map = {row["id"]: row for row in deep_dive_v2["rows"]}
            abstain_map = {row["id"]: row for row in abstention_v2["rows"]}
            """
        ),
        _md(
            """
            ## 1. Vue d'ensemble
            """
        ),
        _code(
            """
            {
                "query_count_v2": len(queries_v2),
                "behavior_counts": abstention_v2["behavior_counts"],
                "doc_retrieval_metrics_supported_only": doc_eval_v2["metrics"],
                "abstention_best_rule": abstention_v2["best_rule"],
            }
            """
        ),
        _md(
            """
            ## 2. Objectif numéro 1: le bon document BOFIP

            Ici on regarde les rares vrais ratés du retrieval documentaire.
            """
        ),
        _code(
            """
            [
                {
                    "id": row["id"],
                    "pattern": row["pattern"],
                    "query": row["query"],
                    "expected_boi": row["expected_boi"],
                    "returned_boi": row["returned_boi"][:5],
                }
                for row in doc_eval_v2["results"]
                if row.get("supported_query") and not row.get("hit@1")
            ]
            """
        ),
        _md(
            """
            ## 3. Helpers d'affichage
            """
        ),
        _code(
            """
            def doc_trace(query_id):
                row = doc_eval_map[query_id]
                return {
                    "query_id": query_id,
                    "query": row["query"],
                    "pattern": row["pattern"],
                    "expected_boi": row.get("expected_boi"),
                    "returned_boi": row["returned_boi"][:5],
                    "top_hits": row["top_hits"][:5],
                }

            def two_stage_trace(query_id):
                row = two_stage_map[query_id]
                return {
                    "query_id": query_id,
                    "query": row["query"],
                    "expected_boi": row.get("expected_boi"),
                    "expected_behavior": query_map[query_id].get("expected_behavior"),
                    "stage1_docs": row["document_hits"][:3],
                    "stage2_chunks": row["chunk_hits"][:6],
                }

            def deep_dive_trace(query_id):
                row = deep_dive_map[query_id]
                return {
                    "query_id": query_id,
                    "query": row["query"],
                    "expected_boi": row.get("expected_boi"),
                    "expected_behavior": row.get("expected_behavior"),
                    "stage1_docs": row["document_hits"][:3],
                    "stage2_top_chunks": row["top_chunk_hits"][:4],
                    "stage3_expanded_context": row["expanded_context"][:10],
                }

            def abstention_trace(query_id):
                row = abstain_map[query_id]
                return {
                    "query_id": query_id,
                    "query": row["query"],
                    "pattern": row["pattern"],
                    "expected_behavior": row["expected_behavior"],
                    "predicted_behavior": row["predicted_behavior"],
                    "decision_correct": row["decision_correct"],
                    "top_doc_boi": row["top_doc_boi"],
                    "doc_margin": row["doc_margin"],
                    "title_overlap_ratio": row["title_overlap_ratio"],
                    "chunk_overlap_ratio": row["chunk_overlap_ratio"],
                    "combined_uncovered_ratio": row["combined_uncovered_ratio"],
                    "top_chunk_text": row["top_chunk_text"],
                }
            """
        ),
        _md(
            """
            ## 4. Cas réels pour l'objectif `#1`

            Succès nets, voisinages thématiques, et ratés documentaires.
            """
        ),
        _code("""doc_trace("q02")"""),
        _code("""doc_trace("q12")"""),
        _code("""doc_trace("q29")"""),
        _code("""doc_trace("q30")"""),
        _code("""doc_trace("q01")"""),
        _md(
            """
            ## 5. Objectif numéro `#2`: le bon extrait du bon document

            On compare:
            - stage 1 document retrieval
            - stage 2 top chunks
            - stage 3 deep dive local borné
            """
        ),
        _code("""deep_dive_trace("q02")"""),
        _code("""deep_dive_trace("q04")"""),
        _code("""deep_dive_trace("q11")"""),
        _code("""deep_dive_trace("q12")"""),
        _code("""deep_dive_trace("q14")"""),
        _code("""deep_dive_trace("q29")"""),
        _code("""deep_dive_trace("q30")"""),
        _md(
            """
            ## 6. Cas non supportés et fausses prémisses

            Ici on regarde ce que le système fait aujourd'hui quand il devrait soit:
            - s'abstenir
            - ou retrouver un document pour corriger la prémisse
            """
        ),
        _code("""two_stage_trace("u01")"""),
        _code("""two_stage_trace("u03")"""),
        _code("""two_stage_trace("u06")"""),
        _code("""two_stage_trace("u17")"""),
        _code("""two_stage_trace("u23")"""),
        _md(
            """
            ## 7. Audit d'abstention

            Important:
            - la gate actuelle n'est pas encore une solution finale
            - c'est un audit transparent des signaux disponibles
            - le meilleur rule actuel repose sur le taux de tokens de requête non couverts par `titre du doc + top chunk`
            """
        ),
        _code(
            """
            {
                "best_rule": abstention_v2["best_rule"],
                "top_candidate_rules": abstention_v2["top_candidate_rules"][:8],
            }
            """
        ),
        _code(
            """
            {
                "correct_abstentions": [
                    {
                        "id": row["id"],
                        "combined_uncovered_ratio": row["combined_uncovered_ratio"],
                        "query": row["query"],
                    }
                    for row in abstention_v2["rows"]
                    if row["expected_behavior"] == "abstain" and row["predicted_behavior"] == "abstain"
                ][:10],
                "wrong_answers_should_abstain": [
                    {
                        "id": row["id"],
                        "combined_uncovered_ratio": row["combined_uncovered_ratio"],
                        "query": row["query"],
                    }
                    for row in abstention_v2["rows"]
                    if row["expected_behavior"] == "abstain" and row["predicted_behavior"] == "answer"
                ][:10],
                "wrong_abstentions_should_answer": [
                    {
                        "id": row["id"],
                        "combined_uncovered_ratio": row["combined_uncovered_ratio"],
                        "query": row["query"],
                    }
                    for row in abstention_v2["rows"]
                    if row["expected_behavior"] == "answer" and row["predicted_behavior"] == "abstain"
                ][:10],
            }
            """
        ),
        _code("""abstention_trace("u01")"""),
        _code("""abstention_trace("u03")"""),
        _code("""abstention_trace("u17")"""),
        _code("""abstention_trace("u23")"""),
        _md(
            """
            ## 8. Conclusion d'ingénieur

            Lecture stricte:
            - `#1` bon document: oui, c'est maintenant la partie la plus solide
            - `#2` bon extrait: meilleur qu'avant grâce au stage 2 `body` et au stage 3 borné, mais encore à auditer
            - abstention: encore insuffisante pour servir de gate finale seule

            Donc la suite rationnelle reste:
            - renforcer encore l'audit du retrieval documentaire sur les voisinages restants
            - continuer à améliorer la qualité du passage local
            - ne brancher le LLM qu'après une gate d'abstention plus crédible
            """
        ),
    ]

    output_path = NOTEBOOKS_DIR / "05_abstention_and_deep_dive_audit.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
