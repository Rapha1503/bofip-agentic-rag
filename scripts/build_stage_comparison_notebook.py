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
            # Stage 1 / Stage 2 Audit

            Audit de transparence sur le retrieval BOFIP clean-room:
            1. stage 1 = selection du bon document BOI
            2. stage 2 = selection du bon passage dans ce document
            3. stage 3 = expansion locale bornee autour du passage

            Portee:
            - pas de LLM
            - pas d'abstention finale
            - focus sur les vrais cas reels du benchmark v3
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

            queries = read_jsonl(PROJECT_ROOT / "data" / "interim" / "retrieval_queries_sample_1000_v3.jsonl")
            query_map = {row["id"]: row for row in queries}

            doc_eval_base = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v3.json")
            doc_eval_sections = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v3__sections.json")
            doc_eval_sections_firstpara = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v3__sections_firstpara.json")
            local_audit_sections = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_local_strategy_audit_retrieval_queries_sample_1000_v3__sections_body.json")
            local_audit_sections_firstpara = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_local_strategy_audit_retrieval_queries_sample_1000_v3__sections_firstpara_body.json")

            two_stage_base_chunk = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_two_stage_probe_retrieval_queries_sample_1000_v3__base_chunk_body.json")
            two_stage_sections_chunk = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_two_stage_probe_retrieval_queries_sample_1000_v3__sections_chunk_body.json")
            two_stage_sections_section = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_two_stage_probe_retrieval_queries_sample_1000_v3__sections_section_then_chunk_body.json")
            two_stage_sections_firstpara_chunk = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_two_stage_probe_retrieval_queries_sample_1000_v3__sections_firstpara_chunk_body.json")

            deep_dive_sections_chunk = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_deep_dive_probe_retrieval_queries_sample_1000_v3__sections_chunk_body.json")
            deep_dive_sections_section = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_deep_dive_probe_retrieval_queries_sample_1000_v3__sections_section_then_chunk_body.json")

            doc_base_map = {row["id"]: row for row in doc_eval_base["results"]}
            doc_sections_map = {row["id"]: row for row in doc_eval_sections["results"]}
            doc_sections_firstpara_map = {row["id"]: row for row in doc_eval_sections_firstpara["results"]}
            stage_base_map = {row["id"]: row for row in two_stage_base_chunk["rows"]}
            stage_sections_chunk_map = {row["id"]: row for row in two_stage_sections_chunk["rows"]}
            stage_sections_section_map = {row["id"]: row for row in two_stage_sections_section["rows"]}
            stage_sections_firstpara_chunk_map = {row["id"]: row for row in two_stage_sections_firstpara_chunk["rows"]}
            deep_chunk_map = {row["id"]: row for row in deep_dive_sections_chunk["rows"]}
            deep_section_map = {row["id"]: row for row in deep_dive_sections_section["rows"]}

            print("Loaded v3 stage-comparison artifacts.")
            """
        ),
        _md(
            """
            ## 1. Global reading

            Ce qu'il faut verifier ici:
            - est-ce que `sections` ameliore vraiment le stage 1 documentaire
            - est-ce que `section_then_chunk` aide vraiment le stage 2 local
            - si non, on ne le garde pas juste parce qu'il est plus sophistique
            """
        ),
        _code(
            """
            {
                "doc_eval_base": doc_eval_base["metrics"],
                "doc_eval_sections": doc_eval_sections["metrics"],
                "doc_eval_sections_firstpara": doc_eval_sections_firstpara["metrics"],
                "local_strategy_sections_body": local_audit_sections["summary"],
                "local_strategy_sections_firstpara_body": local_audit_sections_firstpara["summary"],
            }
            """
        ),
        _code(
            """
            def top_docs(row, limit=3):
                return row["document_hits"][:limit]

            def top_sections(row, limit=4):
                return row.get("section_hits", [])[:limit]

            def top_chunks(row, limit=4):
                key = "chunk_hits" if "chunk_hits" in row else "top_chunk_hits"
                return row[key][:limit]

            def expanded_context(row, limit=6):
                return row["expanded_context"][:limit]

            def compare_case(query_id):
                query = query_map[query_id]
                return {
                    "id": query_id,
                    "query": query["query"],
                    "expected_boi": query.get("expected_boi"),
                    "expected_behavior": query.get("expected_behavior"),
                    "doc_eval_base": {
                        "returned_boi": doc_base_map[query_id]["returned_boi"],
                        "hit@1": doc_base_map[query_id].get("hit@1"),
                        "hit@3": doc_base_map[query_id].get("hit@3"),
                    },
                    "doc_eval_sections": {
                        "returned_boi": doc_sections_map[query_id]["returned_boi"],
                        "hit@1": doc_sections_map[query_id].get("hit@1"),
                        "hit@3": doc_sections_map[query_id].get("hit@3"),
                    },
                    "doc_eval_sections_firstpara": {
                        "returned_boi": doc_sections_firstpara_map[query_id]["returned_boi"],
                        "hit@1": doc_sections_firstpara_map[query_id].get("hit@1"),
                        "hit@3": doc_sections_firstpara_map[query_id].get("hit@3"),
                    },
                    "stage_base_chunk_docs": top_docs(stage_base_map[query_id]),
                    "stage_sections_chunk_docs": top_docs(stage_sections_chunk_map[query_id]),
                    "stage_sections_firstpara_chunk_docs": top_docs(stage_sections_firstpara_chunk_map[query_id]),
                    "stage_sections_chunk_top_chunks": top_chunks(stage_sections_chunk_map[query_id]),
                    "stage_sections_firstpara_chunk_top_chunks": top_chunks(stage_sections_firstpara_chunk_map[query_id]),
                    "stage_sections_section_top_sections": top_sections(stage_sections_section_map[query_id]),
                    "stage_sections_section_top_chunks": top_chunks(stage_sections_section_map[query_id]),
                    "deep_dive_sections_chunk": expanded_context(deep_chunk_map[query_id]),
                    "deep_dive_sections_section": expanded_context(deep_section_map[query_id]),
                }
            """
        ),
        _md(
            """
            ## 2. Stage 1 document retrieval: what improved

            Cas a lire:
            - `q57`: `sections` corrige un vrai parent/child mismatch
            - `q32`: `sections_firstpara` transforme un ancien miss `q12` en confusion parent/enfant plus douce
            - `q01`: vrai top1 miss toujours ouvert
            - `q30`: dernier voisin/famille encore difficile
            """
        ),
        _code("""compare_case("q57")"""),
        _code("""compare_case("q32")"""),
        _code("""compare_case("q01")"""),
        _code("""compare_case("q30")"""),
        _md(
            """
            ## 3. Stage 2 passage selection: chunk vs section_then_chunk

            Ce qu'on cherche:
            - `chunk` doit rester passage-centric
            - `section_then_chunk` ne vaut la peine que s'il sort un extrait meilleur, pas juste un titre de section ou une introduction generique
            """
        ),
        _code("""compare_case("q03")"""),
        _code("""compare_case("q06")"""),
        _code("""compare_case("q14")"""),
        _code("""compare_case("q31")"""),
        _code("""compare_case("q33")"""),
        _code("""compare_case("q42")"""),
        _code("""compare_case("q43")"""),
        _code("""compare_case("q46")"""),
        _md(
            """
            ## 4. Deep-dive local expansion

            Le stage 3 n'est pas une boucle infinie. C'est une expansion locale bornee autour des meilleurs chunks.
            On verifie ici que cette expansion amene du contexte utile sans noyer le passage top1.
            """
        ),
        _code("""compare_case("q04")"""),
        _code("""compare_case("q36")"""),
        _md(
            """
            ## 5. Unsupported / false-premise reminders

            Pas d'abstention finale encore. Donc ces cas montrent surtout pourquoi il ne faut pas brancher le LLM tout de suite.
            """
        ),
        _code("""compare_case("u04")"""),
        _code("""compare_case("u05")"""),
        _code("""compare_case("u10")"""),
        _md(
            """
            ## 6. Engineering conclusion

            Lecture stricte attendue:
            - stage 1: `sections_firstpara` est le meilleur mode documentaire actuel
            - stage 2: `chunk` reste meilleur que `section_then_chunk`
            - stage 3: utile comme expansion locale, pas comme substitut au ranking
            - prochaine etape: continuer l'audit des derniers misses documentaires et stabiliser la gate d'abstention avant tout LLM
            """
        ),
    ]

    output_path = NOTEBOOKS_DIR / "06_stage_comparison_audit.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
