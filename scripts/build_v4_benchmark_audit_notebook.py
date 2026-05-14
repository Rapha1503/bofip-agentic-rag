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
            # V4 Benchmark Audit

            Audit du benchmark `v4` sur `100` questions nouvelles:
            - `70` answerables
            - `30` unsupported

            Objectif:
            - mesurer l'effet d'un jeu de questions plus paraphrase et moins lexicalement confortable
            - comparer `Commentaire-only` et corpus `mixte`
            - comparer lexical, dense et hybride au niveau **documentaire**
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

            def load_json(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))

            commentary_lex_base = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_5666__retrieval_queries_full_v4__base.json")
            commentary_lex_sections = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_5666__retrieval_queries_full_v4__sections.json")
            commentary_lex_sections_firstpara = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_5666__retrieval_queries_full_v4__sections_firstpara.json")
            commentary_lex_sections_leads = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_5666__retrieval_queries_full_v4__sections_leads.json")
            mixed_lex_base = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_6295__retrieval_queries_full_v4__base.json")
            mixed_lex_sections = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_6295__retrieval_queries_full_v4__sections.json")
            mixed_lex_sections_firstpara = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_lexical_eval_raw_docs_sample_6295__retrieval_queries_full_v4__sections_firstpara.json")

            commentary_dense = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_dense_eval_raw_docs_sample_5666__retrieval_queries_full_v4__sections_firstpara__intfloat__multilingual-e5-base.json")
            commentary_chunk_dense = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_dense_eval_chunks_section_window_sample_5666__full__intfloat__multilingual-e5-base.json")
            mixed_dense = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_dense_eval_raw_docs_sample_6295__retrieval_queries_full_v4__sections_firstpara__intfloat__multilingual-e5-base.json")

            commentary_hybrid = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__docsections__densesections_firstpara__intfloat__multilingual-e5-base_lw1p0_dw1p0.json")
            commentary_hybrid_best = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__lexbase_sections_leads__densesections_firstpara__intfloat__multilingual-e5-base__base1p0_chunk_dense2p0_dense1p0_sections_leads2p0.json")
            mixed_hybrid = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_doc_hybrid_eval_raw_docs_sample_6295__retrieval_queries_full_v4__docsections__densesections_firstpara__intfloat__multilingual-e5-base_lw1p0_dw1p0.json")

            commentary_fail = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_failure_analysis_phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__balanced_sections_leads.json")
            mixed_fail = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_failure_analysis_phase3_doc_lexical_eval_raw_docs_sample_6295__retrieval_queries_full_v4__sections.json")

            commentary_local = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_local_strategy_audit_retrieval_queries_full_v4__sections_body_5666.json")
            mixed_local = load_json(PROJECT_ROOT / "data" / "reports" / "phase3_local_strategy_audit_retrieval_queries_full_v4__sections_body_6295.json")
            """
        ),
        _md(
            """
            ## 1. Headline metrics
            """
        ),
        _code(
            """
            {
                "commentary_lexical": {
                    "base": commentary_lex_base["metrics"],
                    "sections": commentary_lex_sections["metrics"],
                    "sections_firstpara": commentary_lex_sections_firstpara["metrics"],
                    "sections_leads": commentary_lex_sections_leads["metrics"],
                },
                "mixed_lexical": {
                    "base": mixed_lex_base["metrics"],
                    "sections": mixed_lex_sections["metrics"],
                    "sections_firstpara": mixed_lex_sections_firstpara["metrics"],
                },
                "commentary_dense": commentary_dense["metrics"],
                "commentary_chunk_dense": commentary_chunk_dense["metrics"],
                "mixed_dense": mixed_dense["metrics"],
                "commentary_hybrid": commentary_hybrid["metrics"],
                "commentary_hybrid_best_current": commentary_hybrid_best["metrics"],
                "mixed_hybrid": mixed_hybrid["metrics"],
            }
            """
        ),
        _md(
            """
            ## 2. Reading the result correctly

            Le benchmark `v4` casse le confort lexical:
            - les questions reprennent moins les formulations des titres BOI
            - elles ressemblent davantage a des demandes utilisateur
            - le benchmark precedent etait donc trop optimiste pour le stage 1 documentaire
            """
        ),
        _code(
            """
            {
                "commentary_miss_categories": commentary_fail["reports"][0]["miss_categories"],
                "mixed_miss_categories": mixed_fail["reports"][0]["miss_categories"],
                "commentary_miss_count": commentary_fail["reports"][0]["miss_count"],
                "mixed_miss_count": mixed_fail["reports"][0]["miss_count"],
            }
            """
        ),
        _md(
            """
            ## 3. Stage 2 local chunk selection under v4
            """
        ),
        _code(
            """
            {
                "commentary_local": commentary_local["summary"],
                "mixed_local": mixed_local["summary"],
            }
            """
        ),
        _md(
            """
            ## 4. Representative misses
            """
        ),
        _code(
            """
            commentary_fail["reports"][0]["misses"][:15]
            """
        ),
        _code(
            """
            mixed_fail["reports"][0]["misses"][:15]
            """
        ),
        _md(
            """
            ## 5. Conclusion

            Ce notebook dit maintenant quelque chose de plus fin:
            - le parsing/chunking restent stables
            - le benchmark `v4` est beaucoup plus exigeant
            - le meilleur baseline actuel combine `base + sections_leads + doc_dense + chunk_dense`
            - on a un vrai gain sur le retrieval documentaire, mais pas encore un niveau satisfaisant pour brancher sereinement un LLM
            """
        ),
    ]

    output_path = NOTEBOOKS_DIR / "08_v4_benchmark_audit.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
