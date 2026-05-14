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
            # LLM Preview Workbench

            Notebook isolé pour tester une réponse LLM au-dessus de la branche retrieval `phase8b`.

            Ce notebook montre:
            - la requête brute
            - le stage 1 documentaire
            - les chunks envoyés au LLM
            - le prompt citations-first
            - la réponse LLM si `OPENAI_API_KEY` est disponible

            Ce notebook n'est **pas** une promotion produit.
            Il sert à auditer un preview end-to-end en mode contrôlé.
            """
        ),
        _code(
            """
            from pathlib import Path
            import sys

            PROJECT_ROOT = Path.cwd().resolve().parent
            SRC_ROOT = PROJECT_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from bofip_cleanroom.env_utils import load_default_env_files
            from bofip_cleanroom.preview_runtime import Phase8bPreviewRuntime
            from bofip_cleanroom.llm_preview import (
                build_citation_prompt,
                generate_preview_answer,
                has_api_key,
            )

            _ = load_default_env_files()
            """
        ),
        _code(
            """
            QUERY = "Notre startup a le statut JEI et porte des travaux de recherche. Peut-elle recuperer sa creance de CIR tout de suite ?"
            RUN_LLM = False
            PROVIDER = "gemini"
            MODEL = "gemini-2.5-flash"
            """
        ),
        _code(
            """
            runtime = Phase8bPreviewRuntime.from_local_corpus(corpus="commentary", device="cpu")
            retrieval = runtime.retrieve(QUERY, top_docs=5, chunks_per_doc=3, max_chunks=8)

            {
                "query": retrieval.query,
                "lexical_query": retrieval.lexical_query,
                "acronym_expansions": retrieval.acronym_expansions,
                "source_confidences": retrieval.source_confidences,
            }
            """
        ),
        _md("## Stage 1 documents"),
        _code(
            """
            [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.boi_reference,
                    "title": hit.title,
                    "sources": hit.sources,
                    "ranks": hit.ranks,
                }
                for hit in retrieval.stage1_hits
            ]
            """
        ),
        _md("## Stage 2 chunks envoyés au LLM"),
        _code(
            """
            [
                {
                    "citation_id": chunk.citation_id,
                    "boi_reference": chunk.boi_reference,
                    "title": chunk.title,
                    "section_path": chunk.section_path,
                    "chunk_kind": chunk.chunk_kind,
                    "text": chunk.text[:1200],
                }
                for chunk in retrieval.stage2_chunks
            ]
            """
        ),
        _md("## Prompt"),
        _code(
            """
            prompt_text = build_citation_prompt(retrieval)
            print(prompt_text)
            """
        ),
        _md("## Réponse LLM"),
        _code(
            """
            if RUN_LLM:
                preview = generate_preview_answer(retrieval, provider=PROVIDER, model=MODEL)
                print(
                    {
                        "api_called": preview.api_called,
                        "provider": preview.provider,
                        "model": preview.model,
                        "contract_version": preview.contract_version,
                        "answer_valid": preview.answer_validation["valid"],
                    }
                )
                print()
                print(preview.answer_text)
            else:
                print({"api_key_present": has_api_key(PROVIDER), "provider": PROVIDER, "run_llm": RUN_LLM})
                print("Passe RUN_LLM = True si la clé du provider est disponible.")
            """
        ),
    ]

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NOTEBOOKS_DIR / "09_llm_preview_workbench.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
