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
            # Gemini Shadow Eval Workbench

            Notebook méthodique pour tester plusieurs cas sur la preview LLM.

            Il montre pour chaque cas:
            - la question
            - les meilleurs documents stage 1
            - les chunks envoyés au LLM
            - le prompt exact
            - la réponse Gemini si la clé est disponible

            Ce notebook sert à auditer la qualité du prompt et du contexte avant toute promotion produit.
            """
        ),
        _code(
            """
            from pathlib import Path
            import sys
            import json

            PROJECT_ROOT = Path.cwd().resolve().parent
            SRC_ROOT = PROJECT_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from bofip_cleanroom.env_utils import load_default_env_files
            from bofip_cleanroom.jsonio import read_jsonl
            from bofip_cleanroom.preview_runtime import Phase8bPreviewRuntime
            from bofip_cleanroom.llm_preview import (
                build_citation_prompt,
                generate_preview_answer_with_retry,
                has_api_key,
                normalize_preview_answer,
            )

            _ = load_default_env_files()
            """
        ),
        _code(
            """
            PROVIDER = "gemini"
            MODEL = "gemini-2.5-flash"
            RUN_LLM = False
            MAX_ATTEMPTS = 5
            BASE_DELAY_SECONDS = 5.0
            REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "phase9_batch_preview_eval_gemini_v1.json"
            USE_CACHED_REPORT = REPORT_PATH.exists() and not RUN_LLM
            CASES = read_jsonl(PROJECT_ROOT / "data" / "interim" / "phase9_shadow_cases_v1.jsonl")
            runtime = None if USE_CACHED_REPORT else Phase8bPreviewRuntime.from_local_corpus(corpus="commentary", device="cpu")
            {"cases": len(CASES), "use_cached_report": USE_CACHED_REPORT, "report_path": str(REPORT_PATH)}
            """
        ),
        _code(
            """
            results = []
            if USE_CACHED_REPORT:
                report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
                by_case_id = {row["case_id"]: row for row in report["rows"]}
                for case in CASES:
                    row = by_case_id[case["case_id"]]
                    results.append(
                        {
                            "case": case,
                            "cached_row": row,
                            "prompt_text": row["prompt_text"],
                            "answer_text": row.get("answer_text", ""),
                            "raw_answer_text": row.get("raw_answer_text", row.get("answer_text", "")),
                            "answer_validation": row.get("answer_validation"),
                        }
                    )
            else:
                for case in CASES:
                    retrieval = runtime.retrieve(case["query"], top_docs=5, chunks_per_doc=3, max_chunks=8)
                    preview = (
                        generate_preview_answer_with_retry(
                            retrieval,
                            provider=PROVIDER,
                            model=MODEL,
                            max_attempts=MAX_ATTEMPTS,
                            base_delay_seconds=BASE_DELAY_SECONDS,
                        )
                        if RUN_LLM
                        else None
                    )
                    results.append(
                        {
                            "case": case,
                            "retrieval": retrieval,
                            "preview": preview,
                            "prompt_text": build_citation_prompt(retrieval),
                            "answer_text": None if preview is None else preview.answer_text,
                            "raw_answer_text": None if preview is None else preview.raw_answer_text,
                            "answer_validation": None if preview is None else preview.answer_validation,
                        }
                    )

            {"api_key_present": has_api_key(PROVIDER), "provider": PROVIDER, "model": MODEL, "cases": len(results), "cached": USE_CACHED_REPORT}
            """
        ),
        _md("## Contrôle rapide du format des réponses"),
        _code(
            """
            summary = []
            for item in results:
                cached_row = item.get("cached_row")
                answer_validation = item.get("answer_validation")
                if answer_validation is None:
                    retrieval_payload = cached_row["retrieval"] if cached_row is not None else Phase8bPreviewRuntime.as_dict(item["retrieval"])
                    _, _, answer_validation = normalize_preview_answer(
                        item.get("raw_answer_text") or item.get("answer_text") or "",
                        retrieval_payload=retrieval_payload,
                    )
                summary.append(
                    {
                        "case_id": item["case"]["case_id"],
                        "category": item["case"]["category"],
                        "answer_status": answer_validation.get("answer_status"),
                        "format_valid": answer_validation.get("valid"),
                        "has_conclusion": answer_validation.get("has_conclusion"),
                        "has_justification": answer_validation.get("has_justification"),
                        "has_limites": answer_validation.get("has_limits"),
                        "citation_count": answer_validation.get("citation_count"),
                        "errors": answer_validation.get("errors"),
                        "warnings": answer_validation.get("warnings"),
                    }
                )

            summary
            """
        ),
        _code(
            """
            for item in results:
                case = item["case"]
                cached_row = item.get("cached_row")
                print("=" * 120)
                print(case["case_id"], case["category"])
                print("QUESTION:", case["query"])
                print("NOTE:", case.get("note", ""))
                print("\\nSTAGE 1 DOCS")
                if cached_row is not None:
                    for hit in cached_row["retrieval"]["stage1_hits"]:
                        print(f"- #{hit['rank']} {hit['boi_reference']} | score={hit['score']:.4f}")
                        print(f"  {hit['title']}")
                else:
                    retrieval = item["retrieval"]
                    for hit in retrieval.stage1_hits:
                        print(f"- #{hit.rank} {hit.boi_reference} | score={hit.score:.4f}")
                        print(f"  {hit.title}")
                print("\\nSTAGE 2 CHUNKS")
                if cached_row is not None:
                    for chunk in cached_row["retrieval"]["stage2_chunks"]:
                        print(f"[{chunk['citation_id']}] {chunk['boi_reference']} | {chunk['section_path']}")
                        print(chunk["text"][:500].replace("\\n", " "))
                        print()
                else:
                    for chunk in retrieval.stage2_chunks:
                        print(f"[{chunk.citation_id}] {chunk.boi_reference} | {chunk.section_path}")
                        print(chunk.text[:500].replace("\\n", " "))
                        print()
                print("PROMPT")
                print(item["prompt_text"][:4000])
                print("\\nANSWER")
                if item["answer_text"] is None:
                    print("RUN_LLM=False")
                else:
                    print(item["answer_text"])
                    raw_answer_text = item.get("raw_answer_text") or ""
                    if raw_answer_text and raw_answer_text.strip() != item["answer_text"].strip():
                        print("\\nRAW MODEL OUTPUT")
                        print(raw_answer_text)
                print()
            """
        ),
    ]

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NOTEBOOKS_DIR / "10_gemini_shadow_eval_workbench.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
