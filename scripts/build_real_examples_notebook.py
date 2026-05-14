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
            # Real Examples: Parsing, Chunking, Retrieval

            Notebook de transparence sur le clean-room BOFIP-only.

            Portée volontaire:
            - parsing réel BOFIP
            - chunking réel
            - retrieval réel
            - pas de génération LLM ici

            Point important:
            - les exemples très faibles du type `BOI-ANNX-*`, `BOI-FORM-*`, `Zone 1 :`, `Numéro SIREN :` proviennent d'anciens essais mixtes ou de stratégies rejetées
            - le baseline actuel retenu pour avancer est:
              - `Commentaire` only
              - `section_window`
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

            def load_jsonl(path):
                return read_jsonl(Path(path))

            raw_docs_50 = [raw_document_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "raw_docs_sample_50.jsonl")]
            raw_docs_200 = [raw_document_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "raw_docs_sample_200.jsonl")]
            raw_docs_1000 = [raw_document_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "raw_docs_sample_1000.jsonl")]

            chunks_10_pp = [chunk_node_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "chunks_paragraph_preserving_sample_10.jsonl")]
            chunks_10_sw = [chunk_node_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_10.jsonl")]
            chunks_50_sw = [chunk_node_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_50.jsonl")]
            chunks_200_sw = [chunk_node_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_200.jsonl")]
            chunks_1000_sw = [chunk_node_from_dict(item) for item in load_jsonl(PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_1000.jsonl")]

            chunk_by_id_200 = {chunk.chunk_id: chunk for chunk in chunks_200_sw}
            chunk_by_id_1000 = {chunk.chunk_id: chunk for chunk in chunks_1000_sw}

            reports = {
                "parse_full": load_json(PROJECT_ROOT / "data" / "reports" / "phase0_full_parse_audit.json"),
                "sample50_summary": load_json(PROJECT_ROOT / "data" / "reports" / "phase1_extract_summary_sample_50.json"),
                "sample200_summary": load_json(PROJECT_ROOT / "data" / "reports" / "phase1_extract_summary_sample_200.json"),
                "sample1000_summary": load_json(PROJECT_ROOT / "data" / "reports" / "phase1_extract_summary_sample_1000.json"),
                "chunk50": load_json(PROJECT_ROOT / "data" / "reports" / "phase2_chunks_section_window_sample_50.json"),
                "chunk200": load_json(PROJECT_ROOT / "data" / "reports" / "phase2_chunks_section_window_sample_200.json"),
                "chunk1000": load_json(PROJECT_ROOT / "data" / "reports" / "phase2_chunks_section_window_sample_1000.json"),
                "lex200": load_json(PROJECT_ROOT / "data" / "reports" / "phase3_batch_eval_chunks_section_window_sample_200.json"),
                "lex1000": load_json(PROJECT_ROOT / "data" / "reports" / "phase3_batch_eval_chunks_section_window_sample_1000.json"),
                "dense50": load_json(PROJECT_ROOT / "data" / "reports" / "phase3_dense_eval_chunks_section_window_sample_50_intfloat__multilingual-e5-base.json"),
                "dense200": load_json(PROJECT_ROOT / "data" / "reports" / "phase3_dense_eval_chunks_section_window_sample_200_intfloat__multilingual-e5-base.json"),
                "dense1000": load_json(PROJECT_ROOT / "data" / "reports" / "phase3_dense_eval_chunks_section_window_sample_1000_intfloat__multilingual-e5-base.json"),
                "hyb200_w": load_json(PROJECT_ROOT / "data" / "reports" / "phase3_hybrid_eval_chunks_section_window_sample_200_intfloat__multilingual-e5-base_lw2p0_dw1p0.json"),
                "hyb1000_w": load_json(PROJECT_ROOT / "data" / "reports" / "phase3_hybrid_eval_chunks_section_window_sample_1000_intfloat__multilingual-e5-base_lw2p0_dw1p0.json"),
            }

            print("Loaded clean-room artifacts.")
            """
        ),
        _code(
            """
            {
                "content_docs_total": reports["parse_full"]["documents_considered"],
                "parse_failures": reports["parse_full"]["failure_count"],
                "sample_50_docs": reports["sample50_summary"]["sample_size"],
                "sample_200_docs": reports["sample200_summary"]["sample_size"],
                "sample_1000_docs": reports["sample1000_summary"]["sample_size"],
                "chunks_50_section_window": reports["chunk50"]["chunk_count"],
                "chunks_200_section_window": reports["chunk200"]["chunk_count"],
                "chunks_1000_section_window": reports["chunk1000"]["chunk_count"],
                "lexical_200": reports["lex200"]["metrics"],
                "dense_200": reports["dense200"]["metrics"],
                "hybrid_weighted_200": reports["hyb200_w"]["metrics"],
                "lexical_1000": reports["lex1000"]["metrics"],
                "dense_1000": reports["dense1000"]["metrics"],
                "hybrid_weighted_1000": reports["hyb1000_w"]["metrics"],
            }
            """
        ),
        _md(
            """
            ## 1. Pourquoi certains petits chunks vus avant étaient mauvais

            La question n'est pas "est-ce un chunk ?" mais "est-ce un chunk acceptable pour le baseline ?".

            Réponse:
            - oui, les extraits très courts étaient bien des chunks
            - non, ils n'étaient pas acceptables comme baseline
            - c'est précisément pour ça que `paragraph_preserving` n'a pas été retenu
            - et pour ça aussi que le baseline actuel ne mélange pas tout BOFIP indistinctement
            """
        ),
        _code(
            """
            def shortest_chunks(chunks, n=12):
                rows = []
                for chunk in sorted(chunks, key=lambda c: (c.token_count, len(c.text), c.boi_reference))[:n]:
                    rows.append({
                        "tokens": chunk.token_count,
                        "boi_reference": chunk.boi_reference,
                        "chunk_kind": chunk.chunk_kind,
                        "text": chunk.text[:160],
                    })
                return rows

            {
                "old_sample10_paragraph_preserving_shortest": shortest_chunks(chunks_10_pp, 12),
                "old_sample10_section_window_shortest": shortest_chunks(chunks_10_sw, 12),
                "current_sample1000_section_window_shortest": shortest_chunks(chunks_1000_sw, 12),
            }
            """
        ),
        _md(
            """
            ## 2. Quels documents réels ont été pris dans le subset `50` commentaires
            """
        ),
        _code(
            """
            [
                {
                    "boi_reference": doc.boi_reference,
                    "title": doc.title,
                    "sections": len(doc.sections),
                    "paragraphs": len(doc.paragraphs),
                    "tables": len(doc.tables),
                    "legal_refs": len(doc.legal_refs),
                }
                for doc in raw_docs_50
            ][:20]
            """
        ),
        _code(
            """
            def get_doc(raw_docs, boi_reference):
                for doc in raw_docs:
                    if doc.boi_reference == boi_reference:
                        return doc
                raise KeyError(boi_reference)

            def doc_overview(raw_docs, boi_reference, section_limit=12, paragraph_limit=12, table_limit=3):
                doc = get_doc(raw_docs, boi_reference)
                return {
                    "boi_reference": doc.boi_reference,
                    "title": doc.title,
                    "content_type": doc.content_type,
                    "publication_date": doc.publication_date,
                    "source_url": doc.source_url,
                    "section_count": len(doc.sections),
                    "paragraph_count": len(doc.paragraphs),
                    "table_count": len(doc.tables),
                    "sections_preview": [
                        {
                            "level": section.level,
                            "title": section.title,
                            "path": section.path,
                        }
                        for section in doc.sections[:section_limit]
                    ],
                    "paragraphs_preview": [
                        {
                            "order_index": p.order_index,
                            "section_id": p.section_id,
                            "paragraph_number": p.paragraph_number,
                            "text": p.text[:240],
                        }
                        for p in doc.paragraphs[:paragraph_limit]
                    ],
                    "tables_preview": [
                        {
                            "caption": table.caption,
                            "headers": table.headers,
                            "rows": table.rows[:5],
                            "linearized_text": table.linearized_text[:400],
                        }
                        for table in doc.tables[:table_limit]
                    ],
                }

            doc_overview(raw_docs_50, "BOI-BIC-CHAMP-80-20-20-20-20240703")
            """
        ),
        _code(
            """
            doc_overview(raw_docs_50, "BOI-IS-GPE-50-10-30-20210811")
            """
        ),
        _code(
            """
            doc_overview(raw_docs_50, "BOI-RES-TVA-000074-20210309")
            """
        ),
        _md(
            """
            ## 3. À quoi ressemble un document complet une fois transformé en chunks

            Ici on regarde **tous** les chunks `section_window` d'un document réel.
            """
        ),
        _code(
            """
            def chunks_for_doc(chunks, boi_reference):
                return [chunk for chunk in chunks if chunk.boi_reference == boi_reference]

            def chunk_view(chunks, boi_reference):
                rows = []
                for idx, chunk in enumerate(chunks_for_doc(chunks, boi_reference), start=1):
                    rows.append({
                        "idx": idx,
                        "chunk_id": chunk.chunk_id,
                        "chunk_kind": chunk.chunk_kind,
                        "tokens": chunk.token_count,
                        "section_path": " > ".join(chunk.section_path),
                        "paragraph_range": chunk.paragraph_range,
                        "text": chunk.text,
                    })
                return rows

            chunk_view(chunks_200_sw, "BOI-BIC-CHAMP-80-20-20-20-20240703")
            """
        ),
        _code(
            """
            chunk_view(chunks_200_sw, "BOI-IS-GPE-20-20-70-20190731")
            """
        ),
        _md(
            """
            ## 4. Exemples réels de retrieval sur le subset `200`

            On montre les sorties réelles:
            - lexical
            - dense
            - hybrid pondéré
            """
        ),
        _code(
            """
            def report_rows(report):
                return {row["id"]: row for row in report["results"]}

            rows_lex_200 = report_rows(reports["lex200"])
            rows_dense_200 = report_rows(reports["dense200"])
            rows_hyb_200 = report_rows(reports["hyb200_w"])

            def chunk_snippet(chunk_map, chunk_id):
                chunk = chunk_map.get(chunk_id)
                if chunk is None:
                    return None
                return {
                    "boi_reference": chunk.boi_reference,
                    "chunk_id": chunk.chunk_id,
                    "chunk_kind": chunk.chunk_kind,
                    "tokens": chunk.token_count,
                    "section_path": " > ".join(chunk.section_path),
                    "text": chunk.text[:320],
                }

            def retrieval_trace_200(query_id):
                lex = rows_lex_200[query_id]
                dense = rows_dense_200[query_id]
                hyb = rows_hyb_200[query_id]
                return {
                    "query_id": query_id,
                    "query": lex["query"],
                    "expected_boi": lex["expected_boi"],
                    "lexical_top3": [chunk_snippet(chunk_by_id_200, hit["chunk_id"]) for hit in lex["top_hits"][:3]],
                    "dense_top3": [chunk_snippet(chunk_by_id_200, hit["chunk_id"]) for hit in dense["top_hits"][:3]],
                    "hybrid_top3_docs": hyb["top_hits"][:3],
                }

            retrieval_trace_200("q01")
            """
        ),
        _code(
            """
            retrieval_trace_200("q09")
            """
        ),
        _code(
            """
            retrieval_trace_200("q12")
            """
        ),
        _code(
            """
            retrieval_trace_200("q22")
            """
        ),
        _md(
            """
            ## 5. Exemples réels de retrieval sur le subset `1000`

            Même logique, mais avec beaucoup plus de distracteurs.
            """
        ),
        _code(
            """
            rows_lex_1000 = report_rows(reports["lex1000"])
            rows_dense_1000 = report_rows(reports["dense1000"])
            rows_hyb_1000 = report_rows(reports["hyb1000_w"])

            def retrieval_trace_1000(query_id):
                lex = rows_lex_1000[query_id]
                dense = rows_dense_1000[query_id]
                hyb = rows_hyb_1000[query_id]
                return {
                    "query_id": query_id,
                    "query": lex["query"],
                    "expected_boi": lex["expected_boi"],
                    "lexical_top3": [chunk_snippet(chunk_by_id_1000, hit["chunk_id"]) for hit in lex["top_hits"][:3]],
                    "dense_top3": [chunk_snippet(chunk_by_id_1000, hit["chunk_id"]) for hit in dense["top_hits"][:3]],
                    "hybrid_top3_docs": hyb["top_hits"][:3],
                }

            retrieval_trace_1000("q01")
            """
        ),
        _code(
            """
            retrieval_trace_1000("q12")
            """
        ),
        _code(
            """
            retrieval_trace_1000("q22")
            """
        ),
        _md(
            """
            ## 6. Conclusion actuelle, sans bullshit

            Ce notebook montre l'état réel du clean-room:
            - parsing: oui, robuste à grande échelle
            - chunking baseline: oui, `section_window`
            - retrieval lexical: bon baseline
            - dense: encore trop confus à mesure qu'on augmente les distracteurs
            - hybrid pondéré: utile, mais pas miracle
            - abstention: pas encore implémentée

            Donc:
            - non, tout n'est pas “déjà bon”
            - oui, on a maintenant un socle transparent et inspectable
            - oui, on peut continuer sans se mentir sur ce qui fonctionne réellement
            """
        ),
    ]

    output_path = NOTEBOOKS_DIR / "03_real_examples_parsing_chunking_retrieval.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
