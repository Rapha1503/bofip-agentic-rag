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
            # Query Workbench

            Notebook interactif pour auditer une requete de bout en bout dans le clean-room BOFiP.

            Ce notebook montre:
            - la requete brute, sa tokenisation et sa version vectorielle
            - le stage 1 lexical (plusieurs modes)
            - le stage 1 dense documentaire
            - le stage 1 dense documentaire derive des chunks si disponible
            - le stage 1 hybride documentaire
            - les liens/familles BOI des meilleurs documents
            - les meilleurs chunks locaux dans les documents remontes

            Ce n'est **pas** un notebook de generation LLM.
            """
        ),
        _code(
            """
            from pathlib import Path
            import json
            import sys
            import numpy as np

            PROJECT_ROOT = Path.cwd().resolve().parent
            SRC_ROOT = PROJECT_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from bofip_cleanroom.dense_retrieval import DenseDocumentIndex, DenseEncoder, DEFAULT_DENSE_MODEL
            from bofip_cleanroom.direct_chunk_retrieval import DirectChunkRetriever, Stage1DocumentHit
            from bofip_cleanroom.family_guided_retrieval import FamilyGuidedRetriever, PriorDocumentHit
            from bofip_cleanroom.hybrid_retrieval import RankedDoc, reciprocal_rank_fuse
            from bofip_cleanroom.jsonio import read_jsonl
            from bofip_cleanroom.lexical_retrieval import (
                DocumentLexicalIndex,
                LexicalBM25Index,
                chunk_search_text_body,
                document_search_text_sections_leads,
                get_document_search_text_fn,
                tokenize,
            )
            from bofip_cleanroom.models import chunk_node_from_dict, raw_document_from_dict
            """
        ),
        _code(
            """
            # User inputs
            QUERY = "Notre startup a le statut JEI et porte des travaux de recherche. Peut-elle recuperer sa creance de CIR tout de suite ?"
            CORPUS = "mixed"  # "commentary" or "mixed"
            TOP_DOCS = 5
            TOP_CHUNKS_PER_DOC = 4
            MODEL_NAME = DEFAULT_DENSE_MODEL
            """
        ),
        _code(
            """
            CORPUS_CONFIG = {
                "commentary": {
                    "raw_docs": PROJECT_ROOT / "data" / "interim" / "raw_docs_sample_5666.jsonl",
                    "chunks": PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_5666.jsonl",
                    "doc_dense_cache": PROJECT_ROOT / "data" / "interim" / "doc_dense_cache_5666_sections_firstpara_e5.npy",
                    "doc_dense_meta": PROJECT_ROOT / "data" / "interim" / "doc_dense_cache_5666_sections_firstpara_e5.json",
                    "chunk_dense_cache": PROJECT_ROOT / "data" / "interim" / "chunk_dense_cache_5666_full_e5.npy",
                    "label": "Commentaire only (5666 docs)",
                },
                "mixed": {
                    "raw_docs": PROJECT_ROOT / "data" / "interim" / "raw_docs_sample_6295.jsonl",
                    "chunks": PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_6295.jsonl",
                    "doc_dense_cache": PROJECT_ROOT / "data" / "interim" / "doc_dense_cache_6295_sections_firstpara_e5.npy",
                    "doc_dense_meta": PROJECT_ROOT / "data" / "interim" / "doc_dense_cache_6295_sections_firstpara_e5.json",
                    "chunk_dense_cache": None,
                    "label": "Full mixed content (6295 docs)",
                },
            }

            if CORPUS not in CORPUS_CONFIG:
                raise ValueError(f"Unsupported corpus: {CORPUS}")

            cfg = CORPUS_CONFIG[CORPUS]
            raw_docs = [raw_document_from_dict(item) for item in read_jsonl(cfg["raw_docs"])]
            chunks = [chunk_node_from_dict(item) for item in read_jsonl(cfg["chunks"])]
            raw_doc_map = {doc.boi_reference: doc for doc in raw_docs}
            chunks_by_ref = {}
            for chunk in chunks:
                chunks_by_ref.setdefault(chunk.boi_reference, []).append(chunk)

            print({"corpus": cfg["label"], "documents": len(raw_docs), "chunks": len(chunks)})
            """
        ),
        _md(
            """
            ## 1. Query normalization and vector
            """
        ),
        _code(
            """
            query_tokens = tokenize(QUERY)
            encoder = DenseEncoder(MODEL_NAME)
            query_embedding = encoder.encode_queries([QUERY])[0]

            {
                "query": QUERY,
                "token_count": len(query_tokens),
                "tokens": query_tokens,
                "embedding_dim": int(query_embedding.shape[0]),
                "embedding_norm": float(np.linalg.norm(query_embedding)),
                "embedding_head": [round(float(x), 6) for x in query_embedding[:16]],
            }
            """
        ),
        _md(
            """
            ## 2. Stage 1 lexical document retrieval
            """
        ),
        _code(
            """
            lexical_modes = ["base", "sections", "sections_firstpara", "sections_leads"]
            lexical_results = {}
            for mode in lexical_modes:
                search_text_fn = document_search_text_sections_leads if mode == "sections_leads" else get_document_search_text_fn(mode)
                index = DocumentLexicalIndex(raw_docs, search_text_fn=search_text_fn)
                hits = index.search_documents(QUERY, top_k=TOP_DOCS)
                lexical_results[mode] = [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 4),
                        "boi_reference": hit.boi_reference,
                        "title": raw_doc_map[hit.boi_reference].title,
                        "publication_date": raw_doc_map[hit.boi_reference].publication_date,
                    }
                    for hit in hits
                ]

            lexical_results
            """
        ),
        _md(
            """
            ## 3. Stage 1 dense document retrieval
            """
        ),
        _code(
            """
            dense_embeddings = np.load(cfg["doc_dense_cache"])
            dense_index = DenseDocumentIndex(raw_docs, dense_embeddings)
            dense_hits = dense_index.search_from_vector(query_embedding, top_k=TOP_DOCS)
            dense_results = [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.boi_reference,
                    "title": hit.document.title,
                    "publication_date": hit.document.publication_date,
                }
                for hit in dense_hits
            ]
            dense_results
            """
        ),
        _md(
            """
            ## 3b. Stage 1 chunk-dense document retrieval
            """
        ),
        _code(
            """
            chunk_dense_results = []
            if cfg["chunk_dense_cache"] and cfg["chunk_dense_cache"].exists():
                from bofip_cleanroom.dense_retrieval import DenseIndex

                chunk_dense_embeddings = np.load(cfg["chunk_dense_cache"])
                chunk_dense_index = DenseIndex(chunks, chunk_dense_embeddings)
                chunk_dense_hits = chunk_dense_index.search_documents_from_vector(query_embedding, top_k=TOP_DOCS)
                chunk_dense_results = [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 6),
                        "boi_reference": hit.boi_reference,
                        "title": raw_doc_map[hit.boi_reference].title,
                        "best_chunk_id": hit.best_chunk.chunk_id,
                        "best_chunk_section": " > ".join(hit.best_chunk.section_path),
                    }
                    for hit in chunk_dense_hits
                ]
            else:
                chunk_dense_results = [{"info": "chunk-dense cache not available for this corpus"}]
            chunk_dense_results
            """
        ),
        _md(
            """
            ## 4. Stage 1 hybrid document retrieval
            """
        ),
        _code(
            """
            hybrid_hits_simple = reciprocal_rank_fuse(
                {
                    "lexical": [
                        RankedDoc(
                            boi_reference=row["boi_reference"],
                            score=float(row["score"]),
                            rank=row["rank"],
                            source="lexical",
                        )
                        for row in lexical_results["sections"]
                    ],
                    "dense": [
                        RankedDoc(
                            boi_reference=row["boi_reference"],
                            score=float(row["score"]),
                            rank=row["rank"],
                            source="dense",
                        )
                        for row in dense_results
                    ],
                },
                top_k=TOP_DOCS,
                rank_constant=60,
                source_weights={"lexical": 1.0, "dense": 1.0},
            )
            hybrid_results_simple = [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.boi_reference,
                    "sources": hit.sources,
                    "ranks": hit.ranks,
                    "title": raw_doc_map[hit.boi_reference].title,
                }
                for hit in hybrid_hits_simple
            ]

            hybrid_hits_promoted = reciprocal_rank_fuse(
                {
                    "base": [
                        RankedDoc(
                            boi_reference=row["boi_reference"],
                            score=float(row["score"]),
                            rank=row["rank"],
                            source="base",
                        )
                        for row in lexical_results["base"]
                    ],
                    "sections_leads": [
                        RankedDoc(
                            boi_reference=row["boi_reference"],
                            score=float(row["score"]),
                            rank=row["rank"],
                            source="sections_leads",
                        )
                        for row in lexical_results["sections_leads"]
                    ],
                    "dense": [
                        RankedDoc(
                            boi_reference=row["boi_reference"],
                            score=float(row["score"]),
                            rank=row["rank"],
                            source="dense",
                        )
                        for row in dense_results
                    ],
                    **(
                        {
                            "chunk_dense": [
                                RankedDoc(
                                    boi_reference=row["boi_reference"],
                                    score=float(row["score"]),
                                    rank=row["rank"],
                                    source="chunk_dense",
                                )
                                for row in chunk_dense_results
                                if "boi_reference" in row
                            ]
                        }
                        if chunk_dense_results and "boi_reference" in chunk_dense_results[0]
                        else {}
                    ),
                },
                top_k=TOP_DOCS,
                rank_constant=60,
                source_weights=(
                    {"base": 1.0, "sections_leads": 2.0, "dense": 1.0, "chunk_dense": 2.0}
                    if chunk_dense_results and "boi_reference" in chunk_dense_results[0]
                    else {"base": 1.0, "sections_leads": 2.0, "dense": 1.0}
                ),
            )
            hybrid_results_promoted = [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.boi_reference,
                    "sources": hit.sources,
                    "ranks": hit.ranks,
                    "title": raw_doc_map[hit.boi_reference].title,
                }
                for hit in hybrid_hits_promoted
            ]

            {
                "simple_sections_plus_dense": hybrid_results_simple,
                "promoted_current_pipeline": hybrid_results_promoted,
            }
            """
        ),
        _md(
            """
            ## 5. Promoted stage 2: direct local chunk retrieval

            Baseline passage promue actuelle:
            - stage 1 documentaire multiview hybride
            - puis recherche locale de chunks dans les top docs stage 1
            - `chunk_mode = full`

            Le family-guided reste plus bas comme variante experimentale.
            """
        ),
        _code(
            """
            direct_retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
            direct_stage2 = direct_retriever.search(
                QUERY,
                lexical_query=QUERY,
                stage1_hits=[
                    Stage1DocumentHit(
                        rank=row["rank"],
                        score=float(row["score"]),
                        boi_reference=row["boi_reference"],
                    )
                    for row in hybrid_results_promoted
                ],
                top_docs=5,
                chunks_per_doc=3,
                max_chunks=8,
            )

            {
                "document_hits": hybrid_results_promoted,
                "direct_chunk_hits": [
                    {
                        "global_rank": hit.global_rank,
                        "boi_reference": hit.boi_reference,
                        "document_rank": hit.document_rank,
                        "local_rank": hit.local_rank,
                        "local_score": round(hit.local_score, 4),
                        "section_path": " > ".join(hit.chunk.section_path),
                        "chunk_id": hit.chunk.chunk_id,
                        "chunk_kind": hit.chunk.chunk_kind,
                        "text": hit.chunk.text[:800],
                    }
                    for hit in direct_stage2.chunk_hits
                ],
            }
            """
        ),
        _md(
            """
            ## 6. Family / related documents for the top hybrid hits
            """
        ),
        _code(
            """
            def ref_core(ref):
                parts = ref.split("-")
                if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
                    parts = parts[:-1]
                return parts

            def parent_refs(ref):
                core = ref_core(ref)
                parents = []
                for cut in range(len(core) - 1, 1, -1):
                    candidate = "-".join(core[:cut])
                    for doc in raw_docs:
                        other = ref_core(doc.boi_reference)
                        if other == core[:cut]:
                            parents.append(doc.boi_reference)
                return sorted(set(parents))

            def child_refs(ref, limit=8):
                core = ref_core(ref)
                children = []
                for doc in raw_docs:
                    other = ref_core(doc.boi_reference)
                    if len(other) > len(core) and other[:len(core)] == core:
                        children.append(doc.boi_reference)
                return sorted(children)[:limit]

            family_rows = []
            for row in hybrid_results_promoted[:3]:
                doc = raw_doc_map[row["boi_reference"]]
                family_rows.append(
                    {
                        "boi_reference": row["boi_reference"],
                        "title": doc.title,
                        "xml_relations": [rel.value for rel in doc.relations[:8]],
                        "parents": parent_refs(row["boi_reference"])[:5],
                        "children": child_refs(row["boi_reference"], limit=8),
                    }
                )

            family_rows
            """
        ),
        _md(
            """
            ## 7. Experimental family-guided variant

            Variante utile pour diagnostiquer les voisinages documentaires BOFiP.
            Elle n'est plus la baseline promue pour retrouver le bon passage.
            """
        ),
        _code(
            """
            family_retriever = FamilyGuidedRetriever(
                raw_docs,
                chunks,
                family_doc_mode="sections_leads",
                family_doc_stem=True,
                local_chunk_mode="body",
            )
            family_guided = family_retriever.search(
                QUERY,
                lexical_query=QUERY,
                stage1_hits=[
                    PriorDocumentHit(
                        rank=row["rank"],
                        score=float(row["score"]),
                        boi_reference=row["boi_reference"],
                    )
                    for row in hybrid_results_promoted
                ],
                family_top_docs=2,
                top_docs=5,
                chunks_per_doc=2,
                max_chunks=8,
                tail_weight=0.25,
                preserve_stage1_top1=True,
            )

            {
                "family_anchor_bois": family_guided.family_selection.anchor_references,
                "family_prefixes": [list(prefix) for prefix in family_guided.family_selection.prefixes],
                "family_members_count": len(family_guided.family_selection.members),
                "family_document_hits": [
                    {
                        "rank": hit.rank,
                        "boi_reference": hit.boi_reference,
                        "combined_score": round(hit.combined_score, 6),
                        "family_rank": hit.family_rank,
                        "prior_rank": hit.prior_rank,
                        "tail_rank": hit.tail_rank,
                        "tail_score": round(hit.tail_score, 6) if hit.tail_score is not None else None,
                        "title": hit.title,
                    }
                    for hit in family_guided.document_hits
                ],
                "family_chunk_hits": [
                    {
                        "global_rank": hit.global_rank,
                        "document_rank": hit.document_rank,
                        "local_rank": hit.local_rank,
                        "boi_reference": hit.boi_reference,
                        "section_path": " > ".join(hit.chunk.section_path),
                        "chunk_id": hit.chunk.chunk_id,
                        "text": hit.chunk.text[:800],
                    }
                    for hit in family_guided.chunk_hits
                ],
            }
            """
        ),
        _md(
            """
            ## 8. Local chunk retrieval inside the top hybrid documents

            Vue document par document, utile pour comprendre ce qui est disponible localement
            avant le rerank direct.
            """
        ),
        _code(
            """
            def local_chunk_hits(boi_reference, top_k=TOP_CHUNKS_PER_DOC):
                doc_chunks = chunks_by_ref.get(boi_reference, [])
                index = LexicalBM25Index(doc_chunks, search_text_fn=chunk_search_text_body)
                hits = index.search(QUERY, top_k=top_k)
                return [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 4),
                        "chunk_id": hit.chunk.chunk_id,
                        "chunk_kind": hit.chunk.chunk_kind,
                        "token_count": hit.chunk.token_count,
                        "section_path": " > ".join(hit.chunk.section_path),
                        "paragraph_range": hit.chunk.paragraph_range,
                        "text": hit.chunk.text[:800],
                    }
                    for hit in hits
                ]

            local_results = {
                row["boi_reference"]: local_chunk_hits(row["boi_reference"])
                for row in hybrid_results_promoted[:3]
            }
            local_results
            """
        ),
        _md(
            """
            ## 9. Legacy lexical two-stage trace

            Reference utile pour comparaison historique. La baseline passage promue n'est plus ce retriever.
            """
        ),
        _code(
            """
            from bofip_cleanroom.two_stage_retrieval import TwoStageLexicalRetriever

            retriever = TwoStageLexicalRetriever(
                raw_docs,
                chunks,
                document_mode="sections_leads",
                local_chunk_mode="body",
                local_strategy="chunk",
            )
            trace = retriever.search(QUERY, top_docs=TOP_DOCS, chunks_per_doc=3, max_chunks=8)

            {
                "document_hits": [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 4),
                        "boi_reference": hit.boi_reference,
                        "title": raw_doc_map[hit.boi_reference].title,
                    }
                    for hit in trace.document_hits
                ],
                "chunk_hits": [
                    {
                        "global_rank": hit.global_rank,
                        "document_rank": hit.document_rank,
                        "local_rank": hit.local_rank,
                        "local_score": round(hit.local_score, 4),
                        "boi_reference": hit.boi_reference,
                        "section_path": " > ".join(hit.chunk.section_path),
                        "chunk_id": hit.chunk.chunk_id,
                        "text": hit.chunk.text[:800],
                    }
                    for hit in trace.chunk_hits
                ],
            }
            """
        ),
        _md(
            """
            ## 10. Reading guide

            Interprete le notebook comme suit:
            - si lexical, dense et hybride divergent fortement sur le top document, la requete est dure
            - si le bon document apparait dans `document_hits` mais que `direct_chunk_hits` ne contient pas le bon passage, le probleme est local au stage 2
            - si la famille BOI est bonne mais le sous-doc exact differe, on a une confusion parent/enfant ou voisin de famille
            - si rien de plausible ne remonte, c'est un vrai miss retrieval
            """
        ),
    ]

    output_path = NOTEBOOKS_DIR / "07_query_workbench.ipynb"
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
