"""
BOFIP Hybrid Retrieval

Combines vector search (ChromaDB) and keyword search (BM25) with French cross-encoder reranking.
Production method: search_simple()
"""

import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RERANK_POOL_SIZE
from src.retrieval.vector_store import BOFIPVectorStore
from src.retrieval.bm25 import get_bm25_index, BOFIPBM25
from src.retrieval.reranker import get_reranker, BOFIPReranker

logger = logging.getLogger(__name__)

MAX_CROSSREF_CHUNKS_PER_ARTICLE = 3


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and BM25.
    """

    def __init__(self, vector_store: Optional[BOFIPVectorStore] = None,
                 bm25_index: Optional[BOFIPBM25] = None,
                 reranker: Optional[BOFIPReranker] = None,
                 alpha: float = 0.5,
                 use_reranker: bool = True,
                 rerank_pool_size: Optional[int] = None):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: ChromaDB vector store (created if None)
            bm25_index: BM25 index (loaded if None)
            reranker: Cross-encoder reranker (created if None and use_reranker=True)
            alpha: Weight for vector search (0=BM25 only, 1=vector only)
            use_reranker: Whether to use reranker (default True)
            rerank_pool_size: Number of candidates sent to reranker
        """
        self.alpha = alpha
        self.use_reranker = use_reranker
        configured_pool = rerank_pool_size if rerank_pool_size is not None else RERANK_POOL_SIZE
        try:
            self.rerank_pool_size = max(1, int(configured_pool))
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid rerank_pool_size={configured_pool}, falling back to default {RERANK_POOL_SIZE}"
            )
            self.rerank_pool_size = max(1, int(RERANK_POOL_SIZE))

        # Initialize vector store
        if vector_store is None:
            logger.info("Initializing vector store...")
            self.vector_store = BOFIPVectorStore()
        else:
            self.vector_store = vector_store

        # Initialize BM25 index
        if bm25_index is None:
            logger.info("Loading BM25 index...")
            self.bm25 = get_bm25_index()
        else:
            self.bm25 = bm25_index

        # Initialize reranker
        if use_reranker:
            if reranker is None:
                logger.info("Loading reranker model...")
                self.reranker = get_reranker()
            else:
                self.reranker = reranker
        else:
            self.reranker = None

        # Build legal reference lookup for cross-reference injection
        self.legal_ref_chunks = self._build_legal_ref_lookup()

        logger.info(
            f"Hybrid retriever ready (alpha={alpha}, reranker={use_reranker}, "
            f"rerank_pool_size={self.rerank_pool_size})"
        )
        logger.info(f"  Vector store: {self.vector_store.get_count()} chunks")
        logger.info(f"  BM25 index: {len(self.bm25.chunks)} chunks")
        logger.info(f"  Legal ref lookup: {len(self.legal_ref_chunks)} article refs")

    def search_simple(self, query: str, n_results: int = 30) -> List[Dict[str, Any]]:
        """
        Simple search: BM25 + Vector, merge by normalized score, return top N.

        No HyDE, no complex post-processing.
        Normalizes scores to 0-1 range before combining.
        Boosts chunks with explicit values (%, â‚¬) for rate/amount queries.

        Args:
            query: Search query
            n_results: Number of results to return (default 30 for better coverage)

        Returns:
            List of chunks sorted by combined score
        """
        # Get BM25 results
        bm25_results = self.bm25.search(query, n_results=30)
        logger.info(f"BM25: {len(bm25_results)} results")

        # Get Vector results with original query (semantic similarity)
        vector_results = self.vector_store.search(query, n_results=30)
        logger.info(f"Vector: {len(vector_results)} results")

        # Diversified CGI/LPF-only vector search (guarantees representation)
        try:
            legal_results = self.vector_store.search(
                query, n_results=5,
                where={"content_type": {"$in": ["CGI", "LPF"]}}
            )
            logger.info(f"Legal vector (CGI/LPF only): {len(legal_results)} results")
        except Exception as e:
            logger.warning(f"CGI/LPF vector search failed: {e}")
            legal_results = []

        # Normalize BM25 scores to 0-1 range
        if bm25_results:
            bm25_max = max(r.get('score', 0) for r in bm25_results)
            bm25_min = min(r.get('score', 0) for r in bm25_results)
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
            for r in bm25_results:
                raw_score = r.get('score', 0)
                r['normalized_score'] = (raw_score - bm25_min) / bm25_range
                r['source'] = 'bm25'

        # Vector scores are already ~0-1, but normalize for consistency
        if vector_results:
            vec_max = max(r.get('score', 0) for r in vector_results)
            vec_min = min(r.get('score', 0) for r in vector_results)
            vec_range = vec_max - vec_min if vec_max != vec_min else 1
            for r in vector_results:
                raw_score = r.get('score', 0)
                r['normalized_score'] = (raw_score - vec_min) / vec_range
                r['source'] = 'vector'

        # Merge: combine scores for duplicates, take best for unique
        chunk_scores = {}  # chunk_id -> {'bm25': score, 'vector': score, 'data': result}
        for r in bm25_results:
            chunk_id = r['chunk_id']
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'bm25': 0, 'vector': 0, 'data': r}
            chunk_scores[chunk_id]['bm25'] = r.get('normalized_score', 0)

        for r in vector_results:
            chunk_id = r['chunk_id']
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'bm25': 0, 'vector': 0, 'data': r}
            chunk_scores[chunk_id]['vector'] = r.get('normalized_score', 0)
            # Update data if vector has it (prefer vector for text)
            if 'data' not in chunk_scores[chunk_id] or chunk_scores[chunk_id]['data'].get('source') != 'vector':
                chunk_scores[chunk_id]['data'] = r

        # Calculate combined score: average of normalized scores (0 if missing)
        # Boost if found in both: max(bm25, vector) + 0.2 * min(bm25, vector)
        for chunk_id, scores in chunk_scores.items():
            bm25_score = scores['bm25']
            vector_score = scores['vector']

            if bm25_score > 0 and vector_score > 0:
                # Found in both - boost score
                combined = max(bm25_score, vector_score) + 0.3 * min(bm25_score, vector_score)
            else:
                # Found in one only
                combined = bm25_score + vector_score

            scores['combined'] = combined
            scores['data']['combined_score'] = combined

        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x['combined'],
            reverse=True
        )

        merged = [s['data'] for s in sorted_chunks]

        # Include table chunks from found documents
        # This ensures critical mapping information (like CO2â†’plafond tables) is included
        # even when the table text doesn't directly match query keywords
        found_doc_ids = set()
        for r in merged[:15]:  # Check top 15 results
            # Only include table chunks from BOFIP docs (CGI/LPF "tables" are just TOCs)
            content_type = r.get('metadata', {}).get('content_type', '')
            if content_type in ('CGI', 'LPF'):
                continue
            doc_id = r.get('metadata', {}).get('doc_id') or r.get('doc_id')
            if doc_id:
                found_doc_ids.add(doc_id)

        result_chunk_ids = set(r['chunk_id'] for r in merged)
        table_chunks_added = 0
        for doc_id in found_doc_ids:
            table_chunks = self._get_table_chunks_for_doc(doc_id)
            for tc in table_chunks:
                if tc['chunk_id'] not in result_chunk_ids:
                    tc['combined_score'] = 0.75  # Good score to include in results
                    tc['source'] = 'table_supplement'
                    merged.append(tc)
                    result_chunk_ids.add(tc['chunk_id'])
                    table_chunks_added += 1

        if table_chunks_added > 0:
            logger.info(f"Added {table_chunks_added} table chunks from found documents")
            # Re-sort to include table chunks in proper position
            merged = sorted(merged, key=lambda x: x.get('combined_score', 0), reverse=True)

        # Diversified legal injection: add top CGI/LPF-only results if not already present
        legal_added = 0
        if legal_results:
            for lr in legal_results:
                if lr['chunk_id'] not in result_chunk_ids:
                    lr['combined_score'] = 0.70  # Moderate score - supplement, don't dominate
                    lr['source'] = 'legal_diversified'
                    merged.append(lr)
                    result_chunk_ids.add(lr['chunk_id'])
                    legal_added += 1
            if legal_added:
                logger.info(f"Added {legal_added} diversified CGI/LPF chunks")
                merged = sorted(merged, key=lambda x: x.get('combined_score', 0), reverse=True)

        # Cross-reference injection: when BOFIP chunks reference CGI/LPF articles, inject them
        xref_added = self._inject_cross_references(merged, result_chunk_ids, query)
        if xref_added > 0:
            merged = sorted(merged, key=lambda x: x.get('combined_score', 0), reverse=True)

        # Value-aware boosting: boost chunks with explicit values for rate/amount queries
        # This helps ensure barÃ¨mes and thresholds are ranked higher when user asks for rates
        merged = self._boost_value_chunks(query, merged)

        # Rerank top candidates with French cross-encoder for final precision
        if self.reranker and self.reranker.is_available():
            rerank_pool_size = min(self.rerank_pool_size, len(merged))
            rerank_pool = merged[:rerank_pool_size]
            reranked = self.reranker.rerank(query, rerank_pool, top_k=n_results)
            logger.info(f"Reranked top {rerank_pool_size} candidates")
            # Combine: reranked top + remaining un-reranked
            reranked_ids = {r['chunk_id'] for r in reranked}
            remaining = [c for c in merged[rerank_pool_size:] if c['chunk_id'] not in reranked_ids]
            merged = reranked + remaining

        logger.info(f"Merged: {len(merged)} unique chunks, returning top {n_results}")

        # Log top results for debugging
        for i, r in enumerate(merged[:5]):
            boi_ref = r.get('metadata', {}).get('boi_reference', 'N/A')
            combined = r.get('combined_score', 0)
            rerank = r.get('rerank_score', '-')
            source = r.get('source', '?')
            logger.debug(f"  {i+1}. {boi_ref} (combined: {combined:.4f}, rerank: {rerank}, source: {source})")

        return merged[:n_results]

    def _get_table_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get chunks containing tables from a document.

        This helps ensure that when a document is found, its table chunks
        (which contain critical mapping information like CO2â†’plafond thresholds)
        are also included even if they don't match the query keywords directly.

        Args:
            doc_id: Document ID (e.g., "4582-PGP")

        Returns:
            List of chunks that have contains_table=True
        """
        all_chunks = self.vector_store.get_chunks_by_doc_id(doc_id)
        return [c for c in all_chunks
                if c.get('metadata', {}).get('contains_table', False)]

    def _boost_value_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Boost chunks containing explicit values (%, â‚¬) for rate/amount queries.

        When user asks for rates, barÃ¨mes, thresholds, or amounts, chunks
        that actually contain percentages or euro values are more likely
        to have the answer.

        NOT hardcoding - uses general patterns to detect:
        - Query keywords: taux, barÃ¨me, seuil, plafond, combien, montant, calcul, impÃ´t
        - Chunk content: digits followed by % or â‚¬ symbols

        Args:
            query: The search query
            chunks: List of chunks with combined_score

        Returns:
            Re-sorted chunks with boosted scores for value-containing chunks
        """
        import re

        # Detect if this is a rate/amount query (general patterns, no hardcoding)
        query_lower = query.lower()
        rate_keywords = ['taux', 'barÃ¨me', 'bareme', 'seuil', 'plafond', 'combien',
                         'montant', 'calcul', 'impÃ´t', 'impot', 'droits', 'payer']

        is_rate_query = any(kw in query_lower for kw in rate_keywords)

        if not is_rate_query:
            return chunks

        logger.info("Rate/amount query detected - boosting value chunks")

        # Boost chunks containing percentages or euro amounts
        boosted_count = 0
        for chunk in chunks:
            text = chunk.get('text', '')

            # Check for percentages (e.g., "11 %", "30%", "45 %")
            has_percentage = bool(re.search(r'\d+\s*%', text))

            # Check for euro amounts (e.g., "11 497 â‚¬", "29315â‚¬", "100 000 euros")
            has_amount = bool(re.search(r'\d+[\s\xa0]*(â‚¬|euros?)\b', text, re.IGNORECASE))

            # Check for barÃ¨me-specific patterns (multiple rate tiers)
            has_bareme_pattern = bool(re.search(r'(0\s*%|11\s*%|30\s*%|41\s*%|45\s*%)', text))

            if has_percentage or has_amount:
                old_score = chunk.get('combined_score', 0)
                # Stronger boost for chunks with barÃ¨me patterns
                boost_factor = 1.3 if has_bareme_pattern else 1.15
                chunk['combined_score'] = old_score * boost_factor
                boosted_count += 1

        if boosted_count > 0:
            logger.info(f"Boosted {boosted_count} chunks with explicit values")
            # Re-sort after boosting
            chunks = sorted(chunks, key=lambda x: x.get('combined_score', 0), reverse=True)

        return chunks

    def _build_legal_ref_lookup(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build in-memory lookup: normalized article ref -> list of chunk dicts.

        Scans all CGI/LPF chunks from BM25 index and indexes them by their
        article reference for fast cross-reference injection.

        Returns:
            Dict mapping normalized refs like "CGI:102 ter" to chunk dicts
        """
        lookup = defaultdict(list)
        for chunk in self.bm25.chunks:
            boi_ref = chunk.boi_reference
            content_type = chunk.content_type

            if content_type not in ('CGI', 'LPF'):
                continue

            # Extract article number from boi_reference
            # CGI: "CGI Art. 102 ter" -> "CGI:102 ter"
            # LPF: "LPF Art. L. 52" -> "LPF:L. 52"
            # Also handles "CGI p.151" (page refs) - skip those
            art_match = re.match(r'(?:CGI|LPF)\s+Art\.\s+(.+)', boi_ref)
            if not art_match:
                continue

            art_num = art_match.group(1).strip()
            key = f"{content_type}:{art_num}"

            # Build chunk dict in BM25 return format
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'text': chunk.text_with_context,
                'metadata': {
                    'doc_id': chunk.doc_id,
                    'boi_reference': chunk.boi_reference,
                    'series': ','.join(chunk.series) if chunk.series else '',
                    'section_title': chunk.section_title or '',
                    'paragraph_number': chunk.paragraph_number or '',
                    'publication_date': chunk.publication_date,
                    'source_url': chunk.source_url,
                    'content_type': chunk.content_type,
                    'contains_table': chunk.contains_table,
                    'token_count': chunk.token_count,
                },
            }
            lookup[key].append(chunk_dict)

        pruned_lookup: Dict[str, List[Dict[str, Any]]] = {}
        for key, chunk_list in lookup.items():
            # Deduplicate by chunk_id then keep the richest chunks per article.
            unique_chunks = {c['chunk_id']: c for c in chunk_list}
            sorted_chunks = sorted(
                unique_chunks.values(),
                key=lambda c: c.get('metadata', {}).get('token_count', 0),
                reverse=True
            )
            pruned_lookup[key] = sorted_chunks[:MAX_CROSSREF_CHUNKS_PER_ARTICLE]

        logger.info(
            f"Built legal ref lookup: {len(pruned_lookup)} unique article refs, "
            f"{sum(len(v) for v in pruned_lookup.values())} chunks after pruning "
            f"(max {MAX_CROSSREF_CHUNKS_PER_ARTICLE}/article)"
        )
        return pruned_lookup

    def _extract_legal_references(self, chunks: List[Dict[str, Any]], query: str) -> set:
        """
        Extract CGI/LPF article references from chunk texts and query.

        Regex patterns tested against 12,388 real CGI references + 1,464 LPF references
        found in BOFIP text.

        Args:
            chunks: Top result chunks to scan for references
            query: The original query (may contain direct article refs)

        Returns:
            Set of normalized reference keys like {"CGI:102 ter", "LPF:L. 64"}
        """
        refs = set()

        # Combine texts to scan: query + top chunk texts
        texts_to_scan = [("query", query)]
        for chunk in chunks:
            text = chunk.get('text', '')
            texts_to_scan.append(("chunk", text))

        for text_source, text in texts_to_scan:
            # Normalize whitespace (non-breaking spaces, newlines) before regex
            text = re.sub(r'[\xa0\n\r\t]+', ' ', text)

            # CGI references: "article 102 ter du CGI", "art. 1600-0 F bis du code general"
            cgi_matches = re.findall(
                r'(?:articles?|art\.?)\s+([\d][\d\w\s\-]*?)(?:[\s,]+(?:et\s+suivants\s+)?du\s+|\s+)(?:CGI|code\s+g[Ã©e]n[Ã©e]ral)',
                text, re.IGNORECASE
            )
            for m in cgi_matches:
                # Split "145 et 216" into separate refs
                for part in re.split(r'\s+et\s+', m.strip().rstrip(',. ')):
                    part = part.strip()
                    if part:
                        refs.add(f"CGI:{part}")

            # LPF references: "article L. 52 du LPF", "article L64 du livre des procedures"
            lpf_matches = re.findall(
                r'(?:articles?|art\.?)\s*([LRA]\*?\.?\s*[\d][\d\w\s\-]*?)(?:[\s,]+(?:et\s+suivants\s+)?du\s+|\s+)(?:LPF|livre\s+des\s+proc[Ã©e]dures)',
                text, re.IGNORECASE
            )
            for m in lpf_matches:
                for part in re.split(r'\s+et\s+', m.strip().rstrip(',. ')):
                    part = part.strip()
                    if part:
                        refs.add(f"LPF:{part}")

            # Query-only shorthand support: "L.64 LPF"
            if text_source == "query":
                lpf_short_matches = re.findall(
                    r'\b([LRA]\.?\s*[\d][\d\w\s\-]*?)\s+(?:du\s+)?LPF\b',
                    text, re.IGNORECASE
                )
                for m in lpf_short_matches:
                    part = m.strip().rstrip(',. ')
                    if part:
                        refs.add(f"LPF:{part}")

        if refs:
            logger.info(f"Extracted {len(refs)} legal references: {refs}")

        return refs

    def _inject_cross_references(self, merged: List[Dict[str, Any]],
                                  result_chunk_ids: set,
                                  query: str) -> int:
        """
        Inject CGI/LPF chunks referenced by BOFIP chunks in results.

        When BOFIP commentary references "article 102 ter du CGI", we look up
        that article and inject it into results. This bridges the vocabulary
        gap between user queries (matched by BOFIP) and legal text (CGI/LPF).

        Args:
            merged: Current merged results list (modified in place)
            result_chunk_ids: Set of chunk IDs already in results (modified in place)
            query: Original query

        Returns:
            Number of chunks injected
        """
        # Extract references from query and top 15 results.
        # Query mentions are treated as stronger intent than contextual mentions.
        query_refs = self._extract_legal_references([], query)
        context_refs = self._extract_legal_references(merged[:15], query)
        refs = query_refs | context_refs

        if not refs:
            return 0

        injected = 0
        boosted_existing = 0
        for ref_key in refs:
            # Try exact match first
            matched_chunks = self.legal_ref_chunks.get(ref_key, [])

            # If no exact match, try normalized lookup (handle spacing differences)
            if not matched_chunks:
                # "LPF:L64" -> try "LPF:L. 64", "LPF:L 64" etc.
                ref_parts = ref_key.split(':', 1)
                if len(ref_parts) == 2:
                    source, art = ref_parts
                    # Try inserting ". " after letter prefix for LPF
                    if source == 'LPF' and re.match(r'^[LRA]\d', art):
                        alt_key = f"{source}:{art[0]}. {art[1:]}"
                        matched_chunks = self.legal_ref_chunks.get(alt_key, [])
                    # Try normalized space comparison
                    if not matched_chunks:
                        art_normalized = re.sub(r'\s+', ' ', art).strip()
                        for lookup_key in self.legal_ref_chunks:
                            lookup_normalized = re.sub(r'\s+', ' ', lookup_key.split(':', 1)[-1]).strip()
                            if lookup_key.startswith(f"{source}:") and lookup_normalized == art_normalized:
                                matched_chunks = self.legal_ref_chunks[lookup_key]
                                break

                    # Try stripping suffix (bis/ter/A/B/C) to find parent article
                    # "L. 80 B" -> "L. 80", "102 ter" -> "102"
                    if not matched_chunks:
                        parent_art = re.sub(r'\s+(?:bis|ter|quater|quinquies|sexies|[A-Z])\s*$', '', art).strip()
                        if parent_art != art:
                            parent_key = f"{source}:{parent_art}"
                            matched_chunks = self.legal_ref_chunks.get(parent_key, [])

            # Avoid flooding with multiple near-duplicate chunks from same legal article.
            for chunk in matched_chunks[:1]:
                if chunk['chunk_id'] in result_chunk_ids:
                    if ref_key in query_refs:
                        for existing in merged:
                            if existing.get('chunk_id') == chunk['chunk_id']:
                                old_score = existing.get('combined_score', 0)
                                if old_score < 0.95:
                                    existing['combined_score'] = 0.95
                                    existing['source'] = 'cross_reference'
                                    boosted_existing += 1
                                break
                    continue

                chunk_copy = chunk.copy()
                if ref_key in query_refs:
                    chunk_copy['combined_score'] = 0.95  # user explicitly asked this law ref
                else:
                    chunk_copy['combined_score'] = 0.90  # LAW > commentary
                chunk_copy['source'] = 'cross_reference'
                merged.append(chunk_copy)
                result_chunk_ids.add(chunk['chunk_id'])
                injected += 1

        if injected or boosted_existing:
            logger.info(
                f"Injected {injected} cross-referenced CGI/LPF chunks "
                f"(boosted existing: {boosted_existing})"
            )

        return injected + boosted_existing

# Singleton instance
_retriever_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create singleton hybrid retriever"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


# Test
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Test hybrid retrieval')
    parser.add_argument('query', nargs='?', default="taux de TVA restauration",
                       help='Search query')
    parser.add_argument('--n', type=int, default=10, help='Number of results')

    args = parser.parse_args()

    retriever = get_hybrid_retriever()

    print(f"\nQuery: '{args.query}'")
    print("=" * 60)

    results = retriever.search_simple(args.query, args.n)

    print(f"Found {len(results)} results\n")

    for i, r in enumerate(results):
        print(f"{i+1}. {r['metadata'].get('boi_reference', 'N/A')}")
        score = r.get('combined_score', r.get('score', 0))
        print(f"   Score: {score:.4f}")
        print(f"   Section: {r['metadata'].get('section_title', 'N/A')}")
        print(f"   Text: {r['text'][:150]}...")
        print()

