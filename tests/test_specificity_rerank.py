from __future__ import annotations

import unittest
from dataclasses import dataclass

from bofip_cleanroom.models import RawDocument
from bofip_cleanroom.specificity_rerank import SpecificityReranker


@dataclass
class _Hit:
    rank: int
    boi_reference: str
    score: float


def _document(boi_reference: str, title: str) -> RawDocument:
    return RawDocument(
        document_id=boi_reference,
        boi_reference=boi_reference,
        title=title,
        document_type="Contenu",
        content_type="Commentaire",
        publication_date="2025-01-01",
        source_url=None,
        language="fr",
    )


class SpecificityRerankTests(unittest.TestCase):
    def test_specificity_rerank_can_promote_more_specific_sibling(self):
        parent = _document(
            "BOI-IF-TU-10-20-20251231",
            "IF - Taxes diverses - Champ d'application",
        )
        child = _document(
            "BOI-IF-TU-10-20-10-20251231",
            "IF - Taxes diverses - Champ d'application - Exonérations",
        )
        reranker = SpecificityReranker([parent, child])
        hits = [
            _Hit(rank=1, boi_reference=parent.boi_reference, score=0.80),
            _Hit(rank=2, boi_reference=child.boi_reference, score=0.78),
        ]

        reranked = reranker.rerank_hits(
            "Quelles exonérations existent ?",
            hits,
            get_reference=lambda hit: hit.boi_reference,
            get_score=lambda hit: hit.score,
            clone_hit=lambda hit, rank, score: _Hit(rank=rank, boi_reference=hit.boi_reference, score=score),
            top_n=2,
            weight=0.10,
        )

        self.assertEqual(reranked[0].boi_reference, child.boi_reference)

    def test_specificity_rerank_does_not_move_isolated_document(self):
        doc1 = _document(
            "BOI-TVA-DECLA-10-10-20200323",
            "TVA - Redevables - Vue d'ensemble",
        )
        doc2 = _document(
            "BOI-IS-GPE-20-20-70-20190731",
            "IS - Groupes - Régime des abandons de créance",
        )
        reranker = SpecificityReranker([doc1, doc2])
        hits = [
            _Hit(rank=1, boi_reference=doc1.boi_reference, score=0.80),
            _Hit(rank=2, boi_reference=doc2.boi_reference, score=0.79),
        ]

        reranked = reranker.rerank_hits(
            "Qui est redevable de la TVA ?",
            hits,
            get_reference=lambda hit: hit.boi_reference,
            get_score=lambda hit: hit.score,
            clone_hit=lambda hit, rank, score: _Hit(rank=rank, boi_reference=hit.boi_reference, score=score),
            top_n=2,
            weight=0.10,
        )

        self.assertEqual([hit.boi_reference for hit in reranked], [doc1.boi_reference, doc2.boi_reference])


if __name__ == "__main__":
    unittest.main()
