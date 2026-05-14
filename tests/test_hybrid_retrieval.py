from __future__ import annotations

import unittest

from bofip_cleanroom.hybrid_retrieval import (
    RankedDoc,
    compute_source_rank_profiles,
    confidence_weighted_reciprocal_rank_fuse,
    reciprocal_rank_fuse,
)


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_fuses_by_document_reference(self) -> None:
        rankings = {
            "lexical": [
                RankedDoc(boi_reference="BOI-A", score=10.0, rank=1, source="lexical"),
                RankedDoc(boi_reference="BOI-B", score=9.0, rank=2, source="lexical"),
            ],
            "dense": [
                RankedDoc(boi_reference="BOI-B", score=0.9, rank=1, source="dense"),
                RankedDoc(boi_reference="BOI-A", score=0.8, rank=2, source="dense"),
            ],
        }

        hits = reciprocal_rank_fuse(rankings, top_k=2, rank_constant=60)

        self.assertEqual([hit.boi_reference for hit in hits], ["BOI-A", "BOI-B"])
        self.assertEqual(hits[0].ranks["lexical"], 1)
        self.assertEqual(hits[0].ranks["dense"], 2)

    def test_rrf_supports_source_weights(self) -> None:
        rankings = {
            "lexical": [
                RankedDoc(boi_reference="BOI-A", score=10.0, rank=1, source="lexical"),
                RankedDoc(boi_reference="BOI-B", score=9.0, rank=2, source="lexical"),
            ],
            "dense": [
                RankedDoc(boi_reference="BOI-A", score=0.9, rank=4, source="dense"),
                RankedDoc(boi_reference="BOI-B", score=0.8, rank=2, source="dense"),
            ],
        }

        default_hits = reciprocal_rank_fuse(rankings, top_k=2, rank_constant=60)
        weighted_hits = reciprocal_rank_fuse(
            rankings,
            top_k=2,
            rank_constant=60,
            source_weights={"lexical": 2.0, "dense": 1.0},
        )

        self.assertEqual(default_hits[0].boi_reference, "BOI-B")
        self.assertEqual(weighted_hits[0].boi_reference, "BOI-A")

    def test_source_rank_profiles_detect_clear_winner(self) -> None:
        rankings = {
            "sections_leads": [
                RankedDoc(boi_reference="BOI-A", score=8.0, rank=1, source="sections_leads"),
                RankedDoc(boi_reference="BOI-B", score=3.0, rank=2, source="sections_leads"),
                RankedDoc(boi_reference="BOI-C", score=2.5, rank=3, source="sections_leads"),
            ],
            "dense": [
                RankedDoc(boi_reference="BOI-B", score=0.82, rank=1, source="dense"),
                RankedDoc(boi_reference="BOI-A", score=0.81, rank=2, source="dense"),
                RankedDoc(boi_reference="BOI-C", score=0.80, rank=3, source="dense"),
            ],
        }

        profiles = compute_source_rank_profiles(rankings)

        self.assertGreater(profiles["sections_leads"].confidence, profiles["dense"].confidence)
        self.assertEqual(profiles["sections_leads"].document_strengths["BOI-A"], 1.0)

    def test_confidence_weighted_rrf_can_promote_confident_source(self) -> None:
        rankings = {
            "sections_leads": [
                RankedDoc(boi_reference="BOI-A", score=8.0, rank=1, source="sections_leads"),
                RankedDoc(boi_reference="BOI-B", score=3.0, rank=2, source="sections_leads"),
                RankedDoc(boi_reference="BOI-C", score=2.5, rank=3, source="sections_leads"),
            ],
            "base": [
                RankedDoc(boi_reference="BOI-B", score=4.3, rank=1, source="base"),
                RankedDoc(boi_reference="BOI-A", score=4.2, rank=2, source="base"),
                RankedDoc(boi_reference="BOI-C", score=4.1, rank=3, source="base"),
            ],
            "dense": [
                RankedDoc(boi_reference="BOI-B", score=0.82, rank=1, source="dense"),
                RankedDoc(boi_reference="BOI-C", score=0.81, rank=2, source="dense"),
                RankedDoc(boi_reference="BOI-A", score=0.80, rank=3, source="dense"),
            ],
        }

        rrf_hits = reciprocal_rank_fuse(rankings, top_k=3, rank_constant=60)
        confidence_hits = confidence_weighted_reciprocal_rank_fuse(
            rankings,
            top_k=3,
            rank_constant=60,
            confidence_alpha=4.0,
            score_alpha=1.0,
        )

        self.assertEqual(rrf_hits[0].boi_reference, "BOI-B")
        self.assertEqual(confidence_hits[0].boi_reference, "BOI-A")


if __name__ == "__main__":
    unittest.main()
