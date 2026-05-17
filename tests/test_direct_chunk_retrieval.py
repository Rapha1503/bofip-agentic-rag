from __future__ import annotations

import unittest

from bofip_agentic.direct_chunk_retrieval import DirectChunkRetriever, Stage1DocumentHit
from bofip_agentic.models import ChunkNode


class DirectChunkRetrievalTests(unittest.TestCase):
    def test_direct_chunk_retrieval_round_robins_top_docs(self) -> None:
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p1"], "dépenses de recherche éligibles", 10, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p2"], "autre paragraphe", 10, "paragraph_window"),
            ChunkNode("c3", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p3"], "redevable de la taxe sur la valeur ajoutée", 10, "paragraph_window"),
            ChunkNode("c4", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p4"], "autre contenu TVA", 10, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="body")
        result = retriever.search(
            "recherche TVA",
            stage1_hits=[
                Stage1DocumentHit(rank=1, score=0.9, boi_reference="BOI-A-10"),
                Stage1DocumentHit(rank=2, score=0.8, boi_reference="BOI-B-10"),
            ],
            top_docs=2,
            chunks_per_doc=2,
            max_candidates=4,
        )

        self.assertEqual([hit.local_rank for hit in result.chunk_hits[:2]], [1, 1])
        self.assertEqual([hit.document_rank for hit in result.chunk_hits[:2]], [1, 2])

    def test_direct_chunk_retrieval_defaults_to_full_mode(self) -> None:
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p1"], "texte", 10, "paragraph_window"),
        ]
        retriever = DirectChunkRetriever(chunks)
        self.assertEqual(retriever.local_chunk_mode, "full")


if __name__ == "__main__":
    unittest.main()
