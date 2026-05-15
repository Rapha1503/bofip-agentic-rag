from __future__ import annotations

import unittest

from bofip_cleanroom.eval_harness import (
    QueryGold,
    _binary_relevance,
    _first_hit_rank,
    _ndcg,
    evaluate,
)


class TestBinaryRelevance(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_binary_relevance([], {"a"}), [])

    def test_mixed(self):
        self.assertEqual(
            _binary_relevance(["a", "b", "c"], {"a", "c"}),
            [1, 0, 1],
        )

    def test_none_found(self):
        self.assertEqual(_binary_relevance(["x", "y"], {"a"}), [0, 0])


class TestFirstHitRank(unittest.TestCase):
    def test_found_first(self):
        self.assertEqual(_first_hit_rank(["a", "b"], {"a"}), 1)

    def test_found_second(self):
        self.assertEqual(_first_hit_rank(["a", "b"], {"b"}), 2)

    def test_not_found(self):
        self.assertIsNone(_first_hit_rank(["a"], {"c"}))

    def test_empty_items(self):
        self.assertIsNone(_first_hit_rank([], {"a"}))

    def test_empty_golds(self):
        self.assertIsNone(_first_hit_rank(["a"], set()))


class TestNDCG(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(_ndcg([1, 0, 0], 3), 1.0, places=5)

    def test_zero_relevance(self):
        self.assertEqual(_ndcg([0, 0, 0], 3), 0.0)

    def test_truncation(self):
        rel = _ndcg([1, 0, 1, 0], 2)
        self.assertGreater(rel, 0.0)
        self.assertLess(rel, 1.0)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.queries = [
            QueryGold(
                query_id="q1",
                query="test 1",
                category="direct",
                gold_doc_refs=["DOC-A", "DOC-B"],
                gold_chunk_ids=["c1", "c2"],
            ),
            QueryGold(
                query_id="q2",
                query="test 2",
                category="direct",
                gold_doc_refs=["DOC-C"],
                gold_chunk_ids=["c3"],
            ),
            QueryGold(
                query_id="q3",
                query="test 3",
                category="unsupported",
                gold_doc_refs=[],
                gold_chunk_ids=[],
            ),
        ]

    def test_perfect_retrieval(self):
        doc_map = {
            "test 1": ["DOC-B", "DOC-A"],
            "test 2": ["DOC-C", "DOC-D"],
            "test 3": ["DOC-E"],
        }
        chunk_map = {
            "test 1": ["c0", "c1", "c2"],
            "test 2": ["c5", "c3"],
            "test 3": ["c9"],
        }
        metrics = evaluate(
            self.queries,
            retrieve_docs=lambda q: doc_map[q],
            retrieve_chunks=lambda q: chunk_map[q],
            k_values=[1, 3],
        )

        self.assertEqual(metrics.queries_count, 3)
        self.assertEqual(metrics.categories_count, {"direct": 2, "unsupported": 1})
        self.assertAlmostEqual(metrics.doc_hit_at[1], 2 / 3, places=5)
        self.assertAlmostEqual(metrics.doc_hit_at[3], 2 / 3, places=5)
        self.assertGreater(metrics.passage_hit_at[3], 0.5)
        self.assertAlmostEqual(metrics.mrr_doc, (1.0 + 1.0 + 0.0) / 3, places=5)

    def test_total_miss(self):
        doc_map = {"test 1": ["X"], "test 2": ["Y"], "test 3": ["Z"]}
        chunk_map = {"test 1": ["x"], "test 2": ["y"], "test 3": ["z"]}
        metrics = evaluate(
            self.queries,
            retrieve_docs=lambda q: doc_map[q],
            retrieve_chunks=lambda q: chunk_map[q],
            k_values=[1, 3],
        )

        self.assertEqual(metrics.doc_hit_at[3], 0.0)
        self.assertEqual(metrics.passage_hit_at[3], 0.0)
        self.assertEqual(metrics.mrr_doc, 0.0)
        self.assertEqual(metrics.mrr_passage, 0.0)
        self.assertEqual(metrics.ndcg_doc_at[3], 0.0)

    def test_unsupported_query_gets_zero_ndcg(self):
        doc_map = {"test 1": ["DOC-A"], "test 2": ["DOC-C"], "test 3": ["X"]}
        chunk_map = {"test 1": ["c1"], "test 2": ["c3"], "test 3": ["x"]}
        metrics = evaluate(
            self.queries,
            retrieve_docs=lambda q: doc_map[q],
            retrieve_chunks=lambda q: chunk_map[q],
            k_values=[1, 3],
        )
        self.assertEqual(metrics.doc_hit_at[1], 2.0 / 3.0)
        self.assertEqual(len(metrics.per_query), 3)

    def test_category_breakdown(self):
        doc_map = {"test 1": ["DOC-A"], "test 2": ["X"], "test 3": ["X"]}
        chunk_map = {"test 1": [], "test 2": [], "test 3": []}
        metrics = evaluate(
            self.queries,
            retrieve_docs=lambda q: doc_map[q],
            retrieve_chunks=lambda q: chunk_map[q],
            k_values=[1],
        )
        direct_hits = any(
            r.doc_hit and r.category == "direct" for r in metrics.per_query
        )
        unsupported = [r for r in metrics.per_query if r.category == "unsupported"]
        self.assertTrue(direct_hits)
        self.assertEqual(len(unsupported), 1)


if __name__ == "__main__":
    unittest.main()
