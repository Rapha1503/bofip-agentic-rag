from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from bofip_cleanroom.reranker import CrossEncoderReranker, RankedItem


class TestCrossEncoderReranker(unittest.TestCase):
    def setUp(self):
        patcher = patch("bofip_cleanroom.reranker.CrossEncoder", autospec=True)
        self.mock_cross_encoder = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_model = self.mock_cross_encoder.return_value

    def test_empty_items_returns_empty(self):
        reranker = CrossEncoderReranker(device="cpu")
        result = reranker.rerank("query", [], get_text=str)
        self.assertEqual(result, [])

    def test_rerank_sorts_by_score_desc(self):
        items = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        self.mock_model.predict.return_value = [0.3, 0.9, 0.5]
        reranker = CrossEncoderReranker(device="cpu")
        result = reranker.rerank("query", items, get_text=lambda x: x["text"], top_k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].item["text"], "b")
        self.assertEqual(result[0].score, 0.9)
        self.assertEqual(result[1].item["text"], "c")
        self.assertEqual(result[2].item["text"], "a")

    def test_rerank_truncates_to_top_k(self):
        items = [{"text": str(i)} for i in range(10)]
        self.mock_model.predict.return_value = [float(i) for i in range(10)]
        reranker = CrossEncoderReranker(device="cpu")
        result = reranker.rerank("query", items, get_text=lambda x: x["text"], top_k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].score, 9.0)
        self.assertEqual(result[1].score, 8.0)

    def test_get_text_is_called_correctly(self):
        items = ["item_a", "item_b"]
        self.mock_model.predict.return_value = [0.5, 0.8]
        reranker = CrossEncoderReranker(device="cpu")
        result = reranker.rerank("hello", items, get_text=lambda x: x.upper())
        self.assertEqual(len(result), 2)
        called_pairs = self.mock_model.predict.call_args[0][0]
        self.assertEqual(called_pairs, [["hello", "ITEM_A"], ["hello", "ITEM_B"]])

    def test_ranked_item_dataclass(self):
        item = RankedItem(item="test", score=0.75)
        self.assertEqual(item.item, "test")
        self.assertEqual(item.score, 0.75)

    def test_scalar_score_becomes_float(self):
        items = [{"x": 1}]
        self.mock_model.predict.return_value = [42]
        reranker = CrossEncoderReranker(device="cpu")
        result = reranker.rerank("q", items, get_text=str)
        self.assertIsInstance(result[0].score, float)
        self.assertEqual(result[0].score, 42.0)


if __name__ == "__main__":
    unittest.main()
