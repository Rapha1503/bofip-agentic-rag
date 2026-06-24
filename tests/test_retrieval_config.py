from __future__ import annotations

import unittest

from bofip_agentic.retrieval_config import selected_retrieval_profile


class RetrievalConfigTests(unittest.TestCase):
    def test_default_profile_is_lexical_full_corpus(self) -> None:
        profile = selected_retrieval_profile({})

        self.assertEqual("lexical", profile.mode)
        self.assertEqual("BM25 full-corpus", profile.label)
        self.assertFalse(profile.load_dense)

    def test_legacy_dense_env_selects_hybrid(self) -> None:
        profile = selected_retrieval_profile({"BOFIP_ENABLE_DENSE": "1"})

        self.assertEqual("hybrid", profile.mode)
        self.assertEqual("BM25 + embeddings E5", profile.label)
        self.assertTrue(profile.load_dense)

    def test_explicit_unknown_mode_falls_back_to_lexical(self) -> None:
        profile = selected_retrieval_profile({"BOFIP_RETRIEVAL_MODE": "experimental"})

        self.assertEqual("lexical", profile.mode)
        self.assertFalse(profile.load_dense)


if __name__ == "__main__":
    unittest.main()
