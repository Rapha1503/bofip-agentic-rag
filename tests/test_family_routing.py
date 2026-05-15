from __future__ import annotations

import unittest

from bofip_cleanroom.family_routing import collect_family_selection, collect_family_union, reference_core


class FamilyRoutingTests(unittest.TestCase):
    def test_reference_core_strips_trailing_date(self) -> None:
        self.assertEqual(
            reference_core("BOI-RPPM-RCM-40-50-60-20240730"),
            ("BOI", "RPPM", "RCM", "40", "50", "60"),
        )

    def test_collect_family_selection_from_parent_keeps_parent_prefix(self) -> None:
        references = [
            "BOI-RPPM-RCM-40-50-20240730",
            "BOI-RPPM-RCM-40-50-20-20240730",
            "BOI-RPPM-RCM-40-50-60-20240730",
            "BOI-RPPM-RCM-40-55-20240730",
        ]

        selection = collect_family_selection("BOI-RPPM-RCM-40-50-20240730", references)

        self.assertEqual(selection.prefix, ("BOI", "RPPM", "RCM", "40", "50"))
        self.assertEqual(
            selection.members,
            [
                "BOI-RPPM-RCM-40-50-20240730",
                "BOI-RPPM-RCM-40-50-20-20240730",
                "BOI-RPPM-RCM-40-50-60-20240730",
            ],
        )

    def test_collect_family_selection_from_child_rolls_up_to_parent_prefix(self) -> None:
        references = [
            "BOI-RPPM-RCM-40-50-20240730",
            "BOI-RPPM-RCM-40-50-20-20240730",
            "BOI-RPPM-RCM-40-50-60-20240730",
            "BOI-RPPM-RCM-40-55-20240730",
        ]

        selection = collect_family_selection("BOI-RPPM-RCM-40-50-20-20240730", references)

        self.assertEqual(selection.prefix, ("BOI", "RPPM", "RCM", "40", "50"))
        self.assertIn("BOI-RPPM-RCM-40-50-20240730", selection.members)
        self.assertIn("BOI-RPPM-RCM-40-50-60-20240730", selection.members)

    def test_collect_family_union_merges_multiple_anchor_families_without_duplicates(self) -> None:
        references = [
            "BOI-RPPM-RCM-40-50-20240730",
            "BOI-RPPM-RCM-40-50-20-20240730",
            "BOI-RPPM-RCM-40-50-60-20240730",
            "BOI-RPPM-RCM-40-55-20240730",
            "BOI-RPPM-RCM-40-55-10-20240730",
        ]

        selection = collect_family_union(
            [
                "BOI-RPPM-RCM-40-50-20-20240730",
                "BOI-RPPM-RCM-40-55-20240730",
            ],
            references,
        )

        self.assertEqual(
            selection.anchor_references,
            [
                "BOI-RPPM-RCM-40-50-20-20240730",
                "BOI-RPPM-RCM-40-55-20240730",
            ],
        )
        self.assertIn(("BOI", "RPPM", "RCM", "40", "50"), selection.prefixes)
        self.assertIn(("BOI", "RPPM", "RCM", "40", "55"), selection.prefixes)
        self.assertEqual(
            selection.members,
            [
                "BOI-RPPM-RCM-40-50-20240730",
                "BOI-RPPM-RCM-40-50-20-20240730",
                "BOI-RPPM-RCM-40-50-60-20240730",
                "BOI-RPPM-RCM-40-55-20240730",
                "BOI-RPPM-RCM-40-55-10-20240730",
            ],
        )

    def test_collect_family_selection_can_expand_one_ancestor_level_when_bounded(self) -> None:
        references = [
            "BOI-RPPM-RCM-40-50-20240730",
            "BOI-RPPM-RCM-40-50-20-10-20240730",
            "BOI-RPPM-RCM-40-50-20-20-20240730",
            "BOI-RPPM-RCM-40-50-20-20240730",
            "BOI-RPPM-RCM-40-50-60-20240730",
            "BOI-RPPM-RCM-40-55-20240730",
        ]

        selection = collect_family_selection(
            "BOI-RPPM-RCM-40-50-20-10-20240730",
            references,
            ancestor_expansion_levels=1,
        )

        self.assertEqual(selection.prefix, ("BOI", "RPPM", "RCM", "40", "50"))
        self.assertIn("BOI-RPPM-RCM-40-50-60-20240730", selection.members)

    def test_collect_family_selection_does_not_expand_when_parent_branch_is_too_large(self) -> None:
        references = [
            "BOI-IF-TU-10-20-30-160-20251231",
            "BOI-IF-TU-10-20-30-150-20251231",
            "BOI-IF-TU-10-20-30-140-20251231",
            "BOI-IF-TU-10-20-30-130-20251231",
            "BOI-IF-TU-10-20-30-120-20251231",
            "BOI-IF-TU-10-20-30-110-20251231",
            "BOI-IF-TU-10-20-30-100-20251231",
            "BOI-IF-TU-10-20-30-10-20251231",
            "BOI-IF-TU-10-20-20251231",
            "BOI-IF-TU-10-20-10-20251231",
            "BOI-IF-TU-10-20-20-20251231",
        ]
        references.extend(
            f"BOI-IF-TU-10-20-{suffix:02d}-20251231"
            for suffix in range(31, 60)
        )

        selection = collect_family_selection(
            "BOI-IF-TU-10-20-30-160-20251231",
            references,
            max_family_docs=18,
            ancestor_expansion_levels=1,
        )

        self.assertEqual(selection.prefix, ("BOI", "IF", "TU", "10", "20", "30"))


if __name__ == "__main__":
    unittest.main()
