from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySelection:
    anchor_reference: str
    prefix: tuple[str, ...]
    members: list[str]


@dataclass(frozen=True)
class FamilyUnionSelection:
    anchor_references: list[str]
    prefixes: list[tuple[str, ...]]
    members: list[str]


def reference_core(boi_reference: str) -> tuple[str, ...]:
    parts = tuple(boi_reference.split("-"))
    if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
        parts = parts[:-1]
    return parts


def collect_family_selection(
    anchor_reference: str,
    all_references: list[str],
    *,
    min_prefix_len: int = 4,
    max_family_docs: int = 25,
    ancestor_expansion_levels: int = 0,
) -> FamilySelection:
    cores = {reference: reference_core(reference) for reference in all_references}
    anchor_core = cores[anchor_reference]

    def members_for_prefix(prefix: tuple[str, ...]) -> list[str]:
        prefix_len = len(prefix)
        members = [
            reference
            for reference, core in cores.items()
            if len(core) >= prefix_len and core[:prefix_len] == prefix
        ]
        members.sort(key=lambda reference: (len(cores[reference]), reference))
        return members

    full_members = members_for_prefix(anchor_core)
    if 1 < len(full_members) <= max_family_docs:
        return FamilySelection(anchor_reference=anchor_reference, prefix=anchor_core, members=full_members)

    for prefix_len in range(len(anchor_core) - 1, min_prefix_len - 1, -1):
        prefix = anchor_core[:prefix_len]
        members = members_for_prefix(prefix)
        if 1 < len(members) <= max_family_docs:
            selection = FamilySelection(anchor_reference=anchor_reference, prefix=prefix, members=members)
            break
    else:
        selection = FamilySelection(anchor_reference=anchor_reference, prefix=anchor_core, members=full_members)

    if ancestor_expansion_levels <= 0:
        return selection

    expanded_prefix = selection.prefix
    expanded_members = selection.members
    levels_left = ancestor_expansion_levels
    current_prefix_len = len(selection.prefix)
    while levels_left > 0 and current_prefix_len - 1 >= min_prefix_len:
        candidate_prefix = anchor_core[: current_prefix_len - 1]
        candidate_members = members_for_prefix(candidate_prefix)
        if 1 < len(candidate_members) <= max_family_docs:
            expanded_prefix = candidate_prefix
            expanded_members = candidate_members
            current_prefix_len -= 1
            levels_left -= 1
            continue
        break

    return FamilySelection(
        anchor_reference=anchor_reference,
        prefix=expanded_prefix,
        members=expanded_members,
    )


def collect_family_union(
    anchor_references: list[str],
    all_references: list[str],
    *,
    min_prefix_len: int = 4,
    max_family_docs: int = 25,
    ancestor_expansion_levels: int = 0,
) -> FamilyUnionSelection:
    seen_members: set[str] = set()
    members: list[str] = []
    prefixes: list[tuple[str, ...]] = []
    kept_anchors: list[str] = []

    for anchor_reference in anchor_references:
        if not anchor_reference:
            continue
        selection = collect_family_selection(
            anchor_reference,
            all_references,
            min_prefix_len=min_prefix_len,
            max_family_docs=max_family_docs,
            ancestor_expansion_levels=ancestor_expansion_levels,
        )
        kept_anchors.append(anchor_reference)
        prefixes.append(selection.prefix)
        for member in selection.members:
            if member in seen_members:
                continue
            seen_members.add(member)
            members.append(member)

    return FamilyUnionSelection(
        anchor_references=kept_anchors,
        prefixes=prefixes,
        members=members,
    )
