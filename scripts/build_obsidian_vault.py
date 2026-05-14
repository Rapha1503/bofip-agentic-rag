from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import textwrap
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl
from bofip_cleanroom.models import raw_document_from_dict


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = textwrap.dedent(content).strip()
    lines = []
    for line in cleaned.splitlines():
        if line.startswith("        "):
            line = line[8:]
        if line.startswith("    "):
            line = line[4:]
        lines.append(line)
    normalized = _fix_mojibake("\n".join(lines).strip())
    path.write_text(normalized + "\n", encoding="utf-8")


def _fix_mojibake(text: str) -> str:
    current = text
    for _ in range(2):
        if not any(marker in current for marker in ("Ã", "â", "Â")):
            break
        try:
            repaired = current.encode("latin1").decode("utf-8")
        except UnicodeError:
            break
        if repaired == current:
            break
        current = repaired
    return current


def _relative_markdown_link(from_path: Path, to_path: Path, label: str) -> str:
    rel = Path(
        *Path(os.path.relpath(to_path.resolve(), start=from_path.parent.resolve())).parts
    )
    return f"[{label}]({rel.as_posix()})"


def _frontmatter(title: str, tags: list[str]) -> str:
    tag_block = "\n".join(f"  - {tag}" for tag in tags)
    return f"---\ntitle: {title}\ntags:\n{tag_block}\n---\n"


def _safe_name(text: str | None) -> str:
    raw = (text or "Unknown").replace("/", " - ").replace("\\", " - ")
    normalized = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Za-z0-9 ._-]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or "Unknown"


def _boi_family(boi_reference: str | None) -> str:
    if not boi_reference:
        return "unknown"
    parts = boi_reference.split("-")
    if len(parts) >= 3:
        return "-".join(parts[1:3])
    if len(parts) >= 2:
        return parts[1]
    return boi_reference


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an Obsidian vault for the BOFIP clean-room project.")
    parser.add_argument(
        "--vault-root",
        type=str,
        default=str(PROJECT_ROOT.parent / "bofip-rag-cleanroom-obsidian"),
    )
    args = parser.parse_args()

    vault_root = Path(args.vault_root).resolve()
    cleanroom_root = PROJECT_ROOT.resolve()

    reports_dir = cleanroom_root / "data" / "reports"
    notebooks_dir = cleanroom_root / "notebooks"
    raw_inventory = _read_json(reports_dir / "raw_inventory.json")
    chunk_summary = _read_json(reports_dir / "phase2_chunks_section_window_sample_5666.json")
    v4_best = _read_json(
        reports_dir
        / "phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__lexbase_sections_leads__densesections_firstpara__intfloat__multilingual-e5-base__base1p0_chunk_dense2p0_dense1p0_sections_leads2p0.json"
    )
    v4_failures = _read_json(
        reports_dir
        / "phase3_failure_analysis_phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__balanced_sections_leads.json"
    )
    targeted_review = _read_json(reports_dir / "phase3_targeted_case_review_raw_docs_sample_5666.json")
    project_status_path = cleanroom_root / "PROJECT_STATUS.md"
    raw_docs_path = cleanroom_root / "data" / "interim" / "raw_docs_sample_5666.jsonl"
    all_docs_path = cleanroom_root / "data" / "interim" / "raw_docs_sample_6295.jsonl"
    raw_docs = [raw_document_from_dict(item) for item in read_jsonl(raw_docs_path)]
    all_docs = [raw_document_from_dict(item) for item in read_jsonl(all_docs_path)]
    doc_by_ref = {doc.boi_reference: doc for doc in raw_docs}
    doc_by_id = {doc.document_id: doc for doc in all_docs}
    duplicate_boi_refs = {
        ref for ref, count in Counter(doc.boi_reference for doc in all_docs).items() if count > 1
    }

    miss_report = v4_failures["reports"][0]
    hard_misses = [miss for miss in miss_report["misses"] if miss["category"] == "true_top1_miss"]
    unsupported_results = [row for row in v4_best["results"] if not row.get("supported_query", True)]
    unsupported_top1_families = Counter(
        _boi_family(row["returned_boi"][0]) for row in unsupported_results if row.get("returned_boi")
    )
    hard_miss_expected_families = Counter(_boi_family(miss["expected_boi"]) for miss in hard_misses)
    family_confusion_count = sum(
        miss_report["miss_categories"].get(key, 0)
        for key in (
            "same_family_neighbor",
            "parent_child_family_confusion",
            "title_equivalent_or_version_confusion",
        )
    )
    targeted_rows = targeted_review.get("rows", [])

    # basic vault structure
    for folder in [
        ".obsidian",
        "00 Home",
        "01 Project",
        "02 Data",
        "03 Retrieval",
        "04 Experiments",
        "05 Queries",
        "06 Decisions",
        "10 BOFiP Docs",
        "99 Templates",
    ]:
        (vault_root / folder).mkdir(parents=True, exist_ok=True)

    _write(
        vault_root / ".obsidian" / "app.json",
        json.dumps(
            {
                "alwaysUpdateLinks": True,
                "newLinkFormat": "relative",
                "showLineNumber": True,
                "attachmentFolderPath": "attachments",
                "readableLineLength": False,
            },
            indent=2,
        ),
    )

    home_path = vault_root / "00 Home" / "00 Home.md"
    _write(
        home_path,
        _frontmatter("BOFIP Clean-Room Home", ["home", "index"])
        + f"""
        # BOFIP Clean-Room Vault

        Ce vault Obsidian sert de couche de connaissance au-dessus du clean-room.

        Règles:
        - source de vérité métrique: rapports JSON du clean-room
        - source de vérité exécutable: code, tests, scripts, notebooks
        - ce vault résume, relie et documente; il ne remplace rien

        ## Navigation

        - [[01 Project/Objective]]
        - [[01 Project/Current Baseline]]
        - [[02 Data/Raw BOFiP Source]]
        - [[02 Data/Corpus Snapshot]]
        - [[02 Data/BOFiP Family Map]]
        - [[03 Retrieval/Baseline Retrieval Pipeline]]
        - [[03 Retrieval/V4 Benchmark]]
        - [[03 Retrieval/Insight Summary]]
        - [[03 Retrieval/Failure Taxonomy]]
        - [[04 Experiments/Experiment Log]]
        - [[05 Queries/Hard Queries]]
        - [[06 Decisions/Decision Log]]
        - [[06 Decisions/Current Priorities]]
        - [[10 BOFiP Docs/Doc Graph Guide]]
        - [[10 BOFiP Docs/Graph Hubs]]

        ## Direct links to executed notebooks

        - {_relative_markdown_link(home_path, notebooks_dir / "07_query_workbench.executed.ipynb", "07 Query Workbench")}
        - {_relative_markdown_link(home_path, notebooks_dir / "08_v4_benchmark_audit.executed.ipynb", "08 V4 Benchmark Audit")}

        ## Current headline

        - Corpus BOFiP contenu: `{raw_inventory["content_documents"]}`
        - Baseline indexé aujourd’hui: `Commentaire` only, `{chunk_summary["document_count"]}` docs
        - Baseline chunking: `{chunk_summary["strategy"]}`
        - Baseline retrieval doc `v4`: `hit@1={v4_best["metrics"]["hit@1"]}`, `hit@3={v4_best["metrics"]["hit@3"]}`, `hit@5={v4_best["metrics"]["hit@5"]}`
        - Vrais top1 misses restants: `{len(hard_misses)}`
        - Confusions intra-famille: `{family_confusion_count}/{miss_report["miss_count"]}`

        ## Source project

        - {_relative_markdown_link(home_path, cleanroom_root / "PROJECT_STATUS.md", "PROJECT_STATUS.md")}
        """,
    )

    objective_path = vault_root / "01 Project" / "Objective.md"
    _write(
        objective_path,
        _frontmatter("Objective", ["project", "objective"])
        + f"""
        # Objective

        Rebuild a BOFiP-only RAG pipeline from raw sources in a clean-room project, without legacy behavior from the original repo.

        ## Scope now

        - Source corpus: BOFiP
        - Current reliable baseline: `Commentaire` only
        - Current focus: retrieval documentaire robuste aux paraphrases
        - Explicitly not yet in scope:
          - LLM generation
          - production abstention policy
          - CGI/LPF reintegration

        ## Source status file

        - {_relative_markdown_link(objective_path, project_status_path, "PROJECT_STATUS.md")}
        """,
    )

    baseline_path = vault_root / "01 Project" / "Current Baseline.md"
    _write(
        baseline_path,
        _frontmatter("Current Baseline", ["project", "baseline"])
        + f"""
        # Current Baseline

        ## Parsing

        - Full BOFiP content parsed without crash: `{raw_inventory["content_documents"]}/{raw_inventory["content_documents"]}`
        - Raw source format: `document.xml` + `data.html`

        ## Chunking

        - Strategy: `{chunk_summary["strategy"]}`
        - Documents: `{chunk_summary["document_count"]}`
        - Chunks: `{chunk_summary["chunk_count"]}`
        - Empty chunks: `{chunk_summary["empty_text_count"]}`
        - Too long chunks: `{chunk_summary["too_long_count"]}`
        - Avg tokens: `{chunk_summary["token_count_stats"]["avg"]}`

        ## Retrieval stage 1

        Promoted documentary fusion on `v4`:
        - lexical `base`
        - lexical `sections_leads`
        - dense document `sections_firstpara`
        - dense chunk-derived document `full`

        Weights:
        - `base = 1`
        - `sections_leads = 2`
        - `dense = 1`
        - `chunk_dense = 2`

        Metrics on commentary full corpus:
        - `hit@1 = {v4_best["metrics"]["hit@1"]}`
        - `hit@3 = {v4_best["metrics"]["hit@3"]}`
        - `hit@5 = {v4_best["metrics"]["hit@5"]}`

        Report:
        - {_relative_markdown_link(baseline_path, reports_dir / "phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__lexbase_sections_leads__densesections_firstpara__intfloat__multilingual-e5-base__base1p0_chunk_dense2p0_dense1p0_sections_leads2p0.json", "Promoted v4 report")}

        ## Retrieval stage 2

        - Local strategy: `chunk`
        - Local chunk mode: `body`
        - Stage 2 is no longer the main bottleneck

        ## Engineering read

        - parsing is stable enough
        - chunking is stable enough
        - the active bottleneck is stage-1 document retrieval under paraphrase
        """,
    )

    raw_source_path = vault_root / "02 Data" / "Raw BOFiP Source.md"
    _write(
        raw_source_path,
        _frontmatter("Raw BOFiP Source", ["data", "source"])
        + f"""
        # Raw BOFiP Source

        The clean-room does **not** use live API calls or page-by-page crawling.

        It reads a local BOFiP dump in read-only mode:
        - raw root: `{raw_inventory["raw_root"]}`

        ## Effective file structure

        - `Contenu/.../<doc_id>/<date>/document.xml`
        - `Contenu/.../<doc_id>/<date>/data.html` or `data.htm`
        - `Attachment/.../document.xml`

        ## Why this is the right ingestion mode

        - reproducible
        - offline
        - stable over time
        - metadata and source HTML are both preserved

        ## Official source reference

        This local dump matches the BOFiP open data distribution model.
        """,
    )

    corpus_snapshot_path = vault_root / "02 Data" / "Corpus Snapshot.md"
    _write(
        corpus_snapshot_path,
        _frontmatter("Corpus Snapshot", ["data", "snapshot"])
        + f"""
        # Corpus Snapshot

        ## Global inventory

        - content documents: `{raw_inventory["content_documents"]}`
        - attachment documents: `{raw_inventory["attachment_documents"]}`
        - HTML documents: `{raw_inventory["html_documents"]}`

        ## Content types

        {chr(10).join(f"- `{k}`: `{v}`" for k, v in raw_inventory["content_type_counts"].items())}

        ## Current retrieval scope

        The baseline is intentionally narrower:
        - included: `Commentaire`
        - excluded from baseline retrieval:
          - `Cartographie`
          - many `Formulaire`
          - some `Annexes`

        ## Chunk snapshot

        - commentary docs: `{chunk_summary["document_count"]}`
        - chunks: `{chunk_summary["chunk_count"]}`
        - chunk kinds:
        {chr(10).join(f"  - `{k}`: `{v}`" for k, v in chunk_summary["chunk_kind_counts"].items())}
        """,
    )

    family_map_path = vault_root / "02 Data" / "BOFiP Family Map.md"
    _write(
        family_map_path,
        _frontmatter("BOFiP Family Map", ["data", "taxonomy", "families"])
        + f"""
        # BOFiP Family Map

        ## Dominant subject families in the raw corpus

        {chr(10).join(f"- `{k}`: `{v}` docs" for k, v in raw_inventory["subject_counts_top20"].items())}

        ## Current retrieval implications

        - commentary retrieval pressure is dominated by large neighboring families such as `BIC`, `TVA`, `IS`, `IF`, `IR`
        - broad lexical tokens like `champ`, `base`, `modalites`, `obligations`, `exoneration` are structurally overloaded
        - this explains why stage-1 errors are often close-family confusions rather than total misses

        ## Notes

        - retrieval is currently scoped to `Commentaire` only
        - this note is useful for understanding corpus pressure, not for selecting runtime filters
        """,
    )

    retrieval_pipeline_path = vault_root / "03 Retrieval" / "Baseline Retrieval Pipeline.md"
    _write(
        retrieval_pipeline_path,
        _frontmatter("Baseline Retrieval Pipeline", ["retrieval", "pipeline"])
        + f"""
        # Baseline Retrieval Pipeline

        ## Stage 1 document retrieval

        Sources fused:
        - lexical `base`
        - lexical `sections_leads`
        - dense document `sections_firstpara`
        - dense chunk-derived document `full`

        ```mermaid
        flowchart TD
            Q[User query]
            Q --> LB[Lexical doc: base]
            Q --> LS[Lexical doc: sections_leads]
            Q --> DD[Dense doc: sections_firstpara]
            Q --> CD[Dense chunk index -> doc aggregation]
            LB --> F[RRF-like weighted fusion]
            LS --> F
            DD --> F
            CD --> F
            F --> D[Top BOFiP documents]
            D --> LC[Local chunk retrieval: chunk/body]
            LC --> E[Evidence chunks]
        ```

        Notes:
        - `sections_leads` improves paraphrase robustness by indexing the first useful sentence of section content
        - `chunk_dense(full)` keeps passage semantics that titles alone miss

        ## Stage 2 local passage retrieval

        - strategy: `chunk`
        - local text: `body`
        - goal: select the best evidence chunk inside the top document set

        ## Why this baseline exists

        Earlier baselines over-relied on:
        - titles only
        - sections only
        - or a single dense source

        The current baseline is the first one that improved `v4` materially without changing parsing or chunking.
        """,
    )

    v4_path = vault_root / "03 Retrieval" / "V4 Benchmark.md"
    _write(
        v4_path,
        _frontmatter("V4 Benchmark", ["retrieval", "benchmark", "v4"])
        + f"""
        # V4 Benchmark

        ## Dataset

        - total questions: `{v4_best["query_count"]}`
        - answerable: `{v4_best["supported_query_count"]}`
        - unsupported: `{v4_best["unsupported_query_count"]}`

        Intent:
        - more user-like phrasing
        - less lexical overlap with BOFiP titles
        - stronger paraphrase stress

        ## Current best metrics

        - `hit@1 = {v4_best["metrics"]["hit@1"]}`
        - `hit@3 = {v4_best["metrics"]["hit@3"]}`
        - `hit@5 = {v4_best["metrics"]["hit@5"]}`

        ## Why it matters

        This benchmark invalidated the earlier comfort zone and forced the project to improve the true stage-1 problem: document retrieval under paraphrase.

        ## Main audit notebook

        - {_relative_markdown_link(v4_path, notebooks_dir / "08_v4_benchmark_audit.executed.ipynb", "08 V4 Benchmark Audit")}
        """,
    )

    insight_summary_path = vault_root / "03 Retrieval" / "Insight Summary.md"
    _write(
        insight_summary_path,
        _frontmatter("Insight Summary", ["retrieval", "insights", "summary"])
        + f"""
        # Insight Summary

        ## What the vault makes obvious

        - the project is no longer blocked by parsing crashes or pathological chunks
        - the current bottleneck is stage-1 document retrieval under paraphrase
        - most misses are still close to the right place

        ## Hard numbers

        - total supported `v4` queries: `{v4_best["supported_query_count"]}`
        - promoted stage-1 `hit@1`: `{v4_best["metrics"]["hit@1"]}`
        - promoted stage-1 `hit@5`: `{v4_best["metrics"]["hit@5"]}`
        - miss count: `{miss_report["miss_count"]}`
        - family-like confusions: `{family_confusion_count}`
        - true top1 misses: `{len(hard_misses)}`

        ## Engineering reading

        - `same_family_neighbor + parent_child_family_confusion + title_equivalent_or_version_confusion` dominate the miss set
        - dense helps, but only when fused with lexical views; dense alone is not trustworthy enough
        - unsupported prompts often collapse into a few semantic basins instead of scattering randomly, which means abstention should reason about ambiguity and family pressure, not just lexical coverage

        ## Current strongest signals

        - lexical `sections_leads`
        - chunk-dense `full`
        - multiview documentary fusion

        ## Current weakest query styles

        {chr(10).join(f"- `{k}`: `{v}` misses" for k, v in miss_report["miss_patterns"].items())}
        """,
    )

    failure_taxonomy_path = vault_root / "03 Retrieval" / "Failure Taxonomy.md"
    _write(
        failure_taxonomy_path,
        _frontmatter("Failure Taxonomy", ["retrieval", "errors"])
        + f"""
        # Failure Taxonomy

        Current miss categories on promoted `v4` baseline:

        {chr(10).join(f"- `{k}`: `{v}`" for k, v in miss_report["miss_categories"].items())}

        Pattern distribution:

        {chr(10).join(f"- `{k}`: `{v}`" for k, v in miss_report["miss_patterns"].items())}

        Reading:
        - many misses are still same-family or parent/child confusions
        - only `{len(hard_misses)}` are currently classified as true top1 misses
        - this means parsing/chunking are no longer the main problem
        """,
    )

    unsupported_pressure_path = vault_root / "03 Retrieval" / "Unsupported Pressure.md"
    _write(
        unsupported_pressure_path,
        _frontmatter("Unsupported Pressure", ["retrieval", "unsupported", "abstention"])
        + f"""
        # Unsupported Pressure

        ## Top families returned for unsupported queries

        {chr(10).join(f"- `{family}`: `{count}` top1 returns" for family, count in unsupported_top1_families.most_common(10))}

        ## Reading

        - unsupported prompts are not random noise
        - they are attracted to semantically broad families already strong in the corpus
        - this is why a naive abstention rule can easily become too aggressive on answerable queries

        ## Implication

        A useful abstention gate will probably need:
        - retrieval margin or entropy
        - family concentration
        - and query/document semantic mismatch
        rather than pure lexical uncovered-ratio alone
        """,
    )

    experiment_log_path = vault_root / "04 Experiments" / "Experiment Log.md"
    _write(
        experiment_log_path,
        _frontmatter("Experiment Log", ["experiments", "log"])
        + """
        # Experiment Log

        ## Accepted

        - `section_window` chunking as the clean chunking baseline
        - `Commentaire`-only as the first reliable BOFiP retrieval scope
        - lexical `sections_leads`
        - dense chunk-derived document signal in `full` mode
        - promoted multiview stage 1 fusion

        ## Rejected

        - chunk-dense `body` as a documentary signal
        - pure chunk-derived lexical document ranking
        - flat hybrid retrieval as the main architecture
        - moving to LLM generation before retrieval stabilizes

        ## Still open

        - better arbitration between lexical and dense confidence per query
        - acronym/paraphrase handling for cases like `PEA`
        - abstention strong enough for generation
        """,
    )

    hard_queries_path = vault_root / "05 Queries" / "Hard Queries.md"
    hard_blocks = []
    for miss in hard_misses:
        top_doc = doc_by_ref.get(miss["top1_boi"])
        expected_doc = doc_by_ref.get(miss["expected_boi"])
        hard_blocks.append(
            f"""## {miss['id']}

- query: `{miss['query']}`
- expected: `{miss['expected_boi']}`
  - {expected_doc.title if expected_doc else miss['expected_title']}
- expected family: `{_boi_family(miss['expected_boi'])}`
- current top1: `{miss['top1_boi']}`
  - {top_doc.title if top_doc else miss['top1_title']}
- returned in top5: `{"yes" if miss['hit@3'] or miss.get('hit@5') else "no"}`
- pattern: `{miss['pattern']}`
"""
        )
    _write(
        hard_queries_path,
        _frontmatter("Hard Queries", ["queries", "hard-cases"])
        + "# Hard Queries\n\n"
        + "\n".join(hard_blocks),
    )

    decision_log_path = vault_root / "06 Decisions" / "Decision Log.md"
    _write(
        decision_log_path,
        _frontmatter("Decision Log", ["decisions", "architecture"])
        + f"""
        # Decision Log

        ## Accepted

        - Keep raw BOFiP ingestion as local `XML + HTML` dump, not live crawling
        - Keep `section_window` as chunking baseline
        - Keep `Commentaire` as the first reliable retrieval scope
        - Keep stage 2 local strategy `chunk/body`
        - Keep promoted stage 1 multiview baseline:
          - `base`
          - `sections_leads`
          - `doc_dense`
          - `chunk_dense(full)`

        ## Rejected

        - using Obsidian as a source of truth
        - jumping to LLM generation before retrieval is stable
        - heavy abstention gate in current state

        ## Current next decision

        Determine whether the next improvement should be:
        - query-side semantic expansion/reformulation under strict audit
        - or better per-query source arbitration in stage 1
        """,
    )

    current_priorities_path = vault_root / "06 Decisions" / "Current Priorities.md"
    _write(
        current_priorities_path,
        _frontmatter("Current Priorities", ["decisions", "priorities", "next-steps"])
        + f"""
        # Current Priorities

        ## What not to touch first

        - do not rewrite the parser again
        - do not rewrite the chunker again
        - do not move to LLM generation yet
        - do not promote the current abstention rule

        ## Why

        The evidence now says:
        - parser and chunker are stable enough
        - stage 2 local chunk selection is no longer the dominant bottleneck
        - stage 1 documentary retrieval under paraphrase is the active problem

        ## Current next bets

        1. Improve query-side semantic normalization under strict audit
        2. Improve per-query arbitration between lexical and dense documentary signals
        3. Re-audit hard acronym and short-keyword families:
           - {", ".join(sorted(hard_miss_expected_families))}

        ## Historical targeted cases still worth remembering

        {chr(10).join(f"- `{row['id']}`: `{row['provisional_classification']}`" for row in targeted_rows)}
        """,
    )

    _write(
        vault_root / "99 Templates" / "Experiment Template.md",
        _frontmatter("Experiment Template", ["template", "experiment"])
        + """
        # Experiment Template

        ## Hypothesis

        ## Change

        ## Dataset / Report

        ## Result

        ## Keep / Reject / Revise

        ## Notes
        """,
    )
    _write(
        vault_root / "99 Templates" / "Decision Template.md",
        _frontmatter("Decision Template", ["template", "decision"])
        + """
        # Decision Template

        ## Context

        ## Decision

        ## Evidence

        ## Tradeoff

        ## Revisit when
        """,
    )
    _write(
        vault_root / "99 Templates" / "Query Review Template.md",
        _frontmatter("Query Review Template", ["template", "query"])
        + """
        # Query Review Template

        ## Query

        ## Expected document

        ## Top returned documents

        ## Category

        ## Why it failed or succeeded

        ## Next action
        """,
    )

    # BOFiP document graph
    docs_root = vault_root / "10 BOFiP Docs"
    note_path_by_id: dict[str, Path] = {}
    for doc in all_docs:
        type_folder = _safe_name(doc.content_type or "Unknown")
        file_stem = doc.boi_reference
        if doc.boi_reference in duplicate_boi_refs:
            file_stem = f"{doc.boi_reference}__{doc.document_id}"
        note_path_by_id[doc.document_id] = docs_root / type_folder / f"{file_stem}.md"

    outbound_by_id: dict[str, list[tuple[str, str, str]]] = {doc.document_id: [] for doc in all_docs}
    unresolved_by_id: dict[str, list[tuple[str, str]]] = {doc.document_id: [] for doc in all_docs}
    inbound_by_id: dict[str, list[tuple[str, str]]] = {doc.document_id: [] for doc in all_docs}
    resolved_relation_counts = Counter()
    unresolved_relation_counts = Counter()

    for doc in all_docs:
        seen_targets: set[tuple[str, str]] = set()
        for relation in doc.relations:
            relation_type = relation.relation_type or "untyped"
            raw_value = relation.value
            target_id = raw_value.split(":", 1)[1] if ":" in raw_value else raw_value
            target_doc = doc_by_id.get(target_id)
            if target_doc and target_doc.document_id != doc.document_id:
                key = (target_doc.document_id, relation_type)
                if key not in seen_targets:
                    outbound_by_id[doc.document_id].append((target_doc.document_id, relation_type, raw_value))
                    inbound_by_id[target_doc.document_id].append((doc.document_id, relation_type))
                    resolved_relation_counts[relation_type] += 1
                    seen_targets.add(key)
            else:
                unresolved_by_id[doc.document_id].append((raw_value, relation_type))
                unresolved_relation_counts[relation_type] += 1

    def _wikilink_for_doc_id(document_id: str) -> str:
        target_doc = doc_by_id[document_id]
        target = note_path_by_id[document_id].relative_to(vault_root).with_suffix("")
        label = target_doc.boi_reference
        if target_doc.boi_reference in duplicate_boi_refs:
            label = f"{target_doc.boi_reference} ({document_id})"
        return f"[[{target.as_posix()}|{label}]]"

    for doc in all_docs:
        note_path = note_path_by_id[doc.document_id]
        outgoing = outbound_by_id[doc.document_id]
        inbound = inbound_by_id[doc.document_id]
        unresolved = unresolved_by_id[doc.document_id]
        section_titles = [section.title for section in doc.sections[:8] if section.title]
        excerpt = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                excerpt = paragraph.text.strip()[:700]
                break
        relation_type_counts = Counter(rel_type for _, rel_type, _ in outgoing)
        tags = [
            "bofip-doc",
            _safe_name(doc.content_type or "Unknown").replace(" ", "_").lower(),
        ]
        tags.extend(_safe_name(subject).replace(" ", "_").lower() for subject in doc.subjects[:3] if subject)
        _write(
            note_path,
            _frontmatter(doc.boi_reference, tags)
            + f"""
            # {doc.boi_reference}

            ## Identity

            - title: {doc.title}
            - document_id: `{doc.document_id}`
            - content_type: `{doc.content_type or "Unknown"}`
            - publication_date: `{doc.publication_date or "Unknown"}`
            - version_status: `{doc.version_status or "Unknown"}`
            - subjects: {", ".join(f"`{subject}`" for subject in doc.subjects) if doc.subjects else "none"}
            - category_path: {" > ".join(doc.category_path) if doc.category_path else "none"}

            ## Structure

            - sections: `{len(doc.sections)}`
            - paragraphs: `{len(doc.paragraphs)}`
            - tables: `{len(doc.tables)}`
            - internal_links: `{len(doc.internal_links)}`
            - legal_refs: `{len(doc.legal_refs)}`
            - raw_text_length: `{doc.raw_text_length}`

            ## Source-derived outgoing document relations

            - resolved outgoing links: `{len(outgoing)}`
            - relation types: {", ".join(f"`{k}={v}`" for k, v in relation_type_counts.items()) if relation_type_counts else "none"}

            {chr(10).join(f"- {_wikilink_for_doc_id(target_id)} (`{rel_type}`)" for target_id, rel_type, _ in outgoing[:50]) if outgoing else "- none"}

            ## Source-derived inbound document relations

            - inbound links: `{len(inbound)}`

            {chr(10).join(f"- {_wikilink_for_doc_id(source_id)} (`{rel_type}`)" for source_id, rel_type in inbound[:50]) if inbound else "- none"}

            ## Unresolved raw relations

            {chr(10).join(f"- `{raw_value}` (`{rel_type}`)" for raw_value, rel_type in unresolved[:20]) if unresolved else "- none"}

            ## Section preview

            {chr(10).join(f"- {title}" for title in section_titles) if section_titles else "- no explicit sections"}

            ## Excerpt preview

            {excerpt if excerpt else "No paragraph excerpt available."}
            """,
        )

    top_inbound = sorted(
        ((document_id, len(inbound_by_id[document_id])) for document_id in inbound_by_id),
        key=lambda item: (-item[1], item[0]),
    )[:30]
    top_outbound = sorted(
        ((document_id, len(outbound_by_id[document_id])) for document_id in outbound_by_id),
        key=lambda item: (-item[1], item[0]),
    )[:30]

    doc_graph_guide_path = docs_root / "Doc Graph Guide.md"
    _write(
        doc_graph_guide_path,
        _frontmatter("Doc Graph Guide", ["graph", "docs", "usage"])
        + f"""
        # Doc Graph Guide

        This note explains how to see the **BOFiP document-to-document graph**, not the project meta graph.

        ## What exists now

        - content-document notes generated: `{len(all_docs)}`
        - duplicate BOI references handled with document-id suffix: `{len(duplicate_boi_refs)}`
        - resolved content-to-content links: `{sum(len(v) for v in outbound_by_id.values())}`
        - unresolved raw relations: `{sum(len(v) for v in unresolved_by_id.values())}`

        ## In Obsidian

        1. Open `Graph view`
        2. In the graph filter box, type exactly:
           - `path:"10 BOFiP Docs"`
        3. To inspect one BOFiP note locally:
           - open any BOI note under `10 BOFiP Docs`
           - open `Local graph`

        ## Good starting notes

        - [[10 BOFiP Docs/Graph Hubs]]
        - {_wikilink_for_doc_id(top_inbound[0][0]) if top_inbound else "- none"}
        - {_wikilink_for_doc_id(top_outbound[0][0]) if top_outbound else "- none"}

        ## Engineering rule

        These links are source-derived from raw BOFiP `dc:relation` resolution whenever possible.
        They are not synthetic semantic neighbors.
        """,
    )

    graph_hubs_path = docs_root / "Graph Hubs.md"
    _write(
        graph_hubs_path,
        _frontmatter("Graph Hubs", ["graph", "docs", "hubs"])
        + f"""
        # Graph Hubs

        ## Relation counts

        - resolved `references`: `{resolved_relation_counts.get("references", 0)}`
        - resolved `requires`: `{resolved_relation_counts.get("requires", 0)}`
        - unresolved `references`: `{unresolved_relation_counts.get("references", 0)}`
        - unresolved `requires`: `{unresolved_relation_counts.get("requires", 0)}`

        ## Top inbound BOFiP documents

        {chr(10).join(f"- {_wikilink_for_doc_id(document_id)}: `{count}` inbound links" for document_id, count in top_inbound)}

        ## Top outbound BOFiP documents

        {chr(10).join(f"- {_wikilink_for_doc_id(document_id)}: `{count}` outbound links" for document_id, count in top_outbound)}

        ## Engineering reading

        High-inbound notes are likely structural hubs in the BOFiP corpus.
        They are good candidates for:
        - graph exploration
        - retrieval pressure analysis
        - future graph-based post-retrieval expansion
        """,
    )

    print(f"Obsidian vault built: {vault_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
