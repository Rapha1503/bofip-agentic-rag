# BOFiP Data Card

## Dataset Purpose

This project uses the public `bofip-vigueur` API from data.economie.gouv.fr to build a retrieval corpus for French tax doctrine question answering. The goal is to test whether a structured RAG pipeline can retrieve and cite relevant BOFiP material for accountant-style answers.

## Current Runtime Scope

| Field | Value |
| --- | --- |
| Runtime corpus | No-filter `bofip-vigueur` source snapshot |
| Raw source rows | 9,048 |
| Stable document IDs | 9,048 |
| Base BOI references | 9,025 |
| Duplicate base-reference extra rows preserved | 23 |
| Chunk rows | 79,160 |
| Chunk strategy | `section_window` |
| Document embedding cache | E5-large, `(9048, 1024)` |
| Chunk embedding cache | E5-large, `(79160, 1024)` |
| Local max publication date observed | `2026-06-17` |

The active runtime stores `document_id` as a stable BOFiP permalink/PGP identity. `boi_reference` remains the legal BOI reference used for citations, grouping, and routing.

## Source Assumptions

The source acquisition script requests the public dataset without a series filter:

```text
https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/bofip-vigueur/records
```

The source identity policy is:

- keep every API row exposed by the no-filter snapshot;
- use the BOFiP permalink PGP id to disambiguate duplicate base references;
- keep `boi_reference` separate from `document_id`.

A production-grade release should continue to publish:

- collection timestamp;
- file hashes;
- document counts by series and content type;
- parser, chunker, and embedding versions;
- explicit exclusions, if any.

## Traceability Preserved

The runtime file `data/interim/raw_docs.jsonl` contains parsed, structured runtime documents. The raw API snapshot used for rebuild/audit is stored separately as `data/raw/latest_snapshot.jsonl` when present locally.

Parsed runtime documents preserve:

- `document_id`;
- `boi_reference`;
- title;
- content type;
- publication date;
- source URL;
- category path;
- sections;
- paragraphs;
- tables;
- internal links;
- legal references.

Chunks preserve:

- source type;
- document id;
- BOI reference;
- document version field;
- strategy;
- section path;
- paragraph range;
- text;
- token count;
- legal references.

## Known Data Quality Risks

- Freshness risk: the local corpus observed during audit ends at `2026-06-17`; official BOFiP may include newer publications after that snapshot.
- Duplicate-reference risk: 23 extra source rows share a base BOI reference with another row; they are preserved via stable `document_id`.
- Empty-source risk: 50 API rows have no useful body text and therefore do not produce chunks.
- Evaluation provenance risk: `data/interim/eval_queries_v1.jsonl` and `data/interim/passage_gold_v3.jsonl` are legacy retrieval gold files preserved for regression checks, not a fresh answer-quality certification.
- Licensing and redistribution risk: heavy raw/model artifacts are intentionally not committed to this repository.

## Phase 2 Data Work

- Track source series counts in the manifest or a signed audit artifact.
- Add table-specific evaluation queries for table-heavy BOFiP pages.
- Publish a tracked evaluation report that ties metrics to a corpus manifest.

## Local Artifact Check

The repository includes a preflight script for the current artifact contract:

```powershell
python scripts/check_setup.py --deep
```

This is not a replacement for a signed manifest, but it catches missing files, wrong JSONL counts, and wrong embedding shapes before the app starts. In deep mode, runtime artifact counts and shapes are checked against the tracked manifest.

The current machine-readable artifact reference is tracked in [full_corpus_manifest.json](full_corpus_manifest.json).
