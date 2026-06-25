# BOFiP Data Card

## Dataset Purpose

This project uses a BOFiP public API snapshot to build a retrieval corpus for French tax doctrine question answering. The goal is to test whether a structured RAG pipeline can retrieve and cite relevant BOFiP commentary for accountant-style answers.

## Current Runtime Scope

| Field | Value |
| --- | --- |
| Runtime corpus | BOFiP commentary documents |
| Raw source rows | 9,048 |
| Base BOI references | 9,025 |
| Chunk rows | 79,160 |
| Chunk strategy | `section_window` |
| Document embedding cache | E5-large, `(9048, 1024)` |
| Chunk embedding cache | E5-large, `(79160, 1024)` |
| Source snapshot | `bofip-vigueur` API, generated 2026-06-23 |

The active hosted runtime uses the no-filter `bofip-vigueur` API snapshot: 9,048 source rows, 9,025 base BOI references, and 23 duplicate base-reference rows preserved as distinct source identities.

## Source Assumptions

The active production bundle comes from the public `bofip-vigueur` API snapshot generated on 2026-06-23. The artifact manifest records the official API URL, row counts, file sizes, embedding shapes, and SHA-256 hashes for the runtime files.

A production-grade data release should still add a fully reproducible acquisition notebook or script, plus parser/chunker/embedding version metadata for every rebuilt bundle.

## Traceability Preserved

Raw documents preserve:

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

- Freshness risk: BOFiP may publish newer content after the 2026-06-23 artifact generation.
- Duplicate-reference risk: 9 BOI references appear more than once with different document IDs or titles.
- Table evidence risk: many BOFiP tables are parsed but not yet first-class chunks.
- Scope risk: the runtime preserves the current API snapshot, but any future API schema change requires a parser check.
- Licensing and redistribution risk: heavy raw/model artifacts are intentionally not committed to this repository.

## Local Artifact Check

The repository includes a preflight script for the current artifact contract:

```powershell
python scripts/check_setup.py --deep
```

This is not a replacement for a signed manifest, but it catches missing files, wrong JSONL counts, and wrong embedding shapes before the app starts.

The current machine-readable artifact reference is tracked in [full_corpus_manifest.json](full_corpus_manifest.json).
