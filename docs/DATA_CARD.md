# BOFiP Data Card

## Dataset Purpose

This project uses a local BOFiP export to build a retrieval corpus for French tax doctrine question answering. The goal is to test whether a structured RAG pipeline can retrieve and cite relevant BOFiP commentary for accountant-style answers.

## Current Runtime Scope

| Field | Value |
| --- | --- |
| Runtime corpus | BOFiP commentary documents |
| Raw document rows | 5,666 |
| Unique BOI references | 5,657 |
| Chunk rows | 66,289 |
| Chunk strategy | `section_window` |
| Document embedding cache | E5-large, `(5666, 1024)` |
| Chunk embedding cache | E5-large, `(66289, 1024)` |
| Local max publication date observed | `2026-01-28` |

The local directory also contains a broader 6,295-document export including non-commentary content types, but the active runtime uses the 5,666-document commentary corpus.

## Source Assumptions

The parsers expect a local BOFiP export shaped like:

```text
Contenu/**/document.xml
Contenu/**/data.html or data.htm
```

The repository does not currently include a public acquisition script for the full export. A production-grade release should add:

- acquisition method;
- collection timestamp;
- official source URL;
- file hashes;
- document counts by content type;
- parser, chunker, and embedding versions;
- explicit exclusions.

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

- Freshness risk: the local corpus observed during audit ends at `2026-01-28`; official BOFiP may include newer publications.
- Duplicate-reference risk: 9 BOI references appear more than once with different document IDs or titles.
- Table evidence risk: many BOFiP tables are parsed but not yet first-class chunks.
- Scope risk: current runtime is commentary-only and excludes some content types available in the broader local export.
- Licensing and redistribution risk: heavy raw/model artifacts are intentionally not committed to this repository.

## Local Artifact Check

The repository includes a preflight script for the current artifact contract:

```powershell
python scripts/check_setup.py --deep
```

This is not a replacement for a signed manifest, but it catches missing files, wrong JSONL counts, and wrong embedding shapes before the app starts.

The current machine-readable artifact reference is tracked in [full_corpus_manifest.json](full_corpus_manifest.json).
