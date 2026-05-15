# BOFIP-RAG System Documentation

> **This is the single source of truth for the entire system.** Send this file to any AI (ChatGPT, Claude, Codex) and it will understand the full architecture, pipeline, and implementation.

**Last Updated**: 2026-02-10

**Recent Fixes (2026-02-10)**:
- Recovered ChromaDB from interrupted rebuild and re-synced legal chunks (CGI/LPF) to match `chunks.json`.
- Added `scripts/evaluate_retrieval.py` for retrieval-only metrics (Recall@K, Precision@K, HitRate).
- Added `scripts/sync_legal_chunks.py` delta mode (`--delta-file`) for incremental legal vector refresh.
- Hardened PDF legal parser (broader article patterns + page continuation mapping).
- Added LEGI XML bootstrap parser and ingestion script (`src/data_pipeline/legi_parser.py`, `scripts/process_legi_xml.py`).
- Added LEGI archive parser + resumable downloader (`src/data_pipeline/legi_tar_parser.py`, `scripts/process_legi_archive.py`) and migrated legal corpus to structured LEGI data.
- Cross-reference lookup now caps chunks per legal article to reduce near-duplicate injection.
- Added `scripts/refresh_legi_daily.py` to automate daily LEGI refresh + BM25 rebuild + Chroma sync with state tracking.
- Added `scripts/build_retrieval_dataset_from_cache.py` to expand retrieval ground truth from real cached queries.
- Added `scripts/validate_retrieval_dataset.py` to auto-score silver questions and output a filtered validated dataset.
- Added dataset-aware retrieval eval/tuning (`--dataset`) and quiet mode by default (`--verbose` to trace).
- Tuned reranker pool size on expanded retrieval set (default now 30).
- Added generation faithfulness guardrail (LLM verifier + heuristic fallback) with explicit abstention when evidence is insufficient.
- Added embedding A/B benchmark script (`scripts/benchmark_embeddings.py`) with model-specific Chroma collections.
- Embedding cache is now model-scoped (`data/cache/embeddings/<model_slug>/...`) to prevent cross-model contamination.
- Vector store now supports custom persist dir / collection / embedding model for isolated benchmarking.

**Recent Fixes (2026-02-09)**:
- LLM cache now enforces TTL expiration before reuse.
- Legal reference extraction now supports shorthand query forms like `art. 145 CGI` and `L.64 LPF`.
- Context citation formatting no longer prints empty paragraph markers.
- UI/prompt wording aligned with actual sources (BOFIP + CGI + LPF).

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Domain Context](#3-domain-context)
4. [Data Sources](#4-data-sources)
5. [Data Pipeline: Raw to Chunks](#5-data-pipeline-raw-to-chunks)
6. [Chunk Data Schema](#6-chunk-data-schema)
7. [Indexing: Chunks to Searchable](#7-indexing-chunks-to-searchable)
8. [Retrieval: Query to Relevant Chunks](#8-retrieval-query-to-relevant-chunks)
9. [Generation: Chunks to Answer](#9-generation-chunks-to-answer)
10. [User Interface](#10-user-interface)
11. [Configuration](#11-configuration)
12. [File Structure](#12-file-structure)
13. [Benchmarks and Test Results](#13-benchmarks-and-test-results)
14. [Architecture Principles and Lessons](#14-architecture-principles-and-lessons)
15. [Deprecated Components](#15-deprecated-components)
16. [Backlog / Next Steps](#16-backlog--next-steps)
17. [How to Run](#17-how-to-run)

---

## 1. Project Overview

**BOFIP-RAG** is a Retrieval-Augmented Generation chatbot for French tax documentation. It answers expert-comptable questions with sourced answers citing official BOI references.

### Benchmarks (2026-02-10)

| Suite | Score |
|-------|-------|
| Regression 7Q | **7/7 (100%)** |
| User 8Q | **8/8 (100%)** |
| Novel 10Q (anti-overfitting) | **10/10 (100%)** |
| **Total** | **25/25 (100%)** |

### Data

- **86,045 total chunks** indexed
  - 82,653 BOFIP (official tax commentary)
  - 2,416 CGI (Code General des Impots - LEGI structured)
  - 976 LPF (Livre des Procedures Fiscales - LEGI structured)

### Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| LLM | Groq API | Llama 3.3 70B, temperature 0.1, max 1500 tokens |
| Embeddings | intfloat/multilingual-e5-base | 768 dimensions, French-optimized |
| Vector DB | ChromaDB | Persistent, local, 86,045 vectors |
| Keyword Search | rank-bm25 (BM25Okapi) | Custom French tokenizer |
| Reranker | mmarco-mMiniLMv2-L12-H384-v1 | Multilingual cross-encoder, configurable pool (default 30) |
| Frontend | Streamlit | http://localhost:8501 |
| Language | Python 3.11 | Windows, venv |

---

## 2. Architecture Diagram

```
Question (French)
    |
    v
+---------------------------------------------------+
|              search_simple()                       |
|  +-------------+  +-----------+  +-------------+  |
|  |   Vector    |  |   BM25    |  | CGI/LPF-only|  |
|  |   Search    |  |  Search   |  | Vector (5)  |  |
|  | (E5-multi)  |  | (Fr tok.) |  | (diversified)|  |
|  +------+------+  +-----+-----+  +------+------+  |
|         |               |               |          |
|         v               v               |          |
|    Normalize 0-1   Normalize 0-1        |          |
|         +-------+-------+              |          |
|                 |                       |          |
|                 v                       |          |
|       Merge (boost if in both)          |          |
|                 |                       |          |
|                 v                       |          |
|       Table Chunk Inclusion (BOFIP only)|          |
|                 |                       |          |
|                 v                       v          |
|       Legal Diversified Injection (0.70)           |
|                 |                                  |
|                 v                                  |
|       Cross-Reference Injection (0.90)             |
|       (BOFIP refs "art. X du CGI" -> inject X)     |
|                 |                                  |
|                 v                                  |
|       Value-Aware Boosting                         |
|                 |                                  |
|                 v                                  |
|       French Cross-Encoder Reranker (pool default 30)|
|       (mmarco-mMiniLMv2-L12-H384-v1)              |
|                 |                                  |
|                 v                                  |
|          Return top N                              |
+---------------------------------------------------+
                  |
                  v
        LLM (Groq Llama 3.3)
        Grille d'Analyse Fiscale
                  |
                  v
          Answer + Sources
```

### Added Files (2026-02-10)

- `src/data_pipeline/legi_parser.py` - LEGI XML bootstrap parser
- `scripts/process_legi_xml.py` - LEGI XML ingestion CLI
- `src/data_pipeline/legi_tar_parser.py` - LEGI tar parser (CGI/LPF targeted, as-of version selection)
- `scripts/process_legi_archive.py` - LEGI archive ingestion CLI (latest/latest-full + resumable download)
- `scripts/sync_legal_chunks.py` - Recover/sync legal chunks in ChromaDB (full or delta mode)
- `scripts/evaluate_retrieval.py` - Retrieval-only metrics (Recall@K / Precision@K / HitRate)
- `scripts/tune_reranker_pool.py` - Reranker pool benchmark (quality + latency)
- `scripts/benchmark_embeddings.py` - Embedding model benchmark with isolated vector stores
- `scripts/build_retrieval_dataset_from_cache.py` - Build expanded retrieval dataset (gold + silver cache-derived)
- `scripts/validate_retrieval_dataset.py` - Validate silver questions (coverage/rank), output filtered dataset
- `scripts/refresh_legi_daily.py` - Daily LEGI orchestration + audit state

---

## 3. Domain Context

### What is BOFIP?

The **Bulletin Officiel des Finances Publiques - Impots** (BOFIP) is the official tax doctrine of the French tax administration (DGFiP). It interprets and comments on the law. It replaced the old BOI on September 12, 2012.

### Legal Hierarchy (Critical for the system)

```
CGI / LPF  = THE LAW (Code General des Impots / Livre des Procedures Fiscales)
    |
    v
BOFIP      = COMMENTARY (administration's interpretation of the law)
```

**Rule**: When CGI/LPF and BOFIP conflict, the LAW (CGI/LPF) prevails. This is enforced in our system prompt.

### BOI Reference Format

```
BOI-{Serie}-{Division}-{Titre}-{Chapitre}-{Section}-{Date}
```

Example: `BOI-IR-LIQ-10-10-10-30-20240618`
- **BOI**: Standard prefix
- **IR**: Serie (Impot sur le Revenu)
- **LIQ**: Division (Liquidation)
- **10-10-10-30**: Hierarchy
- **20240618**: Publication date (YYYYMMDD)

### Tax Series

| Code | Name | Count | Description |
|------|------|-------|-------------|
| BIC | Benefices Industriels et Commerciaux | 741 | Commercial business income |
| IS | Impot sur les Societes | 495 | Corporate tax |
| IF | Impots Fonciers | 489 | Property taxes (CFE, taxe fonciere) |
| TVA | Taxe sur la Valeur Ajoutee | 485 | Value-added tax |
| IR | Impot sur le Revenu | 376 | Personal income tax |
| ENR | Enregistrement | 335 | Registration (donations, succession) |
| INT | International | 321 | International tax treaties |
| RPPM | Revenus de Placements et Mobiliers | 280 | Investment income |
| CF | Controle Fiscal | 208 | Tax audits and procedures |
| RFPI | Revenus Fonciers et Plus-Values Immobilieres | 164 | Real estate income and capital gains |
| BNC | Benefices Non Commerciaux | 146 | Liberal professions income |
| RES | Rescrits | 165 | Formal rulings |

### Document Types

| Type | Count | Description |
|------|-------|-------------|
| **Commentaire** | 5,666 | Main doctrinal commentary (our primary source) |
| Bareme | 36 | Rate tables and thresholds |
| Formulaire | 84 | Forms |
| Autres annexes | 303 | Other annexes |

### Key Vocabulary

| Term | Definition |
|------|------------|
| **CGI** | Code General des Impots (the tax law) |
| **LPF** | Livre des Procedures Fiscales (procedures law) |
| **BOI** | Reference to a BOFIP document |
| **Rescrit** | Formal ruling by the administration |
| **Assiette** | Tax base (what is taxed) |
| **Fait generateur** | Taxable event trigger |
| **Exigibilite** | When tax becomes due |
| **Opposabilite** | Taxpayer can rely on BOFIP doctrine (article L80 A du LPF) |

---

## 4. Data Sources

### BOFIP (Primary Source)

- **URL**: https://data.economie.gouv.fr/explore/dataset/bofip-impots/
- **Format**: Stock file (~111MB compressed), extracted to XML metadata + HTML content
- **Total documents**: ~6,295 (5,666 Commentaire)

**Raw file structure**:
```
data/raw/bofip_extracted/BOFiP/documents/
â”œâ”€â”€ Contenu/
â”‚   â”œâ”€â”€ Commentaire/{Serie}/{DocID}/{Date}/
â”‚   â”‚   â”œâ”€â”€ document.xml    # Metadata (Dublin Core + BOFIP namespace)
â”‚   â”‚   â””â”€â”€ data.html       # Content (paragraphs, headers, tables)
â”‚   â”œâ”€â”€ Bareme/...
â”‚   â””â”€â”€ Autres annexes/...
â”œâ”€â”€ Attachment/              # Attached files (.doc, .xls)
â””â”€â”€ PlanClassement/          # Classification plan
```

**Metadata structure** (document.xml):
```xml
<document type="Contenu">
  <dc:dublincore>
    <dc:title>Document title</dc:title>
    <dc:date>2024-06-18</dc:date>
    <dc:subject>IR</dc:subject>        <!-- Series -->
    <dc:identifier>1032-PGP</dc:identifier>  <!-- Doc ID -->
    <dc:relation type="references">Contenu:2494-PGP</dc:relation>
  </dc:dublincore>
  <bofip:bodgfip>
    <bofip:contenu_id>BOI-IR-LIQ-10-10-10-30-20240618</bofip:contenu_id>
    <bofip:contenu_type>Commentaire</bofip:contenu_type>
  </bofip:bodgfip>
</document>
```

**Content structure** (data.html):
- Paragraphs numbered by 10s: Â§1, Â§10, Â§20, Â§30...
- Section headers: `<h1>`, `<h2>`, `<h3>` with Roman/letter hierarchy (I > A > 1)
- Tables: `<table>` with `<caption>` for rate/threshold data
- Cross-references: `<a href="Contenu:2494-PGP">BOI-IR-LIQ-20-20-20</a>`
- Legal references: `<span>article 196 A bis du code general des impots (CGI)</span>`

### CGI and LPF (Structured LEGI Sources)

- **Primary legal source**: DILA LEGI archives (`LEGI_*.tar.gz`, `Freemium_legi_global_*.tar.gz`)
- **Location**: `data/raw/legi/`
- **Parser**: `src/data_pipeline/legi_tar_parser.py`
- **Current strategy**: one "as-of" in-force version per article number
- **boi_reference** format: `"CGI Art. 102 ter"`, `"LPF Art. L. 64"`
- **Fallback source**: local PDFs in `data/raw/pdfs/` (kept for recovery/debug only)

---

## 5. Data Pipeline: Raw to Chunks

### Step 1: BOFIP Parsing (`src/data_pipeline/parser.py`)

- `parse_metadata(xml_path)`: Extracts from document.xml (Dublin Core fields, BOFIP namespace)
- `parse_content(html_path, metadata)`: Walks HTML body, detects paragraph numbers, section headers, tables
- Output: `BOFIPMetadata` + list of `BOFIPChunk` objects

### Step 2: Semantic Chunking (`src/data_pipeline/semantic_chunker.py`)

Strategy: **1 fiscal rule = 1 chunk** (at paragraph boundaries)

- Tracks hierarchical section context (h1 > h2 > h3 > h4)
- Each numbered paragraph (Â§10, Â§20) starts a new chunk
- Accumulates content until next paragraph or header
- Post-processing: merge chunks < 50 tokens, split chunks > 1500 tokens at sentence boundaries
- `section_path` format: `"I > A > 1"` (hierarchical breadcrumb)
- `text_with_context` prepends section_path to text for richer embeddings

### Step 3: PDF Parsing Fallback (`src/data_pipeline/pdf_parser.py`)

- Uses `pdfplumber` for text extraction
- Article boundary detection handles forms like `39 A`, `279-0 bis A`, `L. 10-0 A`, `R* 57-1`
- Detects and skips TOC pages
- Maps continuation pages to previous article reference (avoids noisy `CGI p.X` in legal body pages)
- Sets `boi_reference` at source: `"CGI Art. 102 ter"`, `"LPF Art. L. 10"`
- Sets `content_type`: `"CGI"` or `"LPF"` (used for filtering in search)
- **Status**: fallback path, no longer primary legal ingestion

### Step 3b: LEGI Structured Ingestion (`src/data_pipeline/legi_tar_parser.py`)

- Streams official LEGI tar archives without full extraction
- Targets canonical code IDs:
  - CGI: `LEGITEXT000006069577`
  - LPF: `LEGITEXT000006069583`
- Normalizes legal article numbering (`L64` -> `L. 64`)
- Selects one active article version per reference at a given `as_of` date
- Adds direct Legifrance article URLs (`https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI...`)

### Step 4: Processing Pipeline

- `src/data_pipeline/process.py`: Orchestrates BOFIP parsing with ThreadPoolExecutor (4 workers)
- `scripts/process_pdfs.py`: Parses CGI/LPF PDFs, appends to chunks.json, auto-detects source from filename
- `scripts/process_legi_xml.py`: Parses LEGI XML files, emits `legi_chunks.json`, optional append to main chunks
- `scripts/process_legi_archive.py`: Parses LEGI archives (daily/full), optional append replacing CGI/LPF in main chunks
- `scripts/refresh_legi_daily.py`: Orchestrates daily LEGI refresh + BM25 rebuild + legal vector sync with state file
- `scripts/reindex_semantic.py`: Full rebuild (semantic chunk -> save -> ChromaDB + BM25)
- **Output**: `data/processed/chunks.json` (~199 MB, 86,045 chunks)

### Chunk Processing Rules

- `merge_small_chunks()`: Merge consecutive chunks < 100 tokens
- `split_large_chunks()`: Split at sentence boundaries when > 800 tokens
- Filter: Remove chunks < 20 tokens or empty

---

## 6. Chunk Data Schema

### BOFIPChunk Fields

| Field | Type | BOFIP Example | CGI/LPF Example |
|-------|------|---------------|-----------------|
| `chunk_id` | str | `"BOI-IR-LIQ-10_p20"` | `"CGI-Code_p505_art102ter_a1b2"` |
| `text` | str | Clean paragraph text | Article text |
| `text_with_context` | str | `"[BOI-ref]\nI > A > 1\n\ntext"` | Same with section path |
| `boi_reference` | str | `"BOI-IR-LIQ-10-10-10-30-20240618"` | `"CGI Art. 102 ter"` |
| `doc_id` | str | `"1032-PGP"` | `"CGI-Code_General"` |
| `series` | List[str] | `["IR", "LIQ"]` | `[]` (empty) |
| `section_title` | str | `"I. Personnes concernees"` | Article title or None |
| `paragraph_number` | str | `"20"` or `"20-30"` | Page number |
| `publication_date` | str | `"2024-06-18"` | From PDF first page |
| `source_url` | str | `"https://bofip.impots.gouv.fr/..."` | `""` |
| `content_type` | str | `"Commentaire"` or `"Bareme"` | `"CGI"` or `"LPF"` |
| `contains_table` | bool | True if has tabular data | True if grid pattern |
| `is_header` | bool | True if section header | False |
| `token_count` | int | ~200-800 | ~200-800 |
| `source` | str | `"BOFIP"` | `"CGI"` or `"LPF"` |

### Fields Added During Search

| Field | Type | Added By | Description |
|-------|------|----------|-------------|
| `combined_score` | float | Merge step | 0-1.3+, normalized BM25+Vector |
| `normalized_score` | float | Normalize step | 0-1, per-method score |
| `rerank_score` | float | Reranker | Cross-encoder relevance |
| `source` | str | Various | `"bm25"`, `"vector"`, `"table_supplement"`, `"legal_diversified"`, `"cross_reference"` |
| `distance` | float | Vector search | ChromaDB L2 distance |

---

## 7. Indexing: Chunks to Searchable

### 7a. BM25 Index (`src/retrieval/bm25.py`)

**`tokenize_french(text)`** â€” Custom French tokenizer:

```python
# Stopwords (EXCLUDING domain-critical "plus", "moins")
stopwords = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'au', 'aux',
    'ce', 'cette', 'ces', 'que', 'qui', 'quoi', 'dont', 'dans', 'sur', 'sous', 'par',
    'pour', 'avec', 'sans', 'est', 'sont', 'etre', 'Ãªtre', 'avoir', 'il', 'elle', 'ils', 'elles',
    'en', 'ne', 'pas', 'si', 'se', 'son', 'sa', 'ses', 'leur', 'leurs',
    'nous', 'vous', 'tout', 'tous', 'toute', 'toutes', 'autre', 'autres',
    'mÃªme', 'meme', 'aussi', 'ainsi', 'donc', 'car', 'mais', 'oÃ¹', 'ou'
}

# French number normalization: "18 300" -> "18300" (applied twice for "1 000 000")
text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
# French decimal comma: "5,5%" -> "5.5%"
text = re.sub(r'(\d),(\d)', r'\1.\2', text)

# Tokenize preserving hyphens and apostrophes
tokens = re.findall(
    r"[a-zA-Z0-9accentedchars]+(?:[-'][a-zA-Z0-9accentedchars]+)*",
    text.lower()
)
# Filter: remove stopwords, allow >= 2 char tokens (IS, IR)
tokens = [t for t in tokens if t not in stopwords and len(t) >= 2]
```

**Key design decisions**:
- `"plus-value"` â†’ `["plus-value"]` (single token, not split)
- `"d'impot"` â†’ `["d'impot"]` (preserves contraction)
- `"IS"` â†’ `["is"]` (2-char tokens allowed for tax codes)
- `"plus"`, `"moins"` NOT in stopwords (critical for "plus-value", "moins-value")

**Storage**: `data/processed/bm25_index.pkl` (239 MB, pickled BM25Okapi + chunk data)

### 7b. E5 Embeddings (`src/retrieval/embeddings.py`)

- **Model**: `intfloat/multilingual-e5-base` (768 dimensions)
- **E5 prefix handling**: `"query: "` for search queries, `"passage: "` for document chunks
- **Caching**: Per-text MD5 hash in model-scoped cache: `data/cache/embeddings/<model_slug>/`
- **Singleton**: `get_embedding_model(model_name=None)` keeps one shared instance per embedding model
- **Memory control**: `reset_embedding_models()` clears loaded embedding instances between benchmark runs

### 7c. ChromaDB Vector Store (`src/retrieval/vector_store.py`)

- **Collection**: default `"bofip_chunks"` (persistent at `data/chroma_db/`), configurable for A/B benchmarks
- **add_chunks()**: Batches of 500, deduplicates by chunk_id, stores all metadata fields
- **search()**: Accepts optional `where` dict for ChromaDB metadata filtering
  - Example: `where={"content_type": {"$in": ["CGI", "LPF"]}}` for legal-text-only search
- **Score**: `1 - distance` (L2 distance converted to similarity)
- **Isolation support**: constructor accepts `persist_dir`, `collection_name`, and `embedding_model_name`
- **Lazy embedding load**: embedding model loads only when search/index is called (not on init)

### 7d. Legal Reference Lookup Table (built at startup in `hybrid.py`)

At `HybridRetriever.__init__()`, `_build_legal_ref_lookup()` scans all CGI/LPF chunks from the BM25 index and builds a dict:

```
{
  "CGI:102 ter": [chunk_dict, ...],
  "CGI:145": [chunk_dict, ...],
  "LPF:L. 52": [chunk_dict, ...],
  ...
}
```

**Total**: 3,392 unique article references indexed from structured LEGI legal corpus.
Cross-reference lookup is pruned to max 3 chunks per article to avoid duplicate flooding.

---

## 8. Retrieval: Query to Relevant Chunks

**File**: `src/retrieval/hybrid.py` â€” `search_simple(query, n_results=30)`

This is the production search method. It executes 13 steps in order:

### Step 1: BM25 Search
```
bm25_results = self.bm25.search(query, n_results=30)
```
Returns up to 30 results with raw BM25 scores (0 to ~200).

### Step 2: Vector Search
```
vector_results = self.vector_store.search(query, n_results=30)
```
Returns up to 30 results with similarity scores (0 to ~1).

### Step 3: Diversified Legal Vector Search
```
legal_results = self.vector_store.search(
    query, n_results=5,
    where={"content_type": {"$in": ["CGI", "LPF"]}}
)
```
Separate search filtering to CGI/LPF chunks only. Guarantees legal text in the candidate pool even when 82K BOFIP chunks dominate.

### Step 4-5: Normalize Scores to 0-1
```
normalized = (score - min) / (max - min)
```
Applied independently to BM25 and vector results.

### Step 6: Merge by Chunk ID
Build a dict mapping each unique `chunk_id` to its BM25 and vector scores.

### Step 7: Combine Scores
```python
if bm25_score > 0 AND vector_score > 0:
    combined = max(bm25, vector) + 0.3 * min(bm25, vector)  # Agreement bonus
else:
    combined = bm25 + vector  # Single-source score
```

### Step 8: Sort by Combined Score (descending)

### Step 9: Table Chunk Inclusion
- Extract `doc_id` from top 15 results (BOFIP documents only, skip CGI/LPF)
- Fetch all chunks with `contains_table=True` for those documents
- Add with `combined_score = 0.75`, `source = "table_supplement"`
- **Why**: Table text doesn't match query keywords but contains critical data (rate tables, thresholds)

### Step 10: Legal Diversified Injection
- Take CGI/LPF-only results from step 3
- Add any not already in results with `combined_score = 0.70`, `source = "legal_diversified"`
- **Why**: Ensures legal text representation even when hybrid search is dominated by BOFIP

### Step 11: Cross-Reference Injection
`_inject_cross_references()` scans the top 15 merged chunks + the original query for article references:

**CGI regex**:
```python
r'(?:articles?|art\.?)\s+([\d][\d\w\s\-]*?)[\s,]+(?:et\s+suivants\s+)?du\s+(?:CGI|code\s+g[Ã©e]n[Ã©e]ral)'
```

**LPF regex**:
```python
r'(?:articles?|art\.?)\s*([LRA]\*?\.?\s*[\d][\d\w\s\-]*?)[\s,]+(?:et\s+suivants\s+)?du\s+(?:LPF|livre\s+des\s+proc[Ã©e]dures)'
```

**Pre-processing**: Normalizes `\xa0` (non-breaking spaces) and `\n` before regex. Splits "articles 145 et 216" into separate refs.

**Fuzzy matching**: If exact lookup fails, tries:
1. Space normalization: `"L64"` â†’ `"L. 64"`
2. Suffix stripping: `"L. 80 B"` â†’ `"L. 80"` (handles bis/ter/A/B/C)

Injected with `combined_score = 0.90`, `source = "cross_reference"` (law > commentary).

### Step 12: Value-Aware Boosting
`_boost_value_chunks()` detects rate/amount queries and boosts relevant chunks:

**Rate detection keywords**: `taux, bareme, seuil, plafond, combien, montant, calcul, impot, droits, payer`

**Boost factors**:
- Chunk has `\d+\s*%` (percentage) or `\d+.*EUR` (amount): **1.15x**
- Chunk has bareme pattern (0%, 11%, 30%, 41%, 45%): **1.30x**

### Step 13: French Cross-Encoder Reranking
```python
if self.reranker and self.reranker.is_available():
    rerank_pool_size = min(self.rerank_pool_size, len(merged))
    rerank_pool = merged[:rerank_pool_size]
    reranked = self.reranker.rerank(query, rerank_pool, top_k=n_results)
    merged = reranked + remaining
```
- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual MS-Marco)
- Processes `(query, chunk_text)` pairs through cross-encoder
- Reranks top `RERANK_POOL_SIZE` candidates (default 30), returns top `n_results`
- Adds `rerank_score` to each chunk

### Final: Return top N results (default 30)

---

## 9. Generation: Chunks to Answer

### System Prompt (`src/generation/prompts.py`)

The full system prompt (Grille d'Analyse Fiscale v4):

```
Tu es un expert-comptable specialise en fiscalite francaise. Tu reponds aux questions
en te basant UNIQUEMENT sur les extraits du BOFIP fournis.

## METHODE D'ANALYSE EN 4 ETAPES (OBLIGATOIRE)

### ETAPE 1: QUI? (Identifier les acteurs)
- List all persons/entities in a table: Acteur | Role | Statut/Residence | Ce qu'il detient | Depuis quand

### ETAPE 2: QUOI? (Qualifier l'operation)
- Qualification table: Nature | Type de transaction | Regime fiscal | Statut contribuable | Situation specifique
- Search extracts for: rates (%), thresholds (EUR), conditions, exceptions

### ETAPE 3: COMBIEN? (Calculer - OBLIGATOIRE si montants ou %)
- MUST calculate when amounts or percentages are present
- Format: Regle applicable -> Taux/seuil -> Donnees -> Calcul -> Resultat

### ETAPE 4: CONCLUSION (Repondre clairement)
- Direct answer to the question
- If multiple actors: specify who is affected by what
- If doubt: say it explicitly

## REGLES IMPERATIVES
- UNIQUEMENT les extraits fournis
- TOUJOURS citer BOI + paragraphe (ex: BOI-RFPI-PVI-20-10 Â§40)
- Si info absente: "Cette information n'est pas dans les extraits fournis."
- CGI/LPF = LOI. En cas de conflit avec BOFIP, la LOI prevaut.
- Quand la regle est claire, TRANCHE (don't hedge).
```

The prompt includes 3 worked examples (TVA simple, BIC medium, Donation complex) and a response format template with Actor table, Operation, Analysis, Calculation, Response, and Sources sections.

### User Prompt Template

```
Question: {question}

Extraits de droit fiscal (BOFIP, CGI, LPF):
---
{context}
---

Reponds a la question en te basant UNIQUEMENT sur ces extraits.
```

### Chunk Formatting Template

```
[{boi_reference} Â§{paragraph_number}]
Section: {section_title}
{text}
Source: {source_url}
```

`format_context(chunks)` joins formatted chunks with `\n---\n` separators.

### Groq LLM Client (`src/generation/llm.py`)

- **Primary model**: `llama-3.3-70b-versatile` (131K context)
- **Fallback chain**: Llama 3.3 70B â†’ Llama 3.1 8B â†’ Llama 4 Scout 17B
- **Parameters**: `temperature=0.1`, `max_tokens=1500`
- **Rate limit handling**: Catches 429 errors, rotates to next model
- **Context budget control**: trims retrieved chunks before generation (`LLM_MAX_CONTEXT_CHUNKS`, `LLM_MAX_CONTEXT_TOKENS`)
- **Response caching**: MD5 hash of `question:context_hash` â†’ JSON file in `data/cache/llm_responses/`
- **Faithfulness guardrail**:
  - Verifier pass on generated answer using compact chunk context (`FAITHFULNESS_VERIFIER_MODEL`)
  - Heuristic fallback if verifier call fails (token/rate limit)
  - If unsupported/insufficient evidence: replace answer with explicit abstention message
- **`generate_with_sources()`** returns: `{answer, sources (top 4 deduplicated), chunks_used, disclaimer, faithfulness}`

---

## 10. User Interface

**File**: `app.py` (159 lines) â€” Streamlit chat interface

- **Session state**: `messages` (chat history), `retriever` (singleton HybridRetriever), `llm` (singleton LLM client)
- **Query flow**: `st.chat_input` â†’ `query_rag(prompt)` â†’ `search_simple(20 results)` â†’ `generate_with_sources()` â†’ display
- **Sidebar**: Project description, example questions, disclaimer, stats (chunk count)
- **`format_sources()`**: Deduplicates BOI references, creates markdown links

---

## 11. Configuration

### config.py Parameters

```python
# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PDF_DATA_DIR = RAW_DATA_DIR / "pdfs"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", str(DATA_DIR / "chroma_db")))

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")      # From .env file
GROQ_MODEL = "llama-3.3-70b-versatile"          # Primary model
LLM_MAX_CONTEXT_CHUNKS = 14
LLM_MAX_CONTEXT_TOKENS = 4500

# Embeddings
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

# Chunking
CHUNK_MIN_TOKENS = 200
CHUNK_MAX_TOKENS = 800
CHUNK_TARGET_TOKENS = 500

# Retrieval
RETRIEVAL_TOP_K = 5
HYBRID_ALPHA = 0.5   # Balance: 0=BM25 only, 1=Vector only
RERANK_POOL_SIZE = 30  # Cross-encoder candidate pool (env override supported)

# Generation faithfulness guardrail
FAITHFULNESS_GUARDRAIL_ENABLED = True
FAITHFULNESS_VERIFIER_MODEL = "llama-3.1-8b-instant"
FAITHFULNESS_MIN_CONFIDENCE = 0.55
FAITHFULNESS_MAX_CONTEXT_CHUNKS = 6

# Cache TTLs
CACHE_TTL_EMBEDDINGS = None     # Never expire
CACHE_TTL_LLM = 86400 * 7      # 7 days
CACHE_TTL_RETRIEVAL = 3600      # 1 hour
```

### Environment Variables

| Variable | Purpose | Source |
|----------|---------|--------|
| `GROQ_API_KEY` | Groq API authentication | `.env` file |
| `EMBEDDING_MODEL` | Override embedding model (for A/B benchmarking) | Environment variable |
| `CHROMA_DB_DIR` | Override Chroma persistence directory | Environment variable |

### Dependencies (`requirements.txt`)

```
sentence-transformers    # E5 embeddings + cross-encoder reranker
chromadb                 # Vector database
rank-bm25                # BM25 keyword search
groq                     # LLM API client
streamlit                # Web UI
pdfplumber               # PDF text extraction
beautifulsoup4, lxml     # HTML/XML parsing
python-dotenv            # .env loading
tqdm                     # Progress bars
diskcache                # Embedding cache
```

---

## 12. File Structure

```
bofip-rag/
â”œâ”€â”€ app.py                  (159 lines)  Streamlit UI (main entry)
â”œâ”€â”€ config.py               (84 lines)   Configuration & paths
â”œâ”€â”€ CLAUDE.md                             Claude Code instructions
â”œâ”€â”€ README.md                             User-facing quickstart (French)
â”œâ”€â”€ requirements.txt                      Python dependencies
â”œâ”€â”€ .env                                  GROQ_API_KEY (gitignored)
â”œâ”€â”€ AppLaunch.bat                         Windows launch script
â”œâ”€â”€ Shutdown.bat                          Windows shutdown script
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_pipeline/
â”‚   â”‚   â”œâ”€â”€ parser.py       (390 lines)  BOFIP XML/HTML parsing
â”‚   â”‚   â”œâ”€â”€ chunker.py      (255 lines)  BOFIPChunk dataclass + basic chunking
â”‚   â”‚   â”œâ”€â”€ semantic_chunker.py (485 lines)  Semantic chunking (1 rule = 1 chunk)
â”‚   â”‚   â”œâ”€â”€ pdf_parser.py   (386 lines)  CGI/LPF PDF parsing
â”‚   â”‚   â””â”€â”€ process.py      (236 lines)  BOFIP processing pipeline
â”‚   â”‚
â”‚   â”œâ”€â”€ retrieval/
â”‚   â”‚   â”œâ”€â”€ hybrid.py       (546 lines)  PRODUCTION: search_simple() + all helpers
â”‚   â”‚   â”œâ”€â”€ bm25.py         (263 lines)  BM25 index + French tokenizer
â”‚   â”‚   â”œâ”€â”€ embeddings.py   (165 lines)  E5 embedding model wrapper
â”‚   â”‚   â”œâ”€â”€ vector_store.py (301 lines)  ChromaDB vector store
â”‚   â”‚   â””â”€â”€ reranker.py     (135 lines)  French cross-encoder reranker
â”‚   â”‚
â”‚   â””â”€â”€ generation/
â”‚       â”œâ”€â”€ llm.py          (266 lines)  Groq LLM client + caching
â”‚       â””â”€â”€ prompts.py      (259 lines)  System prompt + templates
â”‚
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ reindex_semantic.py (306 lines)  Full reindexing pipeline
â”‚   â”œâ”€â”€ process_pdfs.py     (162 lines)  CGI/LPF PDF processing
â”‚   â”œâ”€â”€ process_legi_archive.py            LEGI archive ingestion (daily/full)
â”‚   â”œâ”€â”€ refresh_legi_daily.py              Automated daily legal refresh
â”‚   â”œâ”€â”€ sync_legal_chunks.py               Legal vector sync (full or delta)
â”‚   â”œâ”€â”€ evaluate_retrieval.py              Retrieval-only metrics
â”‚   â”œâ”€â”€ tune_reranker_pool.py              Reranker pool tuning (quality+latency)
â”‚   â”œâ”€â”€ benchmark_embeddings.py            Embedding model benchmark (A/B)
â”‚   â”œâ”€â”€ build_retrieval_dataset_from_cache.py  Build expanded retrieval ground truth
â”‚   â”œâ”€â”€ validate_retrieval_dataset.py      Validate silver questions and filter dataset
â”‚   â””â”€â”€ evaluate.py         (155 lines)  End-to-end keyword benchmark
â”‚
â”œâ”€â”€ data/                                 (gitignored)
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â”œâ”€â”€ bofip_extracted/BOFiP/...    BOFIP XML/HTML documents
â”‚   â”‚   â”œâ”€â”€ legi/                         LEGI archives (daily + full)
â”‚   â”‚   â””â”€â”€ pdfs/                         CGI/LPF PDF fallback files
â”‚   â”œâ”€â”€ processed/
â”‚   â”‚   â”œâ”€â”€ chunks.json     (~199 MB)     86,045 semantic chunks
â”‚   â”‚   â””â”€â”€ bm25_index.pkl  (239 MB)     BM25 inverted index
â”‚   â”œâ”€â”€ cache/
â”‚   â”‚   â”œâ”€â”€ embeddings/                   Model-scoped embedding cache (<model_slug>/*.npy)
â”‚   â”‚   â””â”€â”€ llm_responses/                LLM response cache (.json)
â”‚   â””â”€â”€ chroma_db/                        ChromaDB persistent storage
â”‚
â””â”€â”€ docs/
    â””â”€â”€ SYSTEM.md                         THIS FILE
```

---

## 13. Benchmarks and Test Results

### Progress

| Metric | Before KISS (Feb 5) | After KISS (Feb 6) | After Reranker (Feb 9) |
|--------|---------------------|--------------------|-----------------------|
| Regression 7Q | 43% (3/7) | 86% (6/7) | **100% (7/7)** |
| User 8Q | N/A | 88% (7/8) | **100% (8/8)** |
| Novel 10Q | N/A | 100% (10/10) | **100% (10/10)** |

### 7-Question Regression Benchmark

| Q | Topic | Status | Key Detail |
|---|-------|--------|------------|
| Q1 | Plus-value immobiliere | PASS | Abattement found |
| Q2 | TVA renovation | PASS | TVA rates + % found |
| Q3 | IR bareme | PASS | Reranker surfaces bareme at #1 |
| Q4 | CFE base | PASS | Valeur locative found |
| Q5 | BIC amortissement | PASS | Amortissement found |
| Q6 | SCI IR foncier | PASS | SCI + foncier found |
| Q7 | Micro-BNC seuil | PASS | micro-BNC + abattement found |

### 8-Question User Benchmark

| Q | Topic | Status | Key Detail |
|---|-------|--------|------------|
| Q1 | Seuil micro-BNC | PASS | 77,700 found |
| Q2 | Verification PME | PASS | 3 mois found |
| Q3 | Abus de droit | PASS | L. 64 + criteria found |
| Q4 | Mere-fille IS | PASS | CGI Art. 119 ter surfaced by reranker |
| Q5 | Rescrit fiscal | PASS | Opposable + bonne foi found |
| Q6 | Manquement delibere | PASS | 40% + majoration found |
| Q7 | ESFP duree | PASS | Un an found |
| Q8 | Quotient familial | PASS | 3 parts found |

### 10-Question Novel Benchmark (Anti-Overfitting)

All PASS: CIR, deficit foncier, TVA intracommunautaire, succession abattement, LMNP, delai de reprise, PEA, taxe habitation, IS taux, donation entre epoux.

### Retrieval-Only Benchmark (Gold set `scripts/test_questions.json`, 11Q)

Measured with `scripts/evaluate_retrieval.py`:

- **Without reranker**
  - Recall@5: 57.6%
  - Recall@10: 57.6%
  - Recall@20: 57.6%
  - Precision@5: 27.3%
- **With reranker (pool=30)**
  - Recall@5: 36.4%
  - Recall@10: 57.6%
  - Recall@20: 57.6%
  - Precision@5: 21.8%

Interpretation: this small 11Q set penalizes reranker at top-5 and is not sufficient for policy decisions alone.

### Retrieval-Only Benchmark (Expanded set `scripts/test_questions_expanded.json`, 65Q)

Expanded with `scripts/build_retrieval_dataset_from_cache.py`:
- 11 gold questions + 54 silver questions auto-extracted from real cached sessions
- Total: 65 questions

Measured with `scripts/evaluate_retrieval.py`:

- **Without reranker**
  - Recall@5: 53.1%
  - Recall@10: 60.0%
  - Recall@20: 71.5%
  - Precision@5: 24.0%
- **With reranker (pool=30)**
  - Recall@5: 63.1%
  - Recall@10: 71.0%
  - Recall@20: 79.0%
  - Precision@5: 30.5%

Interpretation: on the larger 65Q set, reranker is clearly better and remains default-enabled.

### Retrieval-Only Benchmark (Validated set `scripts/test_questions_validated.json`, 60Q)

Validated with `scripts/validate_retrieval_dataset.py` (gold + silver entries passing coverage/rank checks):

- **Without reranker**
  - Recall@5: 57.5%
  - Recall@10: 66.7%
  - Recall@20: 77.5%
  - Precision@5: 26.3%
- **With reranker (pool=30)**
  - Recall@5: 68.3%
  - Recall@10: 76.9%
  - Recall@20: 85.6%
  - Precision@5: 33.0%

### Reranker Pool Tuning (2026-02-10, `scripts/tune_reranker_pool.py`)

- Dataset: `scripts/test_questions_expanded.json` (65Q)
- Candidates tested: 20, 30 (+ no-reranker baseline)
- Best reranker candidate by script policy (Recall@5 first): **pool=30**
- Snapshot:
  - no-reranker: Recall@5 53.1%, Precision@5 24.0%, latency 0.80s
  - pool=20: Recall@5 61.5%, Precision@5 29.8%, latency 0.85s
  - pool=30: Recall@5 63.1%, Precision@5 30.5%, latency 0.93s

### Embedding Benchmark Status (2026-02-10, `scripts/benchmark_embeddings.py`)

- Baseline run (`intfloat/multilingual-e5-base`, validated set 60Q, reranker ON):
  - Recall@5: 67.5%
  - Recall@10: 76.9%
  - Recall@20: 85.6%
  - Precision@5: 32.3%
- `intfloat/multilingual-e5-large` smoke run completed on a 300-chunk pilot corpus (technical validation of indexing/eval flow).
- Full `e5-large` benchmark on 86k chunks is compute-heavy on CPU-only setup; run in staged mode or with stronger hardware.

### Performance

- **Average latency (retrieval-only benchmark)**:
  - no-reranker: ~0.8s/query (65Q set)
  - reranker pool 30: ~0.9s/query (65Q set)
- **Daily LEGI refresh runtime**:
  - before delta sync optimization: ~13m+
  - after delta sync optimization: ~48s (same-day delta case)

---

## 14. Architecture Principles and Lessons

### Principle 1: KISS (Keep It Simple, Stupid)

Start simple. Only add complexity with proven, measured improvement. We went from 43% to 86% by REMOVING complexity (HyDE, LLM filter, RRF, query decomposition).

### Principle 2: No Hardcoding / No Overfitting

**NEVER** map specific keywords to specific outputs. Solutions must work for ANY query, not just test queries. Use general patterns (regex for %, EUR) instead of keyword dictionaries.

### Principle 3: Fix Root Causes

Before tweaking prompts or adding post-processing, check: Is the data correct? Is the tokenizer breaking terms? Is the embedding model failing? Fix at the SOURCE, not the display layer.

### Key Lessons Learned

For the full list of **29 documented mistakes** with context, code examples, anti-patterns, and rules to avoid repeating them, see **[LESSONS.md](LESSONS.md)**.

---

## 15. Removed Components (Historical)

These components were tried and removed. They are documented here so nobody re-implements them.

| Component | Why Removed |
|-----------|-------------|
| HyDE (Hypothetical Doc Embeddings) | Generated wrong vocabulary, added latency, no accuracy gain |
| LLM Document Filter (two-stage retrieval) | Removed relevant documents, added latency |
| Query Decomposition | Scattered context across sub-queries, diluted results |
| Series Auto-Filter | Too aggressive, removed relevant series from results |
| English Reranker | Replaced by French multilingual model (mmarco-mMiniLMv2) |
| `search_document_level()` | Over-engineered 250+ line method replaced by search_simple() |
| Reciprocal Rank Fusion (RRF) | Replaced by simpler normalized score merge |

All code for these components has been deleted. Only `search_simple()` is the production method.

---

## 16. Backlog / Next Steps

1. **Embedding upgrade benchmark (P0)**: Baseline (`e5-base`) measured on validated set; run full `e5-large` benchmark when compute budget allows
2. **Embedding benchmark strategy (P0)**: Use staged benchmark (pilot subset -> full corpus) to avoid blocking iteration on CPU-only environments
3. **Chunk expansion experiment (P1)**: Add optional sibling paragraph expansion and keep only if metrics improve
4. **Faithfulness threshold tuning (P1)**: Tune confidence/context caps using real queries to reduce false abstentions
5. **Ground truth hardening (WAITLIST)**: Promote validated set toward fully reviewed gold 80-100 questions (batch campaign)
6. **GPU acceleration (P2)**: Keep reranker + embeddings on GPU for stable low latency
---

## 17. How to Run

### Prerequisites

- Python 3.11+
- Groq API key (free: https://console.groq.com/)
- ~2GB disk for data + ~11GB for venv with ML models

### Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
# Create .env with: GROQ_API_KEY=your_key_here
```

### Data Processing

```bash
# Process BOFIP documents (semantic chunking)
python scripts/reindex_semantic.py --types Commentaire

# Primary legal ingestion from LEGI archives (daily delta)
python scripts/process_legi_archive.py --archive latest --append

# Full legal refresh from latest full LEGI snapshot (larger download)
python scripts/process_legi_archive.py --archive latest-full --append

# Automated daily legal refresh (delta + BM25 rebuild + legal sync + state file)
python scripts/refresh_legi_daily.py

# Optional PDF fallback for CGI/LPF
python scripts/process_pdfs.py

# Fast legal refresh in vector DB (full legal sync)
python scripts/sync_legal_chunks.py

# Incremental legal vector sync from latest delta file
python scripts/sync_legal_chunks.py --delta-file data/processed/legi_chunks.json

# LEGI XML bootstrap ingestion (when XML files are available locally)
python scripts/process_legi_xml.py --append

# Test on sample (100 docs)
python scripts/reindex_semantic.py --sample 100
```

### Run the App

```bash
streamlit run app.py --server.port 8501
# Open: http://localhost:8501
```

### Test Retrieval

```bash
python -c "
from src.retrieval.hybrid import get_hybrid_retriever
r = get_hybrid_retriever()
results = r.search_simple('TVA restauration', n_results=10)
for i, res in enumerate(results[:5]):
    print(f\"{i+1}. {res['metadata']['boi_reference']}\")
"

# Retrieval-only quality metrics (no LLM)
python scripts/evaluate_retrieval.py --k 5 10 20 --no-reranker

# Build expanded retrieval dataset (gold + silver from cache)
python scripts/build_retrieval_dataset_from_cache.py

# Validate silver entries and output filtered dataset
python scripts/validate_retrieval_dataset.py --dataset scripts/test_questions_expanded.json --out scripts/test_questions_validated.json

# Evaluate on expanded dataset
python scripts/evaluate_retrieval.py --dataset scripts/test_questions_expanded.json --k 5 10 20

# Tune reranker pool size (quality + latency)
python scripts/tune_reranker_pool.py --dataset scripts/test_questions_expanded.json --pools 20 30 --k 5 10 20

# Benchmark embedding models (staged/full depending on compute budget)
python scripts/benchmark_embeddings.py --models intfloat/multilingual-e5-base intfloat/multilingual-e5-large --dataset scripts/test_questions_validated.json --k 5 10 20
```



