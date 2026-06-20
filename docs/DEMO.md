# Local Demo Guide

This guide gets a local full-corpus demo running after cloning the repository.

## 1. Install

```powershell
git clone https://github.com/Rapha1503/bofip-agentic-rag.git
cd bofip-agentic-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.local
```

Add one API key to `.env.local`, for example:

```text
DEEPSEEK_API_KEY=...
```

The Streamlit app exposes a model dropdown for each provider. Use a model available on the selected provider account.

## 2. Place Full-Corpus Artifacts

The runtime expects the 5,666-document BOFiP commentary corpus observed through `2026-01-28`.

Recommended path:

```powershell
python scripts/download_artifacts.py
```

Required local files:

```text
data/interim/raw_docs_sample_5666.jsonl
data/interim/chunks_section_window_sample_5666.jsonl
data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy
data/interim/chunk_dense_cache_5666_full_e5large.npy
data/models/intfloat--multilingual-e5-large/
```

Optional quality layer:

```text
data/models/BAAI--bge-reranker-v2-m3/
```

The optional reranker is not exposed in the public Streamlit UI. Retrieval still covers the full 5,666-document corpus without it.

## 3. Check Setup

Fast existence check:

```powershell
python scripts/check_setup.py
```

Deep validation with row counts and embedding shapes:

```powershell
python scripts/check_setup.py --deep
```

The script exits non-zero when required artifacts are missing and prints exactly what to add.

The versioned artifact contract is tracked in [full_corpus_manifest.json](full_corpus_manifest.json).

The public artifact bundle is published as release assets under `full-corpus-v1`.

## 4. Run the App

```powershell
$env:PYTHONPATH='src'
streamlit run app.py
```

The first startup is the slow path because indexes and models are loaded. Later Streamlit reruns reuse cached resources.

## 5. Try a Query

Example:

```text
Quel taux de TVA pour la pose d'une pompe a chaleur chez un particulier ?
```

Expected behavior:

- retrieval trace available in the technical expander;
- cited BOFiP chunks;
- JSON-backed answer rendered as a fiscal reasoning summary;
- explicit status: `supported`, `partial`, or `insufficient_evidence`.

## 6. Run CLI Smoke Checks

```powershell
$env:PYTHONPATH='src'
python scripts/evaluate.py --runtime rag --device cpu --limit 5
python scripts/preview_answer.py --query "Quel taux de TVA pour une pompe a chaleur ?"
```

## Notes

- This is a research prototype, not tax advice.
- BYOK means the API key is sent to the local or hosted server handling the request.
- The current corpus observed during audit is fresh through `2026-01-28`; official BOFiP may be newer.
