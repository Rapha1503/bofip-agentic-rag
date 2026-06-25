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

The runtime expects the 9,048-row BOFiP API snapshot used by the hosted Space.

Recommended path:

```powershell
python scripts/download_artifacts.py
```

Required local files:

```text
data/interim/raw_docs.jsonl
data/interim/chunks.jsonl
data/interim/doc_dense_cache.npy
data/interim/chunk_dense_cache.npy
data/models/intfloat--multilingual-e5-large/
```

Optional quality layer:

```text
data/models/BAAI--bge-reranker-v2-m3/
```

The optional reranker is not exposed in the public Streamlit UI. Retrieval still covers the full 9,048-row corpus without it.

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

The public artifact bundle is published as release assets under `full-corpus-v2`.

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
- explicit status: `supported` or `insufficient_evidence`.

## 6. Run Local Checks

```powershell
$env:PYTHONPATH='src'
python -m compileall app.py scripts src
python scripts/check_setup.py --deep --skip-models
```

## Notes

- This is a research prototype, not tax advice.
- BYOK means the API key is sent to the local or hosted server handling the request.
- The current artifact bundle is generated from the 2026-06-23 BOFiP API snapshot; official BOFiP may be newer.
