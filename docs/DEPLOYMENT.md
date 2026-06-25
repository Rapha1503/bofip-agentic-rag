# Deployment Notes

## Recommended Public Setup

Use two surfaces:

1. **GitHub Pages** for a static portfolio page: project story, architecture, screenshots, metrics, limitations, and a link to the live app.
2. **Hugging Face Spaces** or another Python app host for the Streamlit RAG runtime.

GitHub Pages should not host the RAG runtime because it serves static assets and does not run the Python retrieval stack.

## Full-Corpus Requirement

The hosted app is designed for the full 9,048-row BOFiP API snapshot used by the hosted Space.

Optimize the full-corpus runtime instead:

- prebuild JSONL and `.npy` artifacts;
- validate artifact counts and shapes at startup;
- use cached BM25 indexes;
- prefer memory-mapped embeddings where possible;
- keep the reranker off in the public UI until benchmarks justify the latency;
- report cold-start and query latency honestly;
- fail with clear diagnostics when artifacts are missing.

## BYOK Policy

The app supports user-provided API keys for OpenAI-compatible endpoints. Public copy should be explicit:

- the key is entered per session;
- the key is sent to the server running the Streamlit app for that request;
- the key is not committed, persisted, or logged by the app;
- on Hugging Face Spaces, provider API key fields are intentionally not prefilled from server environment variables;
- prompt and raw JSON debug views stay hidden on the public Space unless `BOFIP_SHOW_DEBUG=1` is explicitly set;
- users should use restricted or low-budget keys for demos.

Do not build a browser-only OpenAI-key workflow for this project. API keys in client-side JavaScript are not an acceptable security model for a public demo.

## Deployment Artifacts

The live app needs these runtime files:

```text
data/interim/raw_docs.jsonl
data/interim/chunks.jsonl
data/interim/doc_dense_cache.npy
data/interim/chunk_dense_cache.npy
data/models/intfloat--multilingual-e5-large/
```

Optional reranker artifact:

```text
data/models/BAAI--bge-reranker-v2-m3/
```

The full-corpus file contract is versioned in [full_corpus_manifest.json](full_corpus_manifest.json).

The repository intentionally excludes those large artifacts. A deployment should either:

- download them during startup from the `full-corpus-v2` GitHub release; or
- mount them into the host environment; or
- use an external model/cache volume if the host supports it.

The current default artifact URL is:

```text
https://github.com/Rapha1503/bofip-agentic-rag/releases/download/full-corpus-v2
```

The Docker runtime uses `BOFIP_AUTO_DOWNLOAD_ARTIFACTS=1`.

## Deployment Status

The static portfolio page is served by GitHub Pages and the full-corpus BYOK runtime is prepared for Hugging Face Spaces.

Use [DEMO.md](DEMO.md) for local full-corpus testing. The setup checker is:

```powershell
python scripts/check_setup.py --deep
```
