# Deployment Notes

## Recommended Public Setup

Use two surfaces:

1. **GitHub Pages** for a static portfolio page: project story, architecture, screenshots, metrics, limitations, and a link to the live app.
2. **Hugging Face Spaces** or another Python app host for the Streamlit RAG runtime.

GitHub Pages should not host the RAG runtime because it serves static assets and does not run the Python retrieval stack.

## Full-Corpus Requirement

The demo must preserve the full 5,666-document commentary corpus. A reduced corpus makes the app unreliable: a user can ask about a BOFiP topic that was removed, and the system would fail for an artificial reason.

Optimize the full-corpus runtime instead:

- prebuild JSONL and `.npy` artifacts;
- validate artifact counts and shapes at startup;
- use cached BM25 indexes;
- prefer memory-mapped embeddings where possible;
- expose reranker mode as configurable;
- report cold-start and query latency honestly;
- fail with clear diagnostics when artifacts are missing.

## BYOK Policy

The app supports user-provided API keys for OpenAI-compatible endpoints. Public copy should be explicit:

- the key is entered per session;
- the key is sent to the server running the Streamlit app for that request;
- the key is not committed, persisted, or logged by the app;
- users should use restricted or low-budget keys for demos.

Do not build a browser-only OpenAI-key workflow for this project. API keys in client-side JavaScript are not an acceptable security model for a public demo.

## Deployment Artifacts

The live app needs these runtime files:

```text
data/interim/raw_docs_sample_5666.jsonl
data/interim/chunks_section_window_sample_5666.jsonl
data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy
data/interim/chunk_dense_cache_5666_full_e5large.npy
data/models/intfloat--multilingual-e5-large/
```

Optional reranker artifact:

```text
data/models/BAAI--bge-reranker-v2-m3/
```

The full-corpus file contract is versioned in [full_corpus_manifest.json](full_corpus_manifest.json).

The repository intentionally excludes those large artifacts. A deployment should either:

- download them during build/startup from a controlled artifact store; or
- mount them into the host environment; or
- use an external model/cache volume if the host supports it.

## Phase 1 Deployment Status

Phase 1 prepares the repository for portfolio review. It does not start a hosted app.

Use [DEMO.md](DEMO.md) for local full-corpus testing. The setup checker is:

```powershell
python scripts/check_setup.py --deep
```

Before deploying:

- publish or mount the large artifacts referenced by the manifest;
- split `app.py` into UI, retrieval, LLM, config, and observability boundaries;
- disable raw query/prompt logging by default;
- add a visible tax-advice disclaimer;
- add screenshots and evaluation report for the static portfolio page.
