# Roadmap

## Phase 1 - Public Portfolio Cleanup

Status: complete.

Goal: make the repository readable, installable, and safe to show.

- Clean Git hygiene and ignored artifacts.
- Add install metadata and `.env.example`.
- Remove unstable tracked profiler code.
- Align README and architecture docs with the actual runtime.
- Add data and deployment documentation.
- Keep ownership and project presentation under Rapha1503.
- Add setup checker, CI workflow, local demo guide, artifact manifest, authorship metadata, and references.

## Phase 2 - Reproducibility and Runtime Contract

Goal: make a fresh clone fail clearly unless the correct artifacts are present.

- Add checksum validation to the corpus manifest workflow.
- Add small fixture tests that exercise the preflight validator.
- Move query rewrite, facet expansion, computation-aware facets, and merging into a shared orchestrator.
- Make CLI, Streamlit, and evaluation use the same retrieval contract.
- Add small fixture tests that do not require the full BOFiP corpus.

## Phase 3 - Data Quality and Legal Traceability

Goal: improve credibility for tax-domain RAG.

- Key documents by stable `document_id` and keep BOI reference as metadata.
- Handle duplicate BOI references explicitly.
- Add table-first chunks with row/header metadata.
- Carry official source URLs, dates, section paths, paragraph numbers, and table IDs into UI citations.
- Add a visible disclaimer: research prototype, not tax advice.

## Phase 4 - Evaluation Upgrade

Goal: move from retrieval-only credibility to answer-level credibility.

- Complete passage gold coverage.
- Publish tracked evaluation reports.
- Add answer-level grading for:
  - citation validity;
  - citation support;
  - unsupported-question abstention;
  - calculation correctness;
  - table-heavy tax cases;
  - freshness-sensitive questions.
- Document failure cases, not only wins.

## Phase 5 - Full-Corpus Public Demo

Goal: deploy a useful demo without sacrificing coverage.

- Keep the full 5,666-document commentary corpus.
- Optimize cold start with prebuilt artifacts and caches.
- Add optional reranker mode for latency control.
- Host the app on a Python-compatible free tier.
- Use GitHub Pages only as the static portfolio surface.
- Keep BYOK handling explicit and non-persistent.
