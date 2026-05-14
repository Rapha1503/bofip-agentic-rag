# BOFIP Cleanroom Status

Last updated: 2026-05-13

## Latest Update

### Phase 10 — RagRuntime + Reranker + DeepSeek + Coverage Checker

**Architecture delivered:**
```
query → LLM rewriting (DeepSeek) → hybrid document retrieval (BM25 + dense + dense-anchor) 
→ local chunk BM25 → cross-encoder reranker (bge-reranker-v2-m3) 
→ coverage-aware LLM answer (DeepSeek)
→ {answer_status: supported|partial|insufficient_evidence, axes_requis/couverts/manquants}
```

**New components:**
- `src/bofip_cleanroom/rag_runtime.py` — Clean runtime (no dead branches). Dense-anchor filter prevents lexical BM25 from flooding results with international convention documents. Source weights configurable.
- `src/bofip_cleanroom/reranker.py` — CrossEncoderReranker wrapping `BAAI/bge-reranker-v2-m3`. GPU-accelerated, section-path-aware chunk scoring.
- `scripts/preview_answer.py` — Self-contained production script. LLM query rewriting (expands acronyms, formalizes French), batch mode with incremental save, resume from partial reports.
- `scripts/evaluate.py` — Standardized retrieval eval CLI using `eval_harness.py`. Supports both `phase8b` and `rag` runtimes.
- `src/bofip_cleanroom/eval_harness.py` — `EvalMetrics`, `QueryGold`, `evaluate()`. doc@1/3/5, passage@1/3/5, MRR, NDCG.

**Test artifacts:**
- `data/interim/eval_queries_v1.jsonl` — 50 diverse queries in 5 categories (direct, paraphrase, cross-document, edge-cases, unsupported)
- `data/interim/passage_gold_v3.jsonl` — Gold passage annotations with real chunk IDs from the 5666-doc corpus
- `data/interim/batch_final.jsonl` — 15-query benchmark input

**Provider change**: Gemini removed. DeepSeek (`deepseek-chat`) is the sole LLM provider. API key stored in `.env.local` as `DEEPSEEK_API_KEY`.

**Coverage checker**: Enhanced prompt requires the LLM to decompose the question into fiscal axes (`axes_requis`), check which axes the evidence covers (`axes_couverts`), identify gaps (`axes_manquants`), and produce a status of `supported`, `partial`, or `insufficient_evidence`. Prevents "confidently wrong" answers like the dictionnaire/Martinique TVA case.

**15-query benchmark results** (`data/reports/batch_final_v1.json`):
- Total: 15 realistic accountant queries across BOFIP domains
- 12 correct · 2 partial · 1 honest · 0 wrong (14/15 format-valid)
- The "dictionnaire en ligne" case still fails — fundamental terminology gap (LLM doesn't know "dictionnaire" = "livre numérique" in BOFIP)

**102 tests** passing (up from 81).

**11-query failure pattern analysis** (diagnostic probe across 7 BOFIP domains):
- Pattern A (convention spam): Fixed via dense-anchor filter
- Pattern B (edge-case sub-documents outranking general docs): Fixed via wider doc net (5→8), section-aware reranker
- Pattern C (terminology gap): Fixed via LLM query rewriting (improved 3/4 failing cases). The "dictionnaire → livres numériques" gap requires domain-specific knowledge beyond generic LLM rewriting.

**Upstream audit**: All 4 failing documents verified — chunking is clean, content is intact, dense embeddings are generated. The failures are retrieval-level, not upstream.

**Hardcoding audit**: 32 issues found, 8 fixed. Corpus paths extracted to `CORPUS_PATHS`, fusion weights and rank constant exposed as parameters.

---

## Historical (pre-Phase 10)

### Phase 9 shadow hardening: compact retry, provider-aware retry delay, and clearer failure typing

Implemented:
- compact retry mode in `src/bofip_cleanroom/llm_preview.py` for truncated JSON outputs
- provider-hint retry delay parsing for Gemini/OpenAI-compatible rate-limit responses
- response metadata and attempt counting in preview/batch reports
- explicit failure typing in review outputs:
  - `valid`
  - `format_invalid`
  - `provider_rate_limit`
  - `provider_timeout`
  - `provider_internal`
  - `missing_api_key`
  - `runtime_error`
- better batch error rows in `scripts/phase9_batch_preview_eval.py` so retrieval context is preserved when the LLM call fails

Verified:
- `81` local tests now pass
- `data/reports/phase9_batch_review_gemini_v9_small.json`
  - `3/3` format-valid on the live gate `sh001`, `sh006`, `sh007`
- `data/reports/phase9_batch_review_gemini_v12_varied5.json`
  - `case_count = 5`
  - `format_valid_count = 3`
  - `format_invalid_count = 0`
  - `provider_rate_limit_count = 2`
  - pending live cases: `sh003`, `sh004`

Interpretation:
- the structured JSON contract is now solid enough for the small live gate
- the current blocker on the varied batch is external Gemini quota, not local format drift
- retrieval remains unchanged in this lot

### Pre-LLM verification + Phase 9 shadow hardening

Implemented:
- `src/bofip_cleanroom/pre_llm_verification.py`
- `scripts/verify_pre_llm_stack.py`
- `scripts/phase9_review_batch_preview.py`
- structured JSON answer contract + local validator in `src/bofip_cleanroom/llm_preview.py`
- notebook builder hardening in:
  - `scripts/build_llm_preview_notebook.py`
  - `scripts/build_phase9_shadow_notebook.py`

Verified:
- `data/reports/pre_llm_verification_summary.json`
  - overall status: `pass`
  - commentary corpus: `5666` docs
  - no empty chunks
  - no duplicate `chunk_id`
  - phase6 official replay reproduces the promoted retrieval baseline
- `data/reports/phase9_batch_review_gemini_v1.json`
  - strict local validator marks `2/8` cached Gemini answers as fully format-valid

Live shadow probe:
- `data/reports/phase9_batch_preview_eval_gemini_v5_sh001.json`
- `data/reports/phase9_batch_review_gemini_v5_sh001.json`
- result:
  - `sh001` validates cleanly under the new structured contract

Transport / quota note:
- a fresh `3`-case live rerun on `sh001`, `sh006`, `sh007` reached Gemini free-tier daily quota
- the batch path now preserves those transient API failures in the report instead of crashing the whole run
- latest quota-limited artifacts:
  - `data/reports/phase9_batch_preview_eval_gemini_v6_small.json`
  - `data/reports/phase9_batch_review_gemini_v6_small.json`

### Phase 9 preview

Implemented:
- `src/bofip_cleanroom/env_utils.py`
- `src/bofip_cleanroom/preview_runtime.py`
- `src/bofip_cleanroom/llm_preview.py`
- `scripts/phase9_preview_answer.py`
- `scripts/phase9_batch_preview_eval.py`
- `scripts/build_llm_preview_notebook.py`
- `scripts/build_phase9_shadow_notebook.py`
- `notebooks/09_llm_preview_workbench.ipynb`
- `notebooks/10_gemini_shadow_eval_workbench.ipynb`

Purpose:
- open an isolated end-to-end preview above the strongest retrieval branch without promoting it as product baseline
- keep retrieval baseline and preview runtime explicitly separated
- require citation-only answers from retrieved BOFiP excerpts
- allow `.env`-based `OPENAI_API_KEY` loading without adding a new dependency
- allow `.env.local`-based `GEMINI_API_KEY` loading for Gemini preview runs

Current preview assumptions:
- corpus: `commentary`
- stage 1: phase `8b` experimental branch
- stage 2: direct local chunk retrieval with `chunk_mode=full`
- default provider for preview experiments: `gemini`
- first selected model: `gemini-2.5-flash`

Smoke run:
- `data/reports/phase9_preview_answer_smoke.json`
- retrieval and prompt construction succeed
- no API call is made when `OPENAI_API_KEY` is absent

Gemini smoke run:
- `data/reports/phase9_preview_answer_gemini_smoke.json`
- real API call succeeds with `gemini-2.5-flash`
- retrieval + prompt + answer generation work end-to-end

Gemini shadow batch:
- `data/reports/phase9_batch_preview_eval_gemini_v1.json`
- `8` cases executed
- `notebooks/10_gemini_shadow_eval_workbench.executed.ipynb` now reads the cached batch report by default
- `data/reports/phase9_batch_review_gemini_v1.json` summarizes output-format compliance

Observed:
- unsupported handling is conservative and acceptable on the tested out-of-scope case
- false-premise correction works on the JEI/CIR negation case
- answer structure compliance is still inconsistent across the `8` cases
- several answers omit `Limites` or citations even when the prompt asks for them
- this means the preview is useful for shadow evaluation, but prompt/schema control is not yet strong enough for promotion

Status:
- preview tooling is ready
- no LLM promotion decision is made yet
- this remains a shadow evaluation track until retrieval Gate A is passed

Latest promoted retrieval baseline remains phase 6.

Latest promoted stage-1 report:
- `data/reports/phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__cfg_058c08ab80ff.json`

Stage-1 exact document retrieval on `v4`:
- `hit@1 = 0.6000`
- `hit@3 = 0.7286`
- `hit@5 = 0.8143`

Latest family-guided stage-2 evaluation:
- `data/reports/phase4_family_guided_eval_phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__cfg_058c08ab80ff__ftop2__cfg.json`

Safe promoted family strategy:
- keep stage-1 top1 document fixed
- expand families from `top2` anchors
- rerank siblings locally with `sections_leads_stem`
- use that ordering only to enrich downstream doc/chunk candidates

Measured impact of that safe family strategy:
- exact document `hit@1` preserved at `0.6000`
- exact document `hit@3` improved to `0.8000`
- exact document `hit@5` improved to `0.8714`
- chunk-document `hit@5` reached `0.8000`

Rejected variant:
- global family rerank replacing stage-1 top1
- reason: it improved some sibling confusions, but degraded too many already-correct exact top1 cases

New experiments on hard families:
- added source-derived lexical view `title_tail` from the last discriminative title segments
- added optional family-local `tail_weight` to combine `sections_leads` with `title_tail` inside the BOFiP family rerank

Conclusions:
- `title_tail` as a new global stage-1 view is promising but **not promoted yet**
  - it improves `v3` and `v5`
  - but regresses `v4` exact top1 on a small number of queries
- family-local `tail_weight = 0.25` is the first useful stage-2 refinement
  - it keeps exact document `top1` unchanged
  - it improves chunk selection on the hard benchmark `v5`
  - it does not regress `v3`
  - it slightly trades one `v4` family `@3` document case for better `chunk@3`

## Objective

Rebuild a BOFIP-only RAG pipeline from raw sources in a clean-room project, without carrying over legacy code or hidden behavior from the original repository.

## What Exists Now

Implemented modules:
- `settings.py`
- `discovery.py`
- `xml_parser.py`
- `html_parser.py`
- `document_builder.py`
- `sampling.py`
- `chunking.py`
- `lexical_retrieval.py`
- `two_stage_retrieval.py`
- `jsonio.py`
- `versioning.py`
- `models.py`

Implemented scripts:
- `scripts/phase0_inventory.py`
- `scripts/phase0_multiset_stress_audit.py`
- `scripts/phase1_extract_raw_documents.py`
- `scripts/phase15_fidelity_report.py`
- `scripts/phase2_build_chunks.py`
- `scripts/phase3_lexical_probe.py`
- `scripts/phase3_batch_lexical_eval.py`
- `scripts/phase3_doc_lexical_eval.py`
- `scripts/phase3_dense_eval.py`
- `scripts/phase3_hybrid_eval.py`
- `scripts/phase3_report_fusion.py`
- `scripts/phase3_failure_analysis.py`
- `scripts/phase3_two_stage_probe.py`
- `scripts/phase3_chunk_order_audit.py`
- `scripts/phase3_deep_dive_probe.py`
- `scripts/phase3_abstention_audit.py`
- `scripts/phase3_local_strategy_audit.py`
- `scripts/build_retrieval_query_set_v2.py`
- `scripts/build_retrieval_query_set_v3.py`
- `scripts/build_real_examples_notebook.py`
- `scripts/build_two_stage_audit_notebook.py`
- `scripts/build_abstention_deepdive_notebook.py`
- `scripts/build_stage_comparison_notebook.py`
- `scripts/build_obsidian_vault.py`

Implemented tests:
- `tests/test_xml_parser.py`
- `tests/test_html_parser.py`
- `tests/test_chunking.py`
- `tests/test_lexical_retrieval.py`
- `tests/test_dense_retrieval.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_two_stage_retrieval.py`

## Executed So Far

### Phase 0

Outputs:
- `data/reports/raw_inventory.json`
- `data/reports/manifest.json`

Observed:
- `6295` content documents
- `141` attachment XMLs
- dominant content type: `Commentaire`
- sample HTML structural estimates generated on `25` docs
- multi-set stress audit added and executed on `75` parsed docs across `5` sets

### Phase 0.b Multi-set stress audit

Output:
- `data/reports/phase0_multiset_stress_audit.json`
- `notebooks/02_multiset_stress_audit.ipynb`

Observed:
- commentary-only sets are structurally much healthier than mixed non-commentary sets
- real commentary docs already reach section depth `6`, so the hierarchy cannot be treated as shallow
- many non-commentary docs are `synthetic_root_only`
- `Cartographie` remains the clearest out-of-scope text retrieval candidate

### Phase 0.c Full parse audit

Output:
- `data/reports/phase0_full_parse_audit.json`

Observed:
- all `6295` BOFIP content documents parse without crashing
- `failure_count = 0`
- `docs_with_zero_paragraphs_count = 52`
- `docs_with_zero_sections_count = 50`
- `docs_with_synthetic_root_only_count = 2746`
- parser stability is now strong enough to treat scope/quality as the main issue, not parser crashes

### Phase 1

Outputs:
- `data/interim/raw_docs_sample_10.jsonl`
- `data/interim/doc_tree_sample_10.jsonl`
- `data/reports/phase1_extract_summary_sample_10.json`
- `data/reports/manifest_phase1_sample_10.json`
- local raw copy in `data/raw/sample_10/`

Observed:
- the sample intentionally spans multiple content types
- commentary docs can expose rich structure
- some non-commentary docs are sparse or non-textual

### Phase 1.5

Output:
- latest fidelity report under `data/reports/phase15_fidelity_report_*.md`

Observed:
- report is useful for manual side-by-side inspection
- `Cartographie` docs currently surface as empty text with the current HTML parser

### Phase 2

Outputs:
- `chunks_paragraph_preserving_sample_10.jsonl`
- `chunks_section_window_sample_10.jsonl`
- `chunks_parent_child_sample_10.jsonl`
- matching JSON summaries in `data/reports/`

Current stats on the `10`-doc sample:
- `paragraph_preserving`: `211` chunks, `too_long_count=0`
- `section_window`: `52` chunks, `too_long_count=0`
- `parent_child`: `263` chunks, `too_long_count=0`

Large random stress tests executed:
- `data/reports/phase2_random_chunk_stress_n200_seed11_filtered.json`
- `data/reports/phase2_random_chunk_stress_n200_seed29.json`
- `data/reports/phase2_random_chunk_stress_n500_seed101_filtered.json`
- `data/reports/phase2_random_chunk_stress_n500_seed73.json`
- `data/reports/phase2_random_chunk_stress_n1000_seed131_filtered.json`
- `data/reports/phase2_random_chunk_stress_n1000_seed137.json`

Current stress-test conclusion:
- `section_window` is the only strategy that stays clean on large random pools
- on `1000` random commentary docs: `too_long_count=0`, `very_short_count_le_5=0`
- on `1000` random mixed docs: `too_long_count=0`, `very_short_count_le_5=0`
- `paragraph_preserving` and `parent_child` still leave too many tiny chunks on large pools, even after cleanup
- current clean-room chunking baseline candidate is therefore `section_window`

### Phase 3A

Status:
- lexical-only probe executed manually on multiple query patterns
- reproducible batch lexical evaluation implemented
- no dense retrieval yet

Observed:
- lexical retrieval is now coherent on the sampled commentary/form/letter/annexe cases
- section-level and parent-child strategies both look usable on the sample
- first smoke batch (`6` queries, supported text docs only) reaches `hit@1 = 1.0` on all three strategies
- this validates the clean-room lexical baseline, but does not yet separate the strategies strongly enough

Expanded execution:
- `50` commentary-only docs extracted with local raw copy
- `section_window` chunks built on this subset: `758` chunks, `too_long_count=0`, `empty_text_count=0`
- lexical benchmark `v1` on `20` supported queries over the `50`-doc commentary subset: `hit@1 = 1.0`
- lexical benchmark `v2` on `25` queries (`20` supported + `5` unsupported/false-premise diagnostics):
- supported queries: `hit@1 = 1.0`
- unsupported queries are now logged explicitly, but not counted as successful retrieval
- current lexical baseline does not abstain; it still returns best-effort matches for unsupported prompts

Dense and hybrid execution:
- on `50` commentary docs:
  - dense `e5-base`: `hit@1 = 0.95`, `hit@3 = 1.0`, `hit@5 = 1.0`
  - hybrid unweighted: `hit@1 = 1.0`, `hit@3 = 1.0`, `hit@5 = 1.0`
- on `200` commentary docs:
  - lexical: `hit@1 = 0.90`, `hit@3 = 1.0`, `hit@5 = 1.0`
  - dense `e5-base`: `hit@1 = 0.90`, `hit@3 = 0.90`, `hit@5 = 1.0`
  - hybrid unweighted: `hit@1 = 0.90`, `hit@3 = 1.0`, `hit@5 = 1.0`
  - hybrid weighted (`lexical=2`, `dense=1`): `hit@1 = 0.95`, `hit@3 = 1.0`, `hit@5 = 1.0`
- on `1000` commentary docs:
  - lexical: `hit@1 = 0.85`, `hit@3 = 1.0`, `hit@5 = 1.0`
  - dense `e5-base`: `hit@1 = 0.65`, `hit@3 = 0.85`, `hit@5 = 0.90`
  - hybrid weighted (`lexical=2`, `dense=1`): `hit@1 = 0.85`, `hit@3 = 0.95`, `hit@5 = 0.95`

Methodology correction:
- lexical and dense evaluation are now measured at the **document level**, not on raw chunk lists with duplicate BOI references
- lexical/dense reports now carry the best supporting chunk for each returned BOI document

Broader `1000`-doc benchmark:
- new query set: `data/interim/retrieval_queries_sample_1000_v1.jsonl`
- size: `55` queries total
  - `45` supported queries with an expected BOI present in the `1000`-doc subset
  - `10` unsupported / false-premise diagnostics
- results on the broader benchmark:
  - lexical: `hit@1 = 0.7333`, `hit@3 = 0.9778`, `hit@5 = 1.0`
  - dense `e5-base`: `hit@1 = 0.7111`, `hit@3 = 0.8889`, `hit@5 = 0.9333`
  - hybrid weighted (`lexical=2`, `dense=1`): `hit@1 = 0.7556`, `hit@3 = 0.9778`, `hit@5 = 1.0`
- conclusion:
  - the previous `20`-query set was too easy
  - on a harder and broader benchmark, the pipeline is still good at retrieving the correct document family in the top 3, but not yet strong enough at top-1 precision

Expanded answerable benchmark `v3`:
- query file:
  - `data/interim/retrieval_queries_sample_1000_v3.jsonl`
- size:
  - `90` queries total
  - `60` supported exact-target BOI queries
  - `30` unsupported / false-premise / broader audit cases
- purpose:
  - keep pressure on objective `#1` with more answerable near-neighbor BOI cases
  - avoid improving retrieval only on unsupported-query additions

Failure analysis:
- report: `data/reports/phase3_failure_analysis_sample_1000.json`
- lexical misses on the broader benchmark: `12`
  - `5` parent/child family confusions
  - `2` same-family neighbor confusions
  - `2` same-domain neighbor confusions
  - `3` true top-1 misses
- hybrid weighted misses on the broader benchmark: `11`
  - `5` parent/child family confusions
  - `1` same-family neighbor confusion
  - `3` same-domain neighbor confusions
  - `2` true top-1 misses
- most remaining failures are therefore not total misses; they are ranking errors among very close BOFIP relatives

Document-level structural retrieval:
- new report:
  - `data/reports/phase3_doc_lexical_eval_raw_docs_sample_1000.json`
- document lexical index uses BOI reference + title + structural metadata from raw documents
- results on the broader `1000`-doc benchmark:
  - document lexical: `hit@1 = 0.9333`, `hit@3 = 0.9778`, `hit@5 = 0.9778`
- failure analysis on document lexical:
  - `3` misses total
  - `1` parent/child family confusion
  - `1` same-family neighbor confusion
  - `1` title-equivalent/version confusion
- conclusion:
  - the strict `>= 0.80` gate is now reached at the document-retrieval stage
  - the main weakness was flat chunk-level ranking, not BOFIP structural discovery

Document retrieval mode comparison on `v3`:
- reports:
  - `data/reports/phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v3.json`
  - `data/reports/phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v3__sections.json`
  - `data/reports/phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v3__sections_firstpara.json`
- results:
  - `base`: `hit@1 = 0.9333`, `hit@3 = 0.9833`, `hit@5 = 0.9833`
  - `sections`: `hit@1 = 0.9500`, `hit@3 = 0.9833`, `hit@5 = 0.9833`
  - `sections_firstpara`: `hit@1 = 0.9500`, `hit@3 = 0.9833`, `hit@5 = 1.0000`
- current best stage-1 document mode:
  - `sections_firstpara`
- current remaining strict top-1 misses on `v3` under `sections_firstpara`:
  - `q01`: true top-1 miss
  - `q30`: same-family neighbor
  - `q32`: parent/child family confusion
- interpretation:
  - stage 1 is now strong and concentrated
  - one real miss remains; the others are close-family ranking errors

Offline fusion experiments:
- `chunk lexical + dense + document lexical` did not beat pure document lexical
- `chunk lexical + document lexical` at `1:1` reaches `hit@1 = 0.8000`
- `chunk lexical + document lexical` at `1:2` reaches `hit@1 = 0.9333`
- conclusion:
  - flat fusion is not the right architecture here
  - document retrieval should become stage 1, not just another peer signal

Two-stage retrieval baseline:
- implemented in `src/bofip_cleanroom/two_stage_retrieval.py`
- probe report:
  - `data/reports/phase3_two_stage_probe_sample_1000_full.json`
  - `data/reports/phase3_two_stage_probe_sample_1000_body.json`
- architecture:
  - stage 1: document retrieval with the structural document lexical index
  - stage 2: lexical chunk retrieval restricted to the top-ranked documents
- current stage-2 default:
  - `body` mode, which uses chunk body text for local passage selection instead of repeating the full BOI title/path
- conclusion:
  - document selection is now strong enough to move past the previous gate
  - the next problem is evidence-chunk selection inside the chosen document

Two-stage retrieval is now configurable:
- stage 1 document mode:
  - `base`
  - `sections`
  - `sections_firstpara`
- local stage 2 strategy:
  - `chunk`
  - `section_then_chunk`
- local chunk ranking now reranks within the selected document/section by:
  - query token overlap
  - then BM25 score
- this keeps the local search bounded and passage-centric, instead of trusting raw BM25 ordering inside a very small local pool

Broader audit set `v2`:
- query file:
  - `data/interim/retrieval_queries_sample_1000_v2.jsonl`
- size:
  - `75` queries total
  - `57` answerable (`45` supported exact-target queries + `12` false-premise answerable diagnostics)
  - `18` abstention candidates
- purpose:
  - keep the strict supported retrieval benchmark unchanged
  - broaden unsupported / false-premise auditing before any LLM generation

Document retrieval on `v2`:
- report:
  - `data/reports/phase3_doc_lexical_eval_raw_docs_sample_1000__retrieval_queries_sample_1000_v2.json`
- supported-query metrics remain:
  - `hit@1 = 0.9333`
  - `hit@3 = 0.9778`
  - `hit@5 = 0.9778`
- conclusion:
  - objective `#1` remains the strongest part of the current pipeline
  - the main document-level misses are still the same close-neighbor family confusions, not broad discovery failures

Bounded third-stage deep dive:
- report:
  - `data/reports/phase3_deep_dive_probe_retrieval_queries_sample_1000_v2_body.json`
- behavior:
  - stage 1: retrieve top BOI documents
  - stage 2: select top local chunks with body-focused ranking
  - stage 3: expand locally around top chunks with a bounded neighbor window
- conclusion:
  - this is the right shape for evidence collection
  - it is bounded and inspectable, not an unbounded “search until satisfied” loop

Abstention audit on `v2`:
- report:
  - `data/reports/phase3_abstention_audit_retrieval_queries_sample_1000_v2.json`
- best current transparent rule:
  - rule family: `combined_uncovered_ratio`
  - abstain if the proportion of query tokens not covered by `top document title + top chunk text` is `>= 0.5`
- resulting metrics on `v2`:
  - overall accuracy: `0.8267`
  - abstain precision: `0.7778`
  - abstain recall: `0.3889`
  - answer recall: `0.9649`
- conclusion:
  - current retrieval signals are strong enough to suggest some abstentions
  - they are not yet strong enough to serve as a final abstention gate
  - unsupported-query handling still needs work before generation can be trusted

Chunk-order audit:
- report:
  - `data/reports/phase3_chunk_order_audit_sample_1000.json`
- audit scope:
  - supported queries where stage-1 document retrieval is already top1-correct
- observed:
  - `full` stage-2 mode returns a title-only opening section as top chunk in `83.33%` of these cases
  - `body` stage-2 mode reduces that to `61.90%`
  - `full` returns an `Actualité liée` opening chunk in `16.67%` of these cases
  - `body` reduces that to `4.76%`
  - `body` changes the top chunk in `17` real benchmark queries and usually surfaces a more substantive passage
- conclusion:
  - keep document retrieval structural
  - keep chunk retrieval passage-centric
  - do not go to LLM generation yet; evidence ordering still needs human review on the new notebook

Local strategy audit on `v3`:
- reports:
  - `data/reports/phase3_local_strategy_audit_retrieval_queries_sample_1000_v3__sections_body.json`
  - `data/reports/phase3_local_strategy_audit_retrieval_queries_sample_1000_v3__sections_firstpara_body.json`
- on the current best stage-1 mode `sections_firstpara`:
  - `chunk`:
    - `doc_hit1_rate = 0.95`
    - `title_only_section_rate = 0.6842`
    - `avg_overlap_ratio = 0.8138`
    - `avg_token_count = 238.0`
  - `section_then_chunk`:
    - `doc_hit1_rate = 0.95`
    - `title_only_section_rate = 0.8421`
    - `avg_overlap_ratio = 0.7269`
    - `avg_token_count = 199.2`
- conclusion:
  - `section_then_chunk` is more complex but worse on the current local evidence proxies
  - keep `chunk` as the default stage-2 local strategy
  - keep `section_then_chunk` only as an experiment

Transparency notebook:
- `notebooks/03_real_examples_parsing_chunking_retrieval.ipynb`
- executed version:
  - `notebooks/03_real_examples_parsing_chunking_retrieval.executed.ipynb`
- this notebook shows:
  - real BOFIP document structures
  - real extracted sections/paragraphs/tables
  - full chunk views for real documents
  - lexical/dense/hybrid retrieval traces on real queries

Two-stage transparency notebook:
- `notebooks/04_two_stage_retrieval_audit.ipynb`
- executed version:
  - `notebooks/04_two_stage_retrieval_audit.executed.ipynb`
- this notebook shows:
  - stage 1 top BOI documents
  - stage 2 top chunks
  - side-by-side comparison of stage-2 `full` vs `body`
  - real success cases, neighbor confusions, and unsupported / false-premise traces

Stage-comparison notebook:
- `notebooks/06_stage_comparison_audit.ipynb`
- executed version:
  - `notebooks/06_stage_comparison_audit.executed.ipynb`
- this notebook shows:
  - stage-1 comparison across `base`, `sections`, and `sections_firstpara`
  - stage-2 comparison across `chunk` vs `section_then_chunk`
  - bounded stage-3 local expansion on the real `v3` benchmark
  - remaining document-level misses and representative local passage differences

Abstention and deep-dive notebook:
- `notebooks/05_abstention_and_deep_dive_audit.ipynb`
- executed version:
  - `notebooks/05_abstention_and_deep_dive_audit.executed.ipynb`
- this notebook shows:
  - objective `#1`: document retrieval on the broader `v2` audit set
  - objective `#2`: stage-2 and stage-3 local evidence selection
  - current abstention behavior and its failure modes

## Bugs Found And Fixed

1. Paragraph-number pollution
- Problem: standalone paragraph markers like `1`, `10`, `20` were extracted as full paragraphs.
- Fix: numeric paragraph markers are now attached as `paragraph_number` metadata on the following paragraph.

2. Parent-child size overflow
- Problem: parent chunks could exceed the configured token ceiling.
- Fix: parent sections are now windowed under the same max-token limit.

3. Accent sensitivity in lexical search
- Problem: queries without French accents underperformed.
- Fix: lexical tokenization now strips diacritics before BM25 tokenization.

4. Trivial fragment pollution in chunking
- Problem: random large pools still produced many chunks like `(60)` or `Cf. BOI-...` and a few oversized single-paragraph chunks.
- Fix: trivial fragments are now absorbed into neighboring substantive text when possible, long paragraphs are split under the hard ceiling, and `section_window` now merges residual boundary fragments across adjacent chunks.

5. Broken parent-child references
- Problem: child chunks referenced a synthetic parent id that did not always correspond to an emitted parent chunk id.
- Fix: parent chunk ids are now emitted deterministically first, and children point to a real emitted parent id.

## Current Open Problems

1. Some BOFIP content types are not text-friendly
- `Cartographie` appears non-textual in current HTML extraction.
- `Bareme` may depend on PDFs or link-based content.
- some `Formulaire` entries can also be effectively empty in inline HTML.

2. Manual fidelity review is still outstanding
- the generated report exists, but the human checklist has not been completed yet.

3. Evidence-chunk selection is not yet validated
- document retrieval is now strong enough
- however, within-document chunk ordering still needs explicit validation before generation can be trusted
- the clean-room now uses:
  - document mode `sections_firstpara`
  - stage-2 local mode `body`
  - stage-2 local strategy `chunk`
- this is a much better retrieval-layer baseline, but still not a proof that the first returned chunk is always the best evidence chunk
- the next retrieval work should therefore target evidence selection and unsupported-query gating, not parser/chunker rewrites

4. `parent_child` is not yet production-ready
- the structural links are now valid, but the strategy still generates too many tiny child chunks on random large pools.
- keep it as an experiment, not as the baseline.

5. Unsupported-query handling is still absent
- lexical retrieval returns plausible top matches even when the query is outside the indexed subset.
- the new `v2` abstention audit shows that a simple transparent rule can catch some of these cases, but recall is still too low for production use.
- false-premise queries should remain answerable with correction, not be collapsed into “unsupported”.

## Current Gate

Do not move to LLM generation until:
- the clean-room baseline stays on `Commentaire`-only + `section_window`
- document-level retrieval remains above `0.80` strict top-1 on the broader `1000`-doc benchmark
- current best strict stage-1 mode is `sections_firstpara` at `0.95 hit@1` and `1.00 hit@5` on `v3`
- chunk selection inside the selected document is judged reliable enough for evidence extraction
- unsupported-query handling and abstention policy are stronger than the current `0.3889` abstain recall audit result

## Next Step

Focus on the remaining retrieval-layer weak spots, in order:
- keep auditing objective `#1` with more answerable near-neighbor BOI queries, not just unsupported additions
- investigate whether the last true miss `q01` can be improved by source-derived structural signals, without introducing query heuristics
- decide whether parent/child family confusions like `q32` should be tolerated at stage 1 and repaired at stage 2/3, or explicitly reduced at document ranking time
- refine objective `#2` by reviewing stage-3 deep-dive examples under the promoted baseline:
  - document mode `sections_firstpara`
  - local strategy `chunk`
  - chunk mode `body`
- only then revisit abstention, ideally with stronger signals than the current lexical coverage rule

## Full Commentary Corpus Revalidation

The stabilized baseline has now been rerun on the full `Commentaire` corpus:
- `5666` raw BOFIP commentary documents
- `66289` `section_window` chunks
- `0` empty chunks
- `0` chunks above the hard token ceiling
- `0` chunks with `token_count <= 5`

Artifacts:
- `data/interim/raw_docs_sample_5666.jsonl`
- `data/interim/doc_tree_sample_5666.jsonl`
- `data/interim/chunks_section_window_sample_5666.jsonl`
- `data/reports/phase1_extract_summary_sample_5666.json`
- `data/reports/phase2_chunks_section_window_sample_5666.json`

Important result:
- the best stage-1 mode on `1000` docs did **not** survive the full commentary corpus
- on `5666` docs with the `v3` benchmark:
  - `base`: `hit@1 = 0.8667`, `hit@3 = 0.9500`, `hit@5 = 0.9667`
  - `sections`: `hit@1 = 0.8333`, `hit@3 = 0.9000`, `hit@5 = 0.9667`
  - `sections_firstpara`: `hit@1 = 0.7333`, `hit@3 = 0.9333`, `hit@5 = 0.9667`

Promoted full-corpus stage-1 baseline:
- document mode: `base`

Full-corpus stage-1 failure profile:
- `8` strict misses
- `0` `true_top1_miss`
- categories:
  - `5` `parent_child_family_confusion`
  - `2` `same_family_neighbor`
  - `1` `same_domain_neighbor`

This is a good sign:
- the parser/chunker still hold at full scale
- remaining misses are BOI-neighbor ranking problems, not gross retrieval failures

### Stage-2 Full-Corpus Update

The stage-2 chunk ordering was corrected:
- old behavior: document-first ordering could hide chunks from document ranks `2/3`
- new behavior: round-robin by `local_rank`, then `document_rank`
- effect: keep the best chunk of the best doc first, while still surfacing evidence from doc ranks `2/3`

Tests:
- `24` tests pass with `PYTHONPATH=src`

Full-corpus local audit under:
- document mode `base`
- local strategy `chunk`
- chunk mode `body`

Results:
- `chunk`:
  - `doc_hit1_rate = 0.8667`
  - `title_only_section_rate = 0.6923`
  - `avg_overlap_ratio = 0.8095`
- `section_then_chunk`:
  - `doc_hit1_rate = 0.8667`
  - `title_only_section_rate = 0.8654`
  - `avg_overlap_ratio = 0.7292`

Decision:
- keep `chunk` as the promoted full-corpus stage-2 strategy

Remaining miss repairability after the chunk-order fix:
- `3` `not_repairable_at_stage2`
- `3` `stage2_repairable_strong`
- `2` `stage2_repairable_weak`

Important examples:
- `q03` and `q57` were previously starved by chunk truncation; they are now visible in stage 2
- `q12`, `q29`, `q45` are now strong stage-2 repairs
- `q01`, `q28`, `q30` remain not repairable with the current `top_docs=3`

### Abstention On Full Commentary

Current best transparent abstention rule on the full-corpus baseline:
- `rule_family = docplus_top2_uncovered_ratio`
- threshold `>= 0.2`

Metrics:
- accuracy `0.8778`
- abstain precision `0.6207`
- abstain recall `1.0`
- answer recall `0.8472`

Interpretation:
- good high-recall unsupported-query filter
- still too many false-positive abstentions for a production generation gate
- useful as an audit signal, not yet as final runtime policy

### Updated Gate

Do not move to LLM generation yet.

What is now validated strongly:
- raw BOFIP ingestion via local `document.xml + data.html`
- full commentary parsing at scale
- `section_window` chunking at scale
- document-level retrieval on full commentary corpus
- stage-2 chunk ordering no longer starves lower-ranked but relevant documents

What is still open before LLM:
- investigate the `3` remaining non-repairable document misses on the full commentary corpus
- decide whether some benchmark expectations are too specific versus correct family/general BOI retrieval
- strengthen abstention precision
- audit final evidence chunk quality on the promoted full-corpus baseline

## V4 Benchmark Re-Audit

A new broader benchmark was added:
- `data/interim/retrieval_queries_full_v4.jsonl`
- `100` new questions
- `70` answerable
- `30` unsupported
- wording is more user-like and less title-shaped than the older benchmark

This benchmark intentionally stresses semantic paraphrase and reduces lexical comfort.

### Full-Corpus V4 Results

Commentary-only (`5666` docs):
- lexical:
  - `base`: `hit@1 = 0.3286`, `hit@3 = 0.5143`, `hit@5 = 0.6143`
  - `sections`: `hit@1 = 0.3714`, `hit@3 = 0.5571`, `hit@5 = 0.6571`
  - `sections_firstpara`: `hit@1 = 0.3429`, `hit@3 = 0.5571`, `hit@5 = 0.6000`
- dense doc (`e5-base`, `sections_firstpara`): `hit@1 = 0.3000`, `hit@3 = 0.5857`, `hit@5 = 0.7000`
- hybrid doc (`sections` lexical + dense `sections_firstpara`, `1:1`): `hit@1 = 0.4286`, `hit@3 = 0.6286`, `hit@5 = 0.7429`

Mixed full content (`6295` docs):
- lexical:
  - `base`: `hit@1 = 0.3143`, `hit@3 = 0.4857`, `hit@5 = 0.5429`
  - `sections`: `hit@1 = 0.3571`, `hit@3 = 0.5571`, `hit@5 = 0.6143`
  - `sections_firstpara`: `hit@1 = 0.3571`, `hit@3 = 0.5571`, `hit@5 = 0.6000`
- dense doc (`e5-base`, `sections_firstpara`): `hit@1 = 0.2857`, `hit@3 = 0.5857`, `hit@5 = 0.6714`
- hybrid doc (`sections` lexical + dense `sections_firstpara`, `1:1`): `hit@1 = 0.4286`, `hit@3 = 0.6286`, `hit@5 = 0.7571`

Interpretation:
- the old benchmark was too easy
- the parser/chunker did not collapse on `v4`
- the real weakness is now clearly stage-1 document retrieval under paraphrase
- document-level hybrid helps, but does not solve the problem
- adding non-commentary BOFIP content does not catastrophically break the system, but it does not save it either

### V4 Error Profile

Commentary-only lexical `sections`:
- `44` misses
- categories:
  - `11` `true_top1_miss`
  - `17` `same_family_neighbor`
  - `7` `same_domain_neighbor`
  - `6` `parent_child_family_confusion`
  - `3` `title_equivalent_or_version_confusion`

Mixed lexical `sections`:
- `45` misses
- categories:
  - `12` `true_top1_miss`
  - `17` `same_family_neighbor`
  - `6` `same_domain_neighbor`
  - `6` `parent_child_family_confusion`
  - `4` `title_equivalent_or_version_confusion`

This is the important shift:
- the clean-room is no longer mostly blocked by near-neighbor BOI families only
- the `v4` benchmark reveals a broader paraphrase-stage retrieval problem

### V4 Local Passage Audit

Local chunk strategy on `v4` under lexical `sections`:
- commentary:
  - `chunk`: `avg_overlap_ratio = 0.6259`
  - `section_then_chunk`: `avg_overlap_ratio = 0.6002`
- mixed:
  - `chunk`: `avg_overlap_ratio = 0.6296`
  - `section_then_chunk`: `avg_overlap_ratio = 0.6029`

Decision remains unchanged:
- keep `chunk` as the promoted local strategy
- `section_then_chunk` remains more complex and not better

### New Transparency Notebooks

Interactive query workbench:
- `notebooks/07_query_workbench.ipynb`
- executed version:
  - `notebooks/07_query_workbench.executed.ipynb`

This notebook shows for one user query:
- tokenization
- query embedding preview
- lexical doc hits
- dense doc hits
- chunk-dense doc hits when available
- hybrid doc hits
- BOI family/relations
- local chunk hits inside top documents
- current lexical two-stage trace

V4 benchmark synthesis:
- `notebooks/08_v4_benchmark_audit.ipynb`
- executed version:
  - `notebooks/08_v4_benchmark_audit.executed.ipynb`

The query workbench now shows the currently strongest document-stage variant:
- lexical views: `base` + `sections_leads`
- dense view: `sections_firstpara`
- chunk-dense view: `full`
- promoted multiview hybrid weights: `base=1`, `sections_leads=2`, `dense=1`, `chunk_dense=2`

### Derived Obsidian Vault

A derived Obsidian knowledge layer now exists alongside the clean-room:
- sibling vault root:
  - `..\bofip-rag-cleanroom-obsidian`
- generator:
  - `scripts/build_obsidian_vault.py`

Purpose:
- summarize the current clean-room state
- centralize architecture decisions, retrieval baseline, hard queries, and failure taxonomy
- link back to executed notebooks and JSON reports

Important rule:
- the vault is **not** a source of truth
- source of truth remains:
  - code
  - tests
  - notebooks
  - reports under `data/reports/`

### Promoted V4 Document-Stage Variant

Added script:
- `scripts/phase3_doc_multiview_hybrid_eval.py`

Supporting additions:
- `scripts/phase3_dense_eval.py` now supports:
  - chunk text modes: `full`, `leaf`, `body`
  - embedding cache reuse
- `src/bofip_cleanroom/lexical_retrieval.py` now supports:
  - `document_mode = sections_leads`
  - this mode builds a richer document lexical view from:
    - BOI reference
    - document title / metadata
    - section titles
    - first paragraph lead snippets per section

Chunk-dense document experiments on `v4` commentary:
- `body` mode:
  - `hit@1 = 0.2714`
  - `hit@3 = 0.3857`
  - `hit@5 = 0.5143`
- `full` mode:
  - `hit@1 = 0.3714`
  - `hit@3 = 0.5286`
  - `hit@5 = 0.6286`

Conclusion:
- `chunk_dense(body)` is rejected
- `chunk_dense(full)` is kept as a useful additional documentary signal

Current promoted document-stage fusion on `v4`:
- commentary:
  - lexical `sections_leads`: `hit@1 = 0.4286`, `hit@3 = 0.6857`, `hit@5 = 0.7571`
  - promoted multiview `base + sections_leads + doc_dense + chunk_dense`:
    - `hit@1 = 0.5286`
    - `hit@3 = 0.7000`
    - `hit@5 = 0.7714`
  - report:
    - `data/reports/phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__lexbase_sections_leads__densesections_firstpara__intfloat__multilingual-e5-base__base1p0_chunk_dense2p0_dense1p0_sections_leads2p0.json`

This improves over:
- lexical-only `sections`
- lexical-only `sections_leads`
- dense-only `sections_firstpara`
- the earlier `base + sections + doc_dense`
- the earlier `base + sections + doc_dense + chunk_dense`

But the improvement is still incremental, not decisive.

Interpretation:
- multi-view document fusion is worth keeping as the current best stage-1 baseline
- it repaired several previously problematic cases such as `q01`, `q28`, `q30`
- it is a valid retrieval improvement, not a final answer

### Targeted Review Of The Current Non-Repairables

Artifacts:
- `data/reports/phase3_targeted_case_review_raw_docs_sample_5666.json`
- `data/reports/phase3_targeted_case_review_raw_docs_sample_6295.json`
- `data/reports/phase3_failure_analysis_phase3_doc_multiview_hybrid_eval_raw_docs_sample_5666__retrieval_queries_full_v4__balanced_sections_leads.json`

Current reading:
- `q01`, `q28`, `q30`: no longer block the promoted baseline
- current remaining strict `true_top1_miss` set on `v4` commentary:
  - `a025`
  - `a026`
  - `a031`
  - `a050`

Reading of these four:
- `a025`:
  - lexical `sections_leads` is correct on its own
  - dense and chunk-dense are wrong
  - current fusion still overweights the wrong dense evidence for this query
- `a026`:
  - dense and chunk-dense are correct
  - lexical is weak / misleading
  - this is the mirror case of `a025`
- `a031`:
  - still a real paraphrase/acronym problem around `PEA`
  - dense surfaces the correct family but not the expected child doc
- `a050`:
  - lexical finds the correct `IF-TU` family but prefers an exemption child
  - dense remains noisy here

So after `v4`, the priority changes:
- the next major effort should target paraphrase-robust stage-1 document retrieval
- especially query-dependent arbitration between strong lexical and strong dense signals
- not parser rewrites
- not chunker rewrites
- not LLM generation

### Passage Gold V1

New artifacts:
- `data/interim/passage_gold_v1.jsonl`
- `data/reports/phase5_stage1_passage_gold_v1_baseline.json`
- `data/reports/phase5_stage1_passage_gold_v1_title_tail.json`
- `data/reports/phase5_passage_gold_eval_baseline_family.json`
- `data/reports/phase5_passage_gold_eval_baseline_family_tail025.json`
- `data/reports/phase5_passage_gold_eval_titletail_family.json`
- `data/reports/phase5_passage_gold_eval_titletail_family_tail025.json`
- `data/reports/phase5_direct_chunk_eval_baseline.json`
- `data/reports/phase5_direct_chunk_eval_baseline_full.json`
- `data/reports/phase5_direct_chunk_eval_baseline_leaf.json`
- `data/reports/phase5_direct_chunk_eval_titletail.json`
- `data/reports/phase5_passage_summary.json`

What changed:
- a first manual passage-level evaluation set was created with `22` answerable queries
- the gold was aligned against the real `section_window` chunks until coverage reached `22 / 22`
- passage matching now uses accent-insensitive and punctuation-robust normalization

Passage-gold stage-1 results:
- promoted stage-1 baseline:
  - doc `hit@1 = 0.2273`
  - doc `hit@3 = 0.5909`
  - doc `hit@5 = 0.7273`
- `title_tail` global variant:
  - doc `hit@1 = 0.1818`
  - doc `hit@3 = 0.6364`
  - doc `hit@5 = 0.7273`

Important conclusion:
- on this passage-level set, `title_tail` improves documentary breadth (`@3`) but hurts exact `@1`
- so it remains an experimental stage-1 variant, not the promoted default

Passage-gold stage-2 findings:
- family-guided `body` on promoted stage-1:
  - passage `hit@1 = 0.0455`
  - passage `hit@3 = 0.1364`
  - passage `hit@5 = 0.1818`
- family-guided `body` with `tail_weight = 0.25`:
  - passage `hit@5 = 0.2273`
- direct stage-1 docs -> local chunks, `body`:
  - passage `hit@1 = 0.0455`
  - passage `hit@3 = 0.3636`
  - passage `hit@5 = 0.3636`
- direct stage-1 docs -> local chunks, `full` or `leaf`:
  - passage `hit@1 = 0.0455`
  - passage `hit@3 = 0.3636`
  - passage `hit@5 = 0.4545`

Reading:
- the current family-guided stage-2 helps on document-family recall in some broad retrieval benchmarks
- but on the first passage-level gold, it is clearly worse than the simpler strategy:
  - keep stage-1 top documents
  - search chunks locally inside those documents
- this is the first concrete sign that:
  - document-family expansion and passage extraction should be treated as separate problems
  - a family-guided document strategy is not automatically a good passage strategy

Loss decomposition on `phase5_passage_gold_eval_baseline_family_tail025.json`:
- `6` queries lost before stage-1 doc top-5
- `4` queries where stage-1 top-5 contains the expected doc but family-guided stage-2 drops it
- `7` queries where the expected doc is still present in chunk candidates but the correct passage is not surfaced in top-5
- only `5 / 22` queries have the correct passage in top-5 under the current family-guided baseline

Current decision:
- keep the promoted stage-1 documentary baseline
- do **not** promote family-guided stage-2 as the default passage retriever
- prefer direct stage-1 docs -> local chunk retrieval as the current clean-room passage baseline
- prefer `chunk_mode = full` or `leaf` over `body` for local passage retrieval

### Phase 6: Stage-2 Promotion And Multi-Benchmark Revalidation

New artifacts:
- `scripts/build_retrieval_query_set_v6.py`
- `data/interim/retrieval_queries_full_v6.jsonl`
- `scripts/build_passage_gold_v2.py`
- `data/interim/passage_gold_v2.jsonl`
- `scripts/phase5_stage2_comparison.py`
- `data/reports/phase5_stage2_comparison__passage_gold_v1.json`
- `data/reports/phase6_stage1_v3.json`
- `data/reports/phase6_stage1_v4.json`
- `data/reports/phase6_stage1_v5.json`
- `data/reports/phase6_stage1_v6.json`
- `data/reports/phase6_stage1_passage_gold_v1.json`
- `data/reports/phase6_stage1_passage_gold_v2.json`
- `data/reports/phase6_family_v3.json`
- `data/reports/phase6_family_v4.json`
- `data/reports/phase6_family_v5.json`
- `data/reports/phase6_family_v6.json`
- `data/reports/phase6_family_passage_v1.json`
- `data/reports/phase6_family_passage_v2.json`
- `data/reports/phase6_direct_passage_v1.json`
- `data/reports/phase6_direct_passage_v2.json`
- `data/reports/phase6_family_passage_eval_v2.json`
- `data/reports/phase6_stage2_comparison__passage_gold_v2.json`
- `data/reports/phase6_cross_benchmark_summary.json`
- `notebooks/07_query_workbench.ipynb`
- `notebooks/07_query_workbench.executed.ipynb`

What changed:
- the query workbench was aligned with the promoted passage baseline:
  - stage 1 multiview document retrieval
  - then direct local chunk retrieval inside stage-1 top docs
  - `chunk_mode = full`
- family-guided retrieval remains available in the workbench, but only as an experimental comparator
- a new document-level benchmark `v6` was added with `100` new queries
- a larger passage-level benchmark `passage_gold_v2` was added with `40` rows
- `passage_gold_v2` reached `40 / 40` chunk coverage on the current `section_window` corpus
- passage matching now supports explicit `chunk_ids_any` constraints to make passage-level evaluation traceable and stable

Stage-2 promotion result:
- on `passage_gold_v1`, direct `top docs -> local chunks(full)` versus family-guided `tail_weight=0.25`:
  - passage `hit@3`: `0.4091` vs `0.1364`
  - passage `hit@5`: `0.5000` vs `0.2273`
  - `9` improved, `0` regressed
- on `passage_gold_v2`, direct `top docs -> local chunks(full)` versus family-guided `tail_weight=0.25`:
  - passage `hit@3`: `0.4250` vs `0.3500`
  - passage `hit@5`: `0.5750` vs `0.4250`
  - `14` improved, `4` regressed

Decision:
- direct local chunk retrieval is now the promoted stage-2 baseline
- family-guided remains experimental and useful only for diagnostic family expansion

Phase-6 cross-benchmark metrics on `commentary`:
- `retrieval_queries_sample_1000_v3`
  - family hit rate: `0.9833`
  - exact doc `@1/@3/@5 = 0.7833 / 0.9833 / 1.0000`
- `retrieval_queries_full_v4`
  - family hit rate: `0.9143`
  - exact doc `@1/@3/@5 = 0.6000 / 0.7286 / 0.8143`
- `retrieval_queries_full_v5`
  - family hit rate: `0.8429`
  - exact doc `@1/@3/@5 = 0.4714 / 0.6571 / 0.7571`
- `retrieval_queries_full_v6`
  - family hit rate: `0.8714`
  - exact doc `@1/@3/@5 = 0.4714 / 0.6286 / 0.8000`
- `passage_gold_v1`
  - family hit rate: `0.6818`
  - exact doc `@1/@3/@5 = 0.2727 / 0.6364 / 0.7727`
  - direct passage `@1/@3/@5 = 0.0909 / 0.4091 / 0.5000`
- `passage_gold_v2`
  - family hit rate: `0.8000`
  - exact doc `@1/@3/@5 = 0.3500 / 0.6000 / 0.8000`
  - direct passage `@1/@3/@5 = 0.2000 / 0.4250 / 0.5750`

Gate A status from `phase6_cross_benchmark_summary.json`:
- `v3` regression guard: passed
- `v4` exact doc `hit@5 >= 0.80`: passed
- `v5` exact doc `hit@5 >= 0.80`: failed
- `v4` family hit rate `>= 0.90`: passed
- `v5` family hit rate `>= 0.90`: failed
- `passage_gold_v2` passage `hit@3 >= 0.45`: failed
- `passage_gold_v2` passage `hit@5 >= 0.65`: failed

Conclusion:
- Gate A is **not** passed
- parsing and chunking remain stable and are no longer the dominant problem
- the promoted stage-2 baseline is now clear and justified
- the main remaining bottleneck is still stage-1 documentary retrieval under paraphrase on `v5` and `v6`
- the next retrieval loop should target:
  - `INT-AEA`
  - `IF-TU`
  - `TVA-DECLA`
  - `IS-GPE`
  - `PAT-IFI`
  - `RPPM`
- no move to abstention or LLM generation until `v5` family/doc metrics and `passage_gold_v2` passage metrics improve materially

### Phase 7: Family Bubble And Stage-1 Retuning Loop

New artifacts:
- `data/reports/phase7_family_v3_anc1.json`
- `data/reports/phase7_family_v4_anc1.json`
- `data/reports/phase7_family_v5_anc1.json`
- `data/reports/phase7_family_v6_anc1.json`
- `data/reports/phase7_family_v5_anc1_overview05.json`
- `data/reports/phase7_family_v6_anc1_overview05.json`
- `data/reports/phase7_stage1_v5_ck30.json`
- `data/reports/phase7_stage1_v6_ck30.json`
- `data/reports/phase7_stage1_v5_tail05.json`
- `data/reports/phase7_stage1_v6_tail05.json`
- `data/reports/phase7_stage1_v5_tail025.json`
- `data/reports/phase7_stage1_v6_tail025.json`
- `data/reports/phase7_stage1_v5_chunkbody.json`
- `data/reports/phase7_stage1_v6_chunkbody.json`

Code changes:
- `family_routing.py`
  - added bounded `ancestor_expansion_levels`
- `family_guided_retrieval.py`
  - threaded bounded ancestor expansion through family-guided retrieval
- `phase4_family_guided_eval.py`
  - added CLI support for ancestor expansion
- `test_family_routing.py`
  - added ancestor expansion and boundedness tests
- `test_family_guided_retrieval.py`
  - added sibling-recovery test with ancestor expansion

What was tested:
- generic ancestor bubble expansion for family retrieval
- ancestor bubble + overview bonus
- stage-1 `candidate_k = 30`
- stage-1 `title_tail` reintroduced at low weights (`0.5`, `0.25`)
- stage-1 chunk-dense source switched from `full` to `body`

Results:
- ancestor bubble `+1`:
  - improves family coverage on hard sets
  - `v5` family hit rate: `0.8429 -> 0.9000`
  - `v6` family hit rate: `0.8714 -> 0.9143`
  - but does **not** improve `v5` exact doc `@5`
  - and degrades `v6` exact doc `@5`: `0.8143 -> 0.7714`
- ancestor bubble + overview bonus:
  - reaches `v5` family-guided doc `@5 = 0.8000`
  - but degrades `v5` doc `@3` and degrades `v6` strongly
  - rejected as default
- `candidate_k = 30`:
  - `v5` exact doc `@5`: `0.7571 -> 0.7571`
  - `v6` exact doc `@5`: `0.8000 -> 0.7714`
  - rejected
- `title_tail = 0.5`:
  - `v5` exact doc `@5`: `0.7571 -> 0.7857`
  - `v6` exact doc `@5`: `0.8000 -> 0.7714`
  - mixed signal, not promoted
- `title_tail = 0.25`:
  - no useful gain over baseline
  - rejected
- chunk-dense `body`:
  - no gain on `v5`
  - degrades `v6`
  - rejected

Conclusion:
- no new stage-1 baseline was promoted in phase 7
- the ancestor bubble is useful as an **experimental diagnostic tool**, not as the default retrieval path
- the strongest remaining weakness is still stage-1 documentary retrieval on broad paraphrase / overview-style queries
- the next useful loop should not keep retuning the same weights blindly
- the next loop should target:
  - better generic query-side semantic normalization or aliasing
  - or a materially different stage-1 signal, such as a stronger dense document encoder

### Phase 8: Stronger Dense Document Signal Without Gold Leakage

New artifacts:
- `data/reports/phase8_doc_dense_v5_e5base.json`
- `data/reports/phase8_doc_dense_v6_e5base.json`
- `data/reports/phase8_doc_dense_v5_e5large_cpu.json`
- `data/reports/phase8_doc_dense_v6_e5large_cpu.json`
- `data/reports/phase8_stage1_v3_e5large_doc_e5chunk.json`
- `data/reports/phase8_stage1_v4_e5large_doc_e5chunk.json`
- `data/reports/phase8_stage1_v5_e5large_doc_e5chunk.json`
- `data/reports/phase8_stage1_v6_e5large_doc_e5chunk.json`
- `data/reports/phase8_stage1_passage_v1_e5large_doc_e5chunk.json`
- `data/reports/phase8_stage1_passage_v2_e5large_doc_e5chunk.json`
- `data/reports/phase8_family_v3_e5large_doc_e5chunk.json`
- `data/reports/phase8_family_v4_e5large_doc_e5chunk.json`
- `data/reports/phase8_family_v5_e5large_doc_e5chunk.json`
- `data/reports/phase8_family_v6_e5large_doc_e5chunk.json`
- `data/reports/phase8_direct_passage_v1_e5large_doc_e5chunk_3x8.json`
- `data/reports/phase8_direct_passage_v2_e5large_doc_e5chunk_3x8.json`
- `data/reports/phase8_cross_benchmark_summary.json`

Code changes:
- `dense_retrieval.py`
  - prompt formatting is now model-aware
  - E5-family models keep `query:` / `passage:` prefixes
  - non-E5 models keep plain text
  - explicit `device` support added to `DenseEncoder`
- `phase3_dense_eval.py`
  - added `--device`
- `phase3_doc_dense_eval.py`
  - added `--device`
- `phase3_doc_multiview_hybrid_eval.py`
  - added `--device`
  - added support for a separate `--chunk-dense-model`
  - added support for a separate `--chunk-dense-device`
  - stage-1 multiview can now mix a document-dense encoder and a chunk-dense encoder cleanly
- `test_dense_retrieval.py`
  - added prompt-style tests for E5 vs non-E5 formatting

What was tested:
- attempted `BAAI/bge-m3`
  - rejected by environment constraints
  - local snapshot was incomplete for safe loading
  - current `torch 2.5.1` environment blocks legacy unsafe loading path
- switched to `intfloat/multilingual-e5-large` as the stronger dense document candidate
- tested dense-only document retrieval with `e5-base` vs `e5-large`
- tested mixed multiview stage-1:
  - lexical `base`
  - lexical `sections_leads`
  - lexical `sections_leads_stem`
  - document-dense `sections_firstpara` with `e5-large`
  - chunk-dense `full` with `e5-base`
- validated on:
  - `v3`
  - `v4`
  - `v5`
  - `v6`
  - `passage_gold_v1`
  - `passage_gold_v2`

Dense-only document insight:
- `e5-large` is materially stronger than `e5-base` as a pure document encoder
- `v5` dense-only doc `@5`: `0.6286 -> 0.7000`
- `v6` dense-only doc `@5`: `0.5714 -> 0.6000`
- this justified evaluating `e5-large` inside the multiview stage-1 pipeline

Stage-1 multiview candidate results:
- `v3`
  - doc `@1/@3/@5 = 0.7667 / 0.9833 / 1.0000`
- `v4`
  - doc `@1/@3/@5 = 0.5429 / 0.7714 / 0.8571`
- `v5`
  - doc `@1/@3/@5 = 0.4857 / 0.6429 / 0.8143`
- `v6`
  - doc `@1/@3/@5 = 0.4857 / 0.6571 / 0.8000`

Compared to the phase-6 promoted baseline:
- `v4`
  - doc `@5`: `0.8143 -> 0.8571`
  - doc `@3`: `0.7286 -> 0.7714`
  - doc `@1`: `0.6000 -> 0.5429`
- `v5`
  - doc `@5`: `0.7571 -> 0.8143`
  - doc `@1`: `0.4714 -> 0.4857`
- `v6`
  - doc `@5`: `0.8000 -> 0.8000`
  - doc `@3`: `0.6286 -> 0.6571`
  - doc `@1`: `0.4714 -> 0.4857`

Family-level effect:
- `v4` expected-in-family rate: `0.9143 -> 0.9286`
- `v5` expected-in-family rate: `0.8429 -> 0.8286`
- `v6` expected-in-family rate: `0.8714 -> 0.8429`

Passage-level effect with direct stage-2 `full`:
- `passage_gold_v1`
  - doc `@5`: `0.7727 -> 0.7273`
  - passage `@3`: `0.4091 -> 0.4545`
  - passage `@5`: `0.5000 -> 0.4545`
- `passage_gold_v2`
  - doc `@5`: `0.8000 -> 0.7500`
  - passage `@3`: `0.4250 -> 0.4750`
  - passage `@5`: `0.5750 -> 0.5250`

Stage-2 validation note:
- the phase-8 candidate was rerun with the same direct stage-2 parameters as the phase-6 baseline:
  - `top_docs=5`
  - `chunks_per_doc=3`
  - `max_chunks=8`
  - `chunk_mode=full`
- this did **not** recover the passage-level loss
- a small grid search on `passage_gold_v1` over:
  - `chunk_mode in {full, leaf, body}`
  - `top_docs in {5, 6}`
  - `chunks_per_doc in {2, 3, 4}`
  - `max_chunks in {6, 8, 10}`
  showed no better stage-2 configuration for this candidate

Gate-A interpretation:
- `v4` doc `@5 >= 0.80`: now passed more comfortably
- `v5` doc `@5 >= 0.80`: now passed
- `v6` doc `@5`: held at `0.8000`
- but Gate A still fails because:
  - `v5` family hit rate remains below `0.90`
  - `passage_gold_v2` passage `@5` remains below `0.65`

Conclusion:
- `e5-large` is a **real improvement** for stage-1 document retrieval
- it is the first candidate to push `v5` doc `@5` above the gate threshold without degrading `v6` doc `@5`
- but it is **not yet safe to promote as the new global baseline**
- reason:
  - it trades exact sub-document / family precision and passage recall for broader document retrieval gains
- decision:
  - keep phase-8 `e5-large doc + e5-base chunk` as the leading experimental branch
  - keep the phase-6 promoted baseline as the current default until passage-level and `v5` family-level regressions are addressed
- next useful loop:
  - target the queries improved by phase 8 vs the queries regressed on passage-level
  - understand whether the stronger dense document signal is surfacing broader overview docs where the previous baseline surfaced the narrower correct sub-doc

### Phase 8b: Generic Specificity Rerank On Top Of The Stronger Dense Branch

New artifacts:
- `data/reports/phase8b_stage1_v5_e5large_doc_e5chunk_spec005.json`
- `data/reports/phase8b_stage1_v6_e5large_doc_e5chunk_spec005.json`
- `data/reports/phase8b_stage1_passage_v2_e5large_doc_e5chunk_spec005.json`
- `data/reports/phase8b_direct_passage_v2_e5large_doc_e5chunk_spec005_3x8.json`
- `data/reports/phase8b_family_v5_e5large_doc_e5chunk_spec005.json`
- `data/reports/phase8b_family_v6_e5large_doc_e5chunk_spec005.json`

Code changes:
- `specificity_rerank.py`
  - new shared helper for generic post-fusion document reranking
  - uses only source-derived signals:
    - title-tail lexical overlap
    - document depth in BOFiP reference hierarchy
    - broadness penalty based on descendant count
- `family_guided_retrieval.py`
  - family-guided rerank can now apply the shared specificity helper
- `phase4_family_guided_eval.py`
  - added CLI support for specificity rerank parameters
- `phase3_doc_multiview_hybrid_eval.py`
  - added CLI support for specificity rerank parameters
- `test_specificity_rerank.py`
  - added focused unit tests for sibling promotion and isolation safety

Method:
- used `passage_gold_v1` as the development set only
- searched a small parameter grid offline on the already-produced phase-8 stage-1 report
- selected:
  - `specificity_rerank_top_n = 5`
  - `specificity_rerank_weight = 0.05`
- validated only afterwards on:
  - `v5`
  - `v6`
  - `passage_gold_v2`

Observed effect:
- on `passage_gold_v1` stage-1:
  - doc `@1`: `0.3182 -> 0.3636`
  - doc `@3`: unchanged at `0.6818`
  - doc `@5`: unchanged at `0.7273`
- on `v5`:
  - stage-1 doc `@1/@3/@5 = 0.4857 / 0.6571 / 0.8143`
  - same `@5` as phase 8, no regression
- on `v6`:
  - stage-1 doc `@1/@3/@5 = 0.5429 / 0.6714 / 0.8000`
  - improvement vs phase 8 on `@1` and `@3`, `@5` unchanged
- on `passage_gold_v2` stage-1:
  - doc `@1/@3/@5 = 0.4250 / 0.7000 / 0.7500`
  - improvement vs phase 8 on `@1` and `@3`

Direct stage-2 `full` on `passage_gold_v2`:
- `stage1_doc_hit@1/@3/@5 = 0.4250 / 0.7000 / 0.7500`
- `passage_hit@1/@3/@5 = 0.2750 / 0.5000 / 0.5250`

Compared to phase 8 direct passage on `passage_gold_v2`:
- passage `@1`: `0.2250 -> 0.2750`
- passage `@3`: `0.4750 -> 0.5000`
- passage `@5`: unchanged at `0.5250`

Family-level effect:
- `v5`
  - expected-in-family rate: `0.8286 -> 0.8429`
  - family doc `@5`: `0.7857 -> 0.8000`
- `v6`
  - expected-in-family rate: `0.8429 -> 0.8286`
  - family doc `@5`: `0.8000 -> 0.8143`

Conclusion:
- the generic specificity rerank is a **useful refinement**
- it improves early precision on the stronger dense branch
- it improves validation passage `@1/@3` without harming `v5` doc `@5`
- but it still does **not** solve the remaining gate blocker:
  - passage `@5` is still too low
  - family coverage on broad paraphrase sets is still below threshold
- decision:
  - keep the specificity rerank as part of the leading experimental branch
  - do not yet promote this branch as the default retrieval baseline
