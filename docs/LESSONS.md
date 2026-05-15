# Lessons Learned

29 documented mistakes organized as **universal engineering principles**.
Each lesson states a general rule, then provides a project-specific example.

**Read this before making any changes to the system.**

---

## 1. Decision-Making

### Lesson 1: Don't Dismiss Ideas Based on One Failed Test

**Principle**: A technique that fails in one configuration may succeed in another. Before ruling something out, identify whether the failure was in the concept or the implementation.

**Rule**: Before agreeing that something "doesn't work", ask:
1. In what configuration was it tested?
2. Could it work in a different configuration?
3. Was the failure in the CONCEPT or the IMPLEMENTATION?

> **Example**: Query decomposition failed when used for retrieval expansion (scattered chunks), but worked perfectly when used for answer structuring (Grille d'Analyse Fiscale). Same concept, different mode.

### Lesson 2: One Technique Can Have Multiple Modes

**Principle**: Most techniques have several implementation strategies. If one mode fails, explore others before abandoning the whole concept.

| Mode | Strategy | Trade-off |
|------|----------|-----------|
| Expand input | Use technique to broaden search | Broader but noisier |
| Aggregate results | Use technique to group findings | Coherent but costly |
| Structure output | Use technique to organize answers | Clean but limited |
| Validate | Use technique to verify completeness | Quality check but extra cost |

> **Example**: Query decomposition â€” Mode 1 (retrieval expansion) failed. Mode 3 (answer structuring via Grille d'Analyse) worked. Mode 2 (document aggregation) was never tested.

### Lesson 3: Explain Disagreements, Don't Just Agree

**Principle**: When evaluating a decision, always explain the specific reasoning â€” what failed, why it failed, and under what conditions it might work.

**Rule**: Never just say "yes" or "no". Always provide:
1. What failed specifically
2. Why it failed
3. Alternatives or conditions where it might work

---

## 2. Debugging

### Lesson 4: Check Infrastructure Before Tuning Parameters

**Principle**: When output is wrong, the problem is more likely in data pipelines, configurations, or silent bugs than in model parameters or prompts. Always check the plumbing first.

**Rule**: Before tweaking prompts or models:
1. Verify data sources are consistent (do all indexes have the same count?)
2. Verify fallback logic is sane (what happens when parsing fails?)
3. Verify transformations actually work (add logging at every step)
4. Verify end-to-end flow (trace one query through the entire pipeline)

> **Example**: Wrong documents were returned. We spent hours on prompt fixes. Root cause: 4 infrastructure bugs â€” mismatched data counts, overly strict regex, silent extraction failure, and truncated aggregation.

### Lesson 5: Silent Failures Are the Worst

**Principle**: Code that fails silently and falls back to a default is harder to debug than code that crashes. Every fallback path must be logged.

**Anti-pattern**:
```python
try:
    result = process(data)
except:
    result = default  # Silent fallback â€” nobody knows this happened
```

**Correct pattern**:
```python
try:
    result = process(data)
    logger.info(f"Processed successfully: {len(result)} items")
except Exception as e:
    result = default
    logger.warning(f"Processing FAILED ({e}), using default")
```

**Rule**: Every `except` block, every `else` branch, every fallback must log what happened and why.

### Lesson 6: Aggregation Must Ensure Diversity

**Principle**: When collecting results from multiple sources and truncating, the first source can dominate if it's larger. Always limit per-source, not just total.

**Anti-pattern**:
```python
for source in sources:
    all_results.extend(get_results(source))
return all_results[:max_results]  # First source dominates!
```

**Correct pattern**:
```python
per_source = max_results // len(sources)
for source in sources:
    all_results.extend(get_results(source)[:per_source])
```

### Lesson 7: Trace Data at Every Pipeline Step

**Principle**: When debugging multi-step pipelines, don't just check input and output. Check every intermediate step to find exactly where the correct result disappears.

**Rule**: For any pipeline with N steps, print/log the relevant result after EACH step:
1. Step 1 output â†’ is the expected item present?
2. Step 2 output â†’ still present?
3. ... and so on until you find the step that loses it.

> **Example**: Correct document existed in BM25 results (step 1) and vector results (step 2), but disappeared after merge (step 3) because score normalization compressed its rank.

---

## 3. No Hardcoding / No Overfitting

### Lesson 8: Never Put Domain Data in Prompts

**Principle**: If a RAG system needs specific facts (rates, thresholds, dates), those facts must come from retrieval, not from the prompt. Hardcoding facts in prompts creates overfitting to test cases and becomes outdated.

**Anti-pattern**: Adding specific values to the system prompt to pass test cases.
**Correct fix**: Improve retrieval so the data IS in the context naturally.

**Rule**: If information is missing from retrieved context, fix the retrieval pipeline â€” not the prompt.

> **Example**: Tax rates were hardcoded in the prompt ("Credit impot: 50%"). This is wrong because rates change yearly, and the whole point of RAG is to retrieve them.

### Lesson 9: Added Complexity Can Lose Good Results

**Principle**: Each post-processing layer (rewriting, filtering, re-ranking) can accidentally demote correct results. More steps = more places for things to go wrong.

**Rule**: Before adding a new processing step, verify that the simpler version already fails. If the base results are correct, you may only need better presentation â€” not more processing.

> **Example**: BM25 found the correct document at rank 1. After adding HyDE + LLM filter + RRF, the correct document dropped to rank 13. Removing all three layers fixed it.

### Lesson 10: KISS â€” Keep It Simple, Stupid

**Principle**: Simpler systems are easier to debug, faster to run, and often more accurate than complex ones. Only add complexity when you have measured proof that it improves results.

**Rule**: Start with the simplest possible implementation. Only add a new layer when:
1. You have a benchmark showing the simple version fails
2. You have a benchmark showing the complex version fixes it
3. The complex version doesn't break other things

> **Example**: Removing 5 components (HyDE, LLM filter, RRF, query decomposition, series filtering) improved accuracy from 43% to 86%.

### Lesson 11: Use Patterns, Not Mappings

**Principle**: When you need to boost or filter results, use general patterns (regex, data types, structural rules) that work for any input â€” not keyword-to-output mappings that only work for known cases.

**Anti-pattern**:
```python
if "bareme IR" in query:
    boost specific_documents  # Only works for this exact query
```

**Correct pattern**:
```python
if contains_rate_keywords(query):
    boost chunks matching r'\d+\s*%' or r'\d+.*EUR'  # Works for ANY rate query
```

**Rule**: Solutions must generalize. If you can't describe the pattern without mentioning a specific query or document, it's hardcoded.

---

## 4. Data & Retrieval

### Lesson 12: Supplementary Content Needs Document-Level Inclusion

**Principle**: Some content types (tables, figures, examples) contain critical data but don't match keyword searches. When a document is found relevant, its supplementary content should be included even without keyword matches.

> **Example**: Query about "vehicle CO2 thresholds" â€” the table with actual thresholds had zero keyword overlap with the query, but was essential for answering.

### Lesson 13: Tokenization Must Be Domain-Aware

**Principle**: Default tokenizers break on domain-specific patterns (compound words, abbreviations, non-standard number formats, special punctuation). Always test tokenization with real domain examples.

**Common tokenization bugs**:
| Problem | Example | Fix |
|---------|---------|-----|
| Stopwords in compounds | "plus-value" â†’ "value" | Remove domain terms from stopwords |
| Splitting compounds | "credit-bail" â†’ "credit", "bail" | Preserve hyphens in regex |
| Splitting contractions | "d'impot" â†’ "d", "impot" | Preserve apostrophes |
| Number formats | "18 300" â†’ "300" | Normalize locale-specific numbers |
| Short codes dropped | "IS" filtered out | Allow short tokens (>=2 chars) |

**Rule**: Before trusting any text processing, test it with 10+ real domain examples and check the output manually.

### Lesson 14: Caching Can Hide Fixed Bugs

**Principle**: If your system caches results, old (wrong) cached answers will be served even after you fix the underlying bug. Always disable cache when testing fixes.

**Rule**: When testing any fix, bypass the cache. Only re-enable caching after verifying the fix works.

> **Example**: Value boosting was working correctly in diagnostics, but the user kept seeing the old wrong answer because the LLM response was cached with the same query hash.

### Lesson 15: Keep Documentation Updated

**Principle**: Outdated documentation causes new contributors (human or AI) to make wrong assumptions, restore deprecated approaches, or miss important architectural decisions.

**Rule**: After any significant change, update:
1. Architecture documentation (what the system looks like now)
2. Lessons learned (what you discovered)
3. Working memory (current status and patterns)

---

## 5. Multi-Source Systems

### Lesson 16: Inject Secondary Sources After Main Scoring

**Principle**: When combining results from different scoring pipelines, secondary sources must be injected AFTER the main normalization/merge step â€” not before. Mixing them before normalization corrupts the score distribution.

**Anti-pattern**:
```python
main_results = search(query)
main_results += secondary_results  # BAD: secondary scores get normalized with main
normalize(main_results)
```

**Correct pattern**:
```python
main_results = search(query)
normalize(main_results)
merged = merge(main_results)
# Inject AFTER merge with a fixed score
for item in secondary_results:
    item['score'] = 0.70
    merged.append(item)
```

### Lesson 17: Same Metadata Flag Can Mean Different Things Across Sources

**Principle**: A boolean flag (like `contains_table`) may have different semantic meanings depending on the data source. Always validate what metadata actually means for each source before applying logic based on it.

> **Example**: `contains_table=True` in BOFIP meant a real data table (rates, thresholds). In CGI/LPF PDFs, it meant a table of contents. Blindly including all "table" chunks flooded results with useless TOC entries.

### Lesson 18: Fix Data at the Source, Not Downstream

**Principle**: When data is wrong in the output, trace it back to where it was created and fix it there. Patching data downstream (in prompts, UI, or display logic) creates fragile workarounds that break on new data.

**Rule**: Always trace bad data upstream to its origin. Fix at the earliest possible point in the pipeline.

> **Example**: Article references were empty in search results. Wrong fix: patch in the prompt template. Right fix: set the reference correctly in the PDF parser during chunk creation.

### Lesson 19: Use Models Trained for Your Target Language

**Principle**: Models trained on a different language or domain cannot understand semantic relationships in your target language. A multilingual or target-language model is essential for non-English systems.

> **Example**: English cross-encoder reranker treated "regime mere-fille" and "CGI Art. 145" as unrelated strings. French multilingual model correctly ranked them as semantically close. One line change: 6/7 â†’ 7/7.

### Lesson 20: Large Sources Drown Small Sources

**Principle**: When one data source has 10x-100x more content than another, the smaller source becomes invisible in merged results. You must add explicit mechanisms to guarantee representation.

**Strategies**:
1. **Filtered queries**: Run a separate search restricted to the minority source
2. **Cross-reference injection**: When the majority source mentions the minority, look it up and inject
3. **Dedicated score slots**: Reserve result positions for each source

> **Example**: 82,653 BOFIP chunks vs 2,335 CGI/LPF chunks. CGI/LPF never appeared in top results naturally. Fixed with dedicated CGI/LPF-only vector search + cross-reference injection.

---

## 6. Testing Methodology

### Lesson 21: Vary Test Inputs Across Sources, Types, and Edge Cases

**Principle**: Testing only with the input you used during development gives false confidence. The fix might work for that specific case but fail on everything else.

**Anti-pattern**:
- Built feature with input X â†’ tested only with input X â†’ declared success

**Correct approach**: After any change, test with inputs that are DIFFERENT from development:
- **Different data sources**: If you changed source A handling, also verify sources B and C
- **Different query types**: Mix rate queries, definition queries, procedure queries
- **Different categories**: Don't test one domain â€” test across all domains
- **Edge cases**: Empty results, very long inputs, ambiguous terms, cross-source queries

**Rule**: If you debugged with input X from source Y, verify with inputs A, B, C from sources Y and Z.

### Lesson 22: Configuration Is Not Behavior Until Enforced in Code

**Principle**: Declaring a setting in `config.py` does nothing unless runtime code actually applies it. Treat every critical config as untrusted until validated in execution.

**Anti-pattern**:
- Define cache TTL in config
- Never check expiry when reading cache
- Assume freshness is guaranteed

**Correct approach**:
1. Read config at runtime where behavior happens
2. Enforce it in logic (e.g., cache expiry check before returning cached answer)
3. Add a direct test proving config changes behavior

**Rule**: For any reliability/safety config (TTL, limits, filtering), verify with a failing-then-passing runtime test.
### Lesson 23: Benchmark Retrieval Separately From Generation

**Principle**: End-to-end success can hide retrieval failures because a strong LLM may answer correctly from prior knowledge. Retrieval quality must be measured independently.

**Anti-pattern**:
- Only evaluate final answer text
- Assume retrieval is correct if answer looks correct

**Correct approach**:
1. Maintain retrieval ground truth (expected BOI/article references)
2. Measure Recall@K, Precision@K, HitRate without LLM
3. Compare with and without reranker to detect top-k regressions

**Rule**: Every major retrieval change must run a retrieval-only benchmark before end-to-end evaluation.

### Lesson 24: Keep Index Artifacts in Sync After Partial Rebuilds

**Principle**: If `chunks.json`, BM25 index, and vector DB are rebuilt at different times, the system can silently serve inconsistent results.

**Anti-pattern**:
- Update chunks but keep old BM25
- Interrupt vector rebuild and continue testing

**Correct approach**:
1. Verify counts on all three artifacts (`chunks.json`, BM25, Chroma)
2. Recover from interrupted vector builds before any benchmark
3. Add focused sync scripts for fast source-specific refreshes (e.g., legal-only)

**Rule**: Treat index-count mismatch as a blocker (P0), not a minor warning.

---
### Lesson 25: Daily Delta Feeds Are Not Full Corpora

**Principle**: Incremental legal data feeds only contain changed records. Treating a delta dump as a full source causes silent data loss.

**Anti-pattern**:
- Parse one daily LEGI archive and assume it contains all CGI/LPF articles
- Replace full legal corpus with that delta

**Correct approach**:
1. Build/refresh from a full snapshot first (e.g., `Freemium_legi_global_...`)
2. Apply daily deltas on top of that baseline
3. If a target source is absent in a daily delta, keep existing data for that source

**Rule**: Distinguish clearly between baseline snapshots and incremental updates in ingestion code.

### Lesson 26: Incremental Data Needs Incremental Index Sync

**Principle**: If upstream ingestion is incremental (daily deltas), downstream vector sync should also be incremental. Full refresh on tiny deltas wastes time and compute.

**Anti-pattern**:
- Apply a daily LEGI delta with 1-2 changed articles
- Delete/re-embed all legal vectors anyway

**Correct approach**:
1. Keep full-rebuild mode for baseline/full snapshot updates
2. Add delta sync mode that upserts only changed legal references
3. Keep count verification (`chunks.json`, BM25, Chroma) after each run

**Rule**: Match sync granularity to update granularity; full sync only when full data changes.

### Lesson 27: Small Benchmarks Can Invert Product Decisions

**Principle**: Tiny evaluation sets can produce the opposite conclusion of larger, more representative sets. Treat policy decisions (defaults, architecture) as invalid until tested on sufficient breadth.

**Anti-pattern**:
- Use a very small retrieval benchmark (e.g., ~10 questions) as the only decision basis
- Freeze defaults (e.g., disable reranker) from that sample alone

**Correct approach**:
1. Keep a small gold debug set for fast checks
2. Build a broader set (50+) from real usage traces, then curate toward gold
3. Re-run decision benchmarks (retrieval-only) before changing defaults

**Rule**: Never finalize retrieval policy on a tiny set; require a broader benchmark pass first.

### Lesson 28: Guardrails Need Their Own Token Budget

**Principle**: Adding a second LLM pass (verification/guardrail) can silently fail on context size even when primary generation works. Guardrails must be budgeted like first-class requests.

**Anti-pattern**:
- Send full retrieved context to a verifier model
- Treat verifier failures (413 payload/rate) as rare edge cases

**Correct approach**:
1. Enforce compact verifier context (max chunks + per-chunk truncation)
2. Add explicit fallback behavior if verifier call fails
3. Log verifier mode (`llm_verifier` vs fallback) for observability

**Rule**: Every additional LLM step must have explicit token controls and fallback strategy.

### Lesson 29: Embedding Caches Must Be Model-Scoped

**Principle**: Embeddings are model-dependent vectors. Reusing cache entries across models silently corrupts evaluation and retrieval quality.

**Anti-pattern**:
- Cache key based only on text (e.g., `md5(text)`)
- Switch embedding model (`e5-base` -> `e5-large`) without cache partition
- Accidentally mix vectors generated by different models

**Correct approach**:
1. Partition cache by model identifier (`data/cache/embeddings/<model_slug>/...`)
2. Use model-specific vector collections for A/B benchmarks
3. Ensure query embeddings and indexed chunk embeddings come from the same model

**Rule**: Include model identity in every embedding artifact boundary (cache, collection, benchmark output).

---
## Summary: Rules for Future Sessions

1. **Don't dismiss ideas from one failure** - Check if the concept failed or just the configuration
2. **Explain disagreements** - Always provide what, why, and alternatives
3. **Check infrastructure first** - Data mismatches, fallback logic, silent failures
4. **Make failures loud** - Every fallback must log what happened
5. **Trace intermediate results** - Find the exact step where data goes wrong
6. **Ensure diversity in aggregation** - Limit per-source, not just total
7. **Never hardcode domain data** - Fix retrieval, not prompts
8. **Verify simplicity first** - Only add complexity with measured improvement
9. **Use patterns, not mappings** - Solutions must generalize to any input
10. **Test tokenization with domain examples** - Default tokenizers break on domain terms
11. **Disable cache when testing** - Old cached answers hide fixed bugs
12. **Keep documentation updated** - After every significant change
13. **Inject secondary sources after scoring** - Not before normalization
14. **Validate metadata per source** - Same flag can mean different things
15. **Fix at the source** - Trace bad data to its origin
16. **Use target-language models** - Not English-only for non-English domains
17. **Guarantee minority source representation** - Explicit mechanisms required
18. **Vary test inputs** - Never test only with the input you developed with
19. **Enforce config at runtime** - A setting is useless until code applies it
20. **Benchmark retrieval alone** - Measure Recall@K/Precision@K independently from LLM answers
21. **Keep index artifacts synchronized** - `chunks.json`, BM25, and vector DB must match counts
22. **Never treat deltas as full data** - Apply daily legal updates only on top of a validated full baseline
23. **Keep sync incremental when updates are incremental** - Upsert changed references instead of full re-embedding
24. **Do not lock product policy on tiny benchmarks** - Re-validate decisions on a broader representative set
25. **Budget every LLM step** - Guardrails/validators need explicit token caps and fallback paths
26. **Scope embedding artifacts by model** - Never share embedding caches/collections across different models
