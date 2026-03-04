# Evaluation Guide

## What Eval Is For

`src/run_eval.py` measures retrieval and grounding quality. It does not train the RAG pipeline by itself.

Use eval runs to:

- catch regressions after ingestion, chunking, retrieval, or answer-logic changes
- compare retrieval settings on the same question slice
- verify that answers stay citation-grounded

Do not use larger eval sets as a substitute for improving the corpus, metadata, chunking, or routing logic.

## Run Strategy

Use the smallest run that answers the question you are testing:

- Smoke test: `--small-test`
- Topic slice: `--topics uae_vat` or `--doc-families vat`
- Specific cases: `--question-ids vat_registration_thresholds,vat_threshold_calculation`
- Full sweep: only after a meaningful pipeline change

Resume interrupted runs instead of restarting:

```powershell
.\.venv\Scripts\python.exe -m src.run_eval --resume-from reports\eval_YYYYMMDD_HHMMSS.jsonl
```

The runner now writes markdown summaries incrementally, so partial runs still leave a readable report.
It also writes a sidecar metadata file next to the report: `eval_YYYYMMDD_HHMMSS.meta.json`.

## Eval Case Fields

You can enrich `eval/golden_questions.jsonl` with optional expectation fields:

- `expected_docs`: documents the final answer should cite
- `preferred_docs`: documents that should outrank secondary guidance when both appear
- `expected_pages`: pages the answer should reach
- `expected_citations`: exact `doc` + `page` pairs the answer should reach

`expected_citations` can be written as objects:

```json
{"doc": "Document Title", "page": 12}
```

## New Metrics

Each eval row now records timing and retrieval-oriented metrics:

- `timings_ms.retrieve`: retrieval time for the query
- `timings_ms.answer`: answer synthesis and guardrail time
- `timings_ms.total`: total time for that case
- `metrics.expected_doc_hit_at_k`: whether any expected document was retrieved
- `metrics.expected_doc_recall_at_k`: fraction of expected documents retrieved
- `metrics.expected_page_hit_at_k`: whether any expected page was retrieved
- `metrics.expected_citation_hit_at_k`: whether any expected citation pair was retrieved
- `metrics.answer_citation_precision`: share of answer citations supported by retrieved context
- `metrics.retrieval_canonical_preference`: whether preferred or more canonical sources outranked weaker sources in retrieval
- `metrics.answer_canonical_preference`: whether the final cited answer basis keeps preferred or more canonical sources first
- `metrics.expected_not_stated_match`: whether the `not_stated` flag matches the expectation when present

Markdown summaries aggregate these by config:

- pass rate
- average retrieve, answer, and total latency
- `doc_hit@k`
- `page_hit@k`
- `citation_hit@k`
- `doc_recall@k`
- citation precision
- retrieval canonical preference
- answer canonical preference
- `not_stated` match rate

## Recommended Workflow

1. Change one layer at a time: corpus, metadata, chunking, retrieval, reranking, or answer logic.
2. Run a targeted eval slice that covers the affected behavior.
3. Check retrieval metrics first, then answer-level pass rate.
4. Only run the full matrix after the targeted slice looks stable.
5. Promote new questions into the eval set only when they cover a real failure mode or edge case.

## Practical Commands

```powershell
.\.venv\Scripts\python.exe -m src.run_eval --small-test
.\.venv\Scripts\python.exe -m src.run_eval --topics uae_vat --max-cases 8
.\.venv\Scripts\python.exe -m src.run_eval --question-ids vat_registration_thresholds,vat_threshold_calculation
.\.venv\Scripts\python.exe -m src.run_eval --doc-families pint_ae --top-k-values 3 --min-citation-values 1
```
