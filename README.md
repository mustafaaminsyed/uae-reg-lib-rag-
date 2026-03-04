# uae-reg-lib-rag

## Setup

Use Python `3.12` (recommended) or `3.13` for this project. Python `3.14` is not currently supported by the pinned `chromadb` dependency stack used here.

1. Create the virtual environment:

```powershell
python -m venv .venv
```

If you use `uv`, it should target Python `3.12` explicitly:

```powershell
uv venv --python 3.12 .venv
```

2. Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

The dependency versions are pinned so the project behavior is more reproducible across machines.

## Usage

1. Add your source files under:
- `docs_raw/uae_vat/`
- `docs_raw/uae_einvoicing/`
- `docs_raw/uae_pint/`

You can place PDFs directly under these folders. You can also place a PINT-AE resource `.zip` under `docs_raw/uae_pint/`. The ingester will index supported files inside the archive, including embedded PDFs, XML examples, schematron files, and code lists.

2. Run ingestion:

```powershell
.\.venv\Scripts\python.exe src/ingest.py
```

This creates:
- `docs_processed/` with extracted and normalized JSON
- `index_store/` with the persistent ChromaDB index

The ingester now preserves both:
- normalized page text (`text`)
- line-preserved page text (`text_line_preserved`) for PDFs

For `e_invoicing` documents, ingestion also adds an extra `line_preserved` chunk variant with smaller chunk windows to keep table/list rows from being split as aggressively. New chunk metadata includes `text_variant` (`normalized` or `line_preserved`).

These generated folders are now allowed in Git so you can optionally back them up to GitHub for portability. If they are already committed, another machine can clone the repo and use the current index directly. If the source corpus changes, re-run ingestion to refresh them.

If you already indexed e-invoicing PDFs before this update, re-run ingestion once so the new line-preserved chunks are added to the index. This change is additive and backward-compatible: older processed files still work, but they do not benefit from the improved table-preserving chunking until you rebuild the index.

3. Run the question interface in interactive mode:

```powershell
.\.venv\Scripts\python.exe src/ask.py
```

4. Run a single question and exit:

```powershell
.\.venv\Scripts\python.exe src/ask.py "What are mandatory fields for UAE electronic invoice?"
```

5. Run the local web UI:

```powershell
.\.venv\Scripts\python.exe -m src.serve_ui
```

Then open `http://127.0.0.1:8000` in your browser.

## Query Modes

- Default mode returns an answer first, then citations used.
- `--retrieval-only` skips answer synthesis and returns retrieval output only.
- `--show-matches` shows the retrieved match previews for debugging.
- `--no-rich` forces plain-text output.
- `src.serve_ui` provides a browser-based local UI for the same local index.

## Answer Modes

- General questions still use the existing citation-grounded answer synthesis path.
- List/table-heavy prompts now use an extraction-first path that attempts to build bullet lists directly from retrieved evidence.
- Each extracted list item is grounded to at least one retrieved chunk using `[doc_title, p. X, chunk Y]`.
- If the retrieved chunks do not contain enough clean list/table evidence, the system now responds with `Not found in retrieved evidence.` instead of inventing or over-compressing list content.

This is especially relevant for prompts such as:
- `What information must appear on a UAE tax invoice?`
- `What fields are required for ... ?`
- `What must include ... ?`

## Useful Commands

```powershell
.\.venv\Scripts\python.exe src/ask.py --top-k 3 --topic uae_einvoicing "What are mandatory fields for UAE electronic invoice?"
.\.venv\Scripts\python.exe src/ask.py --retrieval-only "What are VAT registration thresholds?"
.\.venv\Scripts\python.exe src/ask.py --no-rich "What are mandatory fields for UAE electronic invoice?"
.\.venv\Scripts\python.exe src/ask.py --show-matches --no-rich "What are mandatory fields for UAE electronic invoice?"
.\.venv\Scripts\python.exe src/ask.py --doc-family pint_ae "What does PINT-AE require for invoice line tax fields?"
.\.venv\Scripts\python.exe src/ask.py "What does BTAE-23 mean in PINT-AE?"
.\.venv\Scripts\python.exe src/ask.py "What are the allowed codes in the PINT-AE transaction type code list?"
.\.venv\Scripts\python.exe src/ask.py "What does rule ibr-139-ae say in PINT-AE?"
```

## Evaluation

Use `src/run_eval.py` to measure retrieval and grounding quality after pipeline changes. It is an evaluation harness, not a training loop.

Useful commands:

```powershell
.\.venv\Scripts\python.exe -m src.run_eval --small-test
.\.venv\Scripts\python.exe -m src.run_eval --topics uae_vat --max-cases 8
.\.venv\Scripts\python.exe -m src.run_eval --question-ids vat_registration_thresholds,vat_threshold_calculation
.\.venv\Scripts\python.exe -m src.run_eval --resume-from reports\eval_YYYYMMDD_HHMMSS.jsonl
```

The evaluator now supports case filters, resume-from-report, incremental markdown summaries, a sidecar `.meta.json` run log, and per-config retrieval metrics such as `doc_hit@k`, `page_hit@k`, `citation_hit@k`, citation precision, canonical-source preference, and `not_stated` match rate.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the full workflow and metric definitions.

## Current Capabilities

- Local ingestion of UAE VAT, UAE e-invoicing, and UAE PINT source materials.
- Persistent ChromaDB indexing under `index_store/`.
- Citation-first answers for internal research.
- Extraction-first list/table answers for list-heavy regulatory prompts.
- Retrieval-only validation mode for debugging.
- Structured handling for some high-risk regulatory question classes, including:
- `TIN` / `TRN` / Participant Identifier interpretation questions
- Tax Group / VAT Group identifier questions
- UAE electronic invoice mandatory field counts and field lists
- PINT-AE business term, codelist, and selected schematron rule lookups

## Portability

- The repo now pins exact package versions in `requirements.txt`.
- The generated `docs_processed/` and `index_store/` folders can be committed to GitHub as a backup snapshot.
- This makes it easier to open the project on another machine with the same source corpus, processed data, and current index.
- If you add or change source documents, run ingestion again and commit the refreshed generated folders if you want GitHub to reflect the latest snapshot.
- Future ingestion remains non-disruptive after the text-variant update because the richer processed schema is additive. New documents can be ingested normally; a rebuild is only recommended when you want older indexed PDFs to benefit from the new line-preserved chunks.

## Answering Rules

All answers must include citations in the format `document name + page number`.

For higher-risk interpretive regulatory questions, the preferred response structure is:

```text
Answer:
Regulatory basis:
Explicitly stated:
Inferred:
Not stated:
```

Retrieval output must always show `doc_title`, `page number`, and `source_path`.

## Disclaimer

This system is for internal research only and is not legal advice. Verify important conclusions against primary sources.
