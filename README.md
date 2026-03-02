# uae-reg-lib-rag

## Setup

1. Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

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

3. Run the question interface in interactive mode:

```powershell
.\.venv\Scripts\python.exe src/ask.py
```

4. Run a single question and exit:

```powershell
.\.venv\Scripts\python.exe src/ask.py "What are mandatory fields for UAE electronic invoice?"
```

## Query Modes

- Default mode returns an answer first, then citations used.
- `--retrieval-only` skips answer synthesis and returns retrieval output only.
- `--show-matches` shows the retrieved match previews for debugging.
- `--no-rich` forces plain-text output.

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

## Current Capabilities

- Local ingestion of UAE VAT, UAE e-invoicing, and UAE PINT source materials.
- Persistent ChromaDB indexing under `index_store/`.
- Citation-first answers for internal research.
- Retrieval-only validation mode for debugging.
- Structured handling for some high-risk regulatory question classes, including:
- `TIN` / `TRN` / Participant Identifier interpretation questions
- Tax Group / VAT Group identifier questions
- UAE electronic invoice mandatory field counts and field lists
- PINT-AE business term, codelist, and selected schematron rule lookups

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
