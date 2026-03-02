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

1. Add your source PDF files under:
   - `docs_raw/uae_vat/`
   - `docs_raw/uae_einvoicing/`
   - `docs_raw/uae_pint/`

You can also place a PINT-AE resource `.zip` under `docs_raw/uae_pint/`. The ingester will index supported files inside the archive, including embedded PDFs, XML examples, schematron files, and code lists.

2. Run ingestion:

```powershell
python src/ingest.py
```

3. Run the question-answering interface:

```powershell
python src/ask.py
```

Single-question mode:

```powershell
python src/ask.py "What are mandatory fields for UAE electronic invoice?"
```

Useful options:

```powershell
python src/ask.py --top-k 3 --topic uae_einvoicing "What are mandatory fields for UAE electronic invoice?"
python src/ask.py --retrieval-only "What are VAT registration thresholds?"
python src/ask.py --no-rich "What are mandatory fields for UAE electronic invoice?"
python src/ask.py --no-rich "What does PINT-AE require for invoice line tax fields?"
python src/ask.py --doc-family pint_ae "What does PINT-AE require for invoice line tax fields?"
python src/ask.py "What does BTAE-23 mean in PINT-AE?"
python src/ask.py "What are the allowed codes in the PINT-AE transaction type code list?"
python src/ask.py "What does rule ibr-139-ae say in PINT-AE?"
```

## Answering Rule

All answers must include citations in the format: document name + page number.
Retrieval output must always show `doc_title`, `page number`, and `source_path`.

## Disclaimer

This system is for internal research only and is not legal advice. Verify important conclusions against primary sources.
