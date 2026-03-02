from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable
import zipfile

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_RAW_DIR = BASE_DIR / "docs_raw"
DOCS_PROCESSED_DIR = BASE_DIR / "docs_processed"
INDEX_STORE_DIR = BASE_DIR / "index_store"
COLLECTION_NAME = "uae_reg_library"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
LOW_TEXT_PAGE_CHAR_THRESHOLD = 40
MOSTLY_EMPTY_PAGE_RATIO = 0.8
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUPPORTED_ARCHIVE_TEXT_SUFFIXES = {".xml", ".sch", ".xslt", ".gc", ".txt", ".md", ".json"}


def find_pdf_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def find_zip_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".zip")


def normalize_text(text: str) -> str:
    # Collapse whitespace while preserving readable paragraph spacing.
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def get_topic_name(pdf_path: Path) -> str:
    relative_path = pdf_path.relative_to(DOCS_RAW_DIR)
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "unclassified"


def build_processed_output_path(pdf_path: Path) -> Path:
    relative_path = pdf_path.relative_to(DOCS_RAW_DIR).with_suffix(".json")
    return DOCS_PROCESSED_DIR / relative_path


def build_archive_processed_output_path(archive_path: Path, entry_name: str) -> Path:
    archive_relative = archive_path.relative_to(DOCS_RAW_DIR).with_suffix("")
    entry_relative = Path(*Path(entry_name).parts).with_suffix(".json")
    return DOCS_PROCESSED_DIR / archive_relative / entry_relative


def write_processed_json(pdf_path: Path, topic: str, pages: list[dict[str, object]]) -> None:
    output_path = build_processed_output_path(pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_path": pdf_path.relative_to(BASE_DIR).as_posix(),
        "topic": topic,
        "doc_title": pdf_path.stem,
        "pages": pages,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_processed_json_for_source(
    output_path: Path,
    source_path: str,
    topic: str,
    doc_title: str,
    pages: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_path": source_path,
        "topic": topic,
        "doc_title": doc_title,
        "pages": pages,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_pdf_pages(pdf_path: Path) -> tuple[list[dict[str, object]], bool]:
    reader = PdfReader(str(pdf_path))

    pages: list[dict[str, object]] = []
    low_text_pages = 0

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        normalized = normalize_text(raw_text)
        if len(normalized) < LOW_TEXT_PAGE_CHAR_THRESHOLD:
            low_text_pages += 1
        pages.append({"page_num": page_index, "text": normalized})

    page_count = len(pages)
    mostly_empty = page_count > 0 and (low_text_pages / page_count) >= MOSTLY_EMPTY_PAGE_RATIO
    return pages, mostly_empty


def extract_pdf_pages_from_bytes(pdf_bytes: bytes) -> tuple[list[dict[str, object]], bool]:
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages: list[dict[str, object]] = []
    low_text_pages = 0

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        normalized = normalize_text(raw_text)
        if len(normalized) < LOW_TEXT_PAGE_CHAR_THRESHOLD:
            low_text_pages += 1
        pages.append({"page_num": page_index, "text": normalized})

    page_count = len(pages)
    mostly_empty = page_count > 0 and (low_text_pages / page_count) >= MOSTLY_EMPTY_PAGE_RATIO
    return pages, mostly_empty


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text_pages_from_bytes(data: bytes) -> tuple[list[dict[str, object]], bool]:
    normalized = normalize_text(decode_text_bytes(data))
    pages = [{"page_num": 1, "text": normalized}]
    mostly_empty = len(normalized) < LOW_TEXT_PAGE_CHAR_THRESHOLD
    return pages, mostly_empty


def classify_doc_family(topic: str, source_path: str, doc_title: str) -> str:
    haystack = f"{topic} {source_path} {doc_title}".lower()
    if "pint" in haystack or "peppol" in haystack:
        return "pint_ae"
    if "vat" in haystack:
        return "vat"
    if "einvoic" in haystack or "invoice" in haystack:
        return "e_invoicing"
    return "general"


def iter_chunks(
    pdf_path: Path,
    topic: str,
    doc_title: str,
    pages: Iterable[dict[str, object]],
) -> Iterable[tuple[str, dict[str, object], str]]:
    source_path = pdf_path.relative_to(BASE_DIR).as_posix()

    for page in pages:
        page_num = int(page["page_num"])
        text = str(page["text"])
        for chunk_index, chunk in enumerate(chunk_text(text), start=1):
            metadata = {
                "source_path": source_path,
                "topic": topic,
                "page": page_num,
                "chunk": chunk_index,
                "doc_title": doc_title,
            }
            chunk_id = make_chunk_id(source_path, page_num, chunk_index, chunk)
            yield chunk, metadata, chunk_id


def iter_chunks_for_source(
    source_path: str,
    topic: str,
    doc_title: str,
    pages: Iterable[dict[str, object]],
    source_type: str,
) -> Iterable[tuple[str, dict[str, object], str]]:
    doc_family = classify_doc_family(topic, source_path, doc_title)

    for page in pages:
        page_num = int(page["page_num"])
        text = str(page["text"])
        for chunk_index, chunk in enumerate(chunk_text(text), start=1):
            metadata = {
                "source_path": source_path,
                "topic": topic,
                "page": page_num,
                "chunk": chunk_index,
                "doc_title": doc_title,
                "doc_family": doc_family,
                "source_type": source_type,
            }
            chunk_id = make_chunk_id(source_path, page_num, chunk_index, chunk)
            yield chunk, metadata, chunk_id


def make_chunk_id(source_path: str, page_num: int, chunk_index: int, chunk: str) -> str:
    digest = hashlib.sha1(
        f"{source_path}|{page_num}|{chunk_index}|{chunk}".encode("utf-8")
    ).hexdigest()
    return digest


def build_collection() -> Any:
    INDEX_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_STORE_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Ignore missing collection or incompatible existing state; the new one is created below.
        pass

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "UAE regulatory document library"},
    )


def index_chunks(collection: Any, chunks: list[tuple[str, dict[str, object], str]]) -> int:
    if not chunks:
        return 0

    documents = [item[0] for item in chunks]
    metadatas = [item[1] for item in chunks]
    ids = [item[2] for item in chunks]

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return len(chunks)


def main() -> None:
    DOCS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = find_pdf_files(DOCS_RAW_DIR)
    zip_files = find_zip_files(DOCS_RAW_DIR)
    print(f"PDFs found: {len(pdf_files)}")
    print(f"Zip archives found: {len(zip_files)}")

    if not pdf_files and not zip_files:
        print("No supported source files found under docs_raw.")
        return

    collection = build_collection()
    all_chunks: list[tuple[str, dict[str, object], str]] = []
    scanned_warnings: list[str] = []

    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs", unit="pdf"):
        try:
            pages, mostly_empty = extract_pdf_pages(pdf_path)
        except Exception as exc:
            print(f"Failed to process {pdf_path}: {exc}")
            continue

        topic = get_topic_name(pdf_path)
        write_processed_json(pdf_path, topic, pages)

        if mostly_empty:
            scanned_warnings.append(
                f"Warning: {pdf_path.relative_to(BASE_DIR).as_posix()} yielded mostly empty text and may be scanned."
            )

        source_path = pdf_path.relative_to(BASE_DIR).as_posix()
        all_chunks.extend(
            iter_chunks_for_source(
                source_path=source_path,
                topic=topic,
                doc_title=pdf_path.stem,
                pages=pages,
                source_type="pdf",
            )
        )

    for zip_path in tqdm(zip_files, desc="Ingesting archives", unit="zip"):
        topic = get_topic_name(zip_path)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                entry_names = sorted(
                    name
                    for name in archive.namelist()
                    if not name.endswith("/")
                    and (
                        Path(name).suffix.lower() == ".pdf"
                        or Path(name).suffix.lower() in SUPPORTED_ARCHIVE_TEXT_SUFFIXES
                    )
                )

                for entry_name in entry_names:
                    try:
                        data = archive.read(entry_name)
                        suffix = Path(entry_name).suffix.lower()
                        if suffix == ".pdf":
                            pages, mostly_empty = extract_pdf_pages_from_bytes(data)
                        else:
                            pages, mostly_empty = extract_text_pages_from_bytes(data)
                    except Exception as exc:
                        print(f"Failed to process {zip_path.relative_to(BASE_DIR).as_posix()}::{entry_name}: {exc}")
                        continue

                    archive_source = f"{zip_path.relative_to(BASE_DIR).as_posix()}::{entry_name}"
                    doc_title = Path(entry_name).stem
                    output_path = build_archive_processed_output_path(zip_path, entry_name)
                    write_processed_json_for_source(
                        output_path=output_path,
                        source_path=archive_source,
                        topic=topic,
                        doc_title=doc_title,
                        pages=pages,
                    )

                    if mostly_empty:
                        scanned_warnings.append(
                            f"Warning: {archive_source} yielded mostly empty text and may be scanned or non-text."
                        )

                    all_chunks.extend(
                        iter_chunks_for_source(
                            source_path=archive_source,
                            topic=topic,
                            doc_title=doc_title,
                            pages=pages,
                            source_type="archive",
                        )
                    )
        except Exception as exc:
            print(f"Failed to process archive {zip_path}: {exc}")

    indexed_count = index_chunks(collection, all_chunks)
    print(f"Chunks indexed: {indexed_count}")

    for warning in scanned_warnings:
        print(warning)


if __name__ == "__main__":
    main()
