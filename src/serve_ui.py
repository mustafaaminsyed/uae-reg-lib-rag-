from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from . import ask
except ImportError:
    import ask


BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"
DARIBA_LOGO_PATH = BASE_DIR / "Dariba Technologies LLC Logo.png"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

_collection_lock = threading.Lock()
_collection: Any | None = None


def stable_chunk_id(metadata: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(metadata.get("source_path", "")),
            str(metadata.get("doc_title", "")),
            str(metadata.get("page", "")),
            str(metadata.get("chunk", "")),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"chunk-{digest}"


def score_from_distance(distance: Any) -> float | None:
    if not isinstance(distance, (int, float)):
        return None
    bounded = max(0.0, min(1.0, 1.0 - min(float(distance), 1.0)))
    return round(bounded, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local web UI for the UAE RAG project.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST}).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT}).")
    return parser.parse_args()


def load_collection_once() -> Any:
    global _collection
    if _collection is not None:
        return _collection

    with _collection_lock:
        if _collection is None:
            collection = ask.load_collection(console=None)
            if collection is None:
                raise RuntimeError("Unable to load the Chroma collection. Run ingestion first.")
            _collection = collection
    return _collection


def serialize_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, match in enumerate(matches, start=1):
        metadata = dict(match.get("metadata") or {})
        document_text = str(match.get("document", ""))
        chunk_id = stable_chunk_id(metadata)
        serialized.append(
            {
                "rank": index,
                "chunk_id": chunk_id,
                "doc": metadata.get("doc_title", "Unknown document"),
                "doc_title": metadata.get("doc_title", "Unknown document"),
                "page": metadata.get("page", "n/a"),
                "chunk": metadata.get("chunk", "n/a"),
                "source_path": metadata.get("source_path", ""),
                "distance": match.get("distance"),
                "score": score_from_distance(match.get("distance")),
                "snippet": ask.make_snippet(document_text, max_length=220),
                "text": document_text,
            }
        )
    return serialized


def split_answer_sentences(answer_text: str) -> list[str]:
    answer = str(answer_text or "").strip()
    if not answer:
        return []

    paragraphs = [part.strip() for part in answer.split("\n\n") if part.strip()]
    sentence_groups: list[list[str]] = []
    for paragraph in paragraphs:
        sentences = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", paragraph)
            if segment.strip()
        ]
        sentence_groups.append(sentences or [paragraph])

    return [sentence for group in sentence_groups for sentence in group]


def citation_relevance_score(answer_text: str, citation: dict[str, Any]) -> int:
    answer_words = {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(answer_text).lower())
        if len(token) > 2
    }
    quote_text = " ".join(
        [
            str(citation.get("quote", "")),
            str(citation.get("snippet", "")),
            str(citation.get("doc_title", "")),
        ]
    ).lower()
    quote_words = {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", quote_text)
        if len(token) > 2
    }

    overlap = len(answer_words & quote_words)
    score = overlap

    normalized_answer = " ".join(str(answer_text).lower().split())
    normalized_quote = " ".join(str(citation.get("quote", "")).lower().split())
    if normalized_answer and normalized_quote:
        if normalized_answer in normalized_quote:
            score += 8
        elif normalized_quote in normalized_answer:
            score += 6

    if isinstance(citation.get("score"), (int, float)):
        score += int(float(citation["score"]) * 10)

    return score


def prune_ui_citations(answer_payload: dict[str, Any], citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(citations) <= 2:
        return citations

    answer_text = str(answer_payload.get("answer", "")).strip()
    sentences = split_answer_sentences(answer_text)
    if len(sentences) != 1:
        return citations

    target_sentence = sentences[0]
    ranked = sorted(
        citations,
        key=lambda entry: (
            citation_relevance_score(target_sentence, entry),
            entry.get("score", -1.0) if isinstance(entry.get("score"), (int, float)) else -1.0,
        ),
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    for entry in ranked:
        score = citation_relevance_score(target_sentence, entry)
        if score <= 0 and kept:
            continue
        kept.append(entry)
        if len(kept) >= 2:
            break

    if not kept:
        kept = [ranked[0]]

    kept.sort(key=lambda entry: int(entry.get("citation_number", 0) or 0))
    for index, entry in enumerate(kept, start=1):
        entry["citation_number"] = index
    return kept


def build_ui_citations(
    answer_payload: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    basis = list(answer_payload.get("regulatory_basis") or [])

    for index, entry in enumerate(basis, start=1):
        doc_title = str(entry.get("doc", "Unknown document"))
        page = entry.get("page", "n/a")
        matching_evidence = [
            item
            for item in evidence_items
            if str(item.get("doc_title", "")) == doc_title and str(item.get("page", "n/a")) == str(page)
        ]
        best_match = max(
            matching_evidence,
            key=lambda item: item.get("score", -1.0) if isinstance(item.get("score"), (int, float)) else -1.0,
            default=None,
        )
        snippet_source = str(entry.get("quote", "")).strip() or (best_match or {}).get("snippet", "")

        citations.append(
            {
                "citation_number": index,
                "chunk_id": (best_match or {}).get("chunk_id", ""),
                "doc_title": doc_title,
                "page": page,
                "score": (best_match or {}).get("score"),
                "snippet": ask.make_snippet(snippet_source, max_length=180),
                "quote": str(entry.get("quote", "")).strip(),
            }
        )

    return citations


def split_answer_segments(answer_text: str) -> list[str]:
    answer = str(answer_text or "").strip()
    if not answer:
        return []

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    sentences = split_answer_sentences(answer)
    if sentences:
        return sentences

    return [answer]


def build_answer_segments(
    answer_payload: dict[str, Any],
    citations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int]]:
    segments_text = split_answer_segments(answer_payload.get("answer", ""))
    if not segments_text:
        return [], set()

    segments: list[dict[str, Any]] = []
    used_numbers: set[int] = set()

    for segment_text in segments_text:
        ranked = sorted(
            citations,
            key=lambda entry: (
                citation_relevance_score(segment_text, entry),
                entry.get("score", -1.0) if isinstance(entry.get("score"), (int, float)) else -1.0,
            ),
            reverse=True,
        )

        max_citations = 2 if len(segments_text) == 1 else 1
        selected_numbers: list[int] = []

        for entry in ranked:
            score = citation_relevance_score(segment_text, entry)
            citation_number = int(entry.get("citation_number", 0) or 0)
            if citation_number <= 0:
                continue
            if score <= 0 and selected_numbers:
                continue
            selected_numbers.append(citation_number)
            if len(selected_numbers) >= max_citations:
                break

        if not selected_numbers and ranked:
            fallback_number = int(ranked[0].get("citation_number", 0) or 0)
            if fallback_number > 0:
                selected_numbers = [fallback_number]

        used_numbers.update(selected_numbers)
        segments.append(
            {
                "text": segment_text,
                "citation_numbers": selected_numbers,
            }
        )

    return segments, used_numbers


def reindex_citations(
    citations: list[dict[str, Any]],
    answer_segments: list[dict[str, Any]],
    used_numbers: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not citations:
        return [], answer_segments

    if not used_numbers:
        used_numbers = {int(citations[0].get("citation_number", 0) or 1)}

    kept = [entry.copy() for entry in citations if int(entry.get("citation_number", 0) or 0) in used_numbers]
    kept.sort(key=lambda entry: int(entry.get("citation_number", 0) or 0))

    old_to_new: dict[int, int] = {}
    for new_number, entry in enumerate(kept, start=1):
        old_number = int(entry.get("citation_number", 0) or 0)
        old_to_new[old_number] = new_number
        entry["citation_number"] = new_number

    remapped_segments: list[dict[str, Any]] = []
    for segment in answer_segments:
        remapped_numbers = sorted({
            old_to_new[number]
            for number in segment.get("citation_numbers", [])
            if number in old_to_new
        })
        remapped_segments.append(
            {
                "text": segment.get("text", ""),
                "citation_numbers": remapped_numbers,
            }
        )

    return kept, remapped_segments


def build_answer_markdown_from_segments(answer_segments: list[dict[str, Any]]) -> str:
    if not answer_segments:
        return ""

    lines: list[str] = []
    for segment in answer_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        markers = " ".join(f"[{number}]" for number in segment.get("citation_numbers", []))
        lines.append(f"{text} {markers}".strip())
    return "\n".join(lines).strip()


def build_answer_markdown(answer_payload: dict[str, Any], citations: list[dict[str, Any]]) -> str:
    answer = str(answer_payload.get("answer", "")).strip()
    if not citations:
        return answer
    if not answer:
        return " ".join(f"[{entry['citation_number']}]" for entry in citations)

    paragraphs = [part.strip() for part in answer.split("\n\n") if part.strip()]
    sentence_groups: list[list[str]] = []
    for paragraph in paragraphs:
        sentences = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", paragraph)
            if segment.strip()
        ]
        sentence_groups.append(sentences or [paragraph])

    flattened_sentences = [sentence for group in sentence_groups for sentence in group]
    if not flattened_sentences:
        return answer
    if len(flattened_sentences) == 1:
        marker_text = " ".join(f"[{entry['citation_number']}]" for entry in citations)
        return f"{flattened_sentences[0]} {marker_text}".strip()

    sentence_markers: list[list[str]] = [[] for _ in flattened_sentences]
    for index, entry in enumerate(citations):
        target_index = min(index, len(flattened_sentences) - 1)
        sentence_markers[target_index].append(f"[{entry['citation_number']}]")

    rebuilt_paragraphs: list[str] = []
    sentence_index = 0
    for group in sentence_groups:
        rebuilt_sentences: list[str] = []
        for sentence in group:
            markers = sentence_markers[sentence_index]
            suffix = f" {' '.join(markers)}" if markers else ""
            rebuilt_sentences.append(f"{sentence}{suffix}")
            sentence_index += 1
        rebuilt_paragraphs.append(" ".join(rebuilt_sentences))

    return "\n\n".join(rebuilt_paragraphs)


def build_ui_evidence(serialized_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_items: list[dict[str, Any]] = []
    for item in serialized_matches:
        evidence_items.append(
            {
                "chunk_id": item["chunk_id"],
                "doc_title": item["doc_title"],
                "page": item["page"],
                "score": item["score"],
                "text": item["text"],
                "snippet": item["snippet"],
                "distance": item["distance"],
                "chunk": item["chunk"],
                "rank": item["rank"],
                "source_path": item["source_path"],
            }
        )
    return evidence_items


class UIRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def _serve_dariba_logo(self, include_body: bool) -> bool:
        if self.path != "/dariba-logo.png":
            return False
        if not DARIBA_LOGO_PATH.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Logo not found")
            return True
        data = DARIBA_LOGO_PATH.read_bytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", self.guess_type(str(DARIBA_LOGO_PATH)))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)
        return True

    def do_HEAD(self) -> None:
        if self._serve_dariba_logo(include_body=False):
            return
        return super().do_HEAD()

    def do_GET(self) -> None:
        if self._serve_dariba_logo(include_body=True):
            return
        if self.path in {"", "/"}:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/query":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Request body must be valid JSON."},
            )
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Question is required."})
            return

        top_k = max(1, min(int(payload.get("top_k", 3) or 3), ask.MAX_TOP_K))
        min_citations = max(1, int(payload.get("min_citations", 1) or 1))
        topic = str(payload.get("topic", "")).strip()
        doc_family = str(payload.get("doc_family", "")).strip()
        reranker_enabled = bool(payload.get("reranker_enabled", True))

        try:
            collection = load_collection_once()
            result = ask.build_query_result(
                collection,
                question=question,
                top_k=top_k,
                topic=topic,
                doc_family=doc_family,
                min_citations=min_citations,
                reranker_enabled=reranker_enabled,
            )
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Query execution failed: {exc}"},
            )
            return

        serialized_matches = serialize_matches(result["matches"])
        evidence_items = build_ui_evidence(serialized_matches)
        citation_items = build_ui_citations(result["answer_json"], evidence_items)
        answer_segments, used_numbers = build_answer_segments(result["answer_json"], citation_items)
        citation_items, answer_segments = reindex_citations(citation_items, answer_segments, used_numbers)

        response = {
            "question": result["question"],
            "effective_topic": result["effective_topic"],
            "effective_doc_family": result["effective_doc_family"],
            "request_config": {
                "top_k": top_k,
                "min_citations": min_citations,
                "reranker_enabled": reranker_enabled,
                "topic": topic,
                "doc_family": doc_family,
            },
            "embedding_model": ask.EMBEDDING_MODEL,
            "answer_json": result["answer_json"],
            "answer_segments": answer_segments,
            "answer_markdown": build_answer_markdown_from_segments(answer_segments)
            or build_answer_markdown(result["answer_json"], citation_items),
            "formatted_answer": ask.format_answer_payload(result["answer_json"]),
            "citations": citation_items,
            "citation_labels": ask.citations_from_payload(result["answer_json"]),
            "matches": serialized_matches,
            "evidence": evidence_items,
            "validation_reasons": result["validation_reasons"],
            "error": result["error"],
            "timings_ms": result.get("timings_ms", {}),
        }
        self.send_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ask.suppress_noisy_startup()
    args = parse_args()
    if not UI_DIR.exists():
        raise FileNotFoundError(f"UI directory not found: {UI_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), UIRequestHandler)
    print(f"Serving RAG UI at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
