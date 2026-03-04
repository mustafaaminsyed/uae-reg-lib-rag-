from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from . import ask
except ImportError:
    import ask


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = BASE_DIR / "eval" / "golden_questions.jsonl"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_TOP_K_VALUES = [3, 5]
DEFAULT_MIN_CITATION_VALUES = [1, 2]
SMALL_TEST_CONFIG_COUNT = 2
SMALL_TEST_QUESTION_COUNT = 5
DEFAULT_SUMMARY_EVERY = 10
PRIMARY_SOURCE_HINTS = (
    "federal decree",
    "decree by law",
    "federal law",
    "law no.",
    "cabinet decision",
    "ministerial decision",
    "executive regulation",
    "resolution",
)
AUTHORITATIVE_SPEC_HINTS = (
    "mandatory-fields",
    "mandatory fields",
    "framework",
    "specification",
    "bis",
    "syntax",
    "aligned rules",
)
SECONDARY_SOURCE_HINTS = (
    "guidelines",
    "guide",
    "handbook",
    "consultation",
    "release notes",
    "specialized-release-notes",
    "alert_",
    "compliance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluations over the UAE RAG pipeline.")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help=f"JSONL file containing evaluation prompts (default: {DEFAULT_EVAL_FILE}).",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="3,5",
        help="Comma-separated top_k values for the retrieval grid.",
    )
    parser.add_argument(
        "--min-citation-values",
        type=str,
        default="1,2",
        help="Comma-separated minimum citation counts for the guardrail grid.",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default="",
        help="Optional comma-separated case IDs to run.",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="",
        help="Optional comma-separated topic filters (for example: uae_vat,uae_pint).",
    )
    parser.add_argument(
        "--doc-families",
        type=str,
        default="",
        help="Optional comma-separated doc_family filters (for example: vat,pint_ae).",
    )
    parser.add_argument(
        "--small-test",
        action="store_true",
        help="Run 5 questions x 2 configs for a quick smoke test.",
    )
    parser.add_argument(
        "--repeat-cases",
        type=int,
        default=1,
        help="Repeat the loaded evaluation cases in memory for load testing (default: 1).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on the number of expanded cases to evaluate after repeats (default: 0, no cap).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from an existing JSONL report file and skip completed config/case pairs.",
    )
    parser.add_argument(
        "--save-summary-every",
        type=int,
        default=DEFAULT_SUMMARY_EVERY,
        help="Rewrite the markdown summary after every N new rows (default: 10, 0 disables periodic saves).",
    )
    return parser.parse_args()


def parse_int_list(value: str, fallback: list[int]) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(max(1, int(stripped)))
        except ValueError:
            continue
    return values or fallback


def parse_str_list(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def average_optional(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def normalize_page_token(page: Any) -> str:
    return str(ask.normalize_page_value(page))


def normalize_expected_pages(raw_pages: Any) -> set[str]:
    if not isinstance(raw_pages, list):
        return set()
    return {normalize_page_token(page) for page in raw_pages}


def normalize_expected_citations(raw_citations: Any) -> set[tuple[str, str]]:
    normalized: set[tuple[str, str]] = set()
    if not isinstance(raw_citations, list):
        return normalized

    for entry in raw_citations:
        if isinstance(entry, dict):
            doc = str(entry.get("doc", "")).strip()
            if not doc:
                continue
            normalized.add((doc, normalize_page_token(entry.get("page", "n/a"))))
            continue

        if isinstance(entry, str) and "|" in entry:
            doc, page = entry.split("|", 1)
            doc = doc.strip()
            page = page.strip()
            if doc and page:
                normalized.add((doc, normalize_page_token(page)))
    return normalized


def payload_citation_pairs(payload: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for entry in payload.get("regulatory_basis", []):
        if not isinstance(entry, dict):
            continue
        doc = str(entry.get("doc", "")).strip()
        if not doc:
            continue
        pairs.add((doc, normalize_page_token(entry.get("page", "n/a"))))
    return pairs


def canonical_source_score(doc_title: str, source_path: str) -> int:
    lowered = f"{doc_title} {source_path}".lower()
    if any(hint in lowered for hint in PRIMARY_SOURCE_HINTS):
        return 4
    if any(hint in lowered for hint in AUTHORITATIVE_SPEC_HINTS):
        return 3
    if any(hint in lowered for hint in SECONDARY_SOURCE_HINTS):
        return 1
    return 2


def first_rank_by_doc(items: list[dict[str, Any]], key_name: str) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        doc = str(item.get(key_name, "")).strip()
        if doc and doc not in ranks:
            ranks[doc] = index
    return ranks


def canonical_preference_for_docs(
    ordered_docs: list[dict[str, Any]],
    preferred_docs: set[str],
) -> bool | None:
    if not ordered_docs:
        return None

    doc_ranks = first_rank_by_doc(ordered_docs, "doc")
    if preferred_docs:
        preferred_ranks = [rank for doc, rank in doc_ranks.items() if doc in preferred_docs]
        secondary_ranks = [rank for doc, rank in doc_ranks.items() if doc not in preferred_docs]
        if not preferred_ranks:
            return None
        if not secondary_ranks:
            return True
        return min(preferred_ranks) <= min(secondary_ranks)

    scored_docs = []
    for doc, rank in doc_ranks.items():
        source_path = ""
        for item in ordered_docs:
            if str(item.get("doc", "")).strip() == doc:
                source_path = str(item.get("source_path", ""))
                break
        scored_docs.append((doc, rank, canonical_source_score(doc, source_path)))

    if len(scored_docs) < 2:
        return None

    max_score = max(score for _, _, score in scored_docs)
    best_ranks = [rank for _, rank, score in scored_docs if score == max_score]
    lower_ranks = [rank for _, rank, score in scored_docs if score < max_score]
    if not best_ranks or not lower_ranks:
        return None
    return min(best_ranks) <= min(lower_ranks)


def load_eval_cases(eval_path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    for line_number, raw_line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed JSONL line {line_number} in {eval_path}: {exc}")
            continue
        if "question" not in payload:
            raise ValueError(f"Missing 'question' in {eval_path} line {line_number}")
        payload.setdefault("id", f"q{line_number:03d}")
        cases.append(payload)
    if not cases:
        raise ValueError(f"No valid evaluation cases found in {eval_path}")
    return cases


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed resume row {line_number} in {path}: {exc}")
    return rows


def build_config_grid(top_k_values: list[int], min_citation_values: list[int]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for index, (top_k, reranker_enabled, min_citations) in enumerate(
        product(top_k_values, [False, True], min_citation_values),
        start=1,
    ):
        configs.append(
            {
                "config_id": f"cfg_{index:02d}",
                "top_k": top_k,
                "reranker_enabled": reranker_enabled,
                "min_citations": min_citations,
            }
        )
    return configs


def expand_cases(cases: list[dict[str, Any]], repeat_count: int) -> list[dict[str, Any]]:
    if repeat_count <= 1:
        return cases

    expanded: list[dict[str, Any]] = []
    for repeat_index in range(repeat_count):
        for case in cases:
            cloned = copy.deepcopy(case)
            base_id = str(cloned.get("id", "case"))
            cloned["id"] = f"{base_id}__r{repeat_index + 1:03d}"
            expanded.append(cloned)
    return expanded


def filter_cases(
    cases: list[dict[str, Any]],
    question_ids: set[str],
    topics: set[str],
    doc_families: set[str],
) -> list[dict[str, Any]]:
    filtered = cases
    if question_ids:
        filtered = [case for case in filtered if str(case.get("id", "")) in question_ids]
    if topics:
        filtered = [case for case in filtered if str(case.get("topic", "")).strip() in topics]
    if doc_families:
        filtered = [case for case in filtered if str(case.get("doc_family", "")).strip() in doc_families]
    return filtered


def cap_cases(cases: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    if max_cases <= 0:
        return cases
    return cases[:max_cases]


def trim_for_small_test(
    cases: list[dict[str, Any]],
    configs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_configs: list[dict[str, Any]] = []
    if configs:
        selected_configs.append(configs[0])

    for config in configs[1:]:
        if len(selected_configs) >= SMALL_TEST_CONFIG_COUNT:
            break
        if config["reranker_enabled"] != selected_configs[0]["reranker_enabled"]:
            selected_configs.append(config)

    for config in configs[1:]:
        if len(selected_configs) >= SMALL_TEST_CONFIG_COUNT:
            break
        if config not in selected_configs:
            selected_configs.append(config)

    return (
        cases[:SMALL_TEST_QUESTION_COUNT],
        selected_configs,
    )


def serialize_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for match in matches:
        metadata = dict(match.get("metadata") or {})
        serialized.append(
            {
                "doc": metadata.get("doc_title", "Unknown document"),
                "page": metadata.get("page", "n/a"),
                "source_path": metadata.get("source_path", ""),
                "chunk": metadata.get("chunk", "n/a"),
                "distance": match.get("distance"),
                "metadata": metadata,
            }
        )
    return serialized


def citation_pairs_from_matches(matches: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (
            str((match.get("metadata") or {}).get("doc_title", "Unknown document")),
            str(ask.normalize_page_value((match.get("metadata") or {}).get("page", "n/a"))),
        )
        for match in matches
    }


def evaluate_row_metrics(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    payload = result["answer_json"]
    retrieved_pairs = citation_pairs_from_matches(result["matches"])
    retrieved_pages = {normalize_page_token((match.get("metadata") or {}).get("page", "n/a")) for match in result["matches"]}
    retrieved_docs = {
        str((match.get("metadata") or {}).get("doc_title", "Unknown document")) for match in result["matches"]
    }
    expected_docs = {str(doc) for doc in case.get("expected_docs", [])}
    preferred_docs = {str(doc) for doc in case.get("preferred_docs", case.get("expected_docs", []))}
    expected_pages = normalize_expected_pages(case.get("expected_pages", []))
    expected_citations = normalize_expected_citations(case.get("expected_citations", []))
    answer_basis = [entry for entry in payload.get("regulatory_basis", []) if isinstance(entry, dict)]
    answer_basis_pairs = payload_citation_pairs(payload)
    answer_pages = {normalize_page_token(entry.get("page", "n/a")) for entry in answer_basis}

    supported_answer_citations = 0
    for entry in answer_basis:
        ref_key = (
            str(entry.get("doc", "")),
            normalize_page_token(entry.get("page", "n/a")),
        )
        if ref_key in retrieved_pairs and str(entry.get("quote", "")).strip():
            supported_answer_citations += 1

    answer_basis_count = len(answer_basis)
    answer_citation_precision = None
    if answer_basis_count:
        answer_citation_precision = supported_answer_citations / answer_basis_count

    expected_doc_hit_at_k = None
    expected_doc_recall_at_k = None
    if expected_docs:
        matched_docs = expected_docs.intersection(retrieved_docs)
        expected_doc_hit_at_k = bool(matched_docs)
        expected_doc_recall_at_k = len(matched_docs) / len(expected_docs)

    expected_page_hit_at_k = None
    expected_page_recall_at_k = None
    if expected_pages:
        matched_pages = expected_pages.intersection(retrieved_pages)
        expected_page_hit_at_k = bool(matched_pages)
        expected_page_recall_at_k = len(matched_pages) / len(expected_pages)

    expected_citation_hit_at_k = None
    expected_citation_recall_at_k = None
    if expected_citations:
        matched_citations = expected_citations.intersection(retrieved_pairs)
        expected_citation_hit_at_k = bool(matched_citations)
        expected_citation_recall_at_k = len(matched_citations) / len(expected_citations)

    answer_expected_page_hit = None
    if expected_pages:
        answer_expected_page_hit = bool(expected_pages.intersection(answer_pages))

    answer_expected_citation_hit = None
    if expected_citations:
        answer_expected_citation_hit = bool(expected_citations.intersection(answer_basis_pairs))

    expected_not_stated_match = None
    if "expected_not_stated" in case:
        expected_not_stated_match = bool(case.get("expected_not_stated")) == bool(payload.get("not_stated"))

    ordered_matches = [
        {
            "doc": str((match.get("metadata") or {}).get("doc_title", "Unknown document")),
            "source_path": str((match.get("metadata") or {}).get("source_path", "")),
        }
        for match in result["matches"]
    ]
    ordered_basis_docs = [
        {
            "doc": str(entry.get("doc", "")),
            "source_path": "",
        }
        for entry in answer_basis
    ]

    return {
        "retrieved_unique_docs": len(retrieved_docs),
        "retrieved_unique_citations": len(retrieved_pairs),
        "supported_answer_citations": supported_answer_citations,
        "answer_basis_count": answer_basis_count,
        "answer_citation_precision": answer_citation_precision,
        "expected_doc_hit_at_k": expected_doc_hit_at_k,
        "expected_doc_recall_at_k": expected_doc_recall_at_k,
        "expected_page_hit_at_k": expected_page_hit_at_k,
        "expected_page_recall_at_k": expected_page_recall_at_k,
        "expected_citation_hit_at_k": expected_citation_hit_at_k,
        "expected_citation_recall_at_k": expected_citation_recall_at_k,
        "answer_expected_page_hit": answer_expected_page_hit,
        "answer_expected_citation_hit": answer_expected_citation_hit,
        "retrieval_canonical_preference": canonical_preference_for_docs(ordered_matches, preferred_docs),
        "answer_canonical_preference": canonical_preference_for_docs(ordered_basis_docs, preferred_docs),
        "expected_not_stated_match": expected_not_stated_match,
    }


def validate_eval_case(
    case: dict[str, Any],
    result: dict[str, Any],
    min_citations: int,
) -> list[str]:
    reasons = ask.validate_answer_payload(result["answer_json"], result["matches"], min_citations=min_citations)
    payload = result["answer_json"]
    retrieved_pairs = citation_pairs_from_matches(result["matches"])
    actual_pairs = payload_citation_pairs(payload)
    actual_pages = {
        normalize_page_token(entry.get("page", "n/a"))
        for entry in payload.get("regulatory_basis", [])
        if isinstance(entry, dict)
    }

    for entry in payload.get("regulatory_basis", []):
        if not isinstance(entry, dict):
            continue
        cited_pair = (
            str(entry.get("doc", "")),
            normalize_page_token(entry.get("page", "n/a")),
        )
        if cited_pair not in retrieved_pairs:
            reasons.append("citation_not_in_retrieved_context")
            break

    expected_docs = case.get("expected_docs", [])
    if expected_docs:
        actual_docs = {entry["doc"] for entry in payload.get("regulatory_basis", []) if isinstance(entry, dict)}
        if not actual_docs.intersection(expected_docs):
            reasons.append("expected_doc_not_found")

    expected_pages = normalize_expected_pages(case.get("expected_pages", []))
    if expected_pages and not actual_pages.intersection(expected_pages):
        reasons.append("expected_page_not_found")

    expected_citations = normalize_expected_citations(case.get("expected_citations", []))
    if expected_citations and not actual_pairs.intersection(expected_citations):
        reasons.append("expected_citation_not_found")

    must_contain = [str(item).lower() for item in case.get("must_contain", [])]
    answer_lower = payload.get("answer", "").lower()
    for token in must_contain:
        if token not in answer_lower:
            reasons.append(f"missing_answer_token:{token}")

    if case.get("allow_not_stated") is False and payload.get("not_stated"):
        reasons.append("unexpected_not_stated")
    if "expected_not_stated" in case and bool(case.get("expected_not_stated")) != bool(payload.get("not_stated")):
        reasons.append(
            "expected_not_stated_mismatch:"
            + f"expected={bool(case.get('expected_not_stated'))}|actual={bool(payload.get('not_stated'))}"
        )

    return ask.normalize_notes(reasons)


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def existing_completed_pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        config = row.get("config") or {}
        config_id = str(config.get("config_id", "")).strip()
        question_id = str(row.get("question_id", "")).strip()
        if config_id and question_id:
            pairs.add((config_id, question_id))
    return pairs


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}"


def format_millis(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}ms"


def metadata_path_for_report(report_path: Path) -> Path:
    return report_path.with_suffix(".meta.json")


def load_or_init_run_metadata(meta_path: Path, run_id: str, eval_file: Path) -> dict[str, Any]:
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Run metadata file is not a JSON object: {meta_path}")
        payload.setdefault("run_id", run_id)
        payload.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        payload.setdefault("eval_file", str(eval_file))
        payload.setdefault("invocations", [])
        return payload

    payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_file": str(eval_file),
        "invocations": [],
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def write_run_metadata(meta_path: Path, metadata: dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate_run_details(
    current_run_details: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    invocations = [entry for entry in run_metadata.get("invocations", []) if isinstance(entry, dict)]
    return {
        **current_run_details,
        "run_created_at": str(run_metadata.get("created_at", "")),
        "invocation_count": len(invocations),
        "cumulative_collection_load_seconds": sum(
            float(entry.get("collection_load_seconds", 0.0) or 0.0) for entry in invocations
        ),
        "cumulative_warmup_seconds": sum(float(entry.get("warmup_seconds", 0.0) or 0.0) for entry in invocations),
        "cumulative_completed_pairs": sum(
            int(entry.get("completed_pairs_this_invocation", 0) or 0) for entry in invocations
        ),
        "latest_invocation_started_at": str(invocations[-1].get("started_at", "")) if invocations else "",
    }


def write_markdown_summary(
    path: Path,
    run_id: str,
    configs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    run_details: dict[str, Any],
) -> None:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str((row.get("config") or {}).get("config_id", ""))].append(row)

    ranked: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
    for config in configs:
        config_rows = grouped_rows.get(config["config_id"], [])
        total = len(config_rows)
        passed = sum(1 for row in config_rows if row.get("passed"))
        pass_rate = (passed / total) if total else 0.0
        ranked.append((pass_rate, config, config_rows))

    ranked.sort(key=lambda item: (-item[0], item[1]["config_id"]))

    lines = [
        f"# Eval Report {run_id}",
        "",
        "## Run Overview",
        "",
        f"- Run created: {run_details['run_created_at'] or 'n/a'}",
        f"- Invocation count: {run_details['invocation_count']}",
        f"- Latest invocation started: {run_details['latest_invocation_started_at'] or 'n/a'}",
        f"- Rows written: {len(rows)}",
        f"- Planned pairs after filters: {run_details['planned_pairs']}",
        f"- Pending pairs at start of this invocation: {run_details['pending_pairs']}",
        f"- Resumed rows loaded: {run_details['resumed_rows']}",
        f"- This invocation collection load: {run_details['collection_load_seconds']:.2f}s",
        f"- This invocation warm-up query: {run_details['warmup_seconds']:.2f}s",
        f"- Cumulative collection load: {run_details['cumulative_collection_load_seconds']:.2f}s",
        f"- Cumulative warm-up query: {run_details['cumulative_warmup_seconds']:.2f}s",
        f"- Cumulative completed pairs across invocations: {run_details['cumulative_completed_pairs']}",
        "",
        "## Ranked Configs",
        "",
    ]

    for position, (pass_rate, config, config_rows) in enumerate(ranked, start=1):
        total = len(config_rows)
        passed = sum(1 for row in config_rows if row.get("passed"))
        lines.append(
            f"{position}. `{config['config_id']}` | pass_rate={passed}/{total} ({pass_rate:.0%}) | "
            f"top_k={config['top_k']} | reranker={'on' if config['reranker_enabled'] else 'off'} | "
            f"min_citations={config['min_citations']}"
        )

        retrieval_ms = average_optional(
            [float((row.get("timings_ms") or {}).get("retrieve")) for row in config_rows if row.get("timings_ms")]
        )
        answer_ms = average_optional(
            [float((row.get("timings_ms") or {}).get("answer")) for row in config_rows if row.get("timings_ms")]
        )
        total_ms = average_optional(
            [float((row.get("timings_ms") or {}).get("total")) for row in config_rows if row.get("timings_ms")]
        )
        expected_doc_hit_rate = average_optional(
            [row.get("metrics", {}).get("expected_doc_hit_at_k") for row in config_rows]
        )
        expected_doc_recall = average_optional(
            [row.get("metrics", {}).get("expected_doc_recall_at_k") for row in config_rows]
        )
        not_stated_match_rate = average_optional(
            [row.get("metrics", {}).get("expected_not_stated_match") for row in config_rows]
        )
        citation_precision = average_optional(
            [row.get("metrics", {}).get("answer_citation_precision") for row in config_rows]
        )
        expected_page_hit_rate = average_optional(
            [row.get("metrics", {}).get("expected_page_hit_at_k") for row in config_rows]
        )
        expected_citation_hit_rate = average_optional(
            [row.get("metrics", {}).get("expected_citation_hit_at_k") for row in config_rows]
        )
        retrieval_canonical_preference = average_optional(
            [row.get("metrics", {}).get("retrieval_canonical_preference") for row in config_rows]
        )
        answer_canonical_preference = average_optional(
            [row.get("metrics", {}).get("answer_canonical_preference") for row in config_rows]
        )

        lines.append(
            "   Metrics: "
            + f"retrieve_avg={format_millis(retrieval_ms)} | "
            + f"answer_avg={format_millis(answer_ms)} | "
            + f"total_avg={format_millis(total_ms)} | "
            + f"doc_hit@k={format_percent(expected_doc_hit_rate)} | "
            + f"page_hit@k={format_percent(expected_page_hit_rate)} | "
            + f"citation_hit@k={format_percent(expected_citation_hit_rate)} | "
            + f"doc_recall@k={format_percent(expected_doc_recall)} | "
            + f"citation_precision={format_percent(citation_precision)} | "
            + f"retrieval_canonical_pref={format_percent(retrieval_canonical_preference)} | "
            + f"answer_canonical_pref={format_percent(answer_canonical_preference)} | "
            + f"not_stated_match={format_percent(not_stated_match_rate)}"
        )

        failure_counter: Counter[str] = Counter()
        for row in config_rows:
            if row.get("passed"):
                continue
            failure_counter.update(row.get("validation_reasons", []))

        if failure_counter:
            top_failures = ", ".join(
                f"{reason} x{count}" for reason, count in failure_counter.most_common(5)
            )
            lines.append(f"   Failure reasons: {top_failures}")
        else:
            lines.append("   Failure reasons: none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ask.suppress_noisy_startup()
    args = parse_args()
    cases = load_eval_cases(args.eval_file)
    original_case_count = len(cases)
    repeat_cases = max(1, args.repeat_cases)
    if repeat_cases != args.repeat_cases:
        print(f"Using repeat_cases={repeat_cases} (minimum is 1).")
    cases = expand_cases(cases, repeat_cases)
    cases = filter_cases(
        cases,
        parse_str_list(args.question_ids),
        parse_str_list(args.topics),
        parse_str_list(args.doc_families),
    )
    max_cases = max(0, args.max_cases)
    if max_cases != args.max_cases:
        print(f"Using max_cases={max_cases} (minimum is 0).")
    cases = cap_cases(cases, max_cases)
    configs = build_config_grid(
        parse_int_list(args.top_k_values, DEFAULT_TOP_K_VALUES),
        parse_int_list(args.min_citation_values, DEFAULT_MIN_CITATION_VALUES),
    )

    if args.small_test:
        cases, configs = trim_for_small_test(cases, configs)
    if not cases:
        raise ValueError("No evaluation cases remain after applying filters.")
    if not configs:
        raise ValueError("No evaluation configs were generated.")

    collection_load_start = perf_counter()
    collection = ask.load_collection(console=None)
    collection_load_seconds = perf_counter() - collection_load_start
    if collection is None:
        raise RuntimeError("Unable to load the Chroma collection for evaluation.")

    warmup_case = cases[0]
    warmup_config = configs[0]
    warmup_start = perf_counter()
    ask.build_query_result(
        collection,
        question=str(warmup_case["question"]),
        top_k=min(warmup_config["top_k"], ask.MAX_TOP_K),
        topic=str(warmup_case.get("topic", "")).strip(),
        doc_family=str(warmup_case.get("doc_family", "")).strip(),
        min_citations=max(1, int(warmup_case.get("min_citations", warmup_config["min_citations"]))),
        reranker_enabled=bool(warmup_config["reranker_enabled"]),
    )
    warmup_seconds = perf_counter() - warmup_start

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    resumed_rows: list[dict[str, Any]] = []
    if args.resume_from is not None:
        jsonl_path = args.resume_from
        md_path = args.resume_from.with_suffix(".md")
        run_id = args.resume_from.stem
        resumed_rows = load_existing_rows(jsonl_path)
    else:
        run_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        jsonl_path = REPORTS_DIR / f"{run_id}.jsonl"
        md_path = REPORTS_DIR / f"{run_id}.md"
        jsonl_path.write_text("", encoding="utf-8")
    meta_path = metadata_path_for_report(jsonl_path)
    run_metadata = load_or_init_run_metadata(meta_path, run_id, args.eval_file)

    rows: list[dict[str, Any]] = list(resumed_rows)
    completed_pairs = existing_completed_pairs(resumed_rows)
    pending_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for config in configs:
        for case in cases:
            pair = (config["config_id"], str(case["id"]))
            if pair in completed_pairs:
                continue
            pending_pairs.append((config, case))

    run_details = {
        "planned_pairs": len(configs) * len(cases),
        "pending_pairs": len(pending_pairs),
        "resumed_rows": len(resumed_rows),
        "collection_load_seconds": collection_load_seconds,
        "warmup_seconds": warmup_seconds,
    }
    invocation_record = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "filters": {
            "question_ids": sorted(parse_str_list(args.question_ids)),
            "topics": sorted(parse_str_list(args.topics)),
            "doc_families": sorted(parse_str_list(args.doc_families)),
            "max_cases": max_cases,
            "small_test": bool(args.small_test),
        },
        "planned_pairs": run_details["planned_pairs"],
        "pending_pairs": run_details["pending_pairs"],
        "resumed_rows": run_details["resumed_rows"],
        "collection_load_seconds": collection_load_seconds,
        "warmup_seconds": warmup_seconds,
        "completed_pairs_this_invocation": 0,
        "rows_written_before": len(resumed_rows),
        "rows_written_after": len(resumed_rows),
        "completed_at": "",
    }
    run_metadata.setdefault("invocations", []).append(invocation_record)
    write_run_metadata(meta_path, run_metadata)
    aggregated_run_details = aggregate_run_details(run_details, run_metadata)

    print(
        f"Prepared {len(cases)} case(s) from {original_case_count} loaded | "
        f"{len(configs)} config(s) | {len(pending_pairs)} pending pair(s)"
    )
    print(f"Collection load: {collection_load_seconds:.2f}s | Warm-up: {warmup_seconds:.2f}s")

    summary_every = max(0, args.save_summary_every)
    if summary_every != args.save_summary_every:
        print(f"Using save_summary_every={summary_every} (minimum is 0).")

    if not pending_pairs:
        invocation_record["completed_at"] = datetime.now().isoformat(timespec="seconds")
        write_run_metadata(meta_path, run_metadata)
        aggregated_run_details = aggregate_run_details(run_details, run_metadata)
        write_markdown_summary(md_path, run_id, configs, rows, aggregated_run_details)
        print("No pending pairs remain for this filtered run.")
        print(f"Run ID: {run_id}")
        print(f"JSONL report: {jsonl_path}")
        print(f"Markdown report: {md_path}")
        print(f"Run metadata: {meta_path}")
        return

    completed_runs = 0
    total_runs = len(pending_pairs)

    for config, case in pending_pairs:
        completed_runs += 1
        if completed_runs == 1 or completed_runs % 10 == 0 or completed_runs == total_runs:
            print(f"[{completed_runs}/{total_runs}] {config['config_id']} {case['id']}")
        try:
            min_citations = max(1, int(case.get("min_citations", config["min_citations"])))
            result = ask.build_query_result(
                collection,
                question=str(case["question"]),
                top_k=min(config["top_k"], ask.MAX_TOP_K),
                topic=str(case.get("topic", "")).strip(),
                doc_family=str(case.get("doc_family", "")).strip(),
                min_citations=min_citations,
                reranker_enabled=bool(config["reranker_enabled"]),
            )
            validation_reasons = validate_eval_case(case, result, min_citations=min_citations)
            row = {
                "run_id": run_id,
                "config": config,
                "question_id": case["id"],
                "question": case["question"],
                "retrieved_chunk_metadata": serialize_matches(result["matches"]),
                "answer_json": result["answer_json"],
                "validation_reasons": validation_reasons,
                "guardrail_retry_reasons": result["validation_reasons"],
                "passed": not validation_reasons,
                "error": result["error"],
                "timings_ms": result.get("timings_ms", {}),
                "metrics": evaluate_row_metrics(case, result),
            }
            rows.append(row)
            append_jsonl_row(jsonl_path, row)
        except Exception as exc:
            row = {
                "run_id": run_id,
                "config": config,
                "question_id": case.get("id", "unknown"),
                "question": case.get("question", ""),
                "retrieved_chunk_metadata": [],
                "answer_json": ask.build_not_stated_payload([f"Eval execution failure: {exc}"]),
                "validation_reasons": [f"eval_exception:{exc}"],
                "guardrail_retry_reasons": [],
                "passed": False,
                "error": str(exc),
                "timings_ms": {},
                "metrics": {},
            }
            rows.append(row)
            append_jsonl_row(jsonl_path, row)

        if summary_every and (completed_runs % summary_every == 0 or completed_runs == total_runs):
            invocation_record["completed_pairs_this_invocation"] = completed_runs
            invocation_record["rows_written_after"] = len(rows)
            write_run_metadata(meta_path, run_metadata)
            aggregated_run_details = aggregate_run_details(run_details, run_metadata)
            write_markdown_summary(md_path, run_id, configs, rows, aggregated_run_details)

    invocation_record["completed_pairs_this_invocation"] = completed_runs
    invocation_record["rows_written_after"] = len(rows)
    invocation_record["completed_at"] = datetime.now().isoformat(timespec="seconds")
    write_run_metadata(meta_path, run_metadata)
    aggregated_run_details = aggregate_run_details(run_details, run_metadata)
    write_markdown_summary(md_path, run_id, configs, rows, aggregated_run_details)

    print(f"Run ID: {run_id}")
    print(f"JSONL report: {jsonl_path}")
    print(f"Markdown report: {md_path}")
    print(f"Run metadata: {meta_path}")


if __name__ == "__main__":
    main()
