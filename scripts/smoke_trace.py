from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src import ask


QUERIES = [
    "What is the difference between a TIN and a TRN in UAE e-invoicing?",
    "When must a tax invoice be issued under UAE VAT?",
    "By when must a tax credit note be issued?",
    "Does electronic invoicing apply to free zone entities in the UAE?",
    "What is the legal basis for mandatory UAE electronic invoicing?",
    "List the mandatory fields and cite the source for each item.",
    "What are the mandatory fields for UAE electronic invoices?",
    "Which mandatory fields are for exports only?",
    "What does IBT-024 mean in PINT-AE?",
    "What are VAT registration thresholds in the UAE?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic smoke traces for UAE RAG.")
    parser.add_argument(
        "--question-bank",
        default="eval/question_bank_smoke.jsonl",
        help="Optional JSONL file with {'question': '...'} records. Falls back to built-in queries if missing.",
    )
    return parser.parse_args()


def load_question_bank(path: str) -> list[str]:
    bank_path = Path(path)
    if not bank_path.exists():
        return []

    questions: list[str] = []
    for raw_line in bank_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        question = str(record.get("question", "")).strip()
        if not question:
            continue
        questions.append(question)
    return questions


def classify_warnings(question: str, trace_payload: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    lowered = question.lower().strip()
    intent_enum = str(trace_payload.get("intent_enum", ""))

    if "difference" in lowered and intent_enum != "COMPARE":
        warnings.append("question_contains_difference_but_intent_is_not_COMPARE")
    if lowered.startswith("when") and intent_enum != "RULE_TIMING":
        warnings.append("question_starts_with_when_but_intent_is_not_RULE_TIMING")

    return warnings


def print_trace_summary(trace_payload: dict[str, object]) -> None:
    summary = {
        "question": trace_payload.get("question", ""),
        "intent_enum": trace_payload.get("intent_enum", ""),
        "handler": trace_payload.get("handler", ""),
        "selected_handler": trace_payload.get("selected_handler", ""),
        "list_items_extracted": trace_payload.get("list_items_extracted", 0),
        "list_items_returned": trace_payload.get("list_items_returned", 0),
        "list_items_rejected": trace_payload.get("list_items_rejected", 0),
        "distinct_pages_covered": trace_payload.get("distinct_pages_covered", 0),
        "missing_groups": trace_payload.get("missing_groups", []),
        "timing_rules_found": trace_payload.get("timing_rules_found", 0),
        "timing_rules_returned": trace_payload.get("timing_rules_returned", 0),
        "timing_best_score": trace_payload.get("timing_best_score", 0),
        "timing_best_source_tier": trace_payload.get("timing_best_source_tier", ""),
        "timing_candidates_primary": trace_payload.get("timing_candidates_primary", 0),
        "timing_candidates_official": trace_payload.get("timing_candidates_official", 0),
        "timing_candidates_tertiary": trace_payload.get("timing_candidates_tertiary", 0),
        "timing_best_pattern_hit": trace_payload.get("timing_best_pattern_hit", ""),
        "timing_trigger_found": trace_payload.get("timing_trigger_found", False),
        "timing_primary_selected_score": trace_payload.get("timing_primary_selected_score", 0),
    }
    print(json.dumps(summary, ensure_ascii=False))


def print_warnings(question: str, trace_payload: dict[str, object]) -> None:
    for warning in classify_warnings(question, trace_payload):
        print(json.dumps({"warning": warning, "question": question}, ensure_ascii=False))


def extract_first_citation(answer_text: str) -> dict[str, str]:
    match = re.search(r"\[([^,\]]+),\s*p\.\s*([^\]]+)\]", answer_text)
    if not match:
        return {"doc_title": "", "page": ""}
    return {"doc_title": match.group(1).strip(), "page": match.group(2).strip()}


def extract_first_bullet_text(answer_text: str) -> str:
    for line in answer_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            return stripped[2:].strip()
    return ""


def main() -> None:
    args = parse_args()
    ask.suppress_noisy_startup()
    collection = ask.load_collection(console=None)
    if collection is None:
        raise SystemExit("Index is unavailable. Run ingestion first.")

    questions = load_question_bank(args.question_bank) or QUERIES
    print(json.dumps({"question_bank": args.question_bank, "questions_loaded": len(questions)}, ensure_ascii=False))

    for question in questions:
        trace_context = ask.TraceContext(enabled=True, question=question)
        result = ask.build_query_result(
            collection,
            question,
            top_k=3,
            topic="",
            doc_family="",
            min_citations=ask.DEFAULT_MIN_CITATIONS,
            reranker_enabled=True,
            trace_context=trace_context,
        )
        trace_payload = ask.build_trace_payload(trace_context)
        print_trace_summary(trace_payload)
        print_warnings(question, trace_payload)
        if trace_payload.get("intent_enum") == "NAMED_ENUMERATION":
            answer_text = str((result.get("answer_json") or {}).get("answer", ""))
            headings = [line.rstrip(":") for line in answer_text.splitlines() if line.strip().endswith(":")]
            print(json.dumps({"named_enum_group_headings": headings}, ensure_ascii=False))
        if trace_payload.get("intent_enum") == "RULE_TIMING":
            answer_text = str((result.get("answer_json") or {}).get("answer", ""))
            answer_excerpt = " ".join(answer_text.split())[:240]
            first_citation = extract_first_citation(answer_text)
            first_bullet = extract_first_bullet_text(answer_text)
            first_bullet_excerpt = " ".join(first_bullet.split())[:180]
            first_bullet_pattern = ask.detect_timing_pattern(first_bullet)
            print(
                json.dumps(
                    {
                        "rule_timing_best_source_tier": trace_payload.get("timing_best_source_tier", ""),
                        "rule_timing_first_citation": first_citation,
                        "rule_timing_first_bullet_excerpt": first_bullet_excerpt,
                        "rule_timing_first_bullet_has_timing_pattern": bool(first_bullet_pattern),
                        "rule_timing_first_bullet_pattern": first_bullet_pattern or "",
                        "rule_timing_answer_excerpt": answer_excerpt,
                    },
                    ensure_ascii=False,
                )
            )
        print(json.dumps(trace_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
