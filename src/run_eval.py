from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run overnight evaluations over the UAE RAG pipeline.")
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


def validate_eval_case(
    case: dict[str, Any],
    result: dict[str, Any],
    min_citations: int,
) -> list[str]:
    reasons = ask.validate_answer_payload(result["answer_json"], result["matches"], min_citations=min_citations)
    payload = result["answer_json"]
    retrieved_pairs = citation_pairs_from_matches(result["matches"])

    for entry in payload.get("regulatory_basis", []):
        if not isinstance(entry, dict):
            continue
        cited_pair = (
            str(entry.get("doc", "")),
            str(ask.normalize_page_value(entry.get("page", "n/a"))),
        )
        if cited_pair not in retrieved_pairs:
            reasons.append("citation_not_in_retrieved_context")
            break

    expected_docs = case.get("expected_docs", [])
    if expected_docs:
        actual_docs = {entry["doc"] for entry in payload.get("regulatory_basis", [])}
        if not actual_docs.intersection(expected_docs):
            reasons.append("expected_doc_not_found")

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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    serialized = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if serialized:
        serialized += "\n"
    path.write_text(serialized, encoding="utf-8")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def write_markdown_summary(
    path: Path,
    run_id: str,
    configs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["config"]["config_id"]].append(row)

    ranked: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
    for config in configs:
        config_rows = grouped_rows.get(config["config_id"], [])
        total = len(config_rows)
        passed = sum(1 for row in config_rows if row["passed"])
        pass_rate = (passed / total) if total else 0.0
        ranked.append((pass_rate, config, config_rows))

    ranked.sort(key=lambda item: (-item[0], item[1]["config_id"]))

    lines = [
        f"# Eval Report {run_id}",
        "",
        "## Ranked Configs",
        "",
    ]

    for position, (pass_rate, config, config_rows) in enumerate(ranked, start=1):
        total = len(config_rows)
        passed = sum(1 for row in config_rows if row["passed"])
        lines.append(
            f"{position}. `{config['config_id']}` | pass_rate={passed}/{total} ({pass_rate:.0%}) | "
            f"top_k={config['top_k']} | reranker={'on' if config['reranker_enabled'] else 'off'} | "
            f"min_citations={config['min_citations']}"
        )

        failure_counter: Counter[str] = Counter()
        for row in config_rows:
            if row["passed"]:
                continue
            failure_counter.update(row["validation_reasons"])

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
    repeat_cases = max(1, args.repeat_cases)
    if repeat_cases != args.repeat_cases:
        print(f"Using repeat_cases={repeat_cases} (minimum is 1).")
    cases = expand_cases(cases, repeat_cases)
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

    collection = ask.load_collection(console=None)
    if collection is None:
        raise RuntimeError("Unable to load the Chroma collection for evaluation.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    jsonl_path = REPORTS_DIR / f"{run_id}.jsonl"
    md_path = REPORTS_DIR / f"{run_id}.md"
    jsonl_path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    total_runs = len(configs) * len(cases)
    completed_runs = 0

    for config in configs:
        for case in cases:
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
                passed = not validation_reasons
                row = {
                    "run_id": run_id,
                    "config": config,
                    "question_id": case["id"],
                    "question": case["question"],
                    "retrieved_chunk_metadata": serialize_matches(result["matches"]),
                    "answer_json": result["answer_json"],
                    "validation_reasons": validation_reasons,
                    "guardrail_retry_reasons": result["validation_reasons"],
                    "passed": passed,
                    "error": result["error"],
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
                }
                rows.append(row)
                append_jsonl_row(jsonl_path, row)

    write_markdown_summary(md_path, run_id, configs, rows)

    print(f"Run ID: {run_id}")
    print(f"JSONL report: {jsonl_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
