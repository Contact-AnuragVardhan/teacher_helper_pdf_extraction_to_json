#!/usr/bin/env python3
"""Step 5: publish final JSON and enforce production-readiness gates."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

from physics_common import (
    DEFAULT_DOCUMENT_ID, DEFAULT_DOCUMENT_KEY, build_report, read_json, setup_logging,
    utc_now, validate_output, write_json,
)


def count_unresolved(data: dict[str, Any]) -> int:
    total = 0
    for page in data.get("page_extractions") or []:
        total += int(page.get("unresolved_review_items") or 0)
    return total


def publish_json(input_json: Path, document_id: str, document_key: str, allow_review_required: bool = False) -> tuple[dict[str, Any], str]:
    src = read_json(input_json)
    data = copy.deepcopy(src)
    stats = Counter(data.get("extraction", {}).get("statistics") or {})
    unresolved = count_unresolved(data)
    stats["unresolved_review_items"] = unresolved
    existing_errors: list[str] = []
    validation = validate_output(data, stats, existing_errors)

    final_artifact_blockers = int(stats.get("final_artifact_blockers") or 0)

    if unresolved > 0:
        data["text_accuracy_status"] = "failed_formula_or_diagram_review"
        data["production_status"] = "review_required_not_formula_safe"
        if not allow_review_required:
            validation["errors"].append(
                f"Unresolved formula/table/diagram review items remain: {unresolved}. Fill corrections JSON or rerun with --allow-review-required for non-final review output."
            )
            validation["status"] = "failed"
    elif final_artifact_blockers > 0:
        data["text_accuracy_status"] = "failed_final_artifact_gate"
        data["production_status"] = "review_required_leftover_artifacts"
        if not allow_review_required:
            validation["errors"].append(
                f"Final artifact gate found {final_artifact_blockers} production-text artifact blocker(s). Fix these before final production."
            )
            validation["status"] = "failed"
    else:
        data["text_accuracy_status"] = "formula_safe_text_ready"
        data["production_status"] = "production_ready"

    data["documentId"] = document_id
    data["document_key"] = document_key
    data["production_notes"] = {
        "structure_status": "passed" if not validation.get("metrics", {}).get("subsections_outside_parent_range") else "failed",
        "text_policy": "Exact formulas are included only when reviewed through corrections JSON. Unreviewed risky formula/table/diagram OCR is excluded from production text and retained in raw_extracted_text/line_items.",
        "safe_for": [
            "lesson_planning" if unresolved else "lesson_planning_with_reviewed_formulas",
            "topic_context_embeddings" if unresolved else "formula_reviewed_embeddings",
            "chapter_subsection_page_mapping",
        ],
        "not_safe_for_when_review_required": [
            "exact_formula_QA",
            "student_numerical_answer_validation",
            "formula-search embeddings",
        ] if unresolved else [],
        "unresolved_review_items": unresolved,
        "final_artifact_blockers": int(stats.get("final_artifact_blockers") or 0),
    }
    data["extraction"] = {
        "step": 5,
        "status": "production_ready" if data["production_status"] == "production_ready" else "review_required_output_generated",
        "generated_at": utc_now(),
        "generator": "physics_step_5_publish_production.py",
        "method": "publish_with_strict_formula_review_gate",
        "source_step4_json": str(input_json),
        "statistics": dict(stats),
        "validation": validation,
    }
    return data, build_report(data, validation, "Grade 10 Physics Step 5 production publish report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: publish final Physics JSON with production-readiness gates.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument("--document-key", default=DEFAULT_DOCUMENT_KEY)
    parser.add_argument("--allow-review-required", action="store_true", help="Write output even with unresolved formula review items; status remains review_required_not_formula_safe.")
    parser.add_argument("--fail-on-review-required", action="store_true", help="Exit non-zero if unresolved review items remain.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    for path in [args.output_json, args.report]:
        if path and path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")
    data, report = publish_json(args.input_json.resolve(), args.document_id, args.document_key, args.allow_review_required)
    write_json(args.output_json.resolve(), data)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(f"Final JSON: {args.output_json.resolve()}")
    if args.report:
        print(f"Final report: {args.report.resolve()}")
    print(f"Production status: {data.get('production_status')}")
    if args.fail_on_review_required and data.get("production_status") != "production_ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
