#!/usr/bin/env python3
"""
Step 2: production-safe text isolation.

This is the critical production-grade change for Physics:
- safe prose stays in page.text/text_plain;
- formula/numeric/table/diagram lines are removed from production text and sent to a review queue;
- raw extracted text is preserved in raw_extracted_text and line_items;
- the output is safe for lesson-planning/topic embeddings, but exact formula production requires Step 3 corrections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from physics_common import (
    build_safe_text_from_line_items, make_line_items, read_json, setup_logging, summarize_review_queue,
    utc_now, write_json,
)


def build_step2_json(input_json: Path, review_queue_output: Path | None = None, placeholder_unreviewed: bool = False) -> tuple[dict[str, Any], str, dict[str, Any]]:
    data = read_json(input_json)
    stats = Counter(data.get("extraction", {}).get("statistics") or {})
    page_extractions: list[dict[str, Any]] = []
    for page in data.get("page_extractions") or []:
        page = dict(page)
        if not page.get("include_in_lesson_text"):
            page.setdefault("raw_extracted_text", page.get("selectable_text") or page.get("text") or "")
            page["production_text_policy"] = "not_lesson_content"
            page_extractions.append(page)
            continue
        line_items, line_stats = make_line_items(page)
        stats.update(line_stats)
        safe_result = build_safe_text_from_line_items(
            line_items,
            reviewed_blocks={},
            placeholder_unreviewed=placeholder_unreviewed,
        )
        # physics_common.py newer versions return:
        # (text, unresolved, safe_used, reviewed_used, discarded_review_used)
        # Older versions returned only 4 values. Support both so mixed file updates do not crash.
        if len(safe_result) == 4:
            safe_text, unresolved, safe_used, reviewed_used = safe_result
            discarded_review_used = 0
        else:
            safe_text, unresolved, safe_used, reviewed_used, discarded_review_used = safe_result

        page["pre_production_safety_text"] = page.get("text") or ""
        page["line_items"] = line_items
        page["unsafe_text_blocks"] = [item for item in line_items if item.get("type") == "review_required"]
        page["unresolved_review_items"] = len(unresolved)
        page["safe_lines_used"] = safe_used
        page["reviewed_items_applied"] = reviewed_used
        page["discarded_review_items_applied"] = discarded_review_used
        page["production_text_policy"] = "safe_prose_only_unreviewed_math_isolated"
        page["text"] = safe_text
        page["text_plain"] = safe_text
        page["production_safe_text"] = safe_text
        page["include_in_embeddings"] = bool(safe_text)
        page["embedding_readiness"] = "ready_for_safe_text_embedding" if safe_text and not unresolved else "review_required_before_formula_safe_embedding"
        page["text_length_chars"] = len(safe_text)
        page.setdefault("quality_flags", [])
        if unresolved:
            if "formula_or_diagram_text_isolated_for_review" not in page["quality_flags"]:
                page["quality_flags"].append("formula_or_diagram_text_isolated_for_review")
            stats["pages_with_unresolved_review_items"] += 1
            stats["unresolved_review_items"] += len(unresolved)
        page_extractions.append(page)
    data["page_extractions"] = page_extractions
    review_queue = summarize_review_queue(page_extractions)
    validation = {
        "status": "passed",
        "errors": [],
        "metrics": {
            "page_extractions": len(page_extractions),
            "teaching_pages": stats.get("teaching_pages", 0),
            "safe_lines": stats.get("safe_lines", 0),
            "review_required_lines": stats.get("review_required_lines", 0),
            "artifact_lines": stats.get("artifact_lines", 0),
            "formula_review_lines": stats.get("formula_review_lines", 0),
            "table_diagram_review_lines": stats.get("table_diagram_review_lines", 0),
            "table_diagram_context_lines": stats.get("table_diagram_context_lines", 0),
            "discarded_noise_lines": stats.get("discarded_noise_lines", 0),
            "discarded_non_content_lines": stats.get("discarded_non_content_lines", 0),
            "pages_with_unresolved_review_items": stats.get("pages_with_unresolved_review_items", 0),
            "unresolved_review_items": stats.get("unresolved_review_items", 0),
        },
    }
    data["text_accuracy_status"] = "review_required" if stats.get("unresolved_review_items", 0) else "formula_safe_text_ready"
    data["production_status"] = "review_required_not_formula_safe" if stats.get("unresolved_review_items", 0) else "production_ready"
    data["extraction"] = {
        "step": 2,
        "status": "step2_safe_text_isolated",
        "generated_at": utc_now(),
        "generator": "physics_step_2_production_text.py",
        "method": "line_level_safe_prose_extraction_with_formula_table_diagram_review_queue",
        "source_step1_json": str(input_json),
        "review_queue_output": str(review_queue_output) if review_queue_output else None,
        "statistics": dict(stats),
        "validation": validation,
    }
    return data, build_step2_report(data, review_queue), review_queue


def build_step2_report(data: dict[str, Any], review_queue: dict[str, Any]) -> str:
    metrics = data.get("extraction", {}).get("validation", {}).get("metrics", {})
    lines = [
        "Grade 10 Physics Step 2 production-safe text isolation report",
        "=" * 72,
        f"Generated at: {data.get('extraction', {}).get('generated_at')}",
        f"book_title: {data.get('book_title')}",
        f"text_accuracy_status: {data.get('text_accuracy_status')}",
        f"production_status: {data.get('production_status')}",
        "",
        "Metrics:",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "Important:",
        "- This step removes formula/table/diagram-risk text from production text instead of silently trusting corrupted OCR.",
        "- Exact Physics/formula production requires reviewing the generated review queue and applying Step 3 corrections.",
        f"- unresolved_review_items: {review_queue.get('unresolved_review_items')}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: isolate formula/table/diagram-risk text into a review queue.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--review-queue", type=Path, default=None)
    parser.add_argument("--placeholder-unreviewed", action="store_true", help="Insert placeholders into text where formula review lines were removed.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    for path in [args.output_json, args.report, args.review_queue]:
        if path and path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")
    data, report, queue = build_step2_json(args.input_json.resolve(), args.review_queue.resolve() if args.review_queue else None, args.placeholder_unreviewed)
    write_json(args.output_json.resolve(), data)
    if args.review_queue:
        write_json(args.review_queue.resolve(), queue)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(f"Step 2 JSON: {args.output_json.resolve()}")
    if args.review_queue:
        print(f"Step 2 review queue: {args.review_queue.resolve()}")
    if args.report:
        print(f"Step 2 report: {args.report.resolve()}")
    print(f"Step 2 production status: {data.get('production_status')}")


if __name__ == "__main__":
    main()
