#!/usr/bin/env python3
"""
Step 3: apply curated formula/table/diagram text corrections.

Corrections file schema:
{
  "reviewed_blocks": {
    "p015_l003": "Exact reviewed text to use instead of the raw corrupted line"
  },
  "discard_review_ids": [
    "p015_l004"
  ],
  "global_replacements": [{"bad":"...", "good":"..."}]
}

Use discard_review_ids only for visually reviewed non-content, duplicated crop fragments,
or pure OCR garbage. Do not discard real formulas just to pass validation.

Without reviewed_blocks, exact formula text remains unresolved and Step 5 will not mark production_ready.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from physics_common import (
    apply_safe_global_replacements, build_safe_text_from_line_items, read_json, setup_logging,
    summarize_review_queue, utc_now, write_json,
)


def build_step3_json(input_json: Path, corrections_json: Path | None, review_queue_output: Path | None = None, placeholder_unreviewed: bool = False) -> tuple[dict[str, Any], str, dict[str, Any]]:
    data = read_json(input_json)
    corrections = read_json(corrections_json) if corrections_json and corrections_json.exists() else {"reviewed_blocks": {}, "global_replacements": []}
    raw_reviewed_blocks = corrections.get("reviewed_blocks") or {}
    reviewed_blocks: dict[str, str] = {}
    for k, v in raw_reviewed_blocks.items():
        # Support both old schema: "id": "text" and richer schema: "id": {"text": "..."}
        if isinstance(v, dict):
            text = v.get("text") or v.get("reviewed_text") or ""
        else:
            text = v
        if str(text).strip():
            reviewed_blocks[str(k)] = str(text)
    discard_review_ids = {str(x) for x in (corrections.get("discard_review_ids") or []) if str(x).strip()}
    # Also support dict form: "discarded_blocks": {"id": "reason"}
    discard_review_ids.update(str(k) for k in (corrections.get("discarded_blocks") or {}).keys())
    global_replacements = corrections.get("global_replacements") or []
    stats = Counter(data.get("extraction", {}).get("statistics") or {})
    # Recompute these so current corrections determine the final review counts.
    stats["unresolved_review_items"] = 0
    stats["reviewed_items_applied"] = 0
    stats["discarded_review_items_applied"] = 0
    stats["pages_with_unresolved_review_items"] = 0
    stats["pages_with_reviewed_items"] = 0

    page_extractions: list[dict[str, Any]] = []
    for page in data.get("page_extractions") or []:
        page = dict(page)
        if not page.get("include_in_lesson_text"):
            page_extractions.append(page)
            continue
        line_items = page.get("line_items") or []
        # Apply safe global replacements to safe lines and reviewed block text.
        for item in line_items:
            if item.get("type") == "safe_text":
                item["text"] = apply_safe_global_replacements(item.get("text") or "", global_replacements)
        corrected_blocks = {rid: apply_safe_global_replacements(text, global_replacements) for rid, text in reviewed_blocks.items()}
        text, unresolved, safe_used, reviewed_used, discarded_review_used = build_safe_text_from_line_items(
            line_items,
            reviewed_blocks=corrected_blocks,
            discard_review_ids=discard_review_ids,
            placeholder_unreviewed=placeholder_unreviewed,
        )
        resolved_ids = set(corrected_blocks.keys())
        for item in line_items:
            if item.get("type") == "review_required":
                rid = str(item.get("review_id"))
                item["resolved"] = (rid in resolved_ids and bool(corrected_blocks.get(rid))) or rid in discard_review_ids
                if rid in resolved_ids and bool(corrected_blocks.get(rid)):
                    item["reviewed_text"] = corrected_blocks[rid]
                    item["resolution_type"] = "reviewed_text"
                elif rid in discard_review_ids:
                    item["reviewed_text"] = ""
                    item["resolution_type"] = "discarded_non_content_or_duplicate"
        page["line_items"] = line_items
        page["unsafe_text_blocks"] = [item for item in line_items if item.get("type") == "review_required" and not item.get("resolved")]
        page["unresolved_review_items"] = len(unresolved)
        page["reviewed_items_applied"] = reviewed_used
        page["discarded_review_items_applied"] = discarded_review_used
        page["safe_lines_used"] = safe_used
        page["text"] = text
        page["text_plain"] = text
        page["production_safe_text"] = text
        page["include_in_embeddings"] = bool(text)
        page["embedding_readiness"] = "ready_for_formula_reviewed_embedding" if text and not unresolved else "review_required_before_formula_safe_embedding"
        page["text_length_chars"] = len(text)
        page.setdefault("quality_flags", [])
        # Remove old flag if all resolved; add reviewed/corrections flags.
        page["quality_flags"] = [f for f in page["quality_flags"] if f != "formula_or_diagram_text_isolated_for_review"]
        if unresolved:
            page["quality_flags"].append("formula_or_diagram_text_isolated_for_review")
            stats["pages_with_unresolved_review_items"] += 1
            stats["unresolved_review_items"] += len(unresolved)
        if reviewed_used:
            page["quality_flags"].append("curated_formula_or_diagram_text_applied")
            stats["pages_with_reviewed_items"] += 1
            stats["reviewed_items_applied"] += reviewed_used
        if discarded_review_used:
            page["quality_flags"].append("reviewed_non_content_or_duplicate_discarded")
            stats["discarded_review_items_applied"] += discarded_review_used
        page_extractions.append(page)

    data["page_extractions"] = page_extractions
    review_queue = summarize_review_queue(page_extractions)
    data["text_accuracy_status"] = "formula_safe_text_ready" if stats.get("unresolved_review_items", 0) == 0 else "review_required"
    data["production_status"] = "production_ready" if stats.get("unresolved_review_items", 0) == 0 else "review_required_not_formula_safe"
    validation = {
        "status": "passed",
        "errors": [],
        "metrics": {
            "page_extractions": len(page_extractions),
            "reviewed_blocks_in_corrections_json": len(reviewed_blocks),
            "discard_review_ids_in_corrections_json": len(discard_review_ids),
            "reviewed_items_applied": stats.get("reviewed_items_applied", 0),
            "discarded_review_items_applied": stats.get("discarded_review_items_applied", 0),
            "unresolved_review_items": stats.get("unresolved_review_items", 0),
            "pages_with_unresolved_review_items": stats.get("pages_with_unresolved_review_items", 0),
            "pages_with_reviewed_items": stats.get("pages_with_reviewed_items", 0),
        },
    }
    data["extraction"] = {
        "step": 3,
        "status": "step3_curated_corrections_applied",
        "generated_at": utc_now(),
        "generator": "physics_step_3_apply_corrections.py",
        "method": "apply_reviewed_formula_blocks_and_global_replacements",
        "source_step2_json": str(input_json),
        "corrections_json": str(corrections_json) if corrections_json else None,
        "review_queue_output": str(review_queue_output) if review_queue_output else None,
        "statistics": dict(stats),
        "validation": validation,
    }
    return data, build_step3_report(data), review_queue


def build_step3_report(data: dict[str, Any]) -> str:
    metrics = data.get("extraction", {}).get("validation", {}).get("metrics", {})
    lines = [
        "Grade 10 Physics Step 3 curated correction report",
        "=" * 72,
        f"Generated at: {data.get('extraction', {}).get('generated_at')}",
        f"text_accuracy_status: {data.get('text_accuracy_status')}",
        f"production_status: {data.get('production_status')}",
        "",
        "Metrics:",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    if data.get("production_status") != "production_ready":
        lines.extend([
            "",
            "Remaining work:",
            "- Fill reviewed_blocks in Grade10_Physics_formula_corrections.json for all unresolved review IDs.",
            "- Rerun the pipeline. Step 5 will mark production_ready only when unresolved_review_items is 0.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: apply curated formula/table/diagram corrections.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--corrections-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--review-queue", type=Path, default=None)
    parser.add_argument("--placeholder-unreviewed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    for path in [args.output_json, args.report, args.review_queue]:
        if path and path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")
    data, report, queue = build_step3_json(args.input_json.resolve(), args.corrections_json.resolve() if args.corrections_json else None, args.review_queue.resolve() if args.review_queue else None, args.placeholder_unreviewed)
    write_json(args.output_json.resolve(), data)
    if args.review_queue:
        write_json(args.review_queue.resolve(), queue)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(f"Step 3 JSON: {args.output_json.resolve()}")
    if args.review_queue:
        print(f"Step 3 remaining review queue: {args.review_queue.resolve()}")
    if args.report:
        print(f"Step 3 report: {args.report.resolve()}")
    print(f"Step 3 production status: {data.get('production_status')}")


if __name__ == "__main__":
    main()
