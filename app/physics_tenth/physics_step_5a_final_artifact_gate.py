#!/usr/bin/env python3
"""Step 5A: final production-text artifact gate.

This gate runs after Step 4 and before final publish. It scans only production-facing
lesson text, not raw_extracted_text/selectable_text/layout_lines. Its purpose is to
catch OCR/math artifacts that slipped through safe_text classification or were copied
into reviewed corrections.

If blockers are found, the script writes a review queue and exits non-zero unless
--allow-artifacts is provided. The final JSON must not be marked production_ready while
this gate has blockers.
"""
from __future__ import annotations

import argparse
import copy
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from physics_step_5_publish_production import (
    _cleanup_decorative_ocr_artifacts,
    _scan_remaining_decorative_ocr_artifacts,
)

from physics_common import (
    CONTEXTUAL_FINAL_ARTIFACT_RE,
    FINAL_ARTIFACT_RE,
    compact_line,
    line_is_formula_or_numeric_risk,
    read_json,
    setup_logging,
    utc_now,
    write_json,
)

# Extra signatures specific to the PDF QA failures reported after the first pass.
# These are intentionally narrow and are only applied to production-facing text.
EXTRA_FINAL_ARTIFACTS: list[tuple[str, re.Pattern[str]]] = [
    ("latex_fraction_or_control_sequence", re.compile(r"\\\\(?:frac|[A-Za-z]{1,8})\b")),
    ("bad_scientific_notation_percent", re.compile(r"(?i)\b0%\s*(?:J\s*/\s*s|W|J)?\b")),
    ("bad_scientific_notation_star", re.compile(r"(?i)\b0\s*\*\b")),
    ("bad_scientific_notation_decimal_percent", re.compile(r"(?i)\b0\.0%\b")),
    ("known_ocr_token", re.compile(r"\b(?:AID|Alii|Lda|NowP|WALA|Vag)\b")),
    ("bad_near_point_00_cm", re.compile(r"(?i)\bnear\s+point\s+(?:is\s+)?00\s*cm\b")),
    ("damaged_lens_formula_noise", re.compile(r"(?i)\b\d?\s*ee\s*ot\b|(?<!\d)\.0(?!\d)")),
    ("bad_ow_f_fragment", re.compile(r"\bow\s+f\b")),
    ("bad_power_formula_350_200", re.compile(r"\bP\s*=\s*350\s*/\s*200\b")),
    ("bad_standalone_10_minus_15", re.compile(r"^\s*10\s*-\s*15\s*=\s*$")),
    ("tiny_junk_safe_text_line", re.compile(r"(?im)^\s*(?:ata|ay|cath|ov|oe|peek|som)\s*$")),
]

PRODUCTION_TEXT_KEYS = {
    "text",
    "text_plain",
    "production_text",
    "production_text_plain",
    "subsection_text",
    "subsection_text_plain",
    "chapter_text",
    "chapter_text_plain",
    "section_text",
    "section_text_plain",
}

RAW_OR_AUDIT_KEYS = {
    "raw_extracted_text",
    "selectable_text",
    "ocr_text",
    "layout_lines",
    "line_items",
    "raw_text",
    "original_text",
}


def detect_final_artifacts(line: str) -> list[str]:
    s = compact_line(line)
    if not s:
        return []
    reasons: list[str] = []
    for reason, pattern in EXTRA_FINAL_ARTIFACTS:
        if pattern.search(s):
            reasons.append(reason)
    if FINAL_ARTIFACT_RE.search(s) and "final_artifact_regex" not in reasons:
        reasons.append("final_artifact_regex")
    if CONTEXTUAL_FINAL_ARTIFACT_RE.search(s) and line_is_formula_or_numeric_risk(s):
        reasons.append("contextual_formula_ocr_token")
    # Catch suspicious "00 cm" in human-eye near-point context even if the line does not include the exact phrase.
    if re.search(r"(?i)\b00\s*cm\b", s) and re.search(r"(?i)\b(?:near\s+point|least\s+distance|distinct\s+vision|eye)\b", s):
        reasons.append("suspicious_00_cm_in_eye_context")
    return sorted(set(reasons))


def line_iter(text: str) -> Iterable[tuple[int, str]]:
    for idx, line in enumerate((text or "").splitlines(), start=1):
        clean = compact_line(line)
        if clean:
            yield idx, clean


def add_item(
    items: list[dict[str, Any]],
    seen: set[tuple[str, int | None, str]],
    *,
    source_path: str,
    field: str,
    text: str,
    line_no: int,
    reasons: list[str],
    page: dict[str, Any] | None = None,
) -> None:
    page_number = page.get("page_number") if page else None
    key = (source_path, int(page_number) if page_number is not None else None, text)
    if key in seen:
        return
    seen.add(key)
    artifact_num = len(items) + 1
    item = {
        "artifact_id": f"final_artifact_{artifact_num:04d}",
        "source_path": source_path,
        "field": field,
        "line_no": line_no,
        "reasons": reasons,
        "text": text,
        "instruction": "Fix this upstream: either adjust Step 2 classification/global cleanup, add exact reviewed correction text, or discard only if visually confirmed non-content.",
    }
    if page:
        item.update({
            "page_number": page.get("page_number"),
            "printed_page_number": page.get("printed_page_number"),
            "printed_page_label": page.get("printed_page_label"),
            "chapter_title": page.get("chapter_title"),
            "section_title": page.get("section_title"),
        })
    items.append(item)


def scan_page_extractions(data: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, str]] = set()
    for page_idx, page in enumerate(data.get("page_extractions") or []):
        if not page.get("include_in_lesson_text"):
            continue
        for field in ["text", "text_plain", "production_text", "production_text_plain", "production_safe_text", "safe_text", "safe_text_plain"]:
            value = page.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            for line_no, line in line_iter(value):
                reasons = detect_final_artifacts(line)
                if reasons:
                    add_item(
                        items,
                        seen,
                        source_path=f"page_extractions[{page_idx}]",
                        field=field,
                        text=line,
                        line_no=line_no,
                        reasons=reasons,
                        page=page,
                    )
    return items


def scan_production_fields_recursive(data: Any, path: str = "$", *, items: list[dict[str, Any]], seen: set[tuple[str, int | None, str]]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in RAW_OR_AUDIT_KEYS:
                continue
            child_path = f"{path}.{key}"
            if isinstance(value, str) and key.startswith("production_"):
                for line_no, line in line_iter(value):
                    reasons = detect_final_artifacts(line)
                    if reasons:
                        add_item(items, seen, source_path=path, field=key, text=line, line_no=line_no, reasons=reasons)
            elif isinstance(value, (dict, list)):
                scan_production_fields_recursive(value, child_path, items=items, seen=seen)
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            scan_production_fields_recursive(value, f"{path}[{idx}]", items=items, seen=seen)


def scan_final_artifacts(data: dict[str, Any]) -> list[dict[str, Any]]:
    page_items = scan_page_extractions(data)
    items = list(page_items)
    seen = {(i.get("source_path", ""), i.get("page_number"), i.get("text", "")) for i in items}
    # Recursively scan chapters/section_index production-facing fields too. This can surface
    # artifacts introduced during aggregation even when page fields look clean.
    for root_key in ["chapters", "section_index"]:
        if root_key in data:
            scan_production_fields_recursive(data[root_key], root_key, items=items, seen=seen)
    return items


def build_queue(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_page = Counter(str(i.get("page_number")) for i in items if i.get("page_number") is not None)
    by_reason = Counter(reason for i in items for reason in i.get("reasons", []))
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "gate": "final_production_text_artifact_gate",
        "status": "passed" if not items else "failed",
        "blocker_count": len(items),
        "instructions": (
            "These are leftover artifacts found in production-facing lesson text only. "
            "They must be fixed upstream before production_ready is valid. Raw PDF/audit fields are not scanned."
        ),
        "summary": {
            "by_page": dict(by_page.most_common()),
            "by_reason": dict(by_reason.most_common()),
        },
        "review_items": items,
    }


def build_report(queue: dict[str, Any]) -> str:
    lines = [
        "Grade 10 Physics final artifact gate report",
        "=" * 72,
        f"Generated at: {queue.get('generated_at')}",
        f"status: {queue.get('status')}",
        f"blocker_count: {queue.get('blocker_count')}",
        "",
        "Top reasons:",
    ]
    for reason, count in (queue.get("summary", {}).get("by_reason") or {}).items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "Top pages:"])
    for page, count in list((queue.get("summary", {}).get("by_page") or {}).items())[:20]:
        lines.append(f"- PDF page {page}: {count}")
    if queue.get("review_items"):
        lines.extend(["", "First blockers:"])
        for item in queue["review_items"][:50]:
            loc = f"PDF page {item.get('page_number')}" if item.get("page_number") else item.get("source_path")
            lines.append(f"- {item.get('artifact_id')} | {loc} | {item.get('field')} line {item.get('line_no')}: {item.get('text')}")
    return "\n".join(lines) + "\n"


def run_gate(input_json: Path, output_json: Path, review_queue: Path, report: Path | None, allow_artifacts: bool = False) -> tuple[dict[str, Any], dict[str, Any], str]:
    src = read_json(input_json)
    data = copy.deepcopy(src)

    # First remove known decorative OCR divider lines from production-facing text.
    # The old gate reported blocker_count=0 because these lines were not matched
    # by the formula/math artifact regexes and production_safe_text was not scanned.
    decorative_cleanup_summary = _cleanup_decorative_ocr_artifacts(data)
    decorative_blockers = _scan_remaining_decorative_ocr_artifacts(data)

    items = scan_final_artifacts(data)
    for blocker in decorative_blockers:
        items.append({
            "artifact_id": f"final_artifact_{len(items) + 1:04d}",
            "source_path": blocker.get("path"),
            "field": str(blocker.get("path", "")).split(".")[-1],
            "line_no": blocker.get("line"),
            "reasons": ["decorative_ocr_noise_remaining_after_cleanup"],
            "text": blocker.get("text"),
            "instruction": "Fix the final decorative OCR cleanup rule before production publish.",
        })
    queue = build_queue(items)
    report_text = build_report(queue)

    stats = dict(data.get("extraction", {}).get("statistics") or {})
    stats["decorative_ocr_lines_removed"] = int(decorative_cleanup_summary.get("decorative_ocr_lines_removed") or 0)
    stats["decorative_ocr_text_fields_cleaned"] = int(decorative_cleanup_summary.get("decorative_ocr_text_fields_cleaned") or 0)
    stats["final_decorative_ocr_blockers"] = len(decorative_blockers)
    stats["final_artifact_blockers"] = len(items)
    data["final_artifact_gate"] = {
        "status": queue["status"],
        "blocker_count": len(items),
        "review_queue": str(review_queue),
        "checked_scope": "include_in_lesson_text=true page text/text_plain/production_safe_text plus production_* fields",
        "decorative_ocr_cleanup": decorative_cleanup_summary,
        "final_decorative_ocr_blockers": len(decorative_blockers),
    }
    data["extraction"] = dict(data.get("extraction") or {})
    data["extraction"]["statistics"] = stats

    if items:
        data["text_accuracy_status"] = "failed_final_artifact_gate"
        data["production_status"] = "review_required_leftover_artifacts"
        validation = data.get("extraction", {}).get("validation") or {}
        validation = dict(validation)
        errors = list(validation.get("errors") or [])
        errors.append(f"Final artifact gate found {len(items)} production-text artifact blocker(s). See {review_queue}.")
        validation["errors"] = errors
        validation["status"] = "failed"
        data["extraction"]["validation"] = validation
    else:
        data.setdefault("text_accuracy_status", "final_artifact_gate_passed")

    write_json(output_json, data)
    write_json(review_queue, queue)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(report_text, encoding="utf-8")
    return data, queue, report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5A: scan production-facing Physics text for leftover OCR/math artifacts.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-artifacts", action="store_true", help="Write output but do not exit non-zero when blockers are found. Not for final production.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    for path in [args.output_json, args.review_queue, args.report]:
        if path and path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")
    _data, queue, _report = run_gate(args.input_json.resolve(), args.output_json.resolve(), args.review_queue.resolve(), args.report.resolve() if args.report else None, args.allow_artifacts)
    print(f"Final artifact gate output JSON: {args.output_json.resolve()}")
    print(f"Final artifact review queue: {args.review_queue.resolve()}")
    if args.report:
        print(f"Final artifact report: {args.report.resolve()}")
    print(f"Final artifact gate status: {queue.get('status')}; blockers={queue.get('blocker_count')}")
    if queue.get("blocker_count") and not args.allow_artifacts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
