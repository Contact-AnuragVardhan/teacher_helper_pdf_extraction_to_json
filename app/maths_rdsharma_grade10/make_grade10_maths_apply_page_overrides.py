#!/usr/bin/env python3
"""
Apply trusted page override JSON to Grade10_Maths_production_ready.json.

This updates production_safe_text and legacy text fields so DB loaders do not ingest old OCR.
It is the production-safe alternative to hardcoding page transcriptions in Python code.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from make_grade10_maths_step_2_production_gate import rebuild_chapters_and_sections  # noqa: E402
from make_grade10_maths_step_3_vision_repair import load_manual_overrides  # noqa: E402
from make_grade10_maths_step_6_residual_production_audit import (  # noqa: E402
    collect_residual_blockers,
    count_empty_production_subsections,
    run_residual_audit,
    write_csv as write_residual_csv,
    write_report as write_residual_report,
)
from make_grade10_maths_known_page_fixes import apply_known_fixes  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_OVERRIDES = THIS_DIR / "Grade10_Maths_page_overrides.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_page_overrides_apply_report.txt"
DEFAULT_RESIDUAL_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_residual_production_audit_report.txt"
DEFAULT_RESIDUAL_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_residual_production_remaining_pages.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_pages_arg(value: str | None) -> set[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_override_for_page(page: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    page_number = page.get("page_number")
    printed = page.get("printed_page_number")
    if page_number is not None and f"page:{int(page_number)}" in overrides:
        return overrides[f"page:{int(page_number)}"]
    if printed is not None and f"printed:{int(printed)}" in overrides:
        return overrides[f"printed:{int(printed)}"]
    return None


def mark_ready_from_override(page: dict[str, Any], override: dict[str, Any]) -> bool:
    text = str(override.get("production_safe_text") or override.get("text") or "").strip()
    if not text:
        return False
    page["production_safe_text"] = text
    page["text"] = text
    page["text_plain"] = text
    page["include_in_embeddings"] = True
    page["embedding_readiness"] = "ready_for_production_embedding"
    page["production_exclusion_reasons"] = []
    page["math_precision_audit_status"] = "approved_by_trusted_page_override"
    page["math_precision_prompt_version"] = override.get("prompt_version") or "trusted_page_override"
    page["manual_override_applied_at_utc"] = now_utc()
    page["manual_override_notes"] = override.get("notes") or "trusted page override applied"
    page["manual_override_source"] = override.get("source") or "manual_page_override"
    if override.get("math_lines") is not None:
        page["math_lines"] = override.get("math_lines") or []
    if override.get("confidence") is not None:
        page["vision_qa_confidence"] = override.get("confidence")
    flags = set(page.get("quality_flags") or [])
    flags.add("trusted_page_override_applied")
    flags.add("production_embedding_ready")
    flags.add("legacy_text_fields_synced_to_production_safe_text")
    flags.discard("production_embedding_excluded_until_math_precision_qa")
    flags.discard("production_embedding_excluded_until_vision_qa")
    page["quality_flags"] = sorted(flags)
    return True


def sync_legacy_sections(data: dict[str, Any]) -> int:
    changed = 0
    extraction = data.get("extraction", {})
    for chapter in extraction.get("chapters", []) or []:
        prod_ch = chapter.get("production_chapter_text") or chapter.get("chapter_text") or ""
        if prod_ch:
            for field in ("chapter_text", "chapter_text_plain", "text_plain"):
                if chapter.get(field) != prod_ch:
                    chapter[field] = prod_ch
                    changed += 1
        for sub in chapter.get("subsections", []) or []:
            prod = sub.get("production_subsection_text") or ""
            if prod:
                for field in ("subsection_text", "subsection_text_plain", "text_plain"):
                    if sub.get(field) != prod:
                        sub[field] = prod
                        changed += 1
    for key in ("lessons", "sections", "subsections"):
        for item in extraction.get(key, []) or []:
            prod = item.get("production_subsection_text") or item.get("production_lesson_text") or item.get("production_text") or ""
            if prod:
                for field in ("subsection_text", "subsection_text_plain", "lesson_text", "text_plain"):
                    if field in item and item.get(field) != prod:
                        item[field] = prod
                        changed += 1
    return changed


def refresh_policy(data: dict[str, Any], *, applied_pages: list[int], legacy_synced_count: int, residual_summary: dict[str, Any]) -> dict[str, Any]:
    """Record override application and keep policy consistent with the residual final gate.

    Earlier versions marked production_complete_ready immediately after overrides were applied.
    That was wrong because stale math audit counters and unrelated ready-page corruption could
    still exist. Now production_complete_ready is allowed only when the residual production
    audit is clean.
    """
    extraction = data.setdefault("extraction", {})
    pages = extraction.get("page_extractions", []) or []
    lesson_body = [p for p in pages if p.get("content_type") == "lesson_body" and p.get("include_in_lesson_text") is True]
    ready = [p for p in lesson_body if p.get("include_in_embeddings") is True and p.get("embedding_readiness") == "ready_for_production_embedding" and str(p.get("production_safe_text") or "").strip()]
    excluded = [p for p in lesson_body if p not in ready]
    residual_remaining = int(residual_summary.get("remaining_residual_pages") or 0)
    empty_subsections = int(residual_summary.get("empty_production_subsections") or 0)
    complete = len(excluded) == 0 and residual_remaining == 0 and empty_subsections == 0

    qs = extraction.setdefault("quality_summary", {})
    qs["trusted_page_overrides"] = {
        "generated_at_utc": now_utc(),
        "applied_page_count": len(applied_pages),
        "applied_pages": applied_pages,
        "legacy_text_fields_synced_count": legacy_synced_count,
        "lesson_body_pages": len(lesson_body),
        "ready_lesson_body_pages": len(ready),
        "excluded_lesson_body_pages": len(excluded),
        "residual_remaining_pages_after_overrides": residual_remaining,
        "empty_production_subsections_after_overrides": empty_subsections,
    }
    policy = extraction.setdefault("production_embedding_policy", {})
    policy["trusted_page_overrides_applied"] = True
    policy["trusted_page_overrides_applied_at_utc"] = now_utc()
    policy["production_complete"] = complete
    policy["status"] = "production_complete_ready" if complete else "production_safe_gated_needs_residual_production_qa"
    policy["excluded_lesson_body_pages"] = len(excluded)
    policy["ready_lesson_body_pages"] = len(ready)
    policy["residual_production_remaining_pages"] = residual_remaining
    policy["residual_production_empty_subsections"] = empty_subsections

    # Clear stale math precision failure state only when the final residual gate is truly clean.
    if complete:
        policy["math_precision_remaining_pages"] = 0
        policy["math_precision_empty_production_subsections"] = 0
        policy["math_precision_errors"] = []
        math = qs.setdefault("math_precision_audit", {})
        math["superseded_by_trusted_page_overrides"] = True
        math["remaining_math_precision_pages"] = 0
        math["excluded_lesson_body_pages"] = 0
        math["empty_production_subsections"] = 0
        math["remaining_reason_counts"] = {}
    extraction["generated_at_utc"] = now_utc()
    return qs["trusted_page_overrides"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply trusted page overrides to production JSON and sync legacy text fields.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--residual-report", type=Path, default=DEFAULT_RESIDUAL_REPORT)
    parser.add_argument("--residual-csv", type=Path, default=DEFAULT_RESIDUAL_CSV)
    parser.add_argument("--pages", default=None, help="Optional PDF pages to apply, e.g. 24,28,503")
    parser.add_argument("--strict-complete", action="store_true")
    args = parser.parse_args()
    data = load_json(args.input)
    overrides = load_manual_overrides(args.overrides)
    if not overrides:
        raise RuntimeError(f"No page overrides loaded from {args.overrides}")
    selected = parse_pages_arg(args.pages)
    pages = data.get("extraction", {}).get("page_extractions", []) or []
    applied: list[int] = []
    for page in pages:
        page_number = int(page.get("page_number") or 0)
        if selected is not None and page_number not in selected:
            continue
        override = get_override_for_page(page, overrides)
        if not override:
            continue
        if mark_ready_from_override(page, override):
            applied.append(page_number)
    if not applied:
        raise RuntimeError("No overrides matched pages in the JSON. Check page_number/printed_page_number.")
    rebuild_chapters_and_sections(data)
    legacy_synced = sync_legacy_sections(data)

    # Trusted page overrides can still reintroduce reviewer-known OCR/math artifacts
    # (for example page 13's cube/square mix). Apply deterministic known fixes after
    # overrides and before the final residual gate.
    known_fix_result = apply_known_fixes(data)
    known_fixed_pages = [int(item.get("page_number") or 0) for item in known_fix_result.get("applied", [])]

    # Final consistency gate after overrides. This catches pages that were already marked
    # ready but still contain corruption, such as the page 128 cross-multiplication OCR
    # fragments. It also syncs page/section legacy text fields.
    residual_summary, residual_blockers, _empty = run_residual_audit(data)
    write_residual_csv(args.residual_csv, residual_blockers)
    write_residual_report(args.residual_report, input_path=args.input, output_path=args.output, remaining_csv=args.residual_csv, summary=residual_summary)

    summary = refresh_policy(data, applied_pages=sorted(applied), legacy_synced_count=legacy_synced, residual_summary=residual_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "Grade10 Maths Trusted Page Override Apply Report",
        f"Generated: {now_utc()}",
        f"Input: {args.input}",
        f"Output: {args.output}",
        f"Overrides: {args.overrides}",
        f"Applied pages: {len(applied)}",
        f"Applied page numbers: {', '.join(map(str, sorted(applied)))}",
        f"Legacy text fields synced: {legacy_synced}",
        f"Known reviewer fixes after overrides: {len(known_fixed_pages)} pages {sorted(known_fixed_pages)}",
        f"Lesson-body pages: {summary['lesson_body_pages']}",
        f"Ready lesson-body pages: {summary['ready_lesson_body_pages']}",
        f"Excluded lesson-body pages: {summary['excluded_lesson_body_pages']}",
        f"Residual remaining pages: {summary.get('residual_remaining_pages_after_overrides')}",
        f"Residual report: {args.residual_report}",
        f"Residual CSV: {args.residual_csv}",
        f"Production complete: {summary['excluded_lesson_body_pages'] == 0 and summary.get('residual_remaining_pages_after_overrides') == 0 and summary.get('empty_production_subsections_after_overrides') == 0}",
    ]
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Applied trusted overrides: {len(applied)} pages")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")
    if args.strict_complete and (summary["excluded_lesson_body_pages"] or summary.get("residual_remaining_pages_after_overrides") or summary.get("empty_production_subsections_after_overrides")):
        raise RuntimeError(
            "Strict complete failed after trusted overrides: "
            f"excluded_lesson_body_pages={summary['excluded_lesson_body_pages']}, "
            f"residual_remaining_pages={summary.get('residual_remaining_pages_after_overrides')}, "
            f"empty_production_subsections={summary.get('empty_production_subsections_after_overrides')}. "
            f"See {args.residual_report} and {args.residual_csv}."
        )


if __name__ == "__main__":
    main()
