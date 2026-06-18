#!/usr/bin/env python3
"""
Finalize Grade10_Maths as true production quality ONLY after the expensive full
image-vs-production-text verifier and the hard regex gate have both passed.

This script is intentionally strict. It is the counterpart to
make_grade10_maths_mark_dev_regex_quality.py:

- dev marker = regex-clean only, not final production
- this finalizer = full-book image verification + hard gate clean

Recommended production flow:
  1) Run Step 7 full image verifier over the full book with --strict-complete.
  2) Re-run Step 6 residual hard gate with --strict-complete.
  3) Run this script with --strict-complete.

It refuses to mark final production if Step 7 was only run for a page subset.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_final_production_quality_report.json"

FINALIZER_VERSION = "final_production_quality_v34_2026_06_17"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def is_final_embedding_page(page: dict[str, Any]) -> bool:
    return (
        page.get("content_type") == "lesson_body"
        and page.get("include_in_lesson_text") is True
        and page.get("include_in_embeddings") is True
        and str(page.get("embedding_readiness") or "") == "ready_for_production_embedding"
        and bool(str(page.get("production_safe_text") or "").strip())
    )


def collect_final_page_stats(data: dict[str, Any]) -> dict[str, Any]:
    pages = (((data.get("extraction") or {}).get("page_extractions")) or [])
    final_pages = [p for p in pages if is_final_embedding_page(p)]
    verified = [p for p in final_pages if str(p.get("full_image_verification_status") or "").lower() == "pass"]
    missing = [p for p in final_pages if str(p.get("full_image_verification_status") or "").lower() != "pass"]
    return {
        "final_embedding_page_count": len(final_pages),
        "full_image_verified_page_count": len(verified),
        "unverified_final_page_count": len(missing),
        "unverified_final_pages": [int(p.get("page_number")) for p in missing if p.get("page_number") is not None][:500],
    }


def summarize_production_readiness(data: dict[str, Any], *, expected_page_count: int | None) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    qs = extraction.setdefault("quality_summary", {})
    policy = extraction.setdefault("production_embedding_policy", {})

    residual = qs.get("residual_production_audit") or qs.get("hard_production_gate") or {}
    full = qs.get("full_image_production_verify") or {}
    page_stats = collect_final_page_stats(data)

    hard_blocker_count = as_int(policy.get("hard_production_blocker_count"), as_int(residual.get("hard_production_blocker_count")))
    remaining_residual_pages = as_int(policy.get("residual_production_remaining_pages"), as_int(residual.get("remaining_residual_pages")))
    remaining_subsections = as_int(policy.get("residual_production_subsection_blockers"), as_int(residual.get("remaining_residual_subsection_count")))
    empty_subsections = as_int(policy.get("residual_production_empty_subsections"), as_int(residual.get("empty_production_subsections")))
    excluded_lesson_body_pages = as_int(policy.get("excluded_lesson_body_pages"), as_int(residual.get("excluded_lesson_body_pages")))

    full_applied = bool(policy.get("full_image_production_verify_applied") or full.get("applied"))
    full_failed_count = as_int(policy.get("full_image_failed_page_count"), as_int(full.get("remaining_failed_page_count")))
    full_checked_count = as_int(full.get("checked_page_count"))
    full_strict_all = bool(full.get("strict_scope_all_ready_pages"))

    expected_ok = True
    expected_reason = ""
    if expected_page_count is not None and expected_page_count > 0:
        expected_ok = page_stats["final_embedding_page_count"] == expected_page_count and full_checked_count >= expected_page_count
        if not expected_ok:
            expected_reason = (
                f"expected_page_count={expected_page_count}, "
                f"final_embedding_page_count={page_stats['final_embedding_page_count']}, "
                f"full_checked_count={full_checked_count}"
            )

    failures: list[str] = []
    if not full_applied:
        failures.append("full_image_production_verify_not_applied")
    if not full_strict_all:
        failures.append("full_image_verify_was_not_full_book_scope")
    if full_failed_count != 0:
        failures.append(f"full_image_failed_page_count={full_failed_count}")
    if page_stats["unverified_final_page_count"] != 0:
        failures.append(f"unverified_final_page_count={page_stats['unverified_final_page_count']}")
    if hard_blocker_count != 0:
        failures.append(f"hard_production_blocker_count={hard_blocker_count}")
    if remaining_residual_pages != 0:
        failures.append(f"remaining_residual_pages={remaining_residual_pages}")
    if remaining_subsections != 0:
        failures.append(f"remaining_residual_subsections={remaining_subsections}")
    if empty_subsections != 0:
        failures.append(f"empty_production_subsections={empty_subsections}")
    if excluded_lesson_body_pages != 0:
        failures.append(f"excluded_lesson_body_pages={excluded_lesson_body_pages}")
    if not expected_ok:
        failures.append("expected_full_book_page_count_not_met: " + expected_reason)

    production_ready = not failures
    return {
        "generated_at_utc": now_utc(),
        "finalizer_version": FINALIZER_VERSION,
        "production_ready": production_ready,
        "failures": failures,
        "residual_gate": {
            "hard_production_blocker_count": hard_blocker_count,
            "remaining_residual_pages": remaining_residual_pages,
            "remaining_residual_subsections": remaining_subsections,
            "empty_production_subsections": empty_subsections,
            "excluded_lesson_body_pages": excluded_lesson_body_pages,
        },
        "full_image_gate": {
            "applied": full_applied,
            "strict_scope_all_ready_pages": full_strict_all,
            "checked_page_count": full_checked_count,
            "remaining_failed_page_count": full_failed_count,
            "expected_page_count": expected_page_count,
        },
        **page_stats,
    }


def mark_final_production(data: dict[str, Any], summary: dict[str, Any]) -> None:
    extraction = data.setdefault("extraction", {})
    qs = extraction.setdefault("quality_summary", {})
    policy = extraction.setdefault("production_embedding_policy", {})

    qs["final_production_quality"] = summary

    # Clear/supersede dev/staging marker fields if they exist.
    dev_summary = qs.get("dev_regex_quality")
    if isinstance(dev_summary, dict):
        dev_summary["superseded_by_final_production_quality"] = True
        dev_summary["superseded_at_utc"] = now_utc()

    extraction.update({
        "extraction_quality": "production_image_verified",
        "quality_status": "production_complete_ready",
        "quality_status_updated_at_utc": now_utc(),
        "is_regex_gated_dev": False,
        "is_final_production_verified": True,
        "safe_for_dev_db_embeddings": True,
        "safe_for_final_db_embeddings": True,
        "production_complete": True,
    })

    policy.update({
        "status": "production_complete_ready",
        "production_complete": True,
        "dev_regex_gated": False,
        "dev_embedding_allowed": True,
        "safe_for_dev_db_embeddings": True,
        "safe_for_final_db_embeddings": True,
        "final_production_verified": True,
        "final_production_verified_at_utc": now_utc(),
        "final_production_verification_method": "full_book_image_vs_text_verified_plus_hard_regex_gate",
        "final_production_quality_version": FINALIZER_VERSION,
        "not_final_production_reason": "",
        "allowed_embedding_fields": [
            "page_extractions[].production_safe_text",
            "chapters[].subsections[].production_subsection_text",
        ],
        "do_not_embed_fields": [
            "raw_text", "ocr_text",
            "page_extractions[].raw_text", "page_extractions[].ocr_text",
        ],
    })
    extraction["generated_at_utc"] = now_utc()


def mark_not_final(data: dict[str, Any], summary: dict[str, Any]) -> None:
    extraction = data.setdefault("extraction", {})
    qs = extraction.setdefault("quality_summary", {})
    policy = extraction.setdefault("production_embedding_policy", {})
    qs["final_production_quality"] = summary
    extraction.update({
        "is_final_production_verified": False,
        "safe_for_final_db_embeddings": False,
        "quality_status": "production_needs_full_image_or_hard_gate_qa",
        "quality_status_updated_at_utc": now_utc(),
    })
    policy.update({
        "safe_for_final_db_embeddings": False,
        "final_production_verified": False,
        "final_production_verification_method": "failed_finalizer_checks",
        "not_final_production_reason": "; ".join(summary.get("failures") or []),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Grade10 Maths as production only after full image verifier and hard gate pass.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-page-count", type=int, default=0, help="Optional expected final lesson-body embedding page count, e.g. 744.")
    parser.add_argument("--strict-complete", action="store_true", help="Raise if the artifact cannot be marked final production.")
    args = parser.parse_args()

    data = load_json(args.input)
    expected = args.expected_page_count if args.expected_page_count > 0 else None
    summary = summarize_production_readiness(data, expected_page_count=expected)

    if summary["production_ready"]:
        mark_final_production(data, summary)
    else:
        mark_not_final(data, summary)

    write_json(args.output, data)
    write_json(args.report, summary)

    print(f"Final production ready: {summary['production_ready']}")
    print(f"Full image verified pages: {summary['full_image_verified_page_count']}/{summary['final_embedding_page_count']}")
    print(f"Unverified final pages: {summary['unverified_final_page_count']}")
    print(f"Hard production blocker count: {summary['residual_gate']['hard_production_blocker_count']}")
    print(f"Remaining residual pages: {summary['residual_gate']['remaining_residual_pages']}")
    print(f"Remaining residual subsections: {summary['residual_gate']['remaining_residual_subsections']}")
    print(f"Safe for final DB embeddings: {summary['production_ready']}")
    if summary.get("failures"):
        print("Failures:")
        for item in summary["failures"]:
            print(f"- {item}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")

    if args.strict_complete and not summary["production_ready"]:
        raise RuntimeError("Final production quality check failed: " + "; ".join(summary.get("failures") or []))


if __name__ == "__main__":
    main()
