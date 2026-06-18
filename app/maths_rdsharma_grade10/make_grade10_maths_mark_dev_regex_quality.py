#!/usr/bin/env python3
"""
Mark a Grade10_Maths JSON as regex-gated development/staging output.

This script is intentionally NOT a production verifier. It only records that the
artifact passed the cheap hard regex gate, and it prevents downstream users from
mistaking regex-clean output for final production-verified text.

Recommended dev flow:
  1) Run Step 6 hard regex gate with --strict-complete.
  2) Run this script with --strict-regex-clean.
  3) In DB ingestion, allow this artifact only for development/staging by checking
     production_embedding_policy.dev_embedding_allowed == true and
     production_embedding_policy.final_production_verified == false.
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
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_dev_regex_quality_report.json"

PROMPT_VERSION = "dev_regex_quality_marker_v33_2026_06_17"


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


def count_dev_embedding_pages(data: dict[str, Any]) -> tuple[int, int, int]:
    pages = (((data.get("extraction") or {}).get("page_extractions")) or [])
    lesson_body = [p for p in pages if bool(p.get("lesson_body_page", False))]
    ready = [p for p in lesson_body if bool(p.get("include_in_embeddings")) and str(p.get("embedding_readiness") or "") == "ready_for_production_embedding"]
    excluded = [p for p in lesson_body if p not in ready]
    return len(lesson_body), len(ready), len(excluded)


def summarize_gate(data: dict[str, Any]) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    qs = extraction.setdefault("quality_summary", {})
    policy = extraction.setdefault("production_embedding_policy", {})
    residual = qs.get("residual_production_audit") or {}
    hard_gate = qs.get("hard_production_gate") or residual or {}

    lesson_body_pages, ready_pages, excluded_pages = count_dev_embedding_pages(data)

    hard_blocker_count = as_int(policy.get("hard_production_blocker_count"), as_int(hard_gate.get("hard_production_blocker_count")))
    remaining_pages = as_int(policy.get("residual_production_remaining_pages"), as_int(hard_gate.get("remaining_residual_pages")))
    remaining_subsections = as_int(policy.get("residual_production_subsection_blockers"), as_int(hard_gate.get("remaining_residual_subsection_count")))
    empty_subsections = as_int(policy.get("residual_production_empty_subsections"), as_int(hard_gate.get("empty_production_subsections")))
    policy_excluded_pages = as_int(policy.get("excluded_lesson_body_pages"), excluded_pages)

    regex_clean = (
        hard_blocker_count == 0
        and remaining_pages == 0
        and remaining_subsections == 0
        and empty_subsections == 0
        and policy_excluded_pages == 0
    )

    return {
        "generated_at_utc": now_utc(),
        "prompt_version": PROMPT_VERSION,
        "source_gate_status": str(policy.get("status") or hard_gate.get("gate_status") or ""),
        "regex_clean": regex_clean,
        "hard_production_blocker_count": hard_blocker_count,
        "remaining_residual_pages": remaining_pages,
        "remaining_residual_subsection_count": remaining_subsections,
        "empty_production_subsections": empty_subsections,
        "excluded_lesson_body_pages": policy_excluded_pages,
        "lesson_body_pages": lesson_body_pages,
        "ready_lesson_body_pages": ready_pages,
        "not_final_production_reason": (
            "Regex gate passed, but full image-vs-text or human verification was not run over all pages."
            if regex_clean else
            "Regex gate still has blockers; not ready even for development embeddings."
        ),
    }


def mark_dev_quality(data: dict[str, Any], *, label: str, reviewer_note: str) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    qs = extraction.setdefault("quality_summary", {})
    policy = extraction.setdefault("production_embedding_policy", {})

    summary = summarize_gate(data)
    regex_clean = bool(summary["regex_clean"])

    previous_status = str(policy.get("status") or "")
    status = "regex_gated_dev_ready_not_final_production" if regex_clean else "regex_gated_dev_needs_qa"

    qs["dev_regex_quality"] = {
        **summary,
        "quality_label": label,
        "status": status,
        "safe_for_dev_db_embeddings": regex_clean,
        "safe_for_final_db_embeddings": False,
        "final_production_verified": False,
        "verification_method": "hard_regex_gate_only",
        "allowed_embedding_fields": [
            "page_extractions[].production_safe_text",
            "chapters[].subsections[].production_subsection_text",
        ],
        "blocked_embedding_fields": [
            "raw_text", "ocr_text", "text", "text_plain", "final_ocr_text",
            "page_extractions[].raw_text", "page_extractions[].text", "page_extractions[].text_plain",
        ],
        "reviewer_note": reviewer_note,
    }

    extraction.update({
        "extraction_quality": label,
        "is_regex_gated_dev": regex_clean,
        "is_final_production_verified": False,
        "safe_for_dev_db_embeddings": regex_clean,
        "safe_for_final_db_embeddings": False,
        "quality_status": status,
        "quality_status_updated_at_utc": now_utc(),
    })

    policy.update({
        "status": status,
        "source_status_before_dev_mark": previous_status,
        "production_complete": False,
        "dev_regex_gated": True,
        "dev_regex_quality_label": label,
        "dev_embedding_allowed": regex_clean,
        "safe_for_dev_db_embeddings": regex_clean,
        "safe_for_final_db_embeddings": False,
        "final_production_verified": False,
        "final_production_verification_method": "not_run_full_book_image_or_human_review",
        "not_final_production_reason": summary["not_final_production_reason"],
        "allowed_embedding_fields": [
            "page_extractions[].production_safe_text",
            "chapters[].subsections[].production_subsection_text",
        ],
        "do_not_embed_fields": [
            "raw_text", "ocr_text", "text", "text_plain", "final_ocr_text",
            "page_extractions[].raw_text", "page_extractions[].text", "page_extractions[].text_plain",
        ],
    })

    extraction["generated_at_utc"] = now_utc()
    return qs["dev_regex_quality"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark Grade10_Maths output as regex-gated dev/staging quality, not final production.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--quality-label", default="regex_gated_dev")
    parser.add_argument("--reviewer-note", default="Development/staging artifact: regex-gated only. Not final production until full image-vs-text verification or human review is complete.")
    parser.add_argument("--strict-regex-clean", action="store_true", help="Fail unless Step 6 hard regex gate has zero blockers.")
    args = parser.parse_args()

    data = load_json(args.input)
    summary = mark_dev_quality(data, label=args.quality_label, reviewer_note=args.reviewer_note)
    write_json(args.output, data)
    write_json(args.report, summary)

    print(f"Dev regex status: {summary['status']}")
    print(f"Regex clean: {summary['regex_clean']}")
    print(f"Hard production blocker count: {summary['hard_production_blocker_count']}")
    print(f"Remaining residual pages: {summary['remaining_residual_pages']}")
    print(f"Remaining residual subsections: {summary['remaining_residual_subsection_count']}")
    print(f"Safe for dev DB embeddings: {summary['safe_for_dev_db_embeddings']}")
    print("Safe for final DB embeddings: False")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")

    if args.strict_regex_clean and not summary["regex_clean"]:
        raise RuntimeError(
            "Regex-gated dev marker failed: Step 6 hard regex gate still has blockers. "
            f"hard_production_blocker_count={summary['hard_production_blocker_count']}, "
            f"remaining_residual_pages={summary['remaining_residual_pages']}, "
            f"remaining_residual_subsections={summary['remaining_residual_subsection_count']}"
        )


if __name__ == "__main__":
    main()
