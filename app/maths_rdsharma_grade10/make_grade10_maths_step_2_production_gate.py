#!/usr/bin/env python3
"""
Step 2: production-safe cleanup and embedding gate for Grade10_Maths.

This script does not pretend that OCR has perfect symbolic math fidelity. Instead it:
- applies conservative cleanup only;
- marks risky pages as not safe for embeddings;
- preserves every page in the JSON for later vision/Mathpix/manual QA;
- rebuilds chapter/lesson/subsection production text from only ready pages.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(os.environ.get("GRADE10_MATHS_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("GRADE10_MATHS_OUTPUT_DIR", DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"))
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_step1_base_extraction.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_validation_report.txt"
DEFAULT_QA_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_pages_requiring_vision_qa.csv"

SAFE_NON_ASCII = set("₹°²³√×÷≤≥≠−–—'\"πθαβγ∆Δ∠⊥∥∴±∞∑")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MATH_CONTEXT_RE = re.compile(r"[=+\-−*/×÷^√<>%°]|\b(sin|cos|tan|cot|sec|cosec|log|HCF|LCM)\b", re.I)
AMBIGUOUS_GARBLED_RE = re.compile(r"[�]|(?:[A-Za-z]?\uFFFD)|\b(?:हि|नि|शि|छि|प्|वा|कर|नर|कर्थ|लि)\b")


def slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def compact_author_slug(value: Any) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", str(value or ""))
    if not parts:
        return ""
    if len(parts) >= 2 and all(len(p) == 1 for p in parts[:-1]):
        return "".join(p.lower() for p in parts)
    return slugify(" ".join(parts))


def normalize_school_slug(value: Any) -> str:
    slug = slugify(value)
    slug = re.sub(r"(^|-)school($|-)", r"\1", slug).strip("-")
    return re.sub(r"-+", "-", slug).strip("-")


def build_document_identity(data: dict[str, Any]) -> tuple[str, str]:
    metadata = data.get("metadata") or {}
    extraction = data.get("extraction") or {}
    grade_slug = slugify(metadata.get("grade") or metadata.get("class_name") or "class-10")
    subject_slug = slugify(extraction.get("subject") or "maths")
    publisher_slug = slugify(metadata.get("publisher") or "dhanpat-rai-publications")
    school_slug = normalize_school_slug(metadata.get("school_name") or "mother-miracle")
    author_slug = compact_author_slug(extraction.get("author") or "rd-sharma")
    book_slug = slugify(extraction.get("book_title") or "grade10-maths")
    identity_slug = author_slug or book_slug
    default_document_id = "-".join(part for part in [subject_slug, identity_slug, grade_slug, publisher_slug] if part)
    default_document_key = "-".join(part for part in [school_slug, grade_slug, subject_slug, identity_slug] if part)
    return (
        os.environ.get("GRADE10_MATHS_DOCUMENT_ID", default_document_id),
        os.environ.get("GRADE10_MATHS_DOCUMENT_KEY", default_document_key),
    )


def clean_math_context_line(line: str) -> str:
    if not MATH_CONTEXT_RE.search(line):
        return line
    # Avoid aggressive math rewrites. Only fix isolated OCR confusions in obvious positions.
    line = re.sub(r"(?<=[=+\-×x*/÷(\[{\s])O(?=[\s=+\-×x*/÷)\]}.,;:])", "0", line)
    line = re.sub(r"(?<=\d)O(?=\d)", "0", line)
    line = re.sub(r"(?<=[(\[{+\-×x*/÷=\s])I(?=[)\]}.,\s,+\-×x*/÷=])", "1", line)
    line = re.sub(r"(?<=[(\[{+\-×x*/÷=\s])l(?=[)\]}.,\s,+\-×x*/÷=])", "1", line)
    line = re.sub(r"(?<=\d)l(?=\d)", "1", line)
    line = re.sub(r"(?<=\d)I(?=\d)", "1", line)
    return line


def clean_text(text: str) -> tuple[str, list[str]]:
    if not isinstance(text, str):
        return text, []
    original = text
    fixes: list[str] = []
    text = text.replace("\x00", "")
    text = CONTROL_RE.sub("", text)
    replacements = [
        (r"\bwitha\b", "with a", "witha_to_with_a"),
        (r"\bisa\b", "is a", "isa_to_is_a"),
        (r"\bLeta\b", "Let a", "Leta_to_Let_a"),
        (r"\bIfa\b", "If a", "Ifa_to_If_a"),
        (r"\bwhereQ0\b", "where 0", "whereQ0_to_where_0"),
        (r"\bQ\.E\.D\b", "Q.E.D.", "qed_punctuation"),
        (r"\bearliar\b", "earlier", "earliar_to_earlier"),
        (r"\bconsistsing\b", "consisting", "consistsing_to_consisting"),
        (r"\binvovling\b", "involving", "invovling_to_involving"),
        (r"\bdemertis\b", "demerits", "demertis_to_demerits"),
    ]
    for pat, repl, label in replacements:
        new, n = re.subn(pat, repl, text)
        if n:
            fixes.append(f"{label}:{n}")
            text = new
    cleaned_lines: list[str] = []
    changed = 0
    for line in text.splitlines():
        new_line = clean_math_context_line(line)
        if new_line != line:
            changed += 1
        cleaned_lines.append(new_line.rstrip())
    if changed:
        fixes.append(f"math_context_O_I_l_cleanup_lines:{changed}")
    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if text != original and not fixes:
        fixes.append("generic_text_changed")
    return text, fixes


def clean_any(value: Any) -> tuple[Any, list[str]]:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        out = []
        fixes: list[str] = []
        for x in value:
            y, f = clean_any(x)
            out.append(y)
            fixes.extend(f)
        return out, fixes
    if isinstance(value, dict):
        out = {}
        fixes: list[str] = []
        for k, v in value.items():
            if k == "selectable_text":
                out[k] = v
                continue
            y, f = clean_any(v)
            out[k] = y
            if k in {"text", "text_plain", "ocr_text", "math_lines", "extracted_blocks"}:
                fixes.extend(f)
        return out, fixes
    return value, []


def page_failure_reasons(page: dict[str, Any]) -> list[str]:
    text = page.get("text") or ""
    reasons: list[str] = []
    if page.get("content_type") != "lesson_body":
        reasons.append("not_lesson_body")
    if page.get("assignment_status") not in {"assigned_to_chapter"}:
        reasons.append("not_assigned_to_chapter")
    for flag in page.get("quality_flags") or []:
        flag = str(flag)
        if flag in {"blank_or_no_ocr_text_detected", "tesseract_ocr_error", "suspicious_unicode_in_ocr_text"}:
            reasons.append(flag)
    if not text.strip():
        reasons.append("blank_or_no_ocr_text_detected")
    if len(text.strip()) < 80 and page.get("content_type") == "lesson_body":
        reasons.append("very_low_text_length_requires_qa")
    if DEVANAGARI_RE.search(text):
        reasons.append("remaining_devanagari_or_hindi_ocr_garbage_requires_vision_qa")
    if AMBIGUOUS_GARBLED_RE.search(text):
        reasons.append("remaining_garbled_unicode_requires_vision_qa")
    weird = [ch for ch in text if ord(ch) > 127 and ch not in SAFE_NON_ASCII]
    weird_ratio = len(weird) / max(len(text), 1)
    if weird_ratio > 0.003 or len(weird) >= 15:
        reasons.append(f"too_many_unexpected_non_ascii_chars:{len(weird)}")
    # Very dense symbolic layouts can be readable to humans but unsafe as exact OCR text.
    equation_markers = len(re.findall(r"[=√/^]|\bfrac\b|sin|cos|tan|cot|sec|cosec", text, re.I))
    short_math_lines = sum(1 for line in text.splitlines() if MATH_CONTEXT_RE.search(line) and len(line.strip()) < 90)
    if equation_markers >= 90 or short_math_lines >= 65:
        reasons.append("dense_formula_layout_requires_vision_or_mathpix_qa")
    return sorted(set(reasons))


def format_page_block(page: dict[str, Any], *, text_field: str = "text") -> str:
    pp = page.get("printed_page_number") or page.get("printed_page_label")
    return f"[PDF page {page.get('page_number')} / printed page {pp}]\n{(page.get(text_field) or '').strip()}".strip()


def get_run_scope(data: dict[str, Any]) -> dict[str, Any]:
    return (((data.get("extraction") or {}).get("quality_summary") or {}).get("run_scope") or {})


def validate_run_scope(data: dict[str, Any], *, allow_partial: bool) -> list[str]:
    run_scope = get_run_scope(data)
    blockers: list[str] = []
    if not run_scope:
        blockers.append("missing_step1_run_scope")
    if not run_scope.get("is_full_book_run"):
        blockers.append("not_full_book_run")
    if not run_scope.get("is_full_content_run"):
        blockers.append("not_full_content_run")
    if run_scope.get("missing_selected_pages"):
        blockers.append("missing_selected_pages")
    if run_scope.get("missing_content_page_count", 0):
        blockers.append("missing_content_pages")
    if run_scope.get("chapter_coverage_issues"):
        blockers.append("chapter_coverage_incomplete")
    if run_scope.get("subsection_coverage_issues"):
        blockers.append("subsection_coverage_incomplete")
    if blockers and not allow_partial:
        raise SystemExit(
            "Production gate blocked because Step 1 was not a complete full-book extraction. "
            f"Blockers: {', '.join(sorted(set(blockers)))}. "
            "For smoke tests pass --allow-partial; do not name that output production_ready."
        )
    return sorted(set(blockers))


def rebuild_chapters_and_sections(data: dict[str, Any]) -> None:
    pages = data.get("extraction", {}).get("page_extractions", []) or []
    pages_by_chapter: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in pages:
        if p.get("chapter_number") is not None and p.get("chapter_title"):
            pages_by_chapter[(str(p.get("chapter_number")), str(p.get("chapter_title")))].append(p)

    for chapter in data.get("extraction", {}).get("chapters", []) or []:
        key = (str(chapter.get("chapter_number")), str(chapter.get("chapter_title")))
        cpages = sorted(pages_by_chapter.get(key, []), key=lambda p: int(p.get("page_number") or 0))
        for lesson in chapter.get("lessons", []) or []:
            start = int(lesson.get("start_page") or 0)
            end = int(lesson.get("end_page") or 0)
            lesson_pages = [p for p in cpages if start <= int(p.get("page_number") or 0) <= end and p.get("include_in_lesson_text")]
            production_pages = [p for p in lesson_pages if p.get("include_in_embeddings") is True]
            excluded = [{"page_number": p.get("page_number"), "printed_page_number": p.get("printed_page_number"), "reasons": p.get("production_exclusion_reasons") or []} for p in lesson_pages if p.get("include_in_embeddings") is not True]
            lesson["lesson_text"] = "\n\n".join(format_page_block(p) for p in lesson_pages if (p.get("text") or "").strip())
            lesson["text_plain"] = lesson["lesson_text"]
            lesson["production_lesson_text"] = "\n\n".join(format_page_block(p, text_field="production_safe_text") for p in production_pages if (p.get("production_safe_text") or "").strip())
            lesson["production_indexed_page_numbers"] = [p.get("page_number") for p in production_pages]
            lesson["production_printed_page_numbers"] = [p.get("printed_page_number") for p in production_pages]
            lesson["production_excluded_pages"] = excluded
            lesson["production_page_count"] = len(production_pages)
            lesson["include_in_embeddings"] = len(production_pages) > 0
            flags = set(lesson.get("quality_flags") or [])
            if excluded:
                flags.add("some_pages_excluded_from_production_embeddings")
            if not production_pages:
                flags.add("lesson_requires_vision_qa_before_embedding")
            lesson["quality_flags"] = sorted(flags)
        for sub in chapter.get("subsections", []) or []:
            start = int(sub.get("start_page") or sub.get("start_pdf_page") or 0)
            end = int(sub.get("end_page") or sub.get("end_pdf_page") or 0)
            sub_pages = [p for p in cpages if start <= int(p.get("page_number") or 0) <= end and p.get("include_in_lesson_text")]
            production_pages = [p for p in sub_pages if p.get("include_in_embeddings") is True]
            excluded = [{"page_number": p.get("page_number"), "printed_page_number": p.get("printed_page_number"), "reasons": p.get("production_exclusion_reasons") or []} for p in sub_pages if p.get("include_in_embeddings") is not True]
            sub["subsection_text"] = "\n\n".join(format_page_block(p) for p in sub_pages if (p.get("text") or "").strip())
            sub["subsection_text_plain"] = sub["subsection_text"]
            sub["text_plain"] = sub["subsection_text"]
            sub["production_subsection_text"] = "\n\n".join(format_page_block(p, text_field="production_safe_text") for p in production_pages if (p.get("production_safe_text") or "").strip())
            sub["production_indexed_page_numbers"] = [p.get("page_number") for p in production_pages]
            sub["production_printed_page_numbers"] = [p.get("printed_page_number") for p in production_pages]
            sub["production_excluded_pages"] = excluded
            sub["production_page_count"] = len(production_pages)
            sub["include_in_embeddings"] = len(production_pages) > 0
            flags = set(sub.get("quality_flags") or [])
            if excluded:
                flags.add("some_pages_excluded_from_production_embeddings")
            if not production_pages:
                flags.add("subsection_requires_vision_qa_before_embedding")
            sub["quality_flags"] = sorted(flags)

    section_index = data.get("extraction", {}).get("section_index", []) or []
    by_chapter = {str(ch.get("chapter_number")): copy.deepcopy(ch.get("subsections") or []) for ch in data.get("extraction", {}).get("chapters", []) or []}
    for sec in section_index:
        key = str(sec.get("chapter_number") or sec.get("section_number"))
        if key in by_chapter:
            sec["subsections"] = copy.deepcopy(by_chapter[key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-safe cleanup/gating for Grade10_Maths OCR JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--qa-csv", type=Path, default=DEFAULT_QA_CSV)
    parser.add_argument("--allow-partial", action="store_true", help="Allow smoke/partial inputs, but mark output not production-ready.")
    parser.add_argument("--strict-complete", action="store_true", help="Fail if any lesson_body page is excluded from production embeddings.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    run_scope_blockers = validate_run_scope(data, allow_partial=args.allow_partial)

    document_id, document_key = build_document_identity(data)
    data["documentId"] = document_id
    data["document_key"] = document_key
    metadata = data.setdefault("metadata", {})
    metadata["document_key"] = document_key
    metadata["source_type"] = metadata.get("source_type") or "textbook_pdf"

    cleanup_counter = Counter()
    pages = data.get("extraction", {}).get("page_extractions", []) or []
    for page in pages:
        page_fixes: list[str] = []
        for field in ["text", "text_plain", "ocr_text"]:
            if isinstance(page.get(field), str):
                cleaned, fixes = clean_text(page[field])
                page[field] = cleaned
                page_fixes.extend(fixes)
        if isinstance(page.get("math_lines"), list):
            cleaned, fixes = clean_any(page["math_lines"])
            page["math_lines"] = cleaned
            page_fixes.extend(fixes)
        if isinstance(page.get("extracted_blocks"), list):
            cleaned, fixes = clean_any(page["extracted_blocks"])
            page["extracted_blocks"] = cleaned
            page_fixes.extend(fixes)
        for fix in page_fixes:
            key = fix.split(":", 1)[0]
            val = fix.split(":")[-1]
            cleanup_counter[key] += int(val) if val.isdigit() else 1
        if page_fixes:
            page.setdefault("cleanup_stats", {})["production_cleanup_fixes"] = page_fixes

        reasons = page_failure_reasons(page)
        page["production_exclusion_reasons"] = reasons
        flags = set(page.get("quality_flags") or [])
        if reasons:
            flags.add("production_embedding_excluded_until_vision_qa")
            page["include_in_embeddings"] = False
            page["embedding_readiness"] = "needs_vision_qa_before_production_embedding"
            page["production_safe_text"] = ""
        else:
            flags.add("production_embedding_ready")
            page["include_in_embeddings"] = True
            page["embedding_readiness"] = "ready_for_production_embedding"
            page["production_safe_text"] = page.get("text") or ""
        page["quality_flags"] = sorted(flags)

    rebuild_chapters_and_sections(data)

    ready_pages = [p for p in pages if p.get("include_in_embeddings") is True]
    excluded_pages = [p for p in pages if p.get("include_in_embeddings") is not True]
    reason_counts = Counter()
    for p in excluded_pages:
        reason_counts.update(p.get("production_exclusion_reasons") or [])

    lesson_body_pages = [p for p in pages if p.get("content_type") == "lesson_body"]
    excluded_lesson_body_pages = [p for p in lesson_body_pages if p.get("include_in_embeddings") is not True]
    if args.strict_complete and (run_scope_blockers or excluded_lesson_body_pages):
        raise SystemExit(
            "Strict production gate failed: "
            f"run_scope_blockers={run_scope_blockers}, "
            f"excluded_lesson_body_pages={len(excluded_lesson_body_pages)}. "
            "Review the QA CSV and fix with vision/Mathpix/manual corrections before embedding."
        )

    if run_scope_blockers:
        gate_status = "smoke_or_partial_not_production_ready"
    elif excluded_lesson_body_pages:
        gate_status = "production_safe_gated_needs_qa"
    else:
        gate_status = "production_complete_ready"

    extraction = data.setdefault("extraction", {})
    extraction["production_embedding_policy"] = {
        "status": gate_status,
        "meaning": "Only pages with include_in_embeddings=true and embedding_readiness=ready_for_production_embedding should be embedded.",
        "recommended_embedding_text_field": "production_safe_text",
        "embed_only_when": {"include_in_embeddings": True, "embedding_readiness": "ready_for_production_embedding"},
        "do_not_embed_when_flags_include": ["production_embedding_excluded_until_vision_qa", "dense_formula_layout_requires_vision_or_mathpix_qa", "remaining_garbled_unicode_requires_vision_qa"],
        "note": "This is production-safe for RAG ingestion because risky OCR pages are gated out. It is not a guarantee of perfect symbolic LaTeX for excluded pages.",
        "run_scope_blockers": run_scope_blockers,
        "strict_complete": bool(args.strict_complete),
        "is_smoke_or_partial": bool(run_scope_blockers),
        "production_complete": gate_status == "production_complete_ready",
    }
    qs = extraction.setdefault("quality_summary", {})
    qs["production_cleanup"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "documentId": document_id,
        "document_key": document_key,
        "cleanup_rule_counts": dict(cleanup_counter),
        "total_pages_in_json": len(pages),
        "ready_for_production_embedding_pages": len(ready_pages),
        "excluded_until_vision_qa_pages": len(excluded_pages),
        "lesson_body_pages": len(lesson_body_pages),
        "excluded_lesson_body_pages": len(excluded_lesson_body_pages),
        "run_scope_blockers": run_scope_blockers,
        "gate_status": gate_status,
        "exclusion_reason_counts": dict(reason_counts),
        "safe_for_production_reindex": bool(document_key) and gate_status in {"production_complete_ready", "production_safe_gated_needs_qa"},
    }
    extraction["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.qa_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_number", "printed_page_number", "chapter_title", "embedding_readiness", "reasons", "sample_text"])
        writer.writeheader()
        for p in excluded_pages:
            sample = re.sub(r"\s+", " ", (p.get("text") or "")[:300])
            writer.writerow({
                "page_number": p.get("page_number"),
                "printed_page_number": p.get("printed_page_number"),
                "chapter_title": p.get("chapter_title"),
                "embedding_readiness": p.get("embedding_readiness"),
                "reasons": "; ".join(p.get("production_exclusion_reasons") or []),
                "sample_text": sample,
            })

    report_lines = [
        "Grade10_Maths Production-Safe Validation Report",
        "=" * 60,
        f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Input JSON: {args.input}",
        f"Output JSON: {args.output}",
        f"QA CSV: {args.qa_csv}",
        f"documentId: {document_id}",
        f"document_key: {document_key}",
        "",
        f"Production status: {gate_status}",
        f"Run-scope blockers: {', '.join(run_scope_blockers) if run_scope_blockers else 'none'}",
        "Embed only pages where include_in_embeddings=true and embedding_readiness=ready_for_production_embedding.",
        "Pages excluded from embeddings require vision/Mathpix/manual QA before production embedding.",
        "",
        f"Total pages in JSON: {len(pages)}",
        f"Ready for production embedding: {len(ready_pages)}",
        f"Excluded until vision QA: {len(excluded_pages)}",
        f"Lesson-body pages: {len(lesson_body_pages)}",
        f"Excluded lesson-body pages: {len(excluded_lesson_body_pages)}",
        "",
        "Cleanup rule counts:",
    ]
    for k, v in cleanup_counter.most_common():
        report_lines.append(f"  - {k}: {v}")
    report_lines.append("")
    report_lines.append("Exclusion reason counts:")
    for k, v in reason_counts.most_common():
        report_lines.append(f"  - {k}: {v}")
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")
    print(f"Wrote: {args.qa_csv}")


if __name__ == "__main__":
    main()
