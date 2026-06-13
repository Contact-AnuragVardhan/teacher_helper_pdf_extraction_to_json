#!/usr/bin/env python3
"""
Step 1: raw PDF extraction + chapter/page assignment.

Validates page coverage and structural assignment only. It does not claim formula text is correct.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from physics_common import (
    DEFAULT_BOOK_NAME, DEFAULT_BOOK_TITLE, DEFAULT_CONTENT_START_PDF_PAGE, DEFAULT_GRADE, DEFAULT_SUBJECT,
    add_physical_ranges, as_int, build_page_extractions, extract_pdf_pages, load_static_chapters,
    setup_logging, utc_now, write_json,
)


def build_step1_json(pdf_path: Path, subsections_json: Path, include_question_bank_in_embeddings: bool = False) -> tuple[dict[str, Any], str]:
    spec, chapters_spec = load_static_chapters(subsections_json)
    raw_pages = extract_pdf_pages(pdf_path, include_layout_lines=True)
    pdf_page_count = len(raw_pages)
    add_physical_ranges(chapters_spec, pdf_page_count)
    page_extractions, _, stats = build_page_extractions(raw_pages, chapters_spec, include_question_bank_in_embeddings)
    validation_errors: list[str] = []
    if pdf_page_count != as_int(spec.get("pdf_page_count"), pdf_page_count):
        validation_errors.append(f"PDF page count mismatch: PDF has {pdf_page_count}, static JSON says {spec.get('pdf_page_count')}")
    for chapter in chapters_spec:
        if chapter["start_pdf_page"] < 1 or chapter["end_pdf_page"] > pdf_page_count:
            validation_errors.append(f"Chapter {chapter['sequence']} {chapter['chapter_name']} range outside PDF: {chapter['start_pdf_page']}-{chapter['end_pdf_page']} / pdf={pdf_page_count}")
        if chapter["start_pdf_page"] > chapter["end_pdf_page"]:
            validation_errors.append(f"Chapter {chapter['sequence']} {chapter['chapter_name']} invalid teaching range: {chapter['start_pdf_page']}-{chapter['end_pdf_page']}")
    page_numbers = {int(p["page_number"]) for p in page_extractions}
    missing_pages = [p for p in range(1, pdf_page_count + 1) if p not in page_numbers]
    if missing_pages:
        validation_errors.append(f"Missing page extraction records: {missing_pages}")
    validation = {
        "status": "passed" if not validation_errors else "failed",
        "errors": validation_errors,
        "metrics": {
            "pdf_page_count": pdf_page_count,
            "page_extractions": len(page_extractions),
            "unique_page_numbers": len(page_numbers),
            "front_matter_pages": stats.get("front_matter_pages", 0),
            "teaching_pages": stats.get("teaching_pages", 0),
            "non_teaching_chapter_pages": stats.get("non_teaching_chapter_pages", 0),
            "empty_pages_after_cleanup": stats.get("empty_pages_after_cleanup", 0),
            "noise_lines_removed": stats.get("noise_lines_removed", 0),
            "static_chapters": len(chapters_spec),
        },
    }
    data = {
        "book_name": spec.get("book_name") or DEFAULT_BOOK_NAME,
        "book_title": spec.get("book_title") or DEFAULT_BOOK_TITLE,
        "grade": as_int(spec.get("grade"), DEFAULT_GRADE),
        "subject": spec.get("subject") or DEFAULT_SUBJECT,
        "structure_type": "chapters",
        "source_type": "embedded_text_with_layout_lines",
        "pdf_file": pdf_path.name,
        "pdf_page_count": pdf_page_count,
        "pdf_offset": as_int(spec.get("pdf_offset"), 0) or 0,
        "content_start_pdf_page": DEFAULT_CONTENT_START_PDF_PAGE,
        "subsection_policy": spec.get("subsection_policy") or "practice_exercise_based_static_day_ranges",
        "page_numbering_note": spec.get("page_numbering_note"),
        "extraction": {
            "step": 1,
            "status": "step1_base_extraction_ready" if validation["status"] == "passed" else "step1_validation_failed",
            "generated_at": utc_now(),
            "generator": "physics_step_1_base_extract.py",
            "method": "pymupdf_selectable_text_and_layout_line_extraction_with_static_chapter_assignment",
            "source_subsections_json": str(subsections_json),
            "statistics": dict(stats),
            "validation": validation,
        },
        "static_chapters": chapters_spec,
        "page_extractions": page_extractions,
    }
    return data, build_step1_report(data)


def build_step1_report(data: dict[str, Any]) -> str:
    v = data.get("extraction", {}).get("validation", {})
    metrics = v.get("metrics", {})
    lines = [
        "Grade 10 Physics Step 1 base extraction report",
        "=" * 72,
        f"Generated at: {data.get('extraction', {}).get('generated_at')}",
        f"book_title: {data.get('book_title')}",
        f"pdf_page_count: {data.get('pdf_page_count')}",
        f"validation_status: {v.get('status')}",
        "",
        "Metrics:",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Static chapter map / page assignment:"])
    for chapter in data.get("static_chapters") or []:
        lines.append(f"- Chapter {chapter.get('sequence')} {chapter.get('chapter_name')}: teaching PDF {chapter.get('teaching_start_page')}-{chapter.get('teaching_end_page')}, physical PDF {chapter.get('physical_start_page')}-{chapter.get('physical_end_page')}, days={len(chapter.get('days') or [])}")
    if v.get("errors"):
        lines.extend(["", "Errors:"])
        lines.extend(f"- {err}" for err in v["errors"])
    else:
        lines.extend(["", "No blocking Step 1 validation errors found."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: extract raw Physics PDF pages and assign chapter ranges.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--subsections-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--include-question-bank-in-embeddings", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    pdf_path = args.pdf.resolve()
    subsections_json = args.subsections_json.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report else None
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not subsections_json.exists():
        raise FileNotFoundError(f"Subsections JSON not found: {subsections_json}")
    for path in [output_path, report_path]:
        if path and path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")
    data, report = build_step1_json(pdf_path, subsections_json, args.include_question_bank_in_embeddings)
    write_json(output_path, data)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    print(f"Step 1 JSON: {output_path}")
    if report_path:
        print(f"Step 1 report: {report_path}")
    print(f"Step 1 validation status: {data['extraction']['validation']['status']}")


if __name__ == "__main__":
    main()
