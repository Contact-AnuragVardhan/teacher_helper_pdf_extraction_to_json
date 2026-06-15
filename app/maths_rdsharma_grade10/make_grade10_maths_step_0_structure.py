#!/usr/bin/env python3
"""
Step 0 for Grade10_Maths / R.D. Sharma Mathematics for Class X.

This book has reliable chapter and section starts on the front-matter contents pages.
The maintained static JSON is the source of truth for chapter and TOC-section ranges.

Outputs:
  - Grade10_Maths_chapters.json
  - optional Grade10_Maths_chapters.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

DEFAULT_STATIC = Path(__file__).resolve().parent / "Grade10_Maths_static_subsection_ranges.json"


def load_static(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Static subsection/range JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("chapters"), list) or not data["chapters"]:
        raise ValueError(f"Invalid static JSON: {path}; missing non-empty chapters array")
    return data


def validate_static(data: dict[str, Any], *, total_pdf_pages: int, allow_page_count_drift: bool = False) -> list[str]:
    warnings: list[str] = []
    offset = int(data.get("pdf_offset", 7))
    expected_pages = int(data.get("pdf_page_count", total_pdf_pages))
    if expected_pages != total_pdf_pages:
        msg = f"Static JSON pdf_page_count={expected_pages}, actual PDF pages={total_pdf_pages}"
        if not allow_page_count_drift:
            raise ValueError(msg + "; pass --allow-page-count-drift only if this is intentional")
        warnings.append(msg)

    chapters = data["chapters"]
    starts = [int(ch["start_book_page"]) for ch in chapters]
    ends = [int(ch["end_book_page"]) for ch in chapters]
    if starts != sorted(starts):
        raise ValueError(f"Chapter printed starts are not increasing: {starts}")
    for idx, chapter in enumerate(chapters):
        cnum = str(chapter.get("chapter_number") or chapter.get("sequence"))
        cstart = int(chapter["start_book_page"])
        cend = int(chapter["end_book_page"])
        spdf = int(chapter["start_pdf_page"])
        epdf = int(chapter["end_pdf_page"])
        if spdf != cstart + offset or epdf != cend + offset:
            raise ValueError(f"Chapter {cnum} PDF range does not match offset {offset}: {spdf}-{epdf}")
        if epdf > total_pdf_pages:
            raise ValueError(f"Chapter {cnum} ends after PDF page count: {epdf}>{total_pdf_pages}")
        if idx < len(chapters) - 1:
            next_start = int(chapters[idx + 1]["start_book_page"])
            if cend + 1 != next_start:
                warnings.append(f"Chapter {cnum} end {cend} is not immediately before next start {next_start}")
        days = chapter.get("days") or []
        if not days:
            raise ValueError(f"Chapter {cnum} has no day/section ranges")
        for day in days:
            ds = int(day["start_book_page"])
            de = int(day["end_book_page"])
            if ds < cstart or de > cend or de < ds:
                raise ValueError(f"Chapter {cnum} invalid day range {ds}-{de}; chapter {cstart}-{cend}")
            if int(day["start_pdf_page"]) != ds + offset or int(day["end_pdf_page"]) != de + offset:
                raise ValueError(f"Chapter {cnum} day {day.get('day')} PDF range does not match offset")
    return warnings


def build_chapters_payload(static: dict[str, Any], *, source_pdf: Path, total_pdf_pages: int, warnings: list[str]) -> dict[str, Any]:
    offset = int(static.get("pdf_offset", 7))
    chapters: list[dict[str, Any]] = []
    for chapter in static["chapters"]:
        cnum = str(chapter.get("chapter_number") or chapter.get("sequence"))
        days = []
        for item in chapter.get("days") or []:
            days.append({
                "day": int(item.get("day") or len(days) + 1),
                "day_title": item.get("day_title") or item.get("section_title"),
                "day_type": item.get("day_type") or "toc_section",
                "section_number": str(item.get("section_number") or ""),
                "section_title": item.get("section_title"),
                "printed_start_page": int(item["start_book_page"]),
                "printed_end_page": int(item["end_book_page"]),
                "start_pdf_page": int(item["start_pdf_page"]),
                "end_pdf_page": int(item["end_pdf_page"]),
                "range_source": item.get("range_source") or static.get("subsection_policy"),
                "boundary_overlap_with_previous_day": bool(item.get("boundary_overlap_with_previous_day")),
            })
        chapters.append({
            "chapter_number": cnum,
            "chapter_title": chapter.get("chapter_title") or chapter.get("chapter_name"),
            "printed_start_page": int(chapter["start_book_page"]),
            "printed_end_page": int(chapter["end_book_page"]),
            "start_pdf_page": int(chapter["start_pdf_page"]),
            "end_pdf_page": int(chapter["end_pdf_page"]),
            "chapter_type": "chapter",
            "subsection_count": len(days),
            "days": days,
        })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pdf": str(source_pdf),
        "total_pdf_pages": total_pdf_pages,
        "printed_page_offset": offset,
        "content_start_page": int(static["content_start_pdf_page"]),
        "content_end_page": int(static["content_end_pdf_page"]),
        "printed_start_page": int(static["printed_start_page"]),
        "printed_end_page": int(static["printed_end_page"]),
        "book_title": static.get("book_title"),
        "author": static.get("author"),
        "publisher": static.get("publisher"),
        "grade": static.get("grade"),
        "subject": static.get("subject"),
        "structure_source": "Grade10_Maths_static_subsection_ranges.json",
        "subsection_policy": static.get("subsection_policy"),
        "day_split_policy": static.get("day_split_policy"),
        "total_subsections": sum(len(ch.get("days") or []) for ch in static["chapters"]),
        "warnings": warnings,
        "chapters": chapters,
    }


def write_python_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Auto-generated by make_grade10_maths_step_0_structure.py\n"
        f"# Generated at UTC: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"PRINTED_OFFSET = {payload['printed_page_offset']}\n\n"
        "CHAPTERS = " + json.dumps(payload["chapters"], ensure_ascii=False, indent=4) + "\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grade10_Maths chapter structure from maintained static TOC ranges.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--static-json", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-py", type=Path, default=None)
    parser.add_argument("--allow-page-count-drift", action="store_true")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    static = load_static(args.static_json)
    doc = fitz.open(str(args.pdf))
    warnings = validate_static(static, total_pdf_pages=doc.page_count, allow_page_count_drift=args.allow_page_count_drift)
    payload = build_chapters_payload(static, source_pdf=args.pdf, total_pdf_pages=doc.page_count, warnings=warnings)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_py:
        write_python_config(args.output_py, payload)

    print(f"PDF pages: {doc.page_count}")
    print(f"Printed-page offset: {payload['printed_page_offset']}")
    print(f"Chapters: {len(payload['chapters'])}")
    print(f"Static subsections/days: {sum(len(ch.get('days') or []) for ch in static['chapters'])}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
    print(f"Wrote: {args.output_json}")
    if args.output_py:
        print(f"Wrote: {args.output_py}")


if __name__ == "__main__":
    main()
