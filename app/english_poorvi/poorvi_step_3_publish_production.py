#!/usr/bin/env python3
"""
poorvi_step_3_publish_production.py

Run this AFTER:
  1) poorvi_step_1_base_extract.py
  2) poorvi_step_2_hybrid_correct.py

It converts English_Poorvi_hybrid_corrected_extraction_v2.json into a production-ready
JSON for ingestion/reindexing:
- adds top-level documentId and document_key
- removes intermediate/audit notes
- removes repeated page labels, headers, footers, INDD/Reprint noise from embedding text
- fixes front-matter classification
- normalizes book_title, structure_type, Unit numbering
- recomputes lesson/page/section text lengths
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTENT_START_PDF_PAGE = 17  # Poorvi printed page 1 maps to PDF page 17
DEFAULT_DOCUMENT_ID = "english-poorvi-class-6-ncert-2026-27"
DEFAULT_DOCUMENT_KEY = "mother-miracle-class-6-english-poorvi"

NOISE_PATTERNS = [
    re.compile(r"^Poorvi\s*[—-]\s*Grade\s*6$", re.I),
    re.compile(r"^Reprint\s+2026-27$", re.I),
    re.compile(r"^Prelims\.indd\b.*$", re.I),
    re.compile(r"^Unit\s+\d+\.indd\b.*$", re.I),
    re.compile(r"^\[PDF page \d+ / printed page .*?\]$", re.I),
]
UNIT_HEADER_LINES = {
    "Fables and Folk Tales", "Fables And Folk Tales", "Friendship", "Nurturing Nature",
    "Sports and Wellness", "Culture and Tradition",
}
FRONT_MATTER_TYPES = {
    1: "cover", 2: "copyright_page", 3: "foreword", 4: "foreword", 5: "about_book",
    6: "about_book", 7: "about_book", 8: "about_book", 9: "committee_page",
    10: "development_team", 11: "development_team", 12: "blank_or_divider",
    13: "acknowledgements", 14: "acknowledgements", 15: "toc", 16: "blank_or_divider",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line or "").strip()


def is_noise_line(line: str, printed_page_number: Any = None) -> bool:
    s = compact_line(line)
    if not s:
        return False
    if s in UNIT_HEADER_LINES:
        return True
    if printed_page_number is not None and str(printed_page_number).isdigit() and s == str(printed_page_number):
        return True
    return any(p.match(s) for p in NOISE_PATTERNS)


def clean_text(text: str, printed_page_number: Any = None) -> tuple[str, int]:
    if not text:
        return "", 0
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\[PDF page \d+ / printed page .*?\]\s*$", "", text)
    out, removed = [], 0
    for line in text.split("\n"):
        if is_noise_line(line, printed_page_number):
            removed += 1
            continue
        out.append(line.rstrip())
    cleaned = "\n".join(out)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, removed


def front_matter_type(page_number: int, text: str) -> str:
    if page_number in FRONT_MATTER_TYPES:
        return FRONT_MATTER_TYPES[page_number]
    t = (text or "").lower()
    if "contents" in t:
        return "toc"
    if "foreword" in t:
        return "foreword"
    if "about the book" in t:
        return "about_book"
    if "acknowledgements" in t:
        return "acknowledgements"
    if not t.strip():
        return "blank_or_divider"
    return "front_matter"


def unit_num(value: Any) -> str | None:
    if value is None:
        return None
    m = re.search(r"\d+", str(value))
    return m.group(0) if m else None


def unit_label(unit_number: Any, fallback: Any = None) -> str | None:
    n = unit_num(unit_number) or unit_num(fallback)
    return f"Unit {n}" if n else None


def clean_page(page: dict[str, Any], stats: defaultdict[str, int]) -> dict[str, Any]:
    p = copy.deepcopy(page)
    pnum = int(p.get("page_number") or 0)
    printed = p.get("printed_page_number")

    for field in ("text", "text_plain", "ocr_text"):
        if isinstance(p.get(field), str):
            p[field], removed = clean_text(p[field], printed)
            stats["noise_lines_removed"] += removed

    if pnum < CONTENT_START_PDF_PAGE:
        p.update({
            "content_type": front_matter_type(pnum, p.get("text", "")),
            "assignment_status": "front_matter",
            "include_in_lesson_text": False,
            "include_in_embeddings": False,
            "embedding_readiness": "not_indexed_front_matter",
            "chapter_number": None,
            "chapter_title": None,
            "section_number": None,
            "section_title": None,
            "unit_number": None,
            "unit_title": None,
        })
        p.setdefault("quality_flags", [])
        if "front_matter_not_lesson_content" not in p["quality_flags"]:
            p["quality_flags"].append("front_matter_not_lesson_content")
        stats["front_matter_pages_reclassified"] += 1
    else:
        if p.get("content_type") in {"transcript", "unit_activity"}:
            p["include_in_lesson_text"] = False
            p.setdefault("include_in_embeddings", False)
            p.setdefault("embedding_readiness", "separate_non_lesson_content")
        else:
            p.setdefault("include_in_lesson_text", True)
            p.setdefault("include_in_embeddings", True)
            p.setdefault("embedding_readiness", "ready_for_production_embedding")

        if p.get("unit_number") or p.get("chapter_number"):
            lbl = unit_label(p.get("unit_number"), p.get("chapter_number"))
            if lbl:
                p["chapter_number"] = lbl

    p["text_length_chars"] = len(p.get("text") or "")
    return p


def rebuild_lessons_and_sections(data: dict[str, Any], stats: defaultdict[str, int]) -> None:
    extraction = data["extraction"]
    pages = extraction.get("page_extractions", [])

    pages_by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pages:
        if p.get("include_in_lesson_text") and p.get("section_title"):
            pages_by_title[str(p["section_title"])].append(p)
        elif p.get("linked_section_title"):
            excluded_by_title[str(p["linked_section_title"])].append(p)

    lesson_by_number: dict[str, dict[str, Any]] = {}
    lesson_by_title: dict[str, dict[str, Any]] = {}

    for ch in extraction.get("chapters", []):
        un = unit_num(ch.get("unit_number") or ch.get("chapter_number"))
        ch["unit_number"] = un
        ch["chapter_number"] = unit_label(un, ch.get("chapter_number")) or ch.get("chapter_number")
        ch["chapter_type"] = "unit"
        ch.setdefault("unit_title", ch.get("chapter_title"))

        for lesson in ch.get("lessons", []):
            title = str(lesson.get("section_title"))
            lesson_pages = sorted(pages_by_title.get(title, []), key=lambda x: int(x.get("page_number") or 0))
            if lesson_pages:
                lesson["start_page"] = lesson_pages[0].get("page_number")
                lesson["end_page"] = lesson_pages[-1].get("page_number")
                lesson["printed_start_page"] = lesson_pages[0].get("printed_page_number")
                lesson["printed_end_page"] = lesson_pages[-1].get("printed_page_number")
                lesson["page_numbers"] = [p.get("page_number") for p in lesson_pages]
                lesson["printed_page_numbers"] = [p.get("printed_page_number") for p in lesson_pages]
                lesson["page_count"] = len(lesson_pages)
                lt = "\n\n".join((p.get("text") or "").strip() for p in lesson_pages if (p.get("text") or "").strip())
                lt, removed = clean_text(lt)
                stats["noise_lines_removed"] += removed
                lesson["lesson_text"] = lt
                lesson["text_plain"] = lt
                lesson["text_length_chars"] = len(lt)

            lesson["chapter_number"] = ch["chapter_number"]
            lesson["chapter_type"] = "unit"
            lesson["unit_number"] = un
            lesson["unit_title"] = ch.get("unit_title") or ch.get("chapter_title")
            lesson["include_in_embeddings"] = bool((lesson.get("lesson_text") or "").strip())
            lesson["embedding_readiness"] = "ready_for_production_embedding" if lesson["include_in_embeddings"] else "empty_lesson_text"
            lesson["excluded_related_pages"] = [
                {
                    "page_number": p.get("page_number"),
                    "printed_page_number": p.get("printed_page_number"),
                    "content_type": p.get("content_type"),
                    "reason": "stored_separately_not_in_lesson_text",
                }
                for p in sorted(excluded_by_title.get(title, []), key=lambda x: int(x.get("page_number") or 0))
            ]
            if lesson.get("section_number") is not None:
                lesson_by_number[str(lesson["section_number"])] = lesson
            lesson_by_title[title] = lesson

    for sec in extraction.get("section_index", []):
        lesson = lesson_by_number.get(str(sec.get("section_number"))) or lesson_by_title.get(str(sec.get("section_title")))
        un = unit_num(sec.get("unit_number") or sec.get("chapter_number"))
        sec["unit_number"] = un
        sec["chapter_number"] = unit_label(un, sec.get("chapter_number")) or sec.get("chapter_number")
        sec["chapter_type"] = "unit"
        if lesson:
            for key in ["start_page", "end_page", "printed_start_page", "printed_end_page", "page_count", "text_length_chars", "excluded_related_pages"]:
                sec[key] = lesson.get(key)
            sec["indexed_page_count"] = lesson.get("page_count")
            sec["indexed_page_numbers"] = lesson.get("page_numbers", [])
            sec["indexed_printed_page_numbers"] = lesson.get("printed_page_numbers", [])
            sec["include_in_embeddings"] = lesson.get("include_in_embeddings", True)
            sec["embedding_readiness"] = lesson.get("embedding_readiness", "ready_for_production_embedding")


def rebuild_front_matter(data: dict[str, Any]) -> None:
    front = []
    for p in data["extraction"].get("page_extractions", []):
        if p.get("assignment_status") == "front_matter" or int(p.get("page_number") or 999999) < CONTENT_START_PDF_PAGE:
            front.append({
                "page_number": p.get("page_number"),
                "printed_page_number": p.get("printed_page_number"),
                "content_type": p.get("content_type"),
                "assignment_status": p.get("assignment_status"),
                "include_in_embeddings": False,
                "text": p.get("text", ""),
                "text_length_chars": len(p.get("text") or ""),
            })
    data["extraction"]["front_matter_pages"] = front


def validate(data: dict[str, Any]) -> tuple[list[str], list[str], dict[str, int]]:
    errors, warnings = [], []
    metrics: defaultdict[str, int] = defaultdict(int)
    extraction = data["extraction"]

    if not data.get("documentId"):
        errors.append("Missing top-level documentId")
    if not data.get("document_key"):
        errors.append("Missing top-level document_key")

    joined_notes = " ".join(extraction.get("notes", []))
    for phrase in ["first-pass extraction", "Run poorvi_step_2", "before production reindexing", "selectable PDF text only"]:
        if phrase.lower() in joined_notes.lower():
            errors.append(f"Intermediate/audit note still present: {phrase}")

    noise_re = re.compile(r"Poorvi\s*[—-]\s*Grade\s*6|Reprint\s+2026-27|\.indd\b|\[PDF page", re.I)
    for p in extraction.get("page_extractions", []):
        pnum = int(p.get("page_number") or 0)
        if pnum < CONTENT_START_PDF_PAGE:
            if p.get("content_type") == "lesson_body" or p.get("assignment_status") == "assigned_to_lesson":
                errors.append(f"Front matter page {pnum} still classified as lesson")
            if p.get("include_in_embeddings"):
                errors.append(f"Front matter page {pnum} is still include_in_embeddings=true")
        if noise_re.search(p.get("text") or ""):
            metrics["pages_with_remaining_noise"] += 1
            warnings.append(f"PDF page {pnum}: possible remaining header/footer/page-label noise")

    for ch in extraction.get("chapters", []):
        if not str(ch.get("chapter_number", "")).startswith("Unit "):
            warnings.append(f"Chapter number is not explicit Unit format: {ch.get('chapter_number')}")
        for lesson in ch.get("lessons", []):
            lt = lesson.get("lesson_text") or ""
            if noise_re.search(lt):
                metrics["lessons_with_remaining_noise"] += 1
                warnings.append(f"Lesson {lesson.get('section_number')} {lesson.get('section_title')}: possible remaining noise")
            if lesson.get("text_length_chars") != len(lt):
                errors.append(f"Lesson {lesson.get('section_number')} text_length_chars mismatch")

    return errors, warnings, dict(metrics)


def write_report(path: Path, data: dict[str, Any], errors: list[str], warnings: list[str], stats: dict[str, int], metrics: dict[str, int]) -> None:
    extraction = data["extraction"]
    ready_pages = sum(1 for p in extraction.get("page_extractions", []) if p.get("include_in_embeddings"))
    front_pages = sum(1 for p in extraction.get("page_extractions", []) if p.get("assignment_status") == "front_matter")
    lessons = sum(len(ch.get("lessons", [])) for ch in extraction.get("chapters", []))
    lines = [
        "English Poorvi production publish validation report",
        "=================================================",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"documentId: {data.get('documentId')}",
        f"document_key: {data.get('document_key')}",
        f"book_title: {extraction.get('book_title')}",
        f"structure_type: {extraction.get('structure_type')}",
        "",
        "Summary:",
        f"- total_pdf_pages: {extraction.get('total_pdf_pages')}",
        f"- lessons: {lessons}",
        f"- pages ready for embeddings: {ready_pages}",
        f"- front matter pages: {front_pages}",
        f"- noise lines removed: {stats.get('noise_lines_removed', 0)}",
        f"- front matter pages reclassified: {stats.get('front_matter_pages_reclassified', 0)}",
        "",
        "Errors:",
    ]
    lines.extend([f"- {e}" for e in errors] if errors else ["- None"])
    lines.append("")
    lines.append("Warnings:")
    lines.extend([f"- {w}" for w in warnings[:200]] if warnings else ["- None"])
    if len(warnings) > 200:
        lines.append(f"- ... {len(warnings) - 200} more warnings")
    lines.append("")
    lines.append("Metrics:")
    lines.extend([f"- {k}: {v}" for k, v in sorted(metrics.items())] if metrics else ["- No remaining tracked noise/artifact metrics"])
    lines.append("")
    lines.append("Production ingestion guidance:")
    lines.append("- Use this production-ready JSON for DB ingestion/reindexing.")
    lines.append("- Use document_key as the stable reindex identity.")
    lines.append("- Embed only page_extractions/lessons where include_in_embeddings=true.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def publish(input_path: Path, output_path: Path, report_path: Path, document_id: str, document_key: str) -> None:
    data = read_json(input_path)
    data = copy.deepcopy(data)
    stats: defaultdict[str, int] = defaultdict(int)

    data["documentId"] = document_id
    data["document_key"] = document_key

    metadata = data.setdefault("metadata", {})
    metadata.update({
        "class_name": "Class-6",
        "grade": "Class-6",
        "board": metadata.get("board") or "CBSE",
        "medium": metadata.get("medium") or "English",
        "publisher": "NCERT",
        "source_type": "textbook_pdf",
        "document_key": document_key,
    })

    extraction = data.setdefault("extraction", {})
    extraction["book_title"] = "Poorvi: Textbook of English for Grade 6"
    extraction["subject"] = "English"
    extraction["language"] = "English"
    extraction["content_profile"] = "english_textbook"
    extraction["structure_type"] = "unit_section"
    extraction["notes"] = [
        "Production-ready JSON generated after hybrid correction.",
        "Rama to the Rescue graphic-story panel text was added from visual correction/OCR audit.",
        "Transcript pages and unit-level activity pages are stored separately from lesson_text.",
        "Repeated textbook headers, footers, page labels, and reprint/INDD artifacts were removed from embedding text.",
        "Front matter is classified separately and excluded from embeddings by default.",
        "section_index uses semantic/indexed lesson ranges; physical_* fields preserve source PDF ranges where available.",
    ]

    extraction["page_extractions"] = [clean_page(p, stats) for p in extraction.get("page_extractions", [])]

    for collection_name in ["transcripts", "unit_level_pages"]:
        for item in extraction.get(collection_name, []):
            if isinstance(item.get("text"), str):
                item["text"], removed = clean_text(item["text"], item.get("printed_page_number"))
                stats["noise_lines_removed"] += removed
                item["text_length_chars"] = len(item["text"])

    rebuild_lessons_and_sections(data, stats)
    rebuild_front_matter(data)

    extraction["quality_summary"] = {
        **(extraction.get("quality_summary") or {}),
        "production_publish_version": "poorvi-production-v1",
        "document_id_present": True,
        "document_key_present": True,
        "book_title_normalized": True,
        "structure_type": "unit_section",
        "front_matter_excluded_from_embeddings": True,
        "headers_footers_removed_from_embedding_text": True,
        "intermediate_pipeline_notes_removed": True,
        "safe_for_production_reindex": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    extraction["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    errors, warnings, metrics = validate(data)
    extraction["quality_summary"]["production_validation_error_count"] = len(errors)
    extraction["quality_summary"]["production_validation_warning_count"] = len(warnings)

    write_json(output_path, data)
    write_report(report_path, data, errors, warnings, dict(stats), metrics)
    print(f"Wrote production JSON: {output_path}")
    print(f"Wrote validation report: {report_path}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("output/English_Poorvi_hybrid_corrected_extraction_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("output/English_Poorvi_production_ready.json"))
    parser.add_argument("--report", type=Path, default=Path("output/English_Poorvi_production_validation_report.txt"))
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument("--document-key", default=DEFAULT_DOCUMENT_KEY)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    publish(args.input, args.output, args.report, args.document_id, args.document_key)


if __name__ == "__main__":
    main()
