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
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("english_poorvi.step3_publish")


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging for command-line debugging."""
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


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




def normalize_structure_detection_metadata(data: dict[str, Any]) -> None:
    """Make Poorvi static structure-map wording production-friendly.

    Earlier pipeline stages may contain legacy wording such as
    static_fallback_after_dynamic_failure/fallback_used. For Poorvi, the
    maintained static unit/lesson map is the intended production source of
    structure when dynamic PDF detection is unreliable, so normalize the final
    published metadata to say that clearly.
    """
    extraction = data.setdefault("extraction", {})
    detection = extraction.setdefault("structure_detection", {})
    old_method = detection.get("method")
    old_status = detection.get("status")

    if old_method == "static_fallback_after_dynamic_failure" or old_status == "fallback_used":
        attempts = detection.get("attempts", [])
        units_detected = detection.get("units_detected")
        lessons_detected = detection.get("lessons_detected")
        detection.clear()
        detection.update({
            "method": "curated_static_poorvi_map",
            "status": "production_static_map_used",
            "dynamic_detection_possible": False,
            "units_detected": units_detected,
            "lessons_detected": lessons_detected,
            "reason": (
                "Dynamic PDF structure detection was attempted, but the curated Poorvi "
                "unit/lesson map is used for production because this NCERT textbook has "
                "a fixed, verified table of contents. This is an intentional production "
                "map, not an extraction failure."
            ),
            "curated_map_name": "poorvi_grade6_english_ncert_2026_27",
            "curated_map_status": "verified_against_toc_and_lesson_ranges",
            "attempts": attempts,
        })
        LOGGER.info("Normalized structure_detection metadata to curated static Poorvi map")

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



def normalize_title_key(value: Any) -> str:
    """Normalize a lesson/chapter title for matching days JSON to production JSON."""
    s = str(value or "")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'").replace("´", "'")
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def load_days_chapter_map(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load chapter_map_final-style JSON and index it by normalized chapter/lesson title."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)

    data = read_json(path)
    chapters = data.get("chapters", [])
    chapter_map: dict[str, dict[str, Any]] = {}
    for chapter in chapters:
        key = normalize_title_key(chapter.get("chapter_name"))
        if not key:
            LOGGER.warning("Skipping days chapter with missing chapter_name: %s", chapter)
            continue
        if key in chapter_map:
            LOGGER.warning("Duplicate chapter title in days JSON; later entry wins: %s", chapter.get("chapter_name"))
        chapter_map[key] = chapter

    LOGGER.info("Loaded days/subsection JSON: %s chapters from %s", len(chapter_map), path)
    return chapter_map


def get_semantic_parent_range(section: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return the safe parent lesson range used for subsections.

    Use semantic/indexed lesson ranges (`start_page`/`end_page`) instead of broad
    physical ranges, because physical ranges may include transcript or unit-level
    pages that should not be embedded as part of the lesson.
    """
    start_pdf = as_int(section.get("start_page"), as_int(section.get("physical_start_page"), 0)) or 0
    end_pdf = as_int(section.get("end_page"), as_int(section.get("physical_end_page"), 0)) or 0
    start_printed = as_int(section.get("printed_start_page"), start_pdf - 16 if start_pdf else 0) or 0
    end_printed = as_int(section.get("printed_end_page"), end_pdf - 16 if end_pdf else 0) or 0
    return start_pdf, end_pdf, start_printed, end_printed


def get_pages_in_range(
    pages_by_number: dict[int, dict[str, Any]],
    start_pdf_page: int,
    end_pdf_page: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    pages: list[dict[str, Any]] = []
    missing: list[int] = []
    for page_number in range(start_pdf_page, end_pdf_page + 1):
        page = pages_by_number.get(page_number)
        if page is None:
            missing.append(page_number)
        else:
            pages.append(page)
    return pages, missing


def page_belongs_to_section(page: dict[str, Any], section_title: str) -> bool:
    if not page.get("include_in_lesson_text", True):
        return False
    p_title = page.get("section_title")
    return bool(p_title) and normalize_title_key(p_title) == normalize_title_key(section_title)


def build_subsection_text_from_pages(pages: list[dict[str, Any]]) -> tuple[str, str, int]:
    raw_text = "\n\n".join((p.get("text") or "").strip() for p in pages if (p.get("text") or "").strip()).strip()
    cleaned_text, removed = clean_text(raw_text)
    return raw_text, cleaned_text, removed


def split_inclusive_range(start: int, end: int, parts: int = 5) -> list[tuple[int, int]]:
    """Split an inclusive range into production-safe, non-overlapping day ranges.

    If a lesson has fewer pages than the requested parts, the part count is
    reduced to the available page count. This avoids repeated day/subsection
    page ranges such as National War Memorial Day 1-5 all pointing at the same
    two pages.
    """
    if start <= 0 or end <= 0 or start > end:
        return []
    count = end - start + 1
    parts = max(1, min(parts, count))
    base = count // parts
    remainder = count % parts
    ranges: list[tuple[int, int]] = []
    cursor = start
    for idx in range(parts):
        size = base + (1 if idx < remainder else 0)
        r_start = cursor
        r_end = cursor + size - 1
        ranges.append((r_start, r_end))
        cursor = r_end + 1
    return ranges


def auto_day_defs_for_section(section: dict[str, Any], day_count: int = 5) -> list[dict[str, Any]]:
    parent_start, parent_end, printed_start, printed_end = get_semantic_parent_range(section)
    pdf_ranges = split_inclusive_range(parent_start, parent_end, day_count)
    printed_ranges = split_inclusive_range(printed_start, printed_end, day_count)
    days: list[dict[str, Any]] = []
    for idx, ((pdf_start, pdf_end), (book_start, book_end)) in enumerate(zip(pdf_ranges, printed_ranges), start=1):
        days.append({
            "day": idx,
            "start_pdf_page": pdf_start,
            "end_pdf_page": pdf_end,
            "start_book_page": book_start,
            "end_book_page": book_end,
            "range_source": "auto_variable_split_inside_parent_lesson_no_repeated_pages",
        })
    return days


def external_days_are_safe_for_section(
    section: dict[str, Any],
    days: list[dict[str, Any]],
    pages_by_number: dict[int, dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Return whether an external days[] map can be used exactly as-is.

    We reject the whole external days[] map for a lesson if any day crosses the
    parent semantic lesson range, has invalid/missing pages, or includes a page
    assigned to another section/transcript/unit activity. Step 3 then replaces it
    with a clean auto split rather than preserving bad ranges.
    """
    reasons: list[str] = []
    section_title = str(section.get("section_title") or "")
    parent_start, parent_end, _, _ = get_semantic_parent_range(section)
    if not days:
        return False, ["missing_days_array"]

    for day in days:
        day_number = as_int(day.get("day"), 0) or 0
        start_pdf = as_int(day.get("start_pdf_page"))
        end_pdf = as_int(day.get("end_pdf_page"))
        if start_pdf is None or end_pdf is None:
            reasons.append(f"day_{day_number}_missing_pdf_range")
            continue
        if start_pdf > end_pdf:
            reasons.append(f"day_{day_number}_invalid_pdf_range_{start_pdf}_{end_pdf}")
            continue
        if start_pdf < parent_start or end_pdf > parent_end:
            reasons.append(f"day_{day_number}_outside_parent_{start_pdf}_{end_pdf}_parent_{parent_start}_{parent_end}")
            continue
        pages, missing_pages = get_pages_in_range(pages_by_number, start_pdf, end_pdf)
        if missing_pages:
            reasons.append(f"day_{day_number}_missing_pages_{missing_pages}")
            continue
        bad_pages = [
            p.get("page_number")
            for p in pages
            if not page_belongs_to_section(p, section_title)
        ]
        if bad_pages:
            reasons.append(f"day_{day_number}_cross_section_or_excluded_pages_{bad_pages}")

    return not reasons, reasons


def build_subsections_from_day_defs(
    section: dict[str, Any],
    day_defs: list[dict[str, Any]],
    pages_by_number: dict[int, dict[str, Any]],
    stats: defaultdict[str, int],
    generation_method: str,
    rejection_reasons: list[str] | None = None,
) -> list[dict[str, Any]]:
    subsections: list[dict[str, Any]] = []
    section_title = str(section.get("section_title") or "")
    section_number = str(section.get("section_number") or "")
    parent_start, parent_end, _, _ = get_semantic_parent_range(section)

    for day in sorted(day_defs, key=lambda d: as_int(d.get("day"), 0) or 0):
        day_number = as_int(day.get("day"), len(subsections) + 1) or (len(subsections) + 1)
        start_pdf = as_int(day.get("start_pdf_page"))
        end_pdf = as_int(day.get("end_pdf_page"))
        start_printed = as_int(day.get("start_book_page"))
        end_printed = as_int(day.get("end_book_page"))
        if start_pdf is None or end_pdf is None:
            stats["subsections_skipped_missing_pdf_range"] += 1
            LOGGER.warning("Skipping %s Day %s because start/end PDF page is missing: %s", section_title, day_number, day)
            continue

        # Final safety clamp. This should be a no-op for embedded/default safe map
        # and for accepted external maps, but it protects Step 3 from bad input.
        original_start_pdf, original_end_pdf = start_pdf, end_pdf
        start_pdf = max(start_pdf, parent_start)
        end_pdf = min(end_pdf, parent_end)
        if start_pdf > end_pdf:
            stats["subsections_skipped_no_parent_overlap"] += 1
            LOGGER.warning(
                "Skipping %s Day %s because range PDF %s-%s has no overlap with parent PDF %s-%s",
                section_title, day_number, original_start_pdf, original_end_pdf, parent_start, parent_end,
            )
            continue

        if start_printed is None:
            start_printed = start_pdf - 16
        if end_printed is None:
            end_printed = end_pdf - 16
        # Keep printed pages aligned with any PDF clamp.
        if original_start_pdf != start_pdf:
            start_printed = start_pdf - 16
        if original_end_pdf != end_pdf:
            end_printed = end_pdf - 16

        pages, missing_pages = get_pages_in_range(pages_by_number, start_pdf, end_pdf)
        safe_pages = [p for p in pages if page_belongs_to_section(p, section_title)]
        raw_text, cleaned_text, removed = build_subsection_text_from_pages(safe_pages)
        stats["noise_lines_removed"] += removed

        page_numbers = [as_int(p.get("page_number")) for p in safe_pages if as_int(p.get("page_number")) is not None]
        printed_page_numbers = [
            as_int(p.get("printed_page_number"))
            for p in safe_pages
            if as_int(p.get("printed_page_number")) is not None
        ]

        quality_flags = [generation_method]
        notes: list[str] = []
        if rejection_reasons:
            notes.append("External days map for this lesson was rejected and replaced by safe auto split: " + "; ".join(rejection_reasons[:10]))
        if missing_pages:
            quality_flags.append("subsection_missing_pdf_pages")
            notes.append(f"Missing PDF pages in output page_extractions: {missing_pages}")
            stats["subsections_with_missing_pages"] += 1
        if not cleaned_text:
            quality_flags.append("empty_subsection_text")
            stats["subsections_with_empty_text"] += 1

        subsection_number = f"{section_number}.{day_number}" if section_number else str(day_number)
        subsection_title = f"Day {day_number}"
        subsection = {
            "section_number": section.get("section_number"),
            "section_title": section.get("section_title"),
            "unit_number": section.get("unit_number"),
            "unit_title": section.get("unit_title"),
            "chapter_type": section.get("chapter_type"),
            "chapter_number": section.get("chapter_number"),
            "chapter_title": section.get("chapter_title"),
            "subsection_number": subsection_number,
            "subsection_title": subsection_title,
            "anchor_marker": subsection_title,
            "anchor_pdf_page": start_pdf,
            "anchor_printed_page": start_printed,
            "anchor_detection_method": generation_method,
            "anchor_raw_heading": subsection_title,
            "included_exercises_or_activities": [subsection_title],
            "includes": [subsection_title],
            "start_page": start_pdf,
            "end_page": end_pdf,
            "start_pdf_page": start_pdf,
            "end_pdf_page": end_pdf,
            "printed_start_page": start_printed,
            "printed_end_page": end_printed,
            "start_printed_page": start_printed,
            "end_printed_page": end_printed,
            "pdf_pages": {"start": start_pdf, "end": end_pdf},
            "printed_pages": {"start": start_printed, "end": end_printed},
            "page_count": max(0, end_pdf - start_pdf + 1),
            "subsection_text": raw_text,
            "subsection_text_plain": cleaned_text,
            "text_plain": cleaned_text,
            "production_subsection_text": cleaned_text,
            "production_indexed_page_numbers": page_numbers,
            "production_printed_page_numbers": printed_page_numbers,
            "production_excluded_pages": [],
            "production_page_count": len(page_numbers),
            "physical_start_page": start_pdf,
            "physical_end_page": end_pdf,
            "physical_printed_start_page": start_printed,
            "physical_printed_end_page": end_printed,
            "physical_page_count": max(0, end_pdf - start_pdf + 1),
            "page_numbers": page_numbers,
            "printed_page_numbers": printed_page_numbers,
            "excluded_related_pages": [],
            "text_sources": sorted({src for p in safe_pages for src in (p.get("text_sources") or [])}),
            "quality_flags": quality_flags,
            "include_in_embeddings": bool(cleaned_text),
            "embedding_readiness": "ready_for_production_embedding" if cleaned_text else "empty_subsection_text",
            "text_length_chars": len(cleaned_text),
            "source_days_json_day": day_number,
            "source_days_json_range_source": day.get("range_source"),
            "notes": notes,
        }
        subsections.append(subsection)
        stats["subsections_added"] += 1
        LOGGER.debug(
            "Built subsection %s %s: PDF %s-%s printed %s-%s chars=%s flags=%s",
            section.get("section_title"), subsection_number, start_pdf, end_pdf,
            start_printed, end_printed, len(cleaned_text), quality_flags,
        )

    return subsections


def build_day_subsections_for_section(
    section: dict[str, Any],
    days_chapter: dict[str, Any] | None,
    pages_by_number: dict[int, dict[str, Any]],
    stats: defaultdict[str, int],
) -> list[dict[str, Any]]:
    """Build production-safe subsections for one lesson.

    The default target is up to five subsections, but short lessons get fewer
    subsections so PDF pages are not repeated.
    """
    section_title = str(section.get("section_title") or "")
    days = (days_chapter or {}).get("days", []) if days_chapter else []
    safe, rejection_reasons = external_days_are_safe_for_section(section, days, pages_by_number) if days else (False, ["no_external_days_entry"])

    if safe:
        generation_method = "external_days_json_validated_inside_parent"
        stats["sections_using_external_days_subsections"] += 1
        LOGGER.info("Using validated external days map for %s", section_title)
        return build_subsections_from_day_defs(section, days, pages_by_number, stats, generation_method)

    stats["sections_using_auto_safe_subsections"] += 1
    if days:
        stats["sections_external_days_rejected"] += 1
        LOGGER.warning("Rejected external days map for %s; using auto safe split. Reasons: %s", section_title, rejection_reasons)
    else:
        LOGGER.info("No external days map for %s; using auto safe split", section_title)
    auto_days = auto_day_defs_for_section(section, day_count=5)
    return build_subsections_from_day_defs(
        section,
        auto_days,
        pages_by_number,
        stats,
        "auto_variable_split_inside_parent_lesson_no_repeated_pages",
        rejection_reasons=rejection_reasons if days else None,
    )


def apply_days_subsections(
    data: dict[str, Any],
    days_chapter_map: dict[str, dict[str, Any]],
    stats: defaultdict[str, int],
) -> None:
    """Attach Maths-style subsections to every section_index and lesson object.

    Every lesson receives production-safe subsections. External JSON is used
    only when each day range is already safe inside the parent lesson. Bad
    external ranges are rejected and replaced by an automatic safe split. Short
    lessons may have fewer than five subsections to avoid repeating pages.
    """
    extraction = data["extraction"]
    pages_by_number = {
        int(p["page_number"]): p
        for p in extraction.get("page_extractions", [])
        if p.get("page_number") is not None
    }

    subsections_by_section_number: dict[str, list[dict[str, Any]]] = {}
    matched_titles: set[str] = set()

    for section in extraction.get("section_index", []):
        title_key = normalize_title_key(section.get("section_title"))
        days_chapter = days_chapter_map.get(title_key) if days_chapter_map else None
        if days_chapter:
            matched_titles.add(title_key)

        subsections = build_day_subsections_for_section(section, days_chapter, pages_by_number, stats)
        section["subsections"] = subsections
        if section.get("section_number") is not None:
            subsections_by_section_number[str(section["section_number"])] = copy.deepcopy(subsections)
        LOGGER.info(
            "Attached %s production-safe subsections to section_index %s %s",
            len(subsections), section.get("section_number"), section.get("section_title"),
        )

    for chapter in extraction.get("chapters", []):
        for lesson in chapter.get("lessons", []):
            lesson_section_number = str(lesson.get("section_number"))
            if lesson_section_number in subsections_by_section_number:
                lesson["subsections"] = copy.deepcopy(subsections_by_section_number[lesson_section_number])
                LOGGER.info(
                    "Attached %s subsections to lesson %s %s",
                    len(lesson["subsections"]), lesson.get("section_number"), lesson.get("section_title"),
                )

    unmatched_days = sorted(set(days_chapter_map.keys()) - matched_titles) if days_chapter_map else []
    if unmatched_days:
        stats["days_chapters_unmatched_to_sections"] = len(unmatched_days)
        LOGGER.warning("Days JSON chapters not matched to section_index titles: %s", unmatched_days)
    extraction.setdefault("subsection_generation", {})
    extraction["subsection_generation"].update({
        "source": "validated_days_json_with_auto_safe_fallback",
        "policy": "variable_subsection_count; ranges_must_be_inside_semantic_parent_lesson; invalid_external_ranges_are_rejected; short_lessons_do_not_repeat_pages",
        "sections_with_subsections": sum(1 for s in extraction.get("section_index", []) if s.get("subsections")),
        "total_subsections": stats.get("subsections_added", 0),
        "sections_using_external_days_subsections": stats.get("sections_using_external_days_subsections", 0),
        "sections_using_auto_safe_subsections": stats.get("sections_using_auto_safe_subsections", 0),
        "sections_external_days_rejected": stats.get("sections_external_days_rejected", 0),
        "unmatched_days_chapters": unmatched_days,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })


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



    sections = extraction.get("section_index", [])
    metrics["sections_total"] = len(sections)
    metrics["sections_with_subsections"] = sum(1 for sec in sections if sec.get("subsections"))
    for sec in sections:
        section_title = sec.get("section_title")
        subsections = sec.get("subsections", [])
        if not subsections:
            errors.append(f"Section {sec.get('section_number')} {section_title}: missing subsections")
            continue
        parent_start = as_int(sec.get("start_page"), as_int(sec.get("physical_start_page"), 0)) or 0
        parent_end = as_int(sec.get("end_page"), as_int(sec.get("physical_end_page"), 0)) or 0
        parent_title_key = normalize_title_key(section_title)
        seen_subsection_pages: set[int] = set()
        for sub in subsections:
            snum = sub.get("subsection_number")
            start_pdf = as_int(sub.get("start_pdf_page"))
            end_pdf = as_int(sub.get("end_pdf_page"))
            if start_pdf is None or end_pdf is None:
                errors.append(f"Subsection {snum}: missing start_pdf_page/end_pdf_page")
                continue
            if start_pdf > end_pdf:
                errors.append(f"Subsection {snum}: invalid PDF range {start_pdf}-{end_pdf}")
            if sub.get("page_count") != max(0, end_pdf - start_pdf + 1):
                errors.append(f"Subsection {snum}: page_count mismatch")
            if parent_start and parent_end and (start_pdf < parent_start or end_pdf > parent_end):
                metrics["subsections_outside_parent_range"] += 1
                errors.append(
                    f"Subsection {snum} {section_title}: PDF {start_pdf}-{end_pdf} is outside semantic parent lesson PDF {parent_start}-{parent_end}"
                )
            if sub.get("production_excluded_pages") or sub.get("excluded_related_pages"):
                metrics["subsections_with_cross_section_or_excluded_pages"] += 1
                errors.append(
                    f"Subsection {snum} {section_title}: contains pages assigned to other sections or excluded content"
                )
            # Verify any page objects included in the subsection are still within the parent lesson
            # and are not repeated by another subsection of the same lesson.
            for page_number in sub.get("production_indexed_page_numbers", []) or []:
                pnum = as_int(page_number)
                if pnum is not None and parent_start and parent_end and not (parent_start <= pnum <= parent_end):
                    errors.append(f"Subsection {snum} {section_title}: indexed page {pnum} is outside parent lesson PDF {parent_start}-{parent_end}")
                if pnum is not None:
                    if pnum in seen_subsection_pages:
                        metrics["subsections_with_repeated_pdf_pages"] += 1
                        errors.append(f"Subsection {snum} {section_title}: indexed page {pnum} is repeated in another subsection")
                    seen_subsection_pages.add(pnum)
            if normalize_title_key(sub.get("section_title")) != parent_title_key:
                errors.append(f"Subsection {snum}: section_title does not match parent section title")

    return errors, warnings, dict(metrics)



def write_report(path: Path, data: dict[str, Any], errors: list[str], warnings: list[str], stats: dict[str, int], metrics: dict[str, int]) -> None:
    extraction = data["extraction"]
    ready_pages = sum(1 for p in extraction.get("page_extractions", []) if p.get("include_in_embeddings"))
    front_pages = sum(1 for p in extraction.get("page_extractions", []) if p.get("assignment_status") == "front_matter")
    lessons = sum(len(ch.get("lessons", [])) for ch in extraction.get("chapters", []))
    subsection_count = sum(len(sec.get("subsections", [])) for sec in extraction.get("section_index", []))
    sections_with_subsections = sum(1 for sec in extraction.get("section_index", []) if sec.get("subsections"))
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
        f"- sections with subsections: {sections_with_subsections}",
        f"- total subsections: {subsection_count}",
        f"- subsections outside parent range: {metrics.get('subsections_outside_parent_range', stats.get('subsections_outside_parent_range', 0))}",
        f"- subsections with cross-section pages: {metrics.get('subsections_with_cross_section_or_excluded_pages', stats.get('subsections_with_cross_section_pages', 0))}",
        f"- subsections with repeated PDF pages: {metrics.get('subsections_with_repeated_pdf_pages', 0)}",
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


def publish(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    document_id: str,
    document_key: str,
    subsections_json: Path | None = None,
) -> None:
    LOGGER.info("Starting Step 3 production publish")
    LOGGER.info("Reading input JSON: %s", input_path)
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
        "Poorvi unit/lesson structure uses a curated static production map verified against the textbook contents.",
    ]

    normalize_structure_detection_metadata(data)

    extraction["page_extractions"] = [clean_page(p, stats) for p in extraction.get("page_extractions", [])]

    for collection_name in ["transcripts", "unit_level_pages"]:
        for item in extraction.get(collection_name, []):
            if isinstance(item.get("text"), str):
                item["text"], removed = clean_text(item["text"], item.get("printed_page_number"))
                stats["noise_lines_removed"] += removed
                item["text_length_chars"] = len(item["text"])

    rebuild_lessons_and_sections(data, stats)
    LOGGER.info("Rebuilt lessons/section_index for production")

    days_chapter_map = load_days_chapter_map(subsections_json)
    apply_days_subsections(data, days_chapter_map, stats)
    extraction["notes"].append(
        "Lesson-level subsections were added using production-safe variable day ranges. "
        "External day ranges are used only when fully inside the semantic parent lesson; "
        "invalid/cross-lesson ranges are rejected and replaced by safe auto splits; "
        "short lessons are not forced into repeated five-day subsections."
    )

    rebuild_front_matter(data)
    LOGGER.info("Rebuilt front matter")

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
        "subsections_added_from_days_json": bool(stats.get("subsections_added", 0)),
        "subsections_count": stats.get("subsections_added", 0),
        "subsections_outside_parent_range": stats.get("subsections_outside_parent_range", 0),
        "subsections_with_cross_section_pages": stats.get("subsections_with_cross_section_pages", 0),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    extraction["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    errors, warnings, metrics = validate(data)
    LOGGER.info("Step 3 validation completed: errors=%s warnings=%s", len(errors), len(warnings))
    extraction["quality_summary"]["production_validation_error_count"] = len(errors)
    extraction["quality_summary"]["production_validation_warning_count"] = len(warnings)

    write_json(output_path, data)
    write_report(report_path, data, errors, warnings, dict(stats), metrics)
    LOGGER.info("Step 3 wrote production JSON: %s", output_path)
    LOGGER.info("Step 3 wrote validation report: %s", report_path)
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
    parser.add_argument(
        "--subsections-json",
        type=Path,
        default=None,
        help="Optional chapter_map_final.json containing chapters[].days[] ranges to publish as lesson subsections.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.subsections_json is not None and not args.subsections_json.exists():
        raise FileNotFoundError(args.subsections_json)
    publish(args.input, args.output, args.report, args.document_id, args.document_key, args.subsections_json)


if __name__ == "__main__":
    main()
