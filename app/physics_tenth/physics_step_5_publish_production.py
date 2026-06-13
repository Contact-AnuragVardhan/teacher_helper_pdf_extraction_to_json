#!/usr/bin/env python3
"""Step 5: publish final JSON and enforce production-readiness gates.

The final JSON is intentionally written in the same DB-ingestion shape used by
English Poorvi and Maths RS Aggarwal:

{
  "metadata": {...},
  "extraction": {
    "book_title": "...",
    "subject": "...",
    "structure_type": "chapters",
    "section_index": [...],
    "chapters": [...],
    "page_extractions": [...]
  },
  "documentId": "...",
  "document_key": "..."
}

Earlier pipeline steps keep the legacy/root working shape so OCR, formula review,
artifact gating, and validation are not changed. Only this final publishing step
normalizes the artifact for DB consumption.
"""
from __future__ import annotations

import argparse
import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any

from physics_common import (
    DEFAULT_BOOK_TITLE,
    DEFAULT_DOCUMENT_ID,
    DEFAULT_DOCUMENT_KEY,
    DEFAULT_GRADE,
    DEFAULT_SUBJECT,
    build_report,
    read_json,
    setup_logging,
    utc_now,
    validate_output,
    write_json,
)

DEFAULT_SCHOOL_NAME = "Mother Miracle School"
DEFAULT_BOARD = "CBSE"
DEFAULT_MEDIUM = "English"
DEFAULT_PUBLISHER = "Modern Publishers"
DEFAULT_COPYRIGHT_STATUS = "copyrighted"


DECORATIVE_OCR_TEXT_KEYS = {
    # Page-level DB/embedding text fields
    "text",
    "text_plain",
    "production_text",
    "production_safe_text",
    "safe_text",
    "safe_text_plain",
    # Source/audit fields are also cleaned in the final production artifact so
    # reviewer scans cannot accidentally treat OCR divider noise as usable text.
    "raw_extracted_text",
    "selectable_text",
    "pre_production_safety_text",
    "raw_text",
    "original_text",
    # Chapter/section-level text fields
    "lesson_text",
    "chapter_text",
    "chapter_text_plain",
    "section_text",
    "section_text_plain",
    "production_chapter_text",
    "production_section_text",
    # Subsection/day-level text fields
    "subsection_text",
    "subsection_text_plain",
    "production_subsection_text",
}

_DECORATIVE_JUNK_LETTERS = set("OEHSLDTRIACOPG")
_DECORATIVE_CORE_JUNK_LETTERS = set("OHSEC")
_DECORATIVE_ALLOWED_NOISE_LETTERS = set("OHSECDOPTLRIAG")
_DECORATIVE_PROTECTED_WORDS = {
    # Common real textbook words/headings that may be made from similar letters.
    # These prevent an over-aggressive OCR divider rule from deleting real text.
    "SCHOOL", "CHOOSE", "CHOICE", "CLOSE", "CLOSED", "SOURCE", "SOURCES",
    "SCALE", "SCALES", "CLASS", "CLASSIC", "SOCIAL", "SCIENCE", "SCIENCES",
    "COIL", "COILS", "CELL", "CELLS", "LOAD", "LEAD", "LED", "SOLID",
    "OHM", "OHMS", "OHMSLAW", "LAW", "LAWS", "DIODE", "DIODES",
}
_DECORATIVE_EXACT_JUNK_LINES = {
    # Known divider fragments found in the current Physics production artifact.
    "COOH",
    "DOOOOOOD",
}


def _letters_only_upper(text: str) -> str:
    return "".join(ch.upper() for ch in text if ch.isalpha())


def _compact_line_key(text: str) -> str:
    return _letters_only_upper(text)


def _line_has_protected_word(stripped: str) -> bool:
    compact = _compact_line_key(stripped)
    if compact in _DECORATIVE_PROTECTED_WORDS:
        return True
    tokens = [_compact_line_key(token) for token in stripped.replace("’", "'").split()]
    tokens = [token for token in tokens if token]
    return any(token in _DECORATIVE_PROTECTED_WORDS for token in tokens)


def _has_decorative_noise_signature(letters: str, stripped: str) -> bool:
    """Catch OCR divider fragments, including short/symbol/mixed-case cases.

    This catches examples found in the final Physics JSON:
      COOH
      SH OHOSOOG OOH
      SHHSHHHHHHHSEHHHHSHHHHSHHSHHEHHOHHHEHSOHOHHTEHEHOSHOOOHESSOHOOEOESOEEEEES ES ESE SESE ESES eeoecee
      SOOO HOH O OOS OOH OSE OO OOH HOOHS OOOO OHHH H OHHH OOH OHHH SHOSOOHHOSOSOHSSHOOO HHO OOOO O®
      COCCHOHOHOOE OOS O

    It is intentionally limited to lines whose alphabet is almost entirely the
    decorative OCR alphabet and that have no digits/formula punctuation. Normal
    words/headings are protected by _DECORATIVE_PROTECTED_WORDS.
    """
    if not letters:
        return False

    compact = _compact_line_key(stripped)
    if compact in _DECORATIVE_EXACT_JUNK_LINES:
        return True

    # Do not remove normal heading/prose words like SCHOOL, SOURCES, OHM'S LAW.
    if _line_has_protected_word(stripped):
        return False

    # A real formula/table line usually contains digits or physics/math symbols.
    # The known decorative lines are alphabetic/space plus occasional OCR symbols.
    if re.search(r"[0-9=+\-×÷/\\%₹$]", stripped):
        return False

    if not set(letters) <= _DECORATIVE_ALLOWED_NOISE_LETTERS:
        return False

    core_count = sum(ch in _DECORATIVE_CORE_JUNK_LETTERS for ch in letters)
    core_ratio = core_count / len(letters)
    o_ratio = letters.count("O") / len(letters)
    hse_ratio = (letters.count("H") + letters.count("S") + letters.count("E")) / len(letters)
    repeated_noise_run = bool(re.search(r"(?:O{2,}|E{3,}|H{3,}|S{3,}|C{2,})", letters))
    token_count = len(stripped.split())

    # Short page-divider fragments such as COOH and SH OHOSOOG OOH.
    if len(letters) >= 4 and core_ratio >= 0.78 and repeated_noise_run:
        return True

    # Very short O-heavy chunks such as DOO OOOOD.
    if len(letters) >= 7 and o_ratio >= 0.55 and repeated_noise_run:
        return True

    # Long compact garbage with lowercase tails or OCR symbols like O®.
    if len(letters) >= 18 and core_ratio >= 0.62 and hse_ratio >= 0.18 and repeated_noise_run:
        return True

    # Long multi-token decorative separators, even when a few D/T/L/R/I/A/G spillover
    # letters occur due to OCR mistakes.
    if len(letters) >= 24 and token_count >= 3 and core_ratio >= 0.58 and repeated_noise_run:
        return True

    return False


def _is_decorative_all_caps_ocr_junk_line(line: str) -> bool:
    """Return True for decorative divider OCR noise.

    Unlike the earlier version, this does not require ^[A-Z\\s]+$ because the
    remaining production artifacts include lowercase OCR tails (eeoecee) and
    symbols (O®). The check normalizes to letters first, then applies strict
    decorative-alphabet rules.
    """
    stripped = (line or "").strip()
    if not stripped:
        return False

    letters = _letters_only_upper(stripped)
    if len(letters) < 4:
        return False

    # Fast prefilter: decorative OCR divider junk always has repeated O/H/S/E/C
    # runs. This avoids running the heavier ratio/protected-word checks across
    # thousands of normal textbook prose/formula lines during the final gate.
    if not ("OO" in letters or "HHH" in letters or "SSS" in letters or "EEE" in letters or "CC" in letters):
        return False

    if _has_decorative_noise_signature(letters, stripped):
        return True

    if len(stripped) < 24 or len(letters) < 20:
        return False

    tokens = stripped.split()
    if len(tokens) < 4:
        return False

    if _line_has_protected_word(stripped):
        return False

    if re.search(r"[0-9=+\-×÷/\\%₹$]", stripped):
        return False

    junk_letter_ratio = sum(ch in _DECORATIVE_JUNK_LETTERS for ch in letters) / len(letters)
    o_or_e_ratio = (letters.count("O") + letters.count("E")) / len(letters)
    short_noise_tokens = sum(
        1 for token in tokens
        if len(_letters_only_upper(token)) <= 8
        and set(_letters_only_upper(token)) <= _DECORATIVE_JUNK_LETTERS
    )
    repeated_noise_run = bool(re.search(r"(?:O{3,}|E{3,}|H{3,}|S{3,})", letters))

    if junk_letter_ratio >= 0.82 and o_or_e_ratio >= 0.28 and short_noise_tokens >= max(3, len(tokens) // 2):
        return True

    if repeated_noise_run and junk_letter_ratio >= 0.76 and short_noise_tokens >= max(3, len(tokens) // 2):
        return True

    return False

def _remove_decorative_ocr_lines(text: str) -> tuple[str, int, list[str]]:
    if not isinstance(text, str) or not text:
        return text, 0, []

    kept: list[str] = []
    removed: list[str] = []
    for line in text.splitlines():
        if _is_decorative_all_caps_ocr_junk_line(line):
            removed.append(line.strip())
        else:
            kept.append(line)

    if not removed:
        return text, 0, []

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, len(removed), removed[:10]


def _cleanup_decorative_ocr_artifacts(data: dict[str, Any]) -> dict[str, Any]:
    """Clean decorative OCR divider noise from the final production artifact.

    This is a final pre-DB cleanup gate requested after artifact review. It
    removes O/H/S/E/C-dominated OCR divider lines from DB/embedding text fields
    and from source/audit fields in the published JSON so those lines cannot be
    mistaken as safe_text by downstream validators.
    """
    summary: dict[str, Any] = {
        "decorative_ocr_cleanup_enabled": True,
        "decorative_ocr_lines_removed": 0,
        "decorative_ocr_text_fields_cleaned": 0,
        "decorative_ocr_cleanup_examples": [],
    }

    examples: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        for key, child in list(value.items()):
            if isinstance(child, str) and key in DECORATIVE_OCR_TEXT_KEYS:
                cleaned, removed_count, removed_examples = _remove_decorative_ocr_lines(child)
                if removed_count:
                    value[key] = cleaned
                    summary["decorative_ocr_lines_removed"] += removed_count
                    summary["decorative_ocr_text_fields_cleaned"] += 1
                    for example in removed_examples:
                        if example not in examples:
                            examples.append(example)
            elif isinstance(child, (dict, list)):
                visit(child)

    # Clean the working root shape used before final DB normalization. This covers
    # page_extractions, chapters, section_index, and nested subsections.
    for root_key in ("page_extractions", "chapters", "section_index"):
        visit(data.get(root_key))

    # Do not write the raw removed OCR junk back into the production JSON.
    # Storing those examples caused exact garbage strings to remain searchable
    # in the final artifact even after DB-safe text had been cleaned.
    summary["decorative_ocr_cleanup_examples"] = []
    summary["decorative_ocr_cleanup_example_count"] = len(examples)
    data["decorative_ocr_cleanup"] = summary
    return summary



PRODUCTION_DECORATIVE_SCAN_TEXT_KEYS = {
    "text",
    "text_plain",
    "production_safe_text",
    "production_text",
    "safe_text",
    "safe_text_plain",
    "section_text",
    "section_text_plain",
    "chapter_text",
    "chapter_text_plain",
    "subsection_text",
    "subsection_text_plain",
    "production_section_text",
    "production_chapter_text",
    "production_subsection_text",
}


def _scan_remaining_decorative_ocr_artifacts(data: dict[str, Any], max_examples: int = 50) -> list[dict[str, Any]]:
    """Return remaining decorative OCR blockers in production/DB text fields.

    This is intentionally run *after* final cleanup. Any hit here means the
    published JSON must not be marked production_ready, because the line can flow
    into DB embeddings or lesson text.
    """
    hits: list[dict[str, Any]] = []

    def scan_text(text: str, path: str) -> None:
        if not isinstance(text, str) or not text:
            return
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _is_decorative_all_caps_ocr_junk_line(line):
                if len(hits) < max_examples:
                    hits.append({"path": path, "line": line_no, "text": line.strip()})
                else:
                    hits.append({"path": path, "line": line_no, "text": "<additional blockers omitted>"})
                    return

    def visit(value: Any, path: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(child, str) and key in PRODUCTION_DECORATIVE_SCAN_TEXT_KEYS:
                scan_text(child, child_path)
            elif isinstance(child, (dict, list)):
                visit(child, child_path)

    for root_key in ("page_extractions", "chapters", "section_index"):
        visit(data.get(root_key), root_key)
    return hits

def count_unresolved(data: dict[str, Any]) -> int:
    total = 0
    for page in data.get("page_extractions") or []:
        total += int(page.get("unresolved_review_items") or 0)
    return total


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _class_label(value: Any) -> str:
    """Return grade in the same format used by Poorvi/Maths metadata: Class-10."""
    text = str(value or DEFAULT_GRADE).strip()
    if text.lower().startswith("class"):
        number = text.replace("Class", "", 1).replace("class", "", 1).strip(" -_")
        return f"Class-{number}" if number else "Class-10"
    return f"Class-{text}"


def _max_end_page(items: list[dict[str, Any]], fallback: int | None = None) -> int | None:
    values: list[int] = []
    for item in items:
        for key in ("end_pdf_page", "end_page", "physical_end_page"):
            val = _as_int(item.get(key))
            if val is not None:
                values.append(val)
                break
    return max(values) if values else fallback


def _min_start_page(items: list[dict[str, Any]], fallback: int | None = None) -> int | None:
    values: list[int] = []
    for item in items:
        for key in ("start_pdf_page", "start_page", "physical_start_page"):
            val = _as_int(item.get(key))
            if val is not None:
                values.append(val)
                break
    return min(values) if values else fallback


def _copy_root_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return copy.deepcopy(data.get(key) or [])


def _metadata_block(data: dict[str, Any], document_key: str) -> dict[str, Any]:
    grade_label = _class_label(data.get("grade"))
    return {
        "school_name": data.get("school_name") or DEFAULT_SCHOOL_NAME,
        "class_name": data.get("class_name") or grade_label,
        "grade": data.get("grade_label") or grade_label,
        "board": data.get("board") or DEFAULT_BOARD,
        "medium": data.get("medium") or DEFAULT_MEDIUM,
        "publisher": data.get("publisher") or DEFAULT_PUBLISHER,
        "copyright_status": data.get("copyright_status") or DEFAULT_COPYRIGHT_STATUS,
        "source_file": data.get("pdf_file") or data.get("source_file") or "Grade10_Physics.pdf",
        "source_type": "textbook_pdf",
        "document_key": document_key,
    }


def _quality_summary(data: dict[str, Any], stats: Counter | dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    stats = dict(stats or {})
    metrics = dict(validation.get("metrics") or {})
    return {
        "production_status": data.get("production_status"),
        "text_accuracy_status": data.get("text_accuracy_status"),
        "validation_status": validation.get("status"),
        "statistics": stats,
        "metrics": metrics,
        "errors": validation.get("errors") or [],
    }



def _content_start_page(data: dict[str, Any], section_index: list[dict[str, Any]] | None = None) -> int:
    section_index = section_index or _copy_root_list(data, "section_index")
    return _as_int(data.get("content_start_pdf_page"), _min_start_page(section_index, 1)) or 1


def _global_printed_page_for_pdf(pdf_page: Any, content_start_page: int) -> int | None:
    page = _as_int(pdf_page)
    if page is None:
        return None
    return max(1, page - int(content_start_page) + 1)


def _global_printed_range(start_pdf: Any, end_pdf: Any, content_start_page: int) -> tuple[int | None, int | None]:
    return (
        _global_printed_page_for_pdf(start_pdf, content_start_page),
        _global_printed_page_for_pdf(end_pdf, content_start_page),
    )


def _global_printed_numbers(page_numbers: Any, content_start_page: int) -> list[int]:
    numbers: list[int] = []
    if isinstance(page_numbers, list):
        for item in page_numbers:
            value = _global_printed_page_for_pdf(item, content_start_page)
            if value is not None:
                numbers.append(value)
    return numbers


def _production_text(raw: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _chapter_label_for_item(raw: dict[str, Any], local_page: Any = None) -> str | None:
    label = raw.get("printed_page_label") or raw.get("book_page_label")
    if label:
        return str(label)
    number = raw.get("section_number") or raw.get("chapter_number")
    if number is None or local_page is None:
        return None
    num = str(number).replace("Chapter", "").strip()
    return f"{num}/{local_page}"


def _normalize_printed_fields_on_item(raw: dict[str, Any], content_start_page: int) -> None:
    """Use continuous printed_* fields while preserving chapter-local page labels.

    Physics chapters restart local printed page numbering at 1. Poorvi/Maths use
    continuous printed page numbers, and the DB/UI code tends to treat
    printed_start_page as a continuous comparable number. Therefore the final
    DB JSON exposes continuous printed_* values and keeps the original local
    chapter page numbers under chapter_printed_*.
    """
    if not isinstance(raw, dict):
        return

    # Page-level record.
    if raw.get("page_number") is not None:
        local_page = _as_int(raw.get("printed_page_number"))
        if local_page is not None and raw.get("chapter_printed_page_number") is None:
            raw["chapter_printed_page_number"] = local_page
        if raw.get("chapter_printed_page_label") is None:
            label = _chapter_label_for_item(raw, local_page)
            if label:
                raw["chapter_printed_page_label"] = label
        global_page = _global_printed_page_for_pdf(raw.get("page_number"), content_start_page)
        if global_page is not None:
            raw["printed_page_number"] = global_page
        return

    start_pdf = raw.get("start_pdf_page") or raw.get("start_page") or raw.get("physical_start_page")
    end_pdf = raw.get("end_pdf_page") or raw.get("end_page") or raw.get("physical_end_page")
    local_start = _as_int(raw.get("printed_start_page") or raw.get("start_printed_page"))
    local_end = _as_int(raw.get("printed_end_page") or raw.get("end_printed_page"))
    if local_start is not None and raw.get("chapter_printed_start_page") is None:
        raw["chapter_printed_start_page"] = local_start
    if local_end is not None and raw.get("chapter_printed_end_page") is None:
        raw["chapter_printed_end_page"] = local_end
    if raw.get("chapter_printed_page_label") is None:
        label = _chapter_label_for_item(raw, local_start)
        if label:
            raw["chapter_printed_page_label"] = label

    global_start, global_end = _global_printed_range(start_pdf, end_pdf, content_start_page)
    if global_start is not None:
        raw["printed_start_page"] = global_start
        raw["start_printed_page"] = global_start
    if global_end is not None:
        raw["printed_end_page"] = global_end
        raw["end_printed_page"] = global_end
    if isinstance(raw.get("printed_pages"), dict):
        raw["chapter_printed_pages"] = {
            "start": local_start,
            "end": local_end,
        }
        raw["printed_pages"] = {
            "start": global_start,
            "end": global_end,
        }

    page_numbers = raw.get("page_numbers") or raw.get("indexed_page_numbers") or raw.get("production_indexed_page_numbers")
    local_printed_numbers = raw.get("printed_page_numbers") or raw.get("indexed_printed_page_numbers") or raw.get("production_printed_page_numbers")
    if isinstance(local_printed_numbers, list) and raw.get("chapter_printed_page_numbers") is None:
        raw["chapter_printed_page_numbers"] = copy.deepcopy(local_printed_numbers)
    global_numbers = _global_printed_numbers(page_numbers, content_start_page)
    if global_numbers:
        if "printed_page_numbers" in raw:
            raw["printed_page_numbers"] = global_numbers
        if "indexed_printed_page_numbers" in raw:
            raw["indexed_printed_page_numbers"] = global_numbers
        if "production_printed_page_numbers" in raw:
            raw["production_printed_page_numbers"] = global_numbers


def _normalize_printed_fields_recursive(value: Any, content_start_page: int) -> None:
    if isinstance(value, list):
        for item in value:
            _normalize_printed_fields_recursive(item, content_start_page)
        return
    if not isinstance(value, dict):
        return
    _normalize_printed_fields_on_item(value, content_start_page)
    for child in value.values():
        if isinstance(child, (dict, list)):
            _normalize_printed_fields_recursive(child, content_start_page)


def _copy_selected(raw: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: copy.deepcopy(raw[k]) for k in keys if k in raw and raw[k] is not None}


PAGE_SLIM_KEYS = (
    "page_number", "printed_page_number", "printed_page_label", "chapter_printed_page_number", "chapter_printed_page_label",
    "chapter_number", "chapter_title", "chapter_type", "section_number", "section_title",
    "content_type", "assignment_status", "include_in_chapter_text", "include_in_lesson_text", "include_in_embeddings",
    "embedding_readiness", "text_sources", "quality_flags", "text_length_chars", "unresolved_review_items", "reviewed_items_applied",
)

SECTION_SLIM_KEYS = (
    "section_number", "section_title", "unit_number", "unit_title", "chapter_type", "chapter_number", "chapter_title",
    "book_page", "book_page_label", "chapter_printed_page_label", "start_page", "end_page", "start_pdf_page", "end_pdf_page",
    "printed_start_page", "printed_end_page", "start_printed_page", "end_printed_page",
    "chapter_printed_start_page", "chapter_printed_end_page", "printed_pages", "chapter_printed_pages",
    "physical_start_page", "physical_end_page", "physical_printed_start_page", "physical_printed_end_page",
    "page_count", "physical_page_count", "indexed_page_count", "indexed_page_numbers", "indexed_printed_page_numbers",
    "production_indexed_page_numbers", "production_printed_page_numbers", "chapter_printed_page_numbers",
    "text_sources", "quality_flags", "include_in_embeddings", "embedding_readiness", "text_length_chars",
    "unresolved_review_items", "reviewed_items_applied",
)

SUBSECTION_SLIM_KEYS = (
    "section_number", "section_title", "unit_number", "unit_title", "chapter_type", "chapter_number", "chapter_title",
    "subsection_number", "subsection_title", "subsection_code", "anchor_marker", "anchor_pdf_page", "anchor_printed_page",
    "anchor_detection_method", "included_exercises_or_activities", "includes", "day", "start_page", "end_page", "start_pdf_page", "end_pdf_page",
    "printed_start_page", "printed_end_page", "start_printed_page", "end_printed_page",
    "chapter_printed_start_page", "chapter_printed_end_page", "chapter_printed_page_label", "pdf_pages", "printed_pages", "chapter_printed_pages",
    "page_count", "production_indexed_page_numbers", "production_printed_page_numbers", "production_page_count",
    "physical_start_page", "physical_end_page", "page_numbers", "printed_page_numbers", "chapter_printed_page_numbers",
    "text_sources", "quality_flags", "include_in_embeddings", "embedding_readiness", "text_length_chars",
    "unresolved_review_items", "reviewed_items_applied", "source_days_json_day", "source_days_json_subsection_code",
    "source_days_json_range_source", "filtered_out_page_numbers", "notes",
)

CHAPTER_SLIM_KEYS = (
    "sequence", "chapter_type", "chapter_number", "chapter_title", "chapter_name", "section_number", "section_title",
    "book_page", "book_page_label", "chapter_printed_page_label", "start_page", "end_page", "start_pdf_page", "end_pdf_page",
    "printed_start_page", "printed_end_page", "start_printed_page", "end_printed_page",
    "chapter_printed_start_page", "chapter_printed_end_page", "printed_pages", "chapter_printed_pages",
    "physical_start_page", "physical_end_page", "physical_printed_start_page", "physical_printed_end_page",
    "page_count", "physical_page_count", "production_indexed_page_numbers", "production_printed_page_numbers",
    "chapter_printed_page_numbers", "include_in_embeddings", "embedding_readiness", "text_length_chars",
    "unresolved_review_items", "reviewed_items_applied", "quality_flags",
)


def _slim_page(raw: dict[str, Any], content_start_page: int) -> dict[str, Any]:
    item = _copy_selected(raw, PAGE_SLIM_KEYS)
    text = _production_text(raw, "production_safe_text", "production_text", "text_plain", "text")
    item["text"] = text
    item["text_plain"] = text
    item["production_safe_text"] = text
    _normalize_printed_fields_on_item(item, content_start_page)
    return {k: v for k, v in item.items() if v is not None}


def _slim_subsection(raw: dict[str, Any], content_start_page: int) -> dict[str, Any]:
    item = _copy_selected(raw, SUBSECTION_SLIM_KEYS)
    text = _production_text(raw, "production_subsection_text", "subsection_text_plain", "text_plain", "subsection_text")
    item["subsection_text"] = text
    item["subsection_text_plain"] = text
    item["text_plain"] = text
    item["production_subsection_text"] = text
    _normalize_printed_fields_on_item(item, content_start_page)
    return {k: v for k, v in item.items() if v is not None}


def _slim_section(raw: dict[str, Any], content_start_page: int) -> dict[str, Any]:
    item = _copy_selected(raw, SECTION_SLIM_KEYS)
    text = _production_text(raw, "production_section_text", "section_text_plain", "text_plain", "section_text")
    item["section_text"] = text
    item["section_text_plain"] = text
    item["text_plain"] = text
    item["production_section_text"] = text
    item["subsections"] = [_slim_subsection(sub, content_start_page) for sub in raw.get("subsections") or []]
    _normalize_printed_fields_on_item(item, content_start_page)
    return {k: v for k, v in item.items() if v is not None}


def _slim_chapter(raw: dict[str, Any], content_start_page: int) -> dict[str, Any]:
    item = _copy_selected(raw, CHAPTER_SLIM_KEYS)
    text = _production_text(raw, "production_chapter_text", "chapter_text_plain", "production_section_text", "section_text_plain", "text_plain", "chapter_text")
    item["chapter_text"] = text
    item["chapter_text_plain"] = text
    item["production_chapter_text"] = text
    # Keep chapter-level subsections for compatibility, but do not keep lessons[]
    # because lessons[] is a full duplicate of chapter/section text for Physics.
    item["subsections"] = [_slim_subsection(sub, content_start_page) for sub in raw.get("subsections") or []]
    _normalize_printed_fields_on_item(item, content_start_page)
    return {k: v for k, v in item.items() if v is not None}


def _slim_static_chapter(raw: dict[str, Any], content_start_page: int) -> dict[str, Any]:
    item = _copy_selected(raw, (
        "sequence", "chapter_name", "book_page", "book_page_label", "start_pdf_page", "end_pdf_page",
        "teaching_start_page", "teaching_end_page", "physical_start_page", "physical_end_page", "days",
    ))
    _normalize_printed_fields_recursive(item, content_start_page)
    return item


def _build_common_extraction_payload(
    data: dict[str, Any],
    *,
    pipeline_extraction: dict[str, Any],
    validation: dict[str, Any],
    stats: Counter | dict[str, Any],
    page_extractions: list[dict[str, Any]],
    section_index: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    static_chapters: list[dict[str, Any]],
    include_audit_blocks: bool,
) -> dict[str, Any]:
    total_pdf_pages = _as_int(data.get("pdf_page_count"), len(page_extractions)) or len(page_extractions)
    content_start_page = _content_start_page(data, section_index)
    teaching_content_end_page = _max_end_page(section_index, total_pdf_pages) or total_pdf_pages
    content_end_page = total_pdf_pages
    printed_offset = content_start_page - 1

    notes = [
        "Final JSON has the same DB-ingestion shape as English Poorvi and Maths RS Aggarwal.",
        "OCR, line safety filtering, formula review corrections, and final artifact gate are unchanged; this step only normalizes JSON layout.",
        "A final decorative OCR cleanup gate removes all-caps divider noise from DB/embedding-safe text fields.",
        "printed_start_page/printed_end_page are continuous book-level page numbers for DB/UI consistency.",
        "chapter_printed_start_page/chapter_printed_end_page preserve Physics chapter-local printed pages such as 2/1, 3/1, etc.",
        "Use page_extractions.text, section_index.production_section_text, and subsection production_subsection_text for production-safe embeddings.",
        "Exact formulas are included only after review through the corrections JSON; risky unreviewed formula/table/diagram OCR is excluded from production text.",
    ]
    if include_audit_blocks:
        notes.append("This is the full audit artifact; it intentionally keeps raw/layout/review fields and is not the lean DB-ingestion artifact.")
    else:
        notes.append("This is the slim DB-ingestion artifact; raw/layout/review audit fields are written to Grade10_Physics_production_audit_full.json instead.")

    extraction_payload: dict[str, Any] = {
        "book_title": data.get("book_title") or DEFAULT_BOOK_TITLE,
        "subject": data.get("subject") or DEFAULT_SUBJECT,
        "language": data.get("language") or DEFAULT_MEDIUM,
        "content_profile": "physics_textbook",
        "structure_type": data.get("structure_type") or "chapters",
        "total_pdf_pages": total_pdf_pages,
        "content_start_page": content_start_page,
        "content_end_page": content_end_page,
        "teaching_content_end_page": teaching_content_end_page,
        "printed_page_offset": printed_offset,
        "chapter_local_page_numbering": True,
        "subsection_policy": data.get("subsection_policy") or "practice_exercise_based_static_day_ranges",
        "page_numbering_note": (
            "Physics source pages are chapter-local (1/1, 2/1, ...). "
            "Final printed_* fields are continuous book-level pages; chapter_printed_* fields preserve local chapter page numbers."
        ),
        "notes": notes,
        "section_index": section_index,
        "chapters": chapters,
        "front_matter_pages": [p for p in page_extractions if not p.get("include_in_lesson_text") and p.get("assignment_status") == "front_matter"],
        "page_extractions": page_extractions,
        "static_chapters": static_chapters,
        "quality_summary": _quality_summary(data, stats, validation),
        "decorative_ocr_cleanup": data.get("decorative_ocr_cleanup"),
        "final_decorative_ocr_gate": data.get("final_decorative_ocr_gate"),
        "generated_at_utc": pipeline_extraction.get("generated_at") or utc_now(),
        "pipeline": pipeline_extraction,
        "production_notes": data.get("production_notes"),
        "source_profile": {
            "source_type_before_publish": data.get("source_type"),
            "pdf_file": data.get("pdf_file"),
            "pdf_offset": data.get("pdf_offset"),
            "book_name": data.get("book_name"),
        },
    }
    if include_audit_blocks:
        extraction_payload["final_artifact_gate"] = data.get("final_artifact_gate")
    return {k: v for k, v in extraction_payload.items() if v is not None}


def build_full_audit_json(
    data: dict[str, Any],
    *,
    document_id: str,
    document_key: str,
    pipeline_extraction: dict[str, Any],
    validation: dict[str, Any],
    stats: Counter | dict[str, Any],
) -> dict[str, Any]:
    page_extractions = _copy_root_list(data, "page_extractions")
    chapters = _copy_root_list(data, "chapters")
    section_index = _copy_root_list(data, "section_index")
    static_chapters = _copy_root_list(data, "static_chapters")
    content_start_page = _content_start_page(data, section_index)
    for root in (page_extractions, chapters, section_index, static_chapters):
        _normalize_printed_fields_recursive(root, content_start_page)
    return {
        "metadata": _metadata_block(data, document_key),
        "extraction": _build_common_extraction_payload(
            data,
            pipeline_extraction=pipeline_extraction,
            validation=validation,
            stats=stats,
            page_extractions=page_extractions,
            section_index=section_index,
            chapters=chapters,
            static_chapters=static_chapters,
            include_audit_blocks=True,
        ),
        "documentId": document_id,
        "document_key": document_key,
        "text_accuracy_status": data.get("text_accuracy_status"),
        "production_status": data.get("production_status"),
        "artifact_type": "full_audit_not_for_db_ingestion",
    }


def build_db_consumable_json(
    data: dict[str, Any],
    *,
    document_id: str,
    document_key: str,
    pipeline_extraction: dict[str, Any],
    validation: dict[str, Any],
    stats: Counter | dict[str, Any],
) -> dict[str, Any]:
    """Build the slim, DB-ingestion JSON without changing text extraction output.

    The actual production text is copied exactly from the already-reviewed fields.
    This function only removes raw/layout/review audit fields and duplicate lesson
    branches that are not needed by json_input_loader.
    """
    raw_pages = _copy_root_list(data, "page_extractions")
    raw_section_index = _copy_root_list(data, "section_index")
    raw_chapters = _copy_root_list(data, "chapters")
    raw_static_chapters = _copy_root_list(data, "static_chapters")
    content_start_page = _content_start_page(data, raw_section_index)

    page_extractions = [_slim_page(page, content_start_page) for page in raw_pages]
    section_index = [_slim_section(section, content_start_page) for section in raw_section_index]
    chapters = [_slim_chapter(chapter, content_start_page) for chapter in raw_chapters]
    static_chapters = [_slim_static_chapter(chapter, content_start_page) for chapter in raw_static_chapters]

    return {
        "metadata": _metadata_block(data, document_key),
        "extraction": _build_common_extraction_payload(
            data,
            pipeline_extraction=pipeline_extraction,
            validation=validation,
            stats=stats,
            page_extractions=page_extractions,
            section_index=section_index,
            chapters=chapters,
            static_chapters=static_chapters,
            include_audit_blocks=False,
        ),
        "documentId": document_id,
        "document_key": document_key,
        "text_accuracy_status": data.get("text_accuracy_status"),
        "production_status": data.get("production_status"),
        "artifact_type": "slim_db_ingestion",
    }

def publish_json(input_json: Path, document_id: str, document_key: str, allow_review_required: bool = False) -> tuple[dict[str, Any], dict[str, Any], str]:
    src = read_json(input_json)
    working = copy.deepcopy(src)
    cleanup_summary = _cleanup_decorative_ocr_artifacts(working)
    prior_cleanup_summary = (working.get("final_artifact_gate") or {}).get("decorative_ocr_cleanup") or {}
    cumulative_cleanup_summary = dict(cleanup_summary)
    cumulative_cleanup_summary["decorative_ocr_lines_removed"] = (
        int(prior_cleanup_summary.get("decorative_ocr_lines_removed") or 0)
        + int(cleanup_summary.get("decorative_ocr_lines_removed") or 0)
    )
    cumulative_cleanup_summary["decorative_ocr_text_fields_cleaned"] = (
        int(prior_cleanup_summary.get("decorative_ocr_text_fields_cleaned") or 0)
        + int(cleanup_summary.get("decorative_ocr_text_fields_cleaned") or 0)
    )
    cumulative_cleanup_summary["decorative_ocr_cleanup_example_count"] = (
        int(prior_cleanup_summary.get("decorative_ocr_cleanup_example_count") or 0)
        + int(cleanup_summary.get("decorative_ocr_cleanup_example_count") or 0)
    )
    cumulative_cleanup_summary["decorative_ocr_cleanup_examples"] = []
    working["decorative_ocr_cleanup"] = cumulative_cleanup_summary

    remaining_decorative_blockers = _scan_remaining_decorative_ocr_artifacts(working)
    stats = Counter(working.get("extraction", {}).get("statistics") or {})
    stats["decorative_ocr_lines_removed"] = int(cumulative_cleanup_summary.get("decorative_ocr_lines_removed") or 0)
    stats["decorative_ocr_text_fields_cleaned"] = int(cumulative_cleanup_summary.get("decorative_ocr_text_fields_cleaned") or 0)
    stats["final_decorative_ocr_blockers"] = len(remaining_decorative_blockers)
    stats["final_artifact_blockers"] = int(stats.get("final_artifact_blockers") or 0) + len(remaining_decorative_blockers)
    working["final_decorative_ocr_gate"] = {
        "status": "passed" if not remaining_decorative_blockers else "failed",
        "blocker_count": len(remaining_decorative_blockers),
        "blockers": remaining_decorative_blockers[:50],
    }
    unresolved = count_unresolved(working)
    stats["unresolved_review_items"] = unresolved
    existing_errors: list[str] = []
    validation = validate_output(working, stats, existing_errors)

    final_artifact_blockers = int(stats.get("final_artifact_blockers") or 0)

    if unresolved > 0:
        working["text_accuracy_status"] = "failed_formula_or_diagram_review"
        working["production_status"] = "review_required_not_formula_safe"
        if not allow_review_required:
            validation["errors"].append(
                f"Unresolved formula/table/diagram review items remain: {unresolved}. Fill corrections JSON or rerun with --allow-review-required for non-final review output."
            )
            validation["status"] = "failed"
    elif final_artifact_blockers > 0:
        working["text_accuracy_status"] = "failed_final_artifact_gate"
        working["production_status"] = "review_required_leftover_artifacts"
        if not allow_review_required:
            validation["errors"].append(
                f"Final artifact gate found {final_artifact_blockers} production-text artifact blocker(s). Fix these before final production."
            )
            validation["status"] = "failed"
    else:
        working["text_accuracy_status"] = "formula_safe_text_ready"
        working["production_status"] = "production_ready"

    working["documentId"] = document_id
    working["document_key"] = document_key
    working["production_notes"] = {
        "structure_status": "passed" if not validation.get("metrics", {}).get("subsections_outside_parent_range") else "failed",
        "text_policy": "Exact formulas are included only when reviewed through corrections JSON. Unreviewed risky formula/table/diagram OCR is excluded from production text and retained in raw_extracted_text/line_items.",
        "safe_for": [
            "lesson_planning" if unresolved else "lesson_planning_with_reviewed_formulas",
            "topic_context_embeddings" if unresolved else "formula_reviewed_embeddings",
            "chapter_subsection_page_mapping",
        ],
        "not_safe_for_when_review_required": [
            "exact_formula_QA",
            "student_numerical_answer_validation",
            "formula-search embeddings",
        ] if unresolved else [],
        "unresolved_review_items": unresolved,
        "final_artifact_blockers": int(stats.get("final_artifact_blockers") or 0),
        "final_decorative_ocr_blockers": int(stats.get("final_decorative_ocr_blockers") or 0),
        "decorative_ocr_lines_removed": int(stats.get("decorative_ocr_lines_removed") or 0),
        "decorative_ocr_cleanup_examples": [],
        "decorative_ocr_cleanup_example_count": int(cumulative_cleanup_summary.get("decorative_ocr_cleanup_example_count") or 0),
    }

    pipeline_extraction = {
        "step": 5,
        "status": "production_ready" if working["production_status"] == "production_ready" else "review_required_output_generated",
        "generated_at": utc_now(),
        "generator": "physics_step_5_publish_production.py",
        "method": "publish_with_strict_formula_review_gate_and_db_consumable_schema",
        "source_step4_json": str(input_json),
        "statistics": dict(stats),
        "validation": validation,
    }

    # Build report from the legacy working shape so the existing report helper keeps
    # showing chapter/subsection details. The returned output_json is the slim DB
    # artifact; audit_data is written separately with raw/layout/debug fields.
    report = build_report(working | {"extraction": pipeline_extraction}, validation, "Grade 10 Physics Step 5 production publish report")
    final_data = build_db_consumable_json(
        working,
        document_id=document_id,
        document_key=document_key,
        pipeline_extraction=pipeline_extraction,
        validation=validation,
        stats=stats,
    )
    audit_data = build_full_audit_json(
        working,
        document_id=document_id,
        document_key=document_key,
        pipeline_extraction=pipeline_extraction,
        validation=validation,
        stats=stats,
    )
    return final_data, audit_data, report



def default_audit_json_path(output_json: Path) -> Path:
    """Return sidecar full-audit path for a slim DB output path."""
    stem = output_json.stem
    if stem.endswith("_production_ready"):
        stem = stem[: -len("_production_ready")] + "_production_audit_full"
    else:
        stem = stem + "_audit_full"
    return output_json.with_name(stem + output_json.suffix)

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: publish final Physics JSON with production-readiness gates.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True, help="Slim DB-ingestion production JSON output.")
    parser.add_argument("--audit-json", type=Path, default=None, help="Optional full audit JSON output. Defaults to *_production_audit_full.json next to --output-json.")
    parser.add_argument("--no-audit-json", action="store_true", help="Do not write the full audit sidecar JSON.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument("--document-key", default=DEFAULT_DOCUMENT_KEY)
    parser.add_argument("--allow-review-required", action="store_true", help="Write output even with unresolved formula review items; status remains review_required_not_formula_safe.")
    parser.add_argument("--fail-on-review-required", action="store_true", help="Exit non-zero if unresolved review items remain.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    audit_json = None if args.no_audit_json else (args.audit_json or default_audit_json_path(args.output_json))
    for path in [args.output_json, audit_json, args.report]:
        if path and path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")
    data, audit_data, report = publish_json(args.input_json.resolve(), args.document_id, args.document_key, args.allow_review_required)
    write_json(args.output_json.resolve(), data)
    if audit_json:
        write_json(audit_json.resolve(), audit_data)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(f"Final DB JSON: {args.output_json.resolve()}")
    if audit_json:
        print(f"Full audit JSON: {audit_json.resolve()}")
    if args.report:
        print(f"Final report: {args.report.resolve()}")
    print(f"Production status: {data.get('production_status')}")
    if args.fail_on_review_required and data.get("production_status") != "production_ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
