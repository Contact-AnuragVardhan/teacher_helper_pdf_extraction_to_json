from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF is required. Install with: pip install pymupdf") from exc


REPRINT_RE = re.compile(r"^Reprint\s+\d{4}\s*-\s*\d{2}$", re.I)
SANTOOR_HEADER_RE = re.compile(r"^(?:\d+\s*)?(?:\d+\s*)?Santoor\s+Grade\s+4$", re.I)
INDD_RE = re.compile(r"\b(?:Prelims|Unit\s*\d+|Chapter)\b.*\.indd\b", re.I)
ROMAN_RE = re.compile(r"^(?=[ivxlcdm]+$)[ivxlcdm]+$", re.I)
NUMERIC_RE = re.compile(r"^\d+$")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
PRIVATE_USE_RE = re.compile(r"[\uE000-\uF8FF]")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

UNIT_SPECS = [
    {"unit_number": "1", "unit_title": "My Land", "start_sequence": 1, "end_sequence": 3},
    {"unit_number": "2", "unit_title": "My Beautiful World", "start_sequence": 4, "end_sequence": 6},
    {"unit_number": "3", "unit_title": "Fun with Games", "start_sequence": 7, "end_sequence": 9},
    {"unit_number": "4", "unit_title": "Up High", "start_sequence": 10, "end_sequence": 12},
]

FRONT_MATTER_TYPES = {
    1: "cover",
    2: "copyright",
    3: "foreword",
    4: "foreword",
    5: "about_the_book",
    6: "about_the_book",
    7: "about_the_book",
    8: "about_the_book",
    9: "nstc",
    10: "constitution",
    11: "textbook_development_team",
    12: "blank_or_art_page",
    13: "acknowledgements",
    14: "acknowledgements",
    15: "contents",
    16: "constitution",
}


@dataclass(frozen=True)
class UnitInfo:
    unit_number: str
    unit_title: str
    lesson_index_in_unit: int

    @property
    def chapter_number(self) -> str:
        return f"Unit {self.unit_number}"

    @property
    def chapter_title(self) -> str:
        return self.unit_title

    @property
    def section_number(self) -> str:
        return f"{self.unit_number}.{self.lesson_index_in_unit}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_basic_text(text: str) -> str:
    """Normalize selectable PDF text before any production field is built.

    Santoor uses embedded/private-use font glyphs for bullets and checkmarks on
    some activity pages. Those glyphs are visible in PDF viewers, but selectable
    extraction returns values such as \uf076, \uf0fc, or \x8f. Leaving those inside
    production_lesson_text/subsection_text makes the JSON unsafe for strict DB
    embeddings, so they are converted deterministically here.
    """
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")
    text = text.replace("\xad", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    # Private-use / symbol-font bullets and checkmarks observed in Santoor.
    text = text.replace("\uf0b7", "•")
    text = text.replace("\uf071", "•")
    text = text.replace("\uf076", "•")
    text = text.replace("\x8f", "•")
    text = text.replace("\uf0fc", "✓")
    text = text.replace("", "•")

    # Last-resort cleanup: any other private-use glyph becomes a safe bullet.
    text = PRIVATE_USE_RE.sub("•", text)
    text = CONTROL_RE.sub("", text)
    return text


def _is_noise_line(line: str, printed_page_number: int | None) -> bool:
    s = line.strip()
    if not s:
        return False
    if REPRINT_RE.match(s):
        return True
    if SANTOOR_HEADER_RE.match(MULTI_SPACE_RE.sub(" ", s)):
        return True
    if INDD_RE.search(s):
        return True
    # Standalone page numbers are headers/footers. Do not drop numbers embedded
    # in real content such as "1. Answer the following".
    if printed_page_number is not None:
        if s == str(printed_page_number):
            return True
        if s == f"{printed_page_number} {printed_page_number}":
            return True
    if ROMAN_RE.match(s) and len(s) <= 6:
        return True
    return False


def clean_extracted_text(text: str, *, page_number: int, printed_page_number: int | None) -> str:
    """Clean selectable PDF text without deleting legitimate lesson content."""
    text = normalize_basic_text(text)
    out: list[str] = []
    blank_pending = False
    previous_nonblank: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        line = MULTI_SPACE_RE.sub(" ", line)
        # Clean repeated printed page header like "136 Santoor Grade 4".
        if printed_page_number is not None:
            line = re.sub(rf"^{printed_page_number}\s+Santoor\s+Grade\s+4$", "", line, flags=re.I).strip()
            line = re.sub(rf"^{printed_page_number}\s+{printed_page_number}\s+Santoor\s+Grade\s+4$", "", line, flags=re.I).strip()
        if _is_noise_line(line, printed_page_number):
            continue
        if not line:
            if out and not blank_pending:
                out.append("")
                blank_pending = True
            continue
        # Remove leading printed page number when it is attached to a chapter title:
        # e.g. "1  Together We Can" -> "Together We Can". Keep numbered questions.
        if printed_page_number is not None:
            line = re.sub(rf"^{printed_page_number}\s{{2,}}", "", raw.strip()).strip() or line
            line = MULTI_SPACE_RE.sub(" ", line)
        if previous_nonblank == line:
            continue
        out.append(line)
        previous_nonblank = line
        blank_pending = False
    # Trim blank lines at ends and collapse repeated blank lines.
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def extract_page_text(pdf_path: Path, page_number: int) -> str:
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        # "text" gives stable reading order for this NCERT PDF. The code keeps a
        # single extractor to avoid mixing incompatible ordering models.
        return page.get_text("text") or ""


def extract_all_pages(pdf_path: Path, pdf_offset: int) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            printed = idx - pdf_offset if idx > pdf_offset else None
            selectable = page.get_text("text") or ""
            cleaned = clean_extracted_text(selectable, page_number=idx, printed_page_number=printed)
            pages.append({
                "page_number": idx,
                "printed_page_number": printed,
                "printed_page_label": str(printed) if printed is not None else None,
                "text": cleaned,
                "selectable_text": normalize_basic_text(selectable).strip(),
                "ocr_text": "",
                "text_sources": ["selectable_pdf_text"] if selectable.strip() else [],
                "quality_flags": [],
                "text_length_chars": len(cleaned),
            })
    return pages


def get_pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def unit_info_for_sequence(sequence: int) -> UnitInfo:
    for spec in UNIT_SPECS:
        if spec["start_sequence"] <= sequence <= spec["end_sequence"]:
            lesson_idx = sequence - spec["start_sequence"] + 1
            return UnitInfo(spec["unit_number"], spec["unit_title"], lesson_idx)
    raise ValueError(f"No Santoor unit configured for chapter sequence {sequence}")


def page_numbers_between(start: int, end: int) -> list[int]:
    return list(range(int(start), int(end) + 1))


def join_page_texts(page_lookup: dict[int, dict[str, Any]], start: int, end: int) -> str:
    chunks: list[str] = []
    for pno in range(start, end + 1):
        text = (page_lookup[pno].get("text") or "").strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks).strip()


def detect_page_content_type(page_number: int, text: str, content_start_page: int) -> str:
    if page_number < content_start_page:
        return FRONT_MATTER_TYPES.get(page_number, "front_matter")
    if re.search(r"\bSelf\s+Assessment\b", text, flags=re.I):
        return "self_assessment_merged_with_lesson"
    return "lesson_content"


def make_front_matter_pages(pages: list[dict[str, Any]], content_start_page: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for p in pages:
        if p["page_number"] >= content_start_page:
            break
        result.append({
            "page_number": p["page_number"],
            "printed_page_number": p["printed_page_number"],
            "content_type": FRONT_MATTER_TYPES.get(p["page_number"], "front_matter"),
            "assignment_status": "front_matter",
            "include_in_embeddings": False,
            "text": p["text"],
            "text_length_chars": len(p["text"]),
        })
    return result


def build_section_and_subsections(chapter: dict[str, Any], page_lookup: dict[int, dict[str, Any]]) -> dict[str, Any]:
    sequence = int(chapter["sequence"])
    ui = unit_info_for_sequence(sequence)
    start_pdf = int(chapter["start_pdf_page"])
    end_pdf = int(chapter["end_pdf_page"])
    start_printed = int(chapter["book_page"])
    end_printed = int(chapter["days"][-1]["end_book_page"])
    lesson_text = join_page_texts(page_lookup, start_pdf, end_pdf)
    page_numbers = page_numbers_between(start_pdf, end_pdf)
    printed_numbers = page_numbers_between(start_printed, end_printed)

    section = {
        "section_number": ui.section_number,
        "section_title": chapter["chapter_name"],
        "unit_number": ui.unit_number,
        "unit_title": ui.unit_title,
        "chapter_type": "unit",
        "chapter_number": ui.chapter_number,
        "chapter_title": ui.chapter_title,
        "start_page": start_pdf,
        "end_page": end_pdf,
        "printed_start_page": start_printed,
        "printed_end_page": end_printed,
        "page_count": len(page_numbers),
        "lesson_text": lesson_text,
        "text_plain": lesson_text,
        "production_lesson_text": lesson_text,
        "text_length_chars": len(lesson_text),
        "physical_start_page": start_pdf,
        "physical_end_page": end_pdf,
        "physical_printed_start_page": start_printed,
        "physical_printed_end_page": end_printed,
        "physical_page_count": len(page_numbers),
        "indexed_page_count": len(page_numbers),
        "indexed_page_numbers": page_numbers,
        "indexed_printed_page_numbers": printed_numbers,
        "excluded_related_pages": [],
        "text_sources": ["selectable_pdf_text"],
        "quality_flags": [],
        "include_in_embeddings": True,
        "embedding_readiness": "ready_for_production_embedding",
        "subsections": [],
    }

    subsections: list[dict[str, Any]] = []
    for day in chapter.get("days", []):
        d = int(day["day"])
        s_pdf = max(int(day["start_pdf_page"]), start_pdf)
        e_pdf = min(int(day["end_pdf_page"]), end_pdf)
        s_print = int(day["start_book_page"])
        e_print = int(day["end_book_page"])
        text = join_page_texts(page_lookup, s_pdf, e_pdf)
        pdf_pages = page_numbers_between(s_pdf, e_pdf)
        printed_pages = page_numbers_between(s_print, e_print)
        was_clamped = (s_pdf != int(day["start_pdf_page"]) or e_pdf != int(day["end_pdf_page"]))
        source = day.get("range_source", "maintained_static_json")
        notes: list[str] = []
        if "self_assessment" in source:
            notes.append("Self Assessment pages are intentionally merged into this previous lesson day so this remains a lesson-plan-capable subsection.")
        subsection = {
            "section_number": ui.section_number,
            "section_title": chapter["chapter_name"],
            "unit_number": ui.unit_number,
            "unit_title": ui.unit_title,
            "chapter_type": "unit",
            "chapter_number": ui.chapter_number,
            "chapter_title": ui.chapter_title,
            "subsection_number": f"{ui.section_number}.{d}",
            "subsection_title": f"Day {d}",
            "anchor_marker": f"Day {d}",
            "anchor_pdf_page": s_pdf,
            "anchor_printed_page": s_print,
            "anchor_detection_method": "static_days_json",
            "anchor_raw_heading": f"Day {d}",
            "included_exercises_or_activities": [f"Day {d}"],
            "includes": [f"Day {d}"],
            "start_page": s_pdf,
            "end_page": e_pdf,
            "start_pdf_page": s_pdf,
            "end_pdf_page": e_pdf,
            "printed_start_page": s_print,
            "printed_end_page": e_print,
            "start_printed_page": s_print,
            "end_printed_page": e_print,
            "pdf_pages": {"start": s_pdf, "end": e_pdf},
            "printed_pages": {"start": s_print, "end": e_print},
            "page_count": len(pdf_pages),
            "subsection_text": text,
            "subsection_text_plain": text,
            "text_plain": text,
            "production_subsection_text": text,
            "production_indexed_page_numbers": pdf_pages,
            "production_printed_page_numbers": printed_pages,
            "production_excluded_pages": [],
            "production_page_count": len(pdf_pages),
            "physical_start_page": s_pdf,
            "physical_end_page": e_pdf,
            "physical_printed_start_page": s_print,
            "physical_printed_end_page": e_print,
            "physical_page_count": len(pdf_pages),
            "page_numbers": pdf_pages,
            "printed_page_numbers": printed_pages,
            "excluded_related_pages": [],
            "text_sources": ["selectable_pdf_text"],
            "quality_flags": ["static_days_json"],
            "include_in_embeddings": True,
            "embedding_readiness": "ready_for_production_embedding",
            "text_length_chars": len(text),
            "source_days_json_day": d,
            "source_days_json_range_source": source,
            "source_days_json_start_pdf_page": int(day["start_pdf_page"]),
            "source_days_json_end_pdf_page": int(day["end_pdf_page"]),
            "source_days_json_start_book_page": int(day["start_book_page"]),
            "source_days_json_end_book_page": int(day["end_book_page"]),
            "source_days_json_was_clamped_to_parent": was_clamped,
            "filtered_out_page_numbers": [],
            "notes": notes,
        }
        if len(text) < 120:
            subsection["quality_flags"].append("short_subsection_text_review")
            subsection["embedding_readiness"] = "ready_with_review_note"
        subsections.append(subsection)
    section["subsections"] = subsections
    return section


def assign_page_metadata(page_extractions: list[dict[str, Any]], sections: list[dict[str, Any]], content_start_page: int) -> None:
    by_page: dict[int, dict[str, Any]] = {}
    self_assessment_ranges: dict[int, bool] = {}

    for section in sections:
        section_has_self_assessment = False
        for sub in section.get("subsections", []):
            source = str(sub.get("source_days_json_range_source") or "").lower()
            notes = " ".join(str(n) for n in sub.get("notes", []))
            if "self_assessment" in source or "self assessment" in notes.lower():
                section_has_self_assessment = True
                for pno in sub.get("page_numbers", []):
                    self_assessment_ranges[int(pno)] = True
        for pno in section["indexed_page_numbers"]:
            by_page[pno] = section

    # Pages after the first visible "Self Assessment" heading in a section should
    # remain tagged as self_assessment_merged_with_lesson even when continuation
    # pages do not repeat the heading. This prevents pages 82 and 152 from being
    # mislabeled as ordinary lesson content.
    self_assessment_started_by_section: dict[str, bool] = {}

    for page in page_extractions:
        pno = int(page["page_number"])
        text = page.get("text") or ""
        section = by_page.get(pno)
        content_type = detect_page_content_type(pno, text, content_start_page)

        if section is not None:
            section_key = section["section_number"]
            if re.search(r"\bSelf\s+Assessment\b", text, flags=re.I):
                self_assessment_started_by_section[section_key] = True
            if self_assessment_ranges.get(pno) and self_assessment_started_by_section.get(section_key):
                content_type = "self_assessment_merged_with_lesson"

        if section is None:
            page.update({
                "chapter_type": None,
                "chapter_number": None,
                "chapter_title": None,
                "unit_number": None,
                "unit_title": None,
                "section_number": None,
                "section_title": None,
                "content_type": content_type,
                "assignment_status": "front_matter" if pno < content_start_page else "unassigned",
                "include_in_lesson_text": False,
                "include_in_embeddings": False,
                "linked_section_title": None,
                "linked_section_number": None,
                "unit_level_title": None,
                "embedding_readiness": "not_indexed_front_matter" if pno < content_start_page else "not_ready_unassigned_page",
            })
            if pno < content_start_page:
                page["quality_flags"] = sorted(set(page.get("quality_flags", []) + ["front_matter_not_lesson_content"]))
        else:
            page.update({
                "chapter_type": section["chapter_type"],
                "chapter_number": section["chapter_number"],
                "chapter_title": section["chapter_title"],
                "unit_number": section["unit_number"],
                "unit_title": section["unit_title"],
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "content_type": content_type,
                "assignment_status": "assigned_to_section",
                "include_in_lesson_text": True,
                "include_in_embeddings": True,
                "linked_section_title": section["section_title"],
                "linked_section_number": section["section_number"],
                "unit_level_title": None,
                "embedding_readiness": "ready_for_production_embedding",
            })
            if content_type == "self_assessment_merged_with_lesson":
                page["quality_flags"] = sorted(set(page.get("quality_flags", []) + ["self_assessment_merged_into_previous_lesson_day"]))


def group_units(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for spec in UNIT_SPECS:
        lessons = [deepcopy(s) for s in sections if s["unit_number"] == spec["unit_number"]]
        for l in lessons:
            # The unit lesson objects in Poorvi keep subsection details. Keep them here too.
            pass
        if not lessons:
            continue
        units.append({
            "chapter_type": "unit",
            "chapter_number": f"Unit {spec['unit_number']}",
            "chapter_title": spec["unit_title"],
            "unit_number": spec["unit_number"],
            "unit_title": spec["unit_title"],
            "start_page": min(l["start_page"] for l in lessons),
            "end_page": max(l["end_page"] for l in lessons),
            "printed_start_page": min(l["printed_start_page"] for l in lessons),
            "printed_end_page": max(l["printed_end_page"] for l in lessons),
            "lessons": lessons,
        })
    return units


def validate_static_map(static_map: dict[str, Any], pdf_page_count: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if int(static_map.get("pdf_page_count", -1)) != pdf_page_count:
        errors.append(f"Static JSON pdf_page_count={static_map.get('pdf_page_count')} but PDF has {pdf_page_count} pages")
    pdf_offset = int(static_map.get("pdf_offset", 0))
    seen_pdf_pages: set[int] = set()
    previous_end = None
    for chapter in static_map.get("chapters", []):
        title = chapter.get("chapter_name", "<untitled>")
        start = int(chapter["start_pdf_page"])
        end = int(chapter["end_pdf_page"])
        if start < 1 or end > pdf_page_count or start > end:
            errors.append(f"Invalid chapter PDF range for {title}: {start}-{end}")
        if previous_end is not None and start != previous_end + 1:
            warnings.append(f"Chapter range gap/overlap before {title}: previous end {previous_end}, next start {start}")
        previous_end = end
        day_prev_end = None
        for day in chapter.get("days", []):
            ds = int(day["start_pdf_page"])
            de = int(day["end_pdf_page"])
            bs = int(day["start_book_page"])
            be = int(day["end_book_page"])
            if not (start <= ds <= de <= end):
                errors.append(f"Day {day.get('day')} of {title} is outside parent PDF range: {ds}-{de} vs {start}-{end}")
            if (ds - pdf_offset) != bs or (de - pdf_offset) != be:
                errors.append(f"Day {day.get('day')} of {title} has PDF/book offset mismatch: pdf {ds}-{de}, book {bs}-{be}, offset {pdf_offset}")
            if day_prev_end is not None and ds != day_prev_end + 1:
                warnings.append(f"Day gap/overlap inside {title}: previous end {day_prev_end}, next start {ds}")
            day_prev_end = de
            for p in range(ds, de + 1):
                if p in seen_pdf_pages:
                    errors.append(f"Duplicate PDF page {p} assigned in static day map")
                seen_pdf_pages.add(p)
    return errors, warnings


def _artifact_count(text: str) -> int:
    # Count only characters that should never survive in production JSON text.
    # Normal newlines/tabs are okay; C1 controls and private-use glyphs are not.
    return len(PRIVATE_USE_RE.findall(text or "")) + len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", text or ""))


def validate_output(output: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    extraction = output["extraction"]
    sections = extraction.get("section_index", [])
    if not sections:
        errors.append("section_index is empty")
    for section in sections:
        lesson_text = section.get("production_lesson_text") or ""
        if not lesson_text.strip():
            errors.append(f"Empty lesson text for section {section.get('section_number')} {section.get('section_title')}")
        if _artifact_count(lesson_text):
            errors.append(f"Production text artifact found in lesson {section.get('section_number')} {section.get('section_title')}")
        if len(section.get("subsections", [])) == 0:
            errors.append(f"No subsections for section {section.get('section_number')} {section.get('section_title')}")
        for sub in section.get("subsections", []):
            txt = (sub.get("production_subsection_text") or "").strip()
            if not txt:
                errors.append(f"Empty subsection text for {sub.get('subsection_number')} {section.get('section_title')}")
            elif len(txt) < 120:
                warnings.append(f"Short subsection text ({len(txt)} chars): {sub.get('subsection_number')} {section.get('section_title')}")
            if _artifact_count(txt):
                errors.append(f"Production text artifact found in subsection {sub.get('subsection_number')} {section.get('section_title')}")
    page_extractions = extraction.get("page_extractions", [])
    assigned_pages = [p for p in page_extractions if p.get("include_in_lesson_text")]
    empty_assigned = [p["page_number"] for p in assigned_pages if not (p.get("text") or "").strip()]
    if empty_assigned:
        errors.append(f"Assigned lesson pages have no text: {empty_assigned}")

    page_artifacts = [p["page_number"] for p in page_extractions if _artifact_count(p.get("text") or "")]
    if page_artifacts:
        errors.append(f"Production page text has private-use/control artifacts on pages: {page_artifacts}")

    return errors, warnings


def build_production_json(
    *,
    pdf_path: Path,
    static_json_path: Path,
    document_id: str,
    document_key: str,
    school_name: str = "Mother Miracle School",
    board: str = "CBSE",
    medium: str = "English",
) -> tuple[dict[str, Any], str, list[str], list[str]]:
    static_map = load_json(static_json_path)
    pdf_page_count = get_pdf_page_count(pdf_path)
    static_errors, static_warnings = validate_static_map(static_map, pdf_page_count)
    if static_errors:
        report = "Static map validation failed:\n" + "\n".join(f"ERROR: {e}" for e in static_errors)
        raise ValueError(report)

    pdf_offset = int(static_map["pdf_offset"])
    pages = extract_all_pages(pdf_path, pdf_offset)
    content_start_page = min(int(c["start_pdf_page"]) for c in static_map["chapters"])
    content_end_page = max(int(c["end_pdf_page"]) for c in static_map["chapters"])
    page_lookup = {p["page_number"]: p for p in pages}
    sections = [build_section_and_subsections(ch, page_lookup) for ch in static_map["chapters"]]
    assign_page_metadata(pages, sections, content_start_page)
    front_matter = make_front_matter_pages(pages, content_start_page)
    units = group_units(sections)
    total_subsections = sum(len(s["subsections"]) for s in sections)
    generated = utc_now_iso()

    output = {
        "metadata": {
            "school_name": school_name,
            "class_name": "Class-4",
            "grade": "Class-4",
            "board": board,
            "medium": medium,
            "publisher": "NCERT",
            "copyright_status": "copyrighted_ncert_textbook_reprint_2026_27",
            "source_file": pdf_path.name,
            "source_type": "textbook_pdf",
            "document_key": document_key,
        },
        "extraction": {
            "book_title": "Santoor: Textbook of English for Grade 4",
            "subject": "English",
            "language": "English",
            "content_profile": "english_textbook",
            "structure_type": "unit_section",
            "total_pdf_pages": pdf_page_count,
            "content_start_page": content_start_page,
            "content_end_page": content_end_page,
            "printed_page_offset": pdf_offset,
            "structure_detection": {
                "method": "curated_static_santoor_map",
                "status": "production_static_map_used",
                "dynamic_detection_possible": False,
                "units_detected": len(UNIT_SPECS),
                "lessons_detected": len(sections),
                "reason": "Santoor Grade 4 uses a fixed NCERT table of contents. The maintained static day-range JSON is used as the production source of truth for chapter and day boundaries.",
                "curated_map_name": "santoor_grade4_english_ncert_2026_27",
                "curated_map_status": "verified_against_toc_and_static_day_ranges",
                "attempts": [
                    {"method": "static_days_json", "status": "used_for_production"},
                    {"method": "selectable_pdf_text", "status": "used_for_text_extraction"},
                ],
            },
            "notes": [
                "Production JSON generated from the maintained Santoor static subsection/day JSON.",
                "Self Assessment 1 and 2 are intentionally merged into the previous lesson day with real lesson text, as requested.",
                "Repeated NCERT headers, footers, page-number-only lines, reprint markers, and INDD artifacts are removed from production text.",
                "Front matter is classified separately and excluded from embeddings by default.",
                "section_index uses semantic/indexed lesson ranges; physical_* fields preserve source PDF ranges.",
            ],
            "section_index": sections,
            "chapters": units,
            "front_matter_pages": front_matter,
            "page_extractions": pages,
            "transcripts": [],
            "unit_level_pages": [],
            "detected_transcript_pages": [],
            "quality_summary": {
                "extraction_version": "santoor-production-v1.1",
                "production_publish_version": "santoor-production-v1.1",
                "document_id_present": bool(document_id),
                "document_key_present": bool(document_key),
                "book_title_normalized": True,
                "structure_type": "unit_section",
                "front_matter_excluded_from_embeddings": True,
                "headers_footers_removed_from_embedding_text": True,
                "static_map_validation_warning_count": len(static_warnings),
                "subsections_added_from_days_json": True,
                "subsections_count": total_subsections,
                "subsections_outside_parent_range": 0,
                "subsections_with_cross_section_pages": 0,
                "self_assessment_pages_merged_into_previous_lesson_day": True,
                "generated_at_utc": generated,
                "safe_for_production_reindex": True,  # updated after validation below
                "production_validation_error_count": 0,
                "production_validation_warning_count": 0,
            },
            "generated_at_utc": generated,
            "subsection_generation": {
                "source": "static_days_json_file_with_parent_boundary_clamp",
                "policy": "Static JSON day ranges are used; final day ranges are clamped inside the semantic parent lesson; Self Assessment pages remain part of the previous lesson day where requested.",
                "sections_with_subsections": len(sections),
                "total_subsections": total_subsections,
                "sections_using_static_days_json_subsections": len(sections),
                "sections_using_auto_safe_subsections": 0,
                "sections_static_days_json_needed_boundary_clamp": 0,
                "subsections_clamped_to_parent_range": 0,
                "subsections_with_filtered_pages": 0,
                "unmatched_days_chapters": [],
                "generated_at_utc": generated,
            },
        },
        "documentId": document_id,
        "document_key": document_key,
    }
    validation_errors, validation_warnings = validate_output(output)
    all_warnings = static_warnings + validation_warnings
    qs = output["extraction"]["quality_summary"]
    qs["safe_for_production_reindex"] = len(validation_errors) == 0
    qs["production_validation_error_count"] = len(validation_errors)
    qs["production_validation_warning_count"] = len(all_warnings)
    report = make_report(output, static_errors + validation_errors, all_warnings)
    return output, report, validation_errors, all_warnings


def make_report(output: dict[str, Any], errors: list[str], warnings: list[str]) -> str:
    ext = output["extraction"]
    lines = []
    lines.append("Santoor Grade 4 English production extraction report")
    lines.append("=" * 62)
    lines.append(f"document_id: {output.get('documentId')}")
    lines.append(f"document_key: {output.get('document_key')}")
    lines.append(f"book_title: {ext.get('book_title')}")
    lines.append(f"total_pdf_pages: {ext.get('total_pdf_pages')}")
    lines.append(f"content_start_page: {ext.get('content_start_page')}")
    lines.append(f"content_end_page: {ext.get('content_end_page')}")
    lines.append(f"printed_page_offset: {ext.get('printed_page_offset')}")
    lines.append(f"units: {len(ext.get('chapters', []))}")
    lines.append(f"sections/lessons: {len(ext.get('section_index', []))}")
    lines.append(f"subsections/days: {ext.get('subsection_generation', {}).get('total_subsections')}")
    lines.append(f"page_extractions: {len(ext.get('page_extractions', []))}")
    lines.append(f"safe_for_production_reindex: {ext.get('quality_summary', {}).get('safe_for_production_reindex')}")
    lines.append(f"production_validation_error_count: {len(errors)}")
    lines.append(f"production_validation_warning_count: {len(warnings)}")
    lines.append("")
    if errors:
        lines.append("ERRORS")
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("ERRORS: none")
    lines.append("")
    if warnings:
        lines.append("WARNINGS")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("WARNINGS: none")
    lines.append("")
    lines.append("Section summary")
    for s in ext.get("section_index", []):
        lines.append(
            f"- {s['section_number']} {s['section_title']}: PDF {s['start_page']}-{s['end_page']}, "
            f"printed {s['printed_start_page']}-{s['printed_end_page']}, days {len(s.get('subsections', []))}, chars {s['text_length_chars']}"
        )
    return "\n".join(lines) + "\n"
