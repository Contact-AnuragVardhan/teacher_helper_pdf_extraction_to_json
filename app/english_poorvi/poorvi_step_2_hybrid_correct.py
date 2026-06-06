#!/usr/bin/env python3
"""
poorvi_step_2_hybrid_correct.py

Second-pass correction for NCERT English Poorvi Grade 6.

Purpose:
- Read the step 1 JSON.
- Add missing visual/comic panel text for Rama to the Rescue pages 38-41.
- Remap transcript pages out of the wrong lesson_text.
- Remap unit-level activity pages out of lesson_text.
- Rebuild lesson_text, semantic ranges, and section_index.
- Preserve physical ranges separately in physical_* fields.
- Write validation report.

Run:
  python app/english_poorvi/poorvi_step_2_hybrid_correct.py ^
    --pdf input/English_Poorvi.pdf ^
    --input-json output/English_Poorvi_section_extraction.json ^
    --output-json output/English_Poorvi_hybrid_corrected_extraction_v2.json ^
    --report output/English_Poorvi_hybrid_corrected_extraction_v2_validation_report.txt

Tesseract is optional for this step. If present, it is used as an OCR audit layer for the Rama pages.
The actual corrected visible panel text is included as deterministic visual correction strings.
"""

from __future__ import annotations

import argparse
import logging
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


LOGGER = logging.getLogger("english_poorvi.step2_hybrid_correct")


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging for command-line debugging."""
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )



TITLE_NORMALIZE = {
    "Yoga-A Way of Life": "Yoga—A Way of Life",
    "Hamara Bharat-Incredible India!": "Hamara Bharat—Incredible India!",
}

TRANSCRIPT_PAGE_LINKS = {
    52: {"related_section_title": "A Bottle of Dew", "related_section_number": "1.1"},
    53: {"related_section_title": "The Raven and the Fox", "related_section_number": "1.2"},
    54: {"related_section_title": "Rama to the Rescue", "related_section_number": "1.3"},
    87: {"related_section_title": "The Unlikely Best Friends", "related_section_number": "2.1"},
    88: {"related_section_title": "A Friend’s Prayer", "related_section_number": "2.2"},
    89: {"related_section_title": "The Chair", "related_section_number": "2.3"},
    117: {"related_section_title": "Neem Baba", "related_section_number": "3.1"},
    118: {"related_section_title": "What a Bird Thought / Spices that Heal Us", "related_section_number": "3.2 / 3.3"},
    144: {"related_section_title": "Change of Heart", "related_section_number": "4.1"},
    145: {"related_section_title": "Yoga—A Way of Life", "related_section_number": "4.3"},
    178: {"related_section_title": "Hamara Bharat—Incredible India!", "related_section_number": "5.1"},
    179: {"related_section_title": "The Kites", "related_section_number": "5.2"},
    180: {"related_section_title": "Ila Sachani: Embroidering Dreams with her Feet", "related_section_number": "5.3"},
}

UNIT_ACTIVITY_PAGES = {
    90: {"title": "Save Water", "unit_number": "2", "unit_title": "Friendship"},
    146: {"title": "Who Am I?", "unit_number": "4", "unit_title": "Sports and Wellness"},
}

RAMA_GRAPHIC_VISIBLE_TEXT = {
    38: """[Visible text from graphic panels]
Somebody is trying to get in.
I...! I think he has already got in... The noise has stopped.
Lie down... don't look.
It's a thief... he must have somehow got past Rama.
Rama was the village kotwal.
What should we do?
I'll tell you. Listen...
Meanwhile — Voices! They're awake. I'll have to wait till they fall asleep.
I wonder where they keep their money.""",
    39: """[Visible text from graphic panels]
They're saying something. Perhaps they're talking about their money. I'd better listen closely.
What should we name our child?
If he is a boy we'll call him Rama.
Rama? Yes... that's a good name.
When he's in the house, I'll call out softly to him. Rama! Rama!
But what if he's in the yard?
Then I'll call out a little louder. Rama, Rama!
I wish they would stop this silly game and talk about their money instead.
Or fall asleep at least!
But my dear, what if the boy is not in the house, or in the yard, but in the street?
Oh, then I'll call out very loudly...""",
    40: """[Visible text from graphic panels]
Rama! Rama!
Rama!
Rama, the village kotwal, ran to the house from which he heard his name being called.
Oh, oh! Some thief has dug his way into this house.
Ah, at last they've stopped their chatter!""",
    41: """[Visible text from graphic panels]
Now soon they'll go to sleep and... eh!
You're under arrest.
It's Rama! He heard us!
The plan worked! We're saved.
The man and his wife by their cleverness had saved themselves from being robbed.""",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\u0008", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_title(title: str | None) -> str | None:
    if title is None:
        return None
    return TITLE_NORMALIZE.get(title, title)


def run_tesseract_for_page(pdf_path: Path, page_number: int, dpi: int = 300, psm: str = "11") -> str:
    """Render a PDF page and run Tesseract as an audit layer.

    This uses byte-mode subprocess output so it is safe on Windows code pages.
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name

        pix.save(img_path)

        try:
            proc = subprocess.run(
                ["tesseract", img_path, "stdout", "-l", "eng", "--psm", str(psm)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=90,
            )
            stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
            stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return f"[OCR failed for PDF page {page_number}: {stderr_text[:500]}]"
            return clean_text(stdout_text)
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass
    except Exception as exc:
        return f"[OCR audit unavailable for PDF page {page_number}: {exc}]"


def page_unit_from_chapters(chapters: list[dict[str, Any]], page_number: int) -> tuple[str | None, str | None]:
    for chapter in chapters:
        if chapter.get("start_page") <= page_number <= chapter.get("end_page"):
            unit_number = chapter.get("unit_number") or str(chapter.get("chapter_number")).replace("Unit ", "")
            unit_title = chapter.get("unit_title") or chapter.get("chapter_title")
            return str(unit_number), normalize_title(unit_title)
    return None, None


def classify_normal_page(page: dict[str, Any]) -> str:
    t = (page.get("text") or "").lower()
    if "let us do these activities before we read" in t:
        return "lesson_opener"
    if any(marker in t for marker in [
        "let us discuss",
        "let us think and reflect",
        "let us learn",
        "let us listen",
        "let us speak",
        "let us write",
        "let us explore",
    ]):
        return "lesson_activity"
    return "lesson_body"


def normalize_titles_in_place(extraction: dict[str, Any]) -> None:
    for chapter in extraction.get("chapters", []):
        chapter["chapter_title"] = normalize_title(chapter.get("chapter_title"))
        chapter["unit_title"] = normalize_title(chapter.get("unit_title"))
        for lesson in chapter.get("lessons", []):
            lesson["section_title"] = normalize_title(lesson.get("section_title"))

    for section in extraction.get("section_index", []):
        section["section_title"] = normalize_title(section.get("section_title"))
        section["unit_title"] = normalize_title(section.get("unit_title"))
        section["chapter_title"] = normalize_title(section.get("chapter_title"))


def build_corrected_pages(
    extraction: dict[str, Any],
    pdf_path: Path,
    skip_tesseract_audit: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chapters = extraction["chapters"]

    ocr_cache = {}
    if not skip_tesseract_audit:
        for page_number in RAMA_GRAPHIC_VISIBLE_TEXT:
            ocr_cache[page_number] = run_tesseract_for_page(pdf_path, page_number, dpi=300, psm="11")

    new_pages: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    unit_level_pages: list[dict[str, Any]] = []

    for original_page in extraction.get("page_extractions", []):
        pnum = int(original_page["page_number"])
        page = dict(original_page)

        page["section_title"] = normalize_title(page.get("section_title"))
        page["unit_title"] = normalize_title(page.get("unit_title"))
        page["text"] = clean_text(page.get("text", ""))
        page["selectable_text"] = page.get("selectable_text") or page["text"]
        page["ocr_text"] = page.get("ocr_text", "")
        page["text_sources"] = page.get("text_sources") or ["selectable_pdf_text"]
        page["quality_flags"] = list(page.get("quality_flags") or [])
        page["include_in_lesson_text"] = bool(page.get("include_in_lesson_text", True))
        page["linked_section_title"] = None
        page["linked_section_number"] = None
        page["unit_level_title"] = None

        if pnum in RAMA_GRAPHIC_VISIBLE_TEXT:
            page["content_type"] = "graphic_story"
            page["ocr_text"] = ocr_cache.get(pnum, "")
            page["text_sources"] = ["selectable_pdf_text", "page_ocr_audit", "visual_correction"]
            page["quality_flags"] = ["image_text_added", "comic_panel_text_added", "ocr_required"]
            page["text"] = clean_text(page["selectable_text"] + "\n\n" + RAMA_GRAPHIC_VISIBLE_TEXT[pnum])
            page["assignment_status"] = "assigned_to_lesson"

        elif pnum in TRANSCRIPT_PAGE_LINKS:
            link = TRANSCRIPT_PAGE_LINKS[pnum]
            unit_number, unit_title = page_unit_from_chapters(chapters, pnum)

            page["content_type"] = "transcript"
            page["section_number"] = None
            page["section_title"] = None
            page["linked_section_title"] = normalize_title(link["related_section_title"])
            page["linked_section_number"] = link["related_section_number"]
            page["include_in_lesson_text"] = False
            page["include_in_embeddings"] = True
            page["assignment_status"] = "linked_transcript_not_lesson_body"
            page["quality_flags"] = ["transcript_page_remapped"]

            if unit_number:
                page["unit_number"] = unit_number
                page["unit_title"] = unit_title

            transcripts.append({
                "page_number": pnum,
                "printed_page_number": page.get("printed_page_number"),
                "unit_number": page.get("unit_number"),
                "unit_title": page.get("unit_title"),
                "content_type": "transcript",
                "linked_section_number": page["linked_section_number"],
                "linked_section_title": page["linked_section_title"],
                "text": page["text"],
                "text_sources": page["text_sources"],
                "quality_flags": page["quality_flags"],
            })

        elif pnum in UNIT_ACTIVITY_PAGES:
            info = UNIT_ACTIVITY_PAGES[pnum]

            page["content_type"] = "unit_activity"
            page["section_number"] = None
            page["section_title"] = None
            page["unit_number"] = info["unit_number"]
            page["unit_title"] = info["unit_title"]
            page["unit_level_title"] = info["title"]
            page["include_in_lesson_text"] = False
            page["include_in_embeddings"] = True
            page["assignment_status"] = "unit_level_page_not_lesson_body"
            page["quality_flags"] = ["unit_level_page_remapped"]

            unit_level_pages.append({
                "page_number": pnum,
                "printed_page_number": page.get("printed_page_number"),
                "unit_number": page.get("unit_number"),
                "unit_title": page.get("unit_title"),
                "content_type": "unit_activity",
                "unit_level_title": info["title"],
                "text": page["text"],
                "text_sources": page["text_sources"],
                "quality_flags": page["quality_flags"],
            })

        else:
            page["content_type"] = classify_normal_page(page)
            page["assignment_status"] = "assigned_to_lesson"

        new_pages.append(page)

    return new_pages, transcripts, unit_level_pages


def rebuild_lessons_and_index(extraction: dict[str, Any], new_pages: list[dict[str, Any]]) -> list[str]:
    corrections: list[str] = []
    chapters = extraction["chapters"]

    page_by_section: dict[str, list[dict[str, Any]]] = {}
    excluded_by_section: dict[str, list[dict[str, Any]]] = {}

    for page in new_pages:
        if page.get("include_in_lesson_text") and page.get("section_title"):
            key = normalize_title(page.get("section_title"))
            page_by_section.setdefault(key, []).append(page)
        elif not page.get("include_in_lesson_text") and page.get("linked_section_title"):
            key = normalize_title(page.get("linked_section_title"))
            excluded_by_section.setdefault(key, []).append(page)

    lesson_by_section_number: dict[str, dict[str, Any]] = {}
    lesson_by_title: dict[str, dict[str, Any]] = {}

    for chapter in chapters:
        for lesson in chapter.get("lessons", []):
            title = normalize_title(lesson.get("section_title"))
            section_number = str(lesson.get("section_number"))

            old_page_count = lesson.get("page_count")
            old_start = lesson.get("start_page")
            old_end = lesson.get("end_page")
            old_printed_start = lesson.get("printed_start_page")
            old_printed_end = lesson.get("printed_end_page")

            lesson["physical_start_page"] = old_start
            lesson["physical_end_page"] = old_end
            lesson["physical_printed_start_page"] = old_printed_start
            lesson["physical_printed_end_page"] = old_printed_end
            if old_start is not None and old_end is not None:
                lesson["physical_page_count"] = old_end - old_start + 1

            pages = sorted(page_by_section.get(title, []), key=lambda x: x["page_number"])
            if pages:
                lesson["start_page"] = pages[0]["page_number"]
                lesson["end_page"] = pages[-1]["page_number"]
                lesson["printed_start_page"] = pages[0].get("printed_page_number")
                lesson["printed_end_page"] = pages[-1].get("printed_page_number")
                lesson["page_numbers"] = [p["page_number"] for p in pages]
                lesson["printed_page_numbers"] = [p.get("printed_page_number") for p in pages]
                lesson["page_count"] = len(pages)
                lesson["lesson_text"] = clean_text("\n\n".join(
                    f"[PDF page {p['page_number']} / printed page {p.get('printed_page_number')}]\n{p.get('text', '')}"
                    for p in pages
                ))
                lesson["text_plain"] = lesson["lesson_text"]

                if old_page_count != lesson["page_count"]:
                    corrections.append(f"Lesson {section_number} {title}: page_count {old_page_count} -> {lesson['page_count']}")

            lesson["excluded_related_pages"] = [
                {
                    "page_number": p["page_number"],
                    "printed_page_number": p.get("printed_page_number"),
                    "content_type": p.get("content_type"),
                    "reason": "stored separately from lesson_text",
                }
                for p in sorted(excluded_by_section.get(title, []), key=lambda x: x["page_number"])
            ]

            lesson_by_section_number[section_number] = lesson
            lesson_by_title[title] = lesson

    for section in extraction.get("section_index", []):
        title = normalize_title(section.get("section_title"))
        section_number = str(section.get("section_number"))
        lesson = lesson_by_section_number.get(section_number) or lesson_by_title.get(title)
        if not lesson:
            continue

        old_tuple = (
            section.get("start_page"),
            section.get("end_page"),
            section.get("printed_start_page"),
            section.get("printed_end_page"),
            section.get("page_count"),
        )

        section["section_title"] = normalize_title(section.get("section_title"))
        section["unit_title"] = normalize_title(section.get("unit_title"))
        section["chapter_title"] = normalize_title(section.get("chapter_title"))

        section["physical_start_page"] = section.get("start_page")
        section["physical_end_page"] = section.get("end_page")
        section["physical_printed_start_page"] = section.get("printed_start_page")
        section["physical_printed_end_page"] = section.get("printed_end_page")
        if section.get("start_page") is not None and section.get("end_page") is not None:
            section["physical_page_count"] = section["end_page"] - section["start_page"] + 1

        section["start_page"] = lesson.get("start_page")
        section["end_page"] = lesson.get("end_page")
        section["printed_start_page"] = lesson.get("printed_start_page")
        section["printed_end_page"] = lesson.get("printed_end_page")
        section["page_count"] = lesson.get("page_count")
        section["indexed_page_count"] = lesson.get("page_count")
        section["indexed_page_numbers"] = lesson.get("page_numbers", [])
        section["indexed_printed_page_numbers"] = lesson.get("printed_page_numbers", [])
        section["text_length_chars"] = len(lesson.get("lesson_text", ""))
        section["excluded_related_pages"] = lesson.get("excluded_related_pages", [])

        new_tuple = (
            section.get("start_page"),
            section.get("end_page"),
            section.get("printed_start_page"),
            section.get("printed_end_page"),
            section.get("page_count"),
        )
        if old_tuple != new_tuple:
            corrections.append(f"section_index {section_number} {title}: {old_tuple} -> {new_tuple}")

    return corrections


def validate(extraction: dict[str, Any], corrections: list[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for page in extraction.get("page_extractions", []):
        if page.get("content_type") == "transcript" and page.get("section_title"):
            errors.append(f"transcript page {page['page_number']} still has section_title={page.get('section_title')}")
        if page.get("content_type") == "transcript" and page.get("include_in_lesson_text"):
            errors.append(f"transcript page {page['page_number']} is still included in lesson_text")
        if page.get("content_type") == "unit_activity" and page.get("section_title"):
            errors.append(f"unit activity page {page['page_number']} still has section_title={page.get('section_title')}")

    rama_pages = [
        p for p in extraction.get("page_extractions", [])
        if p.get("section_title") == "Rama to the Rescue" and p.get("include_in_lesson_text")
    ]
    full_rama_text = "\n".join(p.get("text", "") for p in rama_pages)
    for phrase in ["Somebody is trying to get in", "What should we do", "Rama! Rama", "You're under arrest"]:
        if phrase.lower() not in full_rama_text.lower():
            warnings.append(f"expected Rama graphic phrase not found: {phrase}")

    transcript_pages = {p.get("page_number"): p for p in extraction.get("page_extractions", []) if p.get("content_type") == "transcript"}
    for pnum in TRANSCRIPT_PAGE_LINKS:
        page = transcript_pages.get(pnum)
        if not page:
            errors.append(f"expected transcript page {pnum} was not marked as transcript")
        elif page.get("section_title") is not None:
            errors.append(f"transcript page {pnum} still has section_title={page.get('section_title')}")

    rama_section = next((s for s in extraction.get("section_index", []) if s.get("section_number") == "1.3"), None)
    if not rama_section:
        errors.append("Rama to the Rescue section_index entry was not found")
    else:
        expected = (36, 51, 20, 35, 16)
        actual = (
            rama_section.get("start_page"),
            rama_section.get("end_page"),
            rama_section.get("printed_start_page"),
            rama_section.get("printed_end_page"),
            rama_section.get("page_count"),
        )
        if actual != expected:
            errors.append(f"Rama section_index semantic range mismatch: expected {expected}, got {actual}")

    return errors, warnings


def write_report(path: Path, extraction: dict[str, Any], corrections: list[str], errors: list[str], warnings: list[str]) -> None:
    title_note = "section_index titles normalized: Yoga-A Way of Life -> Yoga—A Way of Life; Hamara Bharat-Incredible India! -> Hamara Bharat—Incredible India!"
    if title_note not in corrections:
        corrections.append(title_note)

    rama_lesson = None
    for chapter in extraction.get("chapters", []):
        for lesson in chapter.get("lessons", []):
            if lesson.get("section_number") == "1.3":
                rama_lesson = lesson
                break

    lines = [
        "English_Poorvi_hybrid_corrected_extraction_v2 validation report",
        "========================================================================",
        "",
        "Corrections applied:",
    ]
    lines.extend(f"- {c}" for c in corrections) if corrections else lines.append("- None")

    lines.extend(["", "Validation errors:"])
    lines.extend([f"- {e}" for e in errors] if errors else ["- None"])

    lines.extend(["", "Validation warnings:"])
    lines.extend([f"- {w}" for w in warnings] if warnings else ["- None"])

    lines.extend(["", "Specific check:"])
    if rama_lesson:
        lines.append(
            f"- Rama to the Rescue semantic/indexed range: PDF pages {rama_lesson.get('start_page')}-{rama_lesson.get('end_page')}, "
            f"printed pages {rama_lesson.get('printed_start_page')}-{rama_lesson.get('printed_end_page')}, "
            f"page_count={rama_lesson.get('page_count')}"
        )
        lines.append(
            f"- Rama to the Rescue physical range preserved separately: PDF pages {rama_lesson.get('physical_start_page')}-{rama_lesson.get('physical_end_page')}, "
            f"printed pages {rama_lesson.get('physical_printed_start_page')}-{rama_lesson.get('physical_printed_end_page')}, "
            f"physical_page_count={rama_lesson.get('physical_page_count')}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip-tesseract-audit", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    LOGGER.info("Starting Step 2 hybrid correction")
    LOGGER.debug(
        "Arguments: pdf=%s input_json=%s output_json=%s report=%s skip_tesseract_audit=%s",
        args.pdf, args.input_json, args.output_json, args.report, args.skip_tesseract_audit,
    )

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    if not args.input_json.exists():
        raise FileNotFoundError(args.input_json)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading Step 1 JSON: %s", args.input_json)
    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    extraction = data["extraction"]
    LOGGER.info(
        "Loaded Step 1 JSON: sections=%s pages=%s",
        len(extraction.get("section_index", [])),
        len(extraction.get("page_extractions", [])),
    )

    normalize_titles_in_place(extraction)
    LOGGER.info("Building corrected page extractions")
    new_pages, transcripts, unit_level_pages = build_corrected_pages(
        extraction,
        args.pdf,
        skip_tesseract_audit=args.skip_tesseract_audit,
    )

    extraction["page_extractions"] = new_pages
    extraction["transcripts"] = transcripts
    extraction["unit_level_pages"] = unit_level_pages

    LOGGER.info(
        "Corrected pages built: pages=%s transcripts=%s unit_level_pages=%s",
        len(new_pages), len(transcripts), len(unit_level_pages),
    )
    corrections = rebuild_lessons_and_index(extraction, new_pages)
    LOGGER.info("Rebuilt lessons and section_index: corrections=%s", corrections)

    extraction["notes"] = (extraction.get("notes") or []) + [
        "Hybrid correction applied: graphic-story panel text was added for Rama to the Rescue pages 38-41 using rendered-page OCR audit and visual correction.",
        "Transcript pages were remapped as content_type=transcript and removed from lesson_text to avoid metadata pollution.",
        "Unit-level activity pages such as Save Water and Who Am I? were separated from lesson_text.",
        "section_index now stores semantic/indexed ranges; physical_* fields preserve original physical ranges.",
        "Fields include selectable_text, text_sources, content_type, include_in_lesson_text, and quality_flags for RAG validation.",
    ]

    extraction["quality_summary"] = {
        "hybrid_correction_version": "targeted_hybrid_v2",
        "graphic_story_pages_corrected": sorted(RAMA_GRAPHIC_VISIBLE_TEXT.keys()),
        "transcript_pages_remapped": sorted(TRANSCRIPT_PAGE_LINKS.keys()),
        "unit_level_pages_remapped": sorted(UNIT_ACTIVITY_PAGES.keys()),
        "section_index_uses_semantic_indexed_ranges": True,
        "physical_ranges_preserved_in_physical_fields": True,
        "safe_for_reindex_after_review": True,
        "remaining_caveat": "OCR/visual text was targeted to known image-heavy story pages. For maximum production quality, run full-page OCR or a vision model over all pages and compare outputs.",
    }

    data["metadata"]["publisher"] = "NCERT"
    data["metadata"]["class_name"] = "Class-6"
    data["metadata"]["grade"] = "Class-6"
    data["metadata"]["copyright_status"] = "copyrighted_ncert_textbook_reprint_2026_27"

    errors, warnings = validate(extraction, corrections)
    LOGGER.info("Step 2 validation completed: errors=%s warnings=%s", len(errors), len(warnings))

    args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report, extraction, corrections, errors, warnings)
    LOGGER.info("Step 2 wrote JSON: %s", args.output_json)
    LOGGER.info("Step 2 wrote report: %s", args.report)

    print(f"Wrote {args.output_json} ({args.output_json.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {args.report}")
    print("Validation errors:", "None" if not errors else len(errors))
    if errors:
        for error in errors:
            print("-", error)
    print("Validation warnings:", "None" if not warnings else len(warnings))
    if warnings:
        for warning in warnings:
            print("-", warning)


if __name__ == "__main__":
    main()
