#!/usr/bin/env python3
"""
Shared helpers for Grade 10 Physics production-safe extraction.

This pipeline is intentionally conservative for Physics/math content:
- raw PDF text is preserved for audit;
- formula/numeric/table-like lines are isolated into a review queue;
- final production text never silently trusts corrupted formula OCR;
- a curated corrections JSON can add exact reviewed formula text back.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF

LOGGER = logging.getLogger("physics_tenth")

DEFAULT_DOCUMENT_ID = "modern-abc-physics-class-10-2022-23"
DEFAULT_DOCUMENT_KEY = "mother-miracle-class-10-physics-modern-abc"
DEFAULT_BOOK_NAME = "modern_abc_physics_grade10"
DEFAULT_BOOK_TITLE = "Modern's abc+ Science Physics for Class-X"
DEFAULT_GRADE = 10
DEFAULT_SUBJECT = "Physics"
DEFAULT_CONTENT_START_PDF_PAGE = 8

QUESTION_BANK_MARKERS = [
    ("ncert_file", ["ncert file", "in-text questions", "textbook exercises", "exemplar problems"]),
    ("practice_questions", ["practice questions", "cbse sample questions", "hots"]),
    ("revision_exercise", ["revision exercise", "objective questions", "very short answer questions", "short answer questions", "long answer questions"]),
    ("solution_file", ["solution file", "answers & solutions", "answers and solutions"]),
    ("competition_file", ["competition file", "ntse", "sat"]),
    ("chapter_practice_test", ["chapter practice test"]),
]

NOISE_PATTERNS = [
    re.compile(r"(?i)^(?:ata|ay|cath|ov|oe|peek|som)$"),
    re.compile(r"(?i)^ow\s+f$"),
    re.compile(r"(?i)^om$"),
    re.compile(r"^MODERN'?S\s+abc\s*\+\s*OF\s+PHYSICS-X$", re.I),
    re.compile(r"^SCIENCE\s+PHYSICS\s+NCERT/CBSE\s+SYLLABUS$", re.I),
    re.compile(r"^For\s+Class-X$", re.I),
    re.compile(r"^Er\.\s*SUNIL\s+BATRA$", re.I),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^\|?\s*\d+\s*$"),
    re.compile(r"^\|?\s*[-–—]?\d+\s*%?\s*$"),
    re.compile(r"^\s*\d+\s*,\s*\d+(?:\s*,\s*\d+)*\s*$"),
]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ODD_TOKEN_RE = re.compile(r"[\]\[{}<>¦￾�]|[A-Za-z]*कि[A-Za-z]*|[A-Za-z]*हि[A-Za-z]*|[A-Za-z]*पि[A-Za-z]*")
BROKEN_SCI_NOT_RE = re.compile(
    r"(?i)(?:x\s*0[!\"'`^]?|x\s*10\^?\d{3,}|10\^?\d{3,}|0\s*°\s*C|07\s*°\s*C|0~\s*C|07\s*C)"
)
# Do not treat stand-alone A/V/C/J as formula signals. That caused false positives
# such as: "A property associated with objects due to which they show".
RELATION_OPERATOR_RE = re.compile(r"(=|≈|≤|≥|≠|<|>|→|∝)")
SCIENTIFIC_NOTATION_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:x|×)\s*10(?:\s*\^?\s*[-+]?\d+|\s*[-−]\s*\d+)?")
COMPACT_FORMULA_RE = re.compile(
    r"(?ix)\b(?:I|V|R|Q|W|P|F|E|v|u|s|t|m|q|f|n|p|rho|λ|mu|theta)\s*(?:/|\*|×|÷|\^)\s*(?:I|V|R|Q|W|P|F|E|v|u|s|t|m|q|f|n|p|\d)\b"
)
UNIT_VALUE_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:μC|uC|mC|C|kV|V|mA|A|Ω|ohm|J|W|N|Hz|m/s|kg|cm|mm|m|s|D)\b"
)
TABLE_SIGNAL_RE = re.compile(r"(?i)(\bTable\b|\bFig\.?\b|\bFigure\b|\bdiagram\b|\bcircuit diagram\b|\bray diagram\b|\|)")

# Final OCR/math artifact signatures that must never be allowed into production lesson text.
# These catch lines that looked like safe prose in Step 2, plus malformed text that can be
# introduced by an OCR/LLM correction if it copied broken notation too literally.
FINAL_ARTIFACT_RE = re.compile(
    r"(?ix)"
    r"("
    # Raw LaTeX/control-sequence leakage, e.g. \A or \frac{y}{v}.
    r"\\(?:frac|[A-Za-z]{1,8})\b|"
    # OCR tokens observed in this PDF that are not textbook words.
    r"\b(?:AID|Alii|Lda|NowP|WALA|Vag)\b|"
    # Bad scientific notation / exponent OCR observed in solar-energy and formula areas.
    r"\b0%\s*(?:J\s*/\s*s|W|J)?\b|"
    r"\b0\s*\*\b|"
    r"\b0\.0%\b|"
    # Raw LaTeX/control-sequence leakage is handled above; normal R_1 style
    # subscript notation is not by itself a blocker because reviewed formulas may use it.
    # Known bad numeric OCR: near point should not become 00 cm.
    r"\bnear\s+point\s+(?:is\s+)?00\s*cm\b|"
    # Common residual formula fragments from lens/formula zones.
    r"\b\d?\s*ee\s*ot\b|"
    r"(?<!\d)\.0(?!\d)"
    r")"
)

# Final gate also treats these phrases as suspect only when a line is already numeric/formula-like.
CONTEXTUAL_FINAL_ARTIFACT_RE = re.compile(r"(?i)\b(?:de|ow\s+f)\b")

# Lines that are textbook navigation/sidebar/index references rather than lesson prose.
# These should be dropped from production text and should NOT block final production readiness.
NON_CONTENT_REFERENCE_RE = re.compile(
    r"(?ix)"
    r"^(?:[+•*\-–—]\s*)?"
    r"(?:"
    r"CBSE\s+Sample\s+Questions|"
    r"NCERT\s+File|"
    r"Practice\s+Questions|"
    r"Practice\s+Exercise|"
    r"Revision\s+Exercise|"
    r"Solved\s+Examples?|"
    r"HOTS|"
    r"Competition\s+File|"
    r"Chapter\s+Practice\s+Test|"
    r"Questions\s+based\s+on|"
    r"Answers?\s*(?:&|and)\s*Solutions?"
    r")\b.*\d{1,3}\s*$"
)

SAFE_GLOBAL_REPLACEMENTS = [
    ("Ω's law", "Ohm's law"),
    ("Ω’s law", "Ohm’s law"),
    ("Ohm’s aw", "Ohm’s law"),
    ("Ohm's aw", "Ohm's law"),
]


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_line(line: str) -> str:
    line = (line or "").replace("\u00a0", " ").replace("\r", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def normalize_title_key(value: Any) -> str:
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


def apply_safe_global_replacements(text: str, extra_replacements: list[dict[str, str]] | None = None) -> str:
    out = text or ""
    for bad, good in SAFE_GLOBAL_REPLACEMENTS:
        out = out.replace(bad, good)
    for item in extra_replacements or []:
        bad = item.get("bad")
        good = item.get("good")
        if bad and good is not None:
            out = out.replace(str(bad), str(good))
    return out


def light_clean_text(text: str, chapter_title: str | None = None) -> tuple[str, int]:
    """Light cleanup only. Does not try to repair formulas."""
    if not text:
        return "", 0
    text = text.replace("\x00", "").replace("\u0008", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    removed = 0
    chapter_key = normalize_title_key(chapter_title)
    for raw in text.split("\n"):
        line = compact_line(raw)
        if not line:
            out.append("")
            continue
        if any(p.match(line) for p in NOISE_PATTERNS):
            removed += 1
            continue
        if chapter_key and normalize_title_key(line) == chapter_key and line.isupper():
            removed += 1
            continue
        out.append(line)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, removed


def load_static_chapters(subsections_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = read_json(subsections_json)
    chapters = spec.get("chapters") or []
    if not chapters:
        raise ValueError(f"No chapters[] found in subsections JSON: {subsections_json}")
    normalized: list[dict[str, Any]] = []
    for idx, chapter in enumerate(chapters, start=1):
        c = copy.deepcopy(chapter)
        c["sequence"] = as_int(c.get("sequence"), idx) or idx
        c["chapter_name"] = str(c.get("chapter_name") or f"Chapter {idx}").strip()
        c["start_pdf_page"] = as_int(c.get("start_pdf_page"), 0) or 0
        c["end_pdf_page"] = as_int(c.get("end_pdf_page"), 0) or 0
        c["book_page"] = as_int(c.get("book_page"), 1) or 1
        c["days"] = c.get("days") or []
        normalized.append(c)
    normalized.sort(key=lambda x: (x["start_pdf_page"], x["sequence"]))
    return spec, normalized


def add_physical_ranges(chapters: list[dict[str, Any]], pdf_page_count: int) -> None:
    for idx, chapter in enumerate(chapters):
        next_start = chapters[idx + 1]["start_pdf_page"] if idx + 1 < len(chapters) else pdf_page_count + 1
        chapter["physical_start_page"] = chapter["start_pdf_page"]
        chapter["physical_end_page"] = min(max(chapter["end_pdf_page"], next_start - 1), pdf_page_count)
        chapter["teaching_start_page"] = chapter["start_pdf_page"]
        chapter["teaching_end_page"] = chapter["end_pdf_page"]


def chapter_for_page(chapters: list[dict[str, Any]], page_number: int) -> dict[str, Any] | None:
    for chapter in chapters:
        if int(chapter["physical_start_page"]) <= page_number <= int(chapter["physical_end_page"]):
            return chapter
    return None


def local_book_page_for_pdf(chapter: dict[str, Any], pdf_page: int) -> int:
    return int(chapter.get("book_page") or 1) + (int(pdf_page) - int(chapter["start_pdf_page"]))


def classify_front_matter(page_number: int, raw_text: str) -> str:
    t = (raw_text or "").lower()
    if page_number == 1:
        return "cover"
    if "preface" in t:
        return "preface"
    if "syllabus" in t:
        return "syllabus"
    if "contents" in t:
        return "toc"
    if "addresses" in t or "published by" in t:
        return "publisher_page"
    return "front_matter"


def classify_non_teaching_page(raw_text: str) -> str:
    t = (raw_text or "").lower()
    for content_type, markers in QUESTION_BANK_MARKERS:
        if any(marker in t for marker in markers):
            return content_type
    return "chapter_non_teaching"


def classify_teaching_page(raw_text: str) -> str:
    t = (raw_text or "").lower()
    if "practice exercise" in t:
        return "teaching_with_practice_exercise"
    if "solved example" in t or "example" in t:
        return "teaching_with_solved_examples"
    if "activity" in t:
        return "teaching_activity"
    if "what you have learnt" in t or "chapter summary" in t or "key terms" in t:
        return "chapter_summary"
    return "chapter_teaching"


def extract_pdf_pages(pdf_path: Path, include_layout_lines: bool = True) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            page_number = index + 1
            text = page.get_text("text") or ""
            layout_lines: list[dict[str, Any]] = []
            if include_layout_lines:
                try:
                    info = page.get_text("dict") or {}
                    line_no = 0
                    for block in info.get("blocks", []):
                        if block.get("type") != 0:
                            continue
                        for line in block.get("lines", []):
                            spans = line.get("spans", [])
                            line_text = "".join(span.get("text", "") for span in spans)
                            if not line_text.strip():
                                continue
                            line_no += 1
                            layout_lines.append({
                                "line_no": line_no,
                                "text": line_text,
                                "bbox": list(line.get("bbox") or block.get("bbox") or []),
                                "block_no": block.get("number"),
                            })
                except Exception as exc:
                    LOGGER.warning("Layout extraction failed for page %s: %s", page_number, exc)
            pages.append({
                "page_number": page_number,
                "raw_text": text,
                "selectable_text": text,
                "layout_lines": layout_lines,
                "text_sources": ["selectable_pdf_text", "pymupdf_layout_lines"] if include_layout_lines else ["selectable_pdf_text"],
            })
    return pages


def build_page_extractions(raw_pages: list[dict[str, Any]], chapters: list[dict[str, Any]], include_question_bank_in_embeddings: bool = False) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], Counter]:
    stats: Counter = Counter()
    page_extractions: list[dict[str, Any]] = []
    for raw_page in raw_pages:
        pnum = int(raw_page["page_number"])
        raw_text = raw_page.get("raw_text") or raw_page.get("selectable_text") or ""
        chapter = chapter_for_page(chapters, pnum)
        page: dict[str, Any] = {
            "page_number": pnum,
            "raw_extracted_text": raw_text,
            "selectable_text": raw_text,
            "layout_lines": raw_page.get("layout_lines") or [],
            "ocr_text": "",
            "text_sources": raw_page.get("text_sources") or ["selectable_pdf_text"],
            "quality_flags": [],
        }
        if chapter is None:
            cleaned, removed = light_clean_text(raw_text)
            page.update({
                "printed_page_number": None,
                "printed_page_label": None,
                "chapter_number": None,
                "chapter_title": None,
                "section_number": None,
                "section_title": None,
                "content_type": classify_front_matter(pnum, raw_text),
                "assignment_status": "front_matter",
                "include_in_chapter_text": False,
                "include_in_lesson_text": False,
                "include_in_embeddings": False,
                "embedding_readiness": "not_indexed_front_matter",
                "text": cleaned,
                "text_plain": cleaned,
            })
            page["quality_flags"].append("front_matter_not_lesson_content")
            stats["front_matter_pages"] += 1
            stats["noise_lines_removed"] += removed
        else:
            chapter_number = str(chapter.get("sequence"))
            chapter_title = str(chapter.get("chapter_name"))
            printed_page = local_book_page_for_pdf(chapter, pnum)
            cleaned, removed = light_clean_text(raw_text, chapter_title=chapter_title)
            in_teaching = int(chapter["teaching_start_page"]) <= pnum <= int(chapter["teaching_end_page"])
            if in_teaching:
                content_type = classify_teaching_page(raw_text)
                include_text = True
                include_embeddings = bool(cleaned)
                embedding_readiness = "raw_text_needs_production_safety_filter"
                assignment_status = "assigned_to_chapter_teaching"
                stats["teaching_pages"] += 1
            else:
                content_type = classify_non_teaching_page(raw_text)
                include_text = False
                include_embeddings = bool(cleaned) if include_question_bank_in_embeddings else False
                embedding_readiness = "ready_for_question_bank_embedding" if include_embeddings else "not_indexed_non_teaching_question_bank"
                assignment_status = "chapter_related_non_teaching_excluded_from_lesson_text"
                stats["non_teaching_chapter_pages"] += 1
            page.update({
                "printed_page_number": printed_page,
                "printed_page_label": f"{chapter_number}/{printed_page}",
                "chapter_number": f"Chapter {chapter_number}",
                "chapter_title": chapter_title,
                "chapter_type": "chapter",
                "section_number": chapter_number,
                "section_title": chapter_title,
                "content_type": content_type,
                "assignment_status": assignment_status,
                "include_in_chapter_text": include_text,
                "include_in_lesson_text": include_text,
                "include_in_embeddings": include_embeddings,
                "embedding_readiness": embedding_readiness,
                "text": cleaned,
                "text_plain": cleaned,
            })
            if not in_teaching:
                page["quality_flags"].append("non_teaching_question_bank_page")
            if not cleaned:
                page["quality_flags"].append("empty_text_after_cleanup")
                stats["empty_pages_after_cleanup"] += 1
            stats["noise_lines_removed"] += removed
        page["text_length_chars"] = len(page.get("text_plain") or "")
        page_extractions.append(page)
    return page_extractions, {int(p["page_number"]): p for p in page_extractions}, stats


def get_pages_in_range(pages_by_number: dict[int, dict[str, Any]], start: int, end: int) -> tuple[list[dict[str, Any]], list[int]]:
    pages: list[dict[str, Any]] = []
    missing: list[int] = []
    for pnum in range(int(start), int(end) + 1):
        page = pages_by_number.get(pnum)
        if page is None:
            missing.append(pnum)
        else:
            pages.append(page)
    return pages, missing


def build_text_from_pages(pages: Iterable[dict[str, Any]], title: str | None = None) -> tuple[str, str, int]:
    raw_text = "\n\n".join((p.get("text") or "").strip() for p in pages if (p.get("text") or "").strip()).strip()
    cleaned, removed = light_clean_text(raw_text, chapter_title=title)
    return raw_text, cleaned, removed


def normalize_line_for_safety(line: str) -> tuple[str, list[str]]:
    """Normalize obvious OCR decoration without changing formula semantics.

    This is intentionally conservative. It removes page/scan decoration and
    malformed figure prefixes, but it does not invent formulas.
    """
    s = compact_line(line)
    actions: list[str] = []

    # Remove OCR garbage prefixes before real words, e.g. "उ. INTRODUCTION" -> "INTRODUCTION".
    cleaned_prefix = re.sub(r"^[\s|¦\]\[{}<>:;,.~!\-–—\u0900-\u097F]+(?=[A-Za-z])", "", s).strip()
    if cleaned_prefix != s:
        s = cleaned_prefix
        actions.append("removed_leading_ocr_prefix")

    # Common damaged figure label pattern. Keep the meaningful sentence after the label.
    # Example: "Fig. . During rubbing electrons from glass" -> "During rubbing electrons from glass".
    fig_stripped = re.sub(r"(?i)^\s*(?:fig(?:ure)?\.?\s*[\d.\-–—]*\s*)+(?=[A-Z][a-z])", "", s).strip()
    if fig_stripped != s and len(fig_stripped.split()) >= 3:
        s = fig_stripped
        actions.append("removed_damaged_figure_prefix")

    s2 = re.sub(r"\s+", " ", s).strip()
    if s2 != s:
        s = s2
        actions.append("collapsed_spaces")
    return s, actions

def line_is_discardable_noise(line: str) -> bool:
    """Drop clear OCR/page-decoration fragments instead of sending them to manual review."""
    s = compact_line(line)
    if not s:
        return False
    if re.fullmatch(r"[\W_]+", s):
        return True
    if DEVANAGARI_RE.search(s) and not re.search(r"[A-Za-z0-9]", s):
        return True
    if re.fullmatch(r"[-–—]?\d{1,3}%?", s):
        return True
    if re.fullmatch(r"\d+\s*,\s*\d+(?:\s*,\s*\d+)*", s):
        return True
    # Very short OCR fragments such as "44s", "l0", "0l", "3S" are not reviewable content.
    if re.fullmatch(r"[A-Za-z]?\d{1,3}[A-Za-z]?", s) and len(s) <= 4:
        return True
    if len(s) <= 5 and (DEVANAGARI_RE.search(s) or ODD_TOKEN_RE.search(s)):
        return True
    return False


def line_is_discardable_non_content(line: str) -> bool:
    """Drop sidebar/navigation/index lines that should not block formula-safe production."""
    s = compact_line(line)
    if not s:
        return False
    if NON_CONTENT_REFERENCE_RE.search(s):
        return True
    # Page-navigation entries like "+ CBSE Sample Questions (Solved) 37" after OCR cleanup.
    if re.search(r"(?i)\b(?:sample questions|practice questions|solved|hots|ncert file|revision exercise)\b", s) and re.search(r"\d{1,3}\s*$", s):
        return True
    # Damaged isolated figure/table labels with no useful caption text.
    if re.fullmatch(r"(?i)(?:fig(?:ure)?\.?|table)\s*[\d.\-–—]*", s):
        return True
    return False

def line_has_artifacts(line: str) -> bool:
    s = compact_line(line)
    if DEVANAGARI_RE.search(s) or ODD_TOKEN_RE.search(s) or BROKEN_SCI_NOT_RE.search(s):
        return True
    if FINAL_ARTIFACT_RE.search(s):
        return True
    # Standalone short tokens like "de" are too broad for prose, but in an equation/numeric
    # fragment they are strong OCR artifacts and must go to review.
    if CONTEXTUAL_FINAL_ARTIFACT_RE.search(s) and (RELATION_OPERATOR_RE.search(s) or len(re.findall(r"\d", s)) >= 1):
        return True
    return False


def line_is_formula_or_numeric_risk(line: str) -> bool:
    s = compact_line(line)
    if not s:
        return False
    digits = len(re.findall(r"\d", s))
    operators = len(re.findall(r"[=+\-*/×÷^~<>]", s))
    if BROKEN_SCI_NOT_RE.search(s):
        return True
    if RELATION_OPERATOR_RE.search(s):
        return True
    if SCIENTIFIC_NOTATION_RE.search(s):
        return True
    if COMPACT_FORMULA_RE.search(s):
        return True
    # Numeric value-only fragments with units are usually formula/table fragments.
    # Longer sentences with units remain safe prose unless they contain equations/operators.
    if UNIT_VALUE_RE.search(s) and len(s.split()) <= 8 and not re.search(r"[.!?]$", s):
        return True
    if digits >= 2 and operators >= 1 and len(s.split()) <= 12:
        return True
    return False


def line_is_table_or_diagram_risk(line: str) -> bool:
    """Identify diagram/table captions.

    These are usually not formula-accuracy blockers. Step 2 now keeps meaningful
    non-formula caption text as safe text and only discards pure labels/noise.
    """
    s = compact_line(line)
    if not s:
        return False
    return bool(TABLE_SIGNAL_RE.search(s))

def review_id(page_number: int, line_no: int) -> str:
    return f"p{int(page_number):03d}_l{int(line_no):03d}"


def make_line_items(page: dict[str, Any], global_replacements: list[dict[str, str]] | None = None) -> tuple[list[dict[str, Any]], Counter]:
    """Classify each extracted line for production-safe Physics text.

    Buckets:
    - safe_text: included in production text.
    - discarded_noise: OCR/page-decoration garbage; excluded and not review-blocking.
    - discarded_non_content: sidebar/index/navigation/caption-only text; excluded and not review-blocking.
    - review_required: formula/numeric/OCR-artifact text that must be curated for formula-safe production.

    The key fix is that non-content and diagram-only lines no longer inflate the
    formula review queue. Only lines with formula/scientific-notation risk or real
    OCR artifacts remain blocking review items.
    """
    stats: Counter = Counter()
    page_number = int(page.get("page_number") or 0)
    raw_text = page.get("raw_extracted_text") or page.get("selectable_text") or page.get("text") or ""
    cleaned, _ = light_clean_text(raw_text, chapter_title=page.get("chapter_title"))
    cleaned = apply_safe_global_replacements(cleaned, global_replacements)
    lines = cleaned.splitlines()
    items: list[dict[str, Any]] = []
    for idx, raw_line in enumerate(lines, start=1):
        original = compact_line(raw_line)
        line, normalization_actions = normalize_line_for_safety(raw_line)
        line = compact_line(line)
        if not line:
            items.append({"line_no": idx, "type": "blank", "text": ""})
            continue

        if line_is_discardable_noise(line):
            items.append({
                "line_no": idx,
                "type": "discarded_noise",
                "text": line,
                "original_text": original,
                "normalization_actions": normalization_actions,
                "reasons": ["discardable_ocr_or_page_decoration"],
            })
            stats["discarded_noise_lines"] += 1
            continue

        if line_is_discardable_non_content(line):
            items.append({
                "line_no": idx,
                "type": "discarded_non_content",
                "text": line,
                "original_text": original,
                "normalization_actions": normalization_actions,
                "reasons": ["sidebar_index_or_navigation_text"],
            })
            stats["discarded_non_content_lines"] += 1
            continue

        unsafe_reasons: list[str] = []
        has_artifacts = line_has_artifacts(line)
        has_formula_risk = line_is_formula_or_numeric_risk(line)
        has_diagram_risk = line_is_table_or_diagram_risk(line)

        if has_artifacts:
            unsafe_reasons.append("ocr_artifact_tokens")
            stats["artifact_lines"] += 1
        if has_formula_risk:
            unsafe_reasons.append("formula_or_numeric_risk")
            stats["formula_review_lines"] += 1

        # Table/diagram lines are only blocking when they also have OCR artifacts or formula risk.
        # Otherwise, meaningful captions/sentences are kept as safe text with a non-blocking flag.
        if has_diagram_risk:
            stats["table_diagram_context_lines"] += 1
            if has_artifacts or has_formula_risk:
                unsafe_reasons.append("table_or_diagram_risk")
                stats["table_diagram_review_lines"] += 1

        if unsafe_reasons:
            rid = review_id(page_number, idx)
            items.append({
                "line_no": idx,
                "type": "review_required",
                "review_id": rid,
                "text": line,
                "original_text": original,
                "normalization_actions": normalization_actions,
                "reasons": unsafe_reasons,
            })
            stats["review_required_lines"] += 1
        else:
            item = {
                "line_no": idx,
                "type": "safe_text",
                "text": line,
                "original_text": original,
                "normalization_actions": normalization_actions,
            }
            if has_diagram_risk:
                item["non_blocking_flags"] = ["diagram_or_table_context_text"]
            items.append(item)
            stats["safe_lines"] += 1
    return items, stats

def build_safe_text_from_line_items(
    line_items: list[dict[str, Any]],
    reviewed_blocks: dict[str, str] | None = None,
    discard_review_ids: set[str] | None = None,
    placeholder_unreviewed: bool = False,
) -> tuple[str, list[dict[str, Any]], int, int, int]:
    """Build production-safe text from line items.

    review_required items are handled in three ways:
    1. If review_id exists in reviewed_blocks, insert the exact reviewed text.
    2. If review_id exists in discard_review_ids, resolve it by omitting the line.
       Use this only for confirmed non-content / OCR garbage / duplicated crop fragments.
    3. Otherwise keep it unresolved so strict production fails.

    Returns: text, unresolved_items, safe_used, reviewed_used, discarded_review_used
    """
    reviewed_blocks = reviewed_blocks or {}
    discard_review_ids = discard_review_ids or set()
    output_lines: list[str] = []
    unresolved: list[dict[str, Any]] = []
    reviewed_used = 0
    discarded_review_used = 0
    safe_used = 0
    last_blank = False
    for item in line_items:
        typ = item.get("type")
        if typ == "blank":
            if not last_blank:
                output_lines.append("")
                last_blank = True
            continue
        last_blank = False
        if typ == "safe_text":
            output_lines.append(item.get("text") or "")
            safe_used += 1
            continue
        if typ == "review_required":
            rid = str(item.get("review_id"))
            reviewed = reviewed_blocks.get(rid)
            if reviewed is not None and str(reviewed).strip():
                output_lines.append(str(reviewed).strip())
                reviewed_used += 1
            elif rid in discard_review_ids:
                # Explicitly reviewed as non-content or duplicate visual fragment.
                discarded_review_used += 1
            else:
                unresolved.append(item)
                if placeholder_unreviewed:
                    output_lines.append(f"[FORMULA_OR_DIAGRAM_TEXT_REVIEW_REQUIRED:{rid}]")
    text = "\n".join(output_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, unresolved, safe_used, reviewed_used, discarded_review_used


def summarize_review_queue(pages: list[dict[str, Any]]) -> dict[str, Any]:
    queue: list[dict[str, Any]] = []
    for page in pages:
        if not page.get("include_in_lesson_text"):
            continue
        for item in page.get("line_items") or []:
            if item.get("type") == "review_required" and not item.get("resolved"):
                queue.append({
                    "review_id": item.get("review_id"),
                    "page_number": page.get("page_number"),
                    "printed_page_number": page.get("printed_page_number"),
                    "chapter_title": page.get("chapter_title"),
                    "line_no": item.get("line_no"),
                    "reasons": item.get("reasons") or [],
                    "raw_text": item.get("text") or "",
                    "reviewed_text": ""
                })
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "instructions": "Copy exact reviewed text into reviewed_blocks[review_id], or add confirmed non-content duplicate/garbage review IDs to discard_review_ids, then rerun. Production-ready exact Physics text requires unresolved_review_items = 0.",
        "unresolved_review_items": len(queue),
        "review_items": queue,
    }


def validate_day_ranges_for_chapter(chapter: dict[str, Any], days: list[dict[str, Any]], pdf_page_count: int) -> list[str]:
    reasons: list[str] = []
    parent_start = int(chapter["teaching_start_page"])
    parent_end = int(chapter["teaching_end_page"])
    seen_pages: set[int] = set()
    if not days:
        return ["missing_days_array"]
    for day in sorted(days, key=lambda d: as_int(d.get("day"), 0) or 0):
        day_number = as_int(day.get("day"), 0) or 0
        start_pdf = as_int(day.get("start_pdf_page"))
        end_pdf = as_int(day.get("end_pdf_page"))
        if start_pdf is None or end_pdf is None:
            reasons.append(f"day_{day_number}_missing_pdf_range")
            continue
        if start_pdf > end_pdf:
            reasons.append(f"day_{day_number}_invalid_pdf_range_{start_pdf}_{end_pdf}")
            continue
        if start_pdf < 1 or end_pdf > pdf_page_count:
            reasons.append(f"day_{day_number}_outside_pdf_{start_pdf}_{end_pdf}_pdf_count_{pdf_page_count}")
            continue
        if start_pdf < parent_start or end_pdf > parent_end:
            reasons.append(f"day_{day_number}_outside_parent_teaching_range_{start_pdf}_{end_pdf}_parent_{parent_start}_{parent_end}")
        overlap = sorted(set(range(start_pdf, end_pdf + 1)) & seen_pages)
        if overlap:
            reasons.append(f"day_{day_number}_overlaps_previous_days_{overlap}")
        seen_pages.update(range(start_pdf, end_pdf + 1))
    expected = set(range(parent_start, parent_end + 1))
    missing_from_days = sorted(expected - seen_pages)
    if missing_from_days:
        reasons.append(f"days_do_not_cover_parent_teaching_pages_{missing_from_days}")
    return reasons


def build_subsections_for_chapter(chapter: dict[str, Any], pages_by_number: dict[int, dict[str, Any]], stats: Counter) -> list[dict[str, Any]]:
    subsections: list[dict[str, Any]] = []
    section_number = str(chapter.get("sequence"))
    section_title = str(chapter.get("chapter_name"))
    for day in sorted(chapter.get("days") or [], key=lambda d: as_int(d.get("day"), 0) or 0):
        day_number = as_int(day.get("day"), len(subsections) + 1) or (len(subsections) + 1)
        start_pdf = as_int(day.get("start_pdf_page"))
        end_pdf = as_int(day.get("end_pdf_page"))
        if start_pdf is None or end_pdf is None or start_pdf > end_pdf:
            stats["subsections_skipped_invalid_pdf_range"] += 1
            continue
        start_book = as_int(day.get("start_book_page"), local_book_page_for_pdf(chapter, start_pdf))
        end_book = as_int(day.get("end_book_page"), local_book_page_for_pdf(chapter, end_pdf))
        pages, missing_pages = get_pages_in_range(pages_by_number, start_pdf, end_pdf)
        safe_pages = [p for p in pages if p.get("include_in_lesson_text") and normalize_title_key(p.get("section_title")) == normalize_title_key(section_title)]
        filtered_pages = [p.get("page_number") for p in pages if p not in safe_pages]
        raw_text, cleaned_text, removed = build_text_from_pages(safe_pages, title=section_title)
        stats["noise_lines_removed"] += removed
        unresolved_count = sum(as_int(p.get("unresolved_review_items"), 0) or 0 for p in safe_pages)
        reviewed_count = sum(as_int(p.get("reviewed_items_applied"), 0) or 0 for p in safe_pages)
        quality_flags: list[str] = []
        notes: list[str] = []
        if unresolved_count:
            quality_flags.append("contains_unresolved_formula_or_diagram_review_items")
            stats["subsections_with_unresolved_review_items"] += 1
        if reviewed_count:
            quality_flags.append("contains_curated_formula_corrections")
        if missing_pages:
            quality_flags.append("missing_pages_in_range")
            notes.append(f"Missing PDF pages in range: {missing_pages}")
            stats["subsections_with_missing_pages"] += 1
        if filtered_pages:
            quality_flags.append("some_pages_filtered_out")
            notes.append(f"Filtered out non-teaching/cross-section pages: {filtered_pages}")
            stats["subsections_with_filtered_pages"] += 1
        if not cleaned_text:
            quality_flags.append("empty_subsection_text")
            stats["subsections_with_empty_text"] += 1
        subsection_code = str(day.get("subsection_code") or f"{section_number}{chr(64 + day_number)}")
        subsection_title = f"Day {day_number} ({subsection_code})"
        page_numbers = [int(p["page_number"]) for p in safe_pages]
        printed_page_numbers = [as_int(p.get("printed_page_number")) for p in safe_pages if as_int(p.get("printed_page_number")) is not None]
        subsections.append({
            "section_number": section_number,
            "section_title": section_title,
            "chapter_number": f"Chapter {section_number}",
            "chapter_title": section_title,
            "chapter_type": "chapter",
            "subsection_number": f"{section_number}.{day_number}",
            "subsection_title": subsection_title,
            "subsection_code": subsection_code,
            "anchor_marker": subsection_code,
            "anchor_pdf_page": start_pdf,
            "anchor_printed_page": start_book,
            "anchor_detection_method": "practice_exercise_static_json",
            "included_exercises_or_activities": [subsection_code],
            "includes": [subsection_title, subsection_code],
            "day": day_number,
            "start_page": start_pdf,
            "end_page": end_pdf,
            "start_pdf_page": start_pdf,
            "end_pdf_page": end_pdf,
            "printed_start_page": start_book,
            "printed_end_page": end_book,
            "start_printed_page": start_book,
            "end_printed_page": end_book,
            "pdf_pages": {"start": start_pdf, "end": end_pdf},
            "printed_pages": {"start": start_book, "end": end_book},
            "page_count": max(0, end_pdf - start_pdf + 1),
            "subsection_text": raw_text,
            "subsection_text_plain": cleaned_text,
            "text_plain": cleaned_text,
            "production_subsection_text": cleaned_text,
            "production_indexed_page_numbers": page_numbers,
            "production_printed_page_numbers": printed_page_numbers,
            "production_page_count": len(page_numbers),
            "physical_start_page": start_pdf,
            "physical_end_page": end_pdf,
            "page_numbers": page_numbers,
            "printed_page_numbers": printed_page_numbers,
            "text_sources": sorted({src for p in safe_pages for src in (p.get("text_sources") or [])}),
            "quality_flags": quality_flags,
            "include_in_embeddings": bool(cleaned_text),
            "embedding_readiness": "ready_for_safe_text_embedding" if cleaned_text and not unresolved_count else "review_required_before_formula_safe_embedding",
            "text_length_chars": len(cleaned_text),
            "unresolved_review_items": unresolved_count,
            "reviewed_items_applied": reviewed_count,
            "source_days_json_day": day_number,
            "source_days_json_subsection_code": subsection_code,
            "source_days_json_range_source": day.get("range_source"),
            "filtered_out_page_numbers": filtered_pages,
            "notes": notes,
        })
        stats["subsections_added"] += 1
    return subsections


def build_chapter_and_section_index(chapters_spec: list[dict[str, Any]], pages_by_number: dict[int, dict[str, Any]], pdf_page_count: int, stats: Counter) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    chapters_out: list[dict[str, Any]] = []
    section_index: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for chapter in chapters_spec:
        seq = int(chapter["sequence"])
        title = str(chapter["chapter_name"])
        start_pdf = int(chapter["teaching_start_page"])
        end_pdf = int(chapter["teaching_end_page"])
        physical_start = int(chapter["physical_start_page"])
        physical_end = int(chapter["physical_end_page"])
        book_page = int(chapter.get("book_page") or 1)
        validation_errors.extend(f"Chapter {seq} {title}: {reason}" for reason in validate_day_ranges_for_chapter(chapter, chapter.get("days") or [], pdf_page_count))
        teaching_pages, missing = get_pages_in_range(pages_by_number, start_pdf, end_pdf)
        safe_teaching_pages = [p for p in teaching_pages if p.get("include_in_lesson_text")]
        raw_text, cleaned_text, removed = build_text_from_pages(safe_teaching_pages, title=title)
        stats["noise_lines_removed"] += removed
        if missing:
            validation_errors.append(f"Chapter {seq} {title}: missing teaching pages {missing}")
        if not cleaned_text:
            validation_errors.append(f"Chapter {seq} {title}: empty chapter teaching text")
            stats["chapters_with_empty_text"] += 1
        subsections = build_subsections_for_chapter(chapter, pages_by_number, stats)
        unresolved_count = sum(as_int(p.get("unresolved_review_items"), 0) or 0 for p in safe_teaching_pages)
        reviewed_count = sum(as_int(p.get("reviewed_items_applied"), 0) or 0 for p in safe_teaching_pages)
        chapter_quality_flags: list[str] = []
        if unresolved_count:
            chapter_quality_flags.append("contains_unresolved_formula_or_diagram_review_items")
        if reviewed_count:
            chapter_quality_flags.append("contains_curated_formula_corrections")
        if missing:
            chapter_quality_flags.append("missing_pages_in_chapter_range")
        common = {
            "section_number": str(seq),
            "section_title": title,
            "chapter_number": f"Chapter {seq}",
            "chapter_title": title,
            "chapter_type": "chapter",
            "book_page": book_page,
            "book_page_label": chapter.get("book_page_label") or f"{seq}/{book_page}",
            "start_page": start_pdf,
            "end_page": end_pdf,
            "start_pdf_page": start_pdf,
            "end_pdf_page": end_pdf,
            "printed_start_page": book_page,
            "printed_end_page": local_book_page_for_pdf(chapter, end_pdf),
            "physical_start_page": physical_start,
            "physical_end_page": physical_end,
            "physical_printed_start_page": book_page,
            "physical_printed_end_page": local_book_page_for_pdf(chapter, physical_end),
            "page_count": max(0, end_pdf - start_pdf + 1),
            "physical_page_count": max(0, physical_end - physical_start + 1),
            "section_text": raw_text,
            "section_text_plain": cleaned_text,
            "text_plain": cleaned_text,
            "production_section_text": cleaned_text,
            "production_indexed_page_numbers": [int(p["page_number"]) for p in safe_teaching_pages],
            "production_page_count": len(safe_teaching_pages),
            "include_in_embeddings": bool(cleaned_text),
            "embedding_readiness": "ready_for_safe_text_embedding" if cleaned_text and not unresolved_count else "review_required_before_formula_safe_embedding",
            "text_length_chars": len(cleaned_text),
            "unresolved_review_items": unresolved_count,
            "reviewed_items_applied": reviewed_count,
            "quality_flags": chapter_quality_flags,
            "subsections": subsections,
        }
        section = copy.deepcopy(common)
        lesson = copy.deepcopy(common)
        lesson["lesson_number"] = seq
        lesson["lesson_title"] = title
        chapter_out = copy.deepcopy(common)
        chapter_out.update({
            "sequence": seq,
            "chapter_number": f"Chapter {seq}",
            "chapter_title": title,
            "chapter_name": title,
            "chapter_text": raw_text,
            "chapter_text_plain": cleaned_text,
            "production_chapter_text": cleaned_text,
            "lessons": [lesson],
        })
        chapters_out.append(chapter_out)
        section_index.append(section)
    return chapters_out, section_index, validation_errors


def validate_output(data: dict[str, Any], stats: Counter | dict[str, Any], existing_errors: list[str]) -> dict[str, Any]:
    stats = Counter(stats or {})
    errors = list(existing_errors)
    pages = data.get("page_extractions") or []
    sections = data.get("section_index") or []
    chapters = data.get("chapters") or []
    pdf_page_count = int(data.get("pdf_page_count") or 0)
    page_numbers = [int(p.get("page_number") or 0) for p in pages]
    metrics: dict[str, Any] = {
        "page_extractions": len(pages),
        "unique_page_numbers": len(set(page_numbers)),
        "missing_page_numbers": [p for p in range(1, pdf_page_count + 1) if p not in set(page_numbers)],
        "chapters": len(chapters),
        "sections": len(sections),
        "sections_with_subsections": sum(1 for s in sections if s.get("subsections")),
        "total_subsections": sum(len(s.get("subsections") or []) for s in sections),
        "front_matter_pages": stats.get("front_matter_pages", 0),
        "teaching_pages": stats.get("teaching_pages", 0),
        "non_teaching_chapter_pages": stats.get("non_teaching_chapter_pages", 0),
        "unresolved_review_items": stats.get("unresolved_review_items", 0),
        "reviewed_items_applied": stats.get("reviewed_items_applied", 0),
        "subsections_with_unresolved_review_items": stats.get("subsections_with_unresolved_review_items", 0),
        "subsections_with_empty_text": stats.get("subsections_with_empty_text", 0),
        "subsections_with_filtered_pages": stats.get("subsections_with_filtered_pages", 0),
        "subsections_with_missing_pages": stats.get("subsections_with_missing_pages", 0),
    }
    if len(pages) != pdf_page_count:
        errors.append(f"Expected {pdf_page_count} page_extractions, found {len(pages)}")
    if metrics["missing_page_numbers"]:
        errors.append(f"Missing page extraction records: {metrics['missing_page_numbers']}")
    if not chapters:
        errors.append("No chapters generated")
    if not sections:
        errors.append("No section_index generated")
    if metrics["total_subsections"] == 0:
        errors.append("No subsections generated")
    repeated: list[str] = []
    outside_parent: list[str] = []
    empty_subs: list[str] = []
    for section in sections:
        parent_start = int(section.get("start_pdf_page") or section.get("start_page") or 0)
        parent_end = int(section.get("end_pdf_page") or section.get("end_page") or 0)
        seen: set[int] = set()
        for sub in section.get("subsections") or []:
            sub_name = f"{section.get('section_title')} {sub.get('subsection_number')}"
            s = int(sub.get("start_pdf_page") or 0)
            e = int(sub.get("end_pdf_page") or 0)
            if s < parent_start or e > parent_end:
                outside_parent.append(f"{sub_name}: {s}-{e} outside {parent_start}-{parent_end}")
            overlap = sorted(set(range(s, e + 1)) & seen)
            if overlap:
                repeated.append(f"{sub_name}: repeated pages {overlap}")
            seen.update(range(s, e + 1))
            if not (sub.get("text_plain") or "").strip():
                empty_subs.append(sub_name)
    metrics["subsections_outside_parent_range"] = len(outside_parent)
    metrics["subsections_with_repeated_pdf_pages"] = len(repeated)
    metrics["empty_subsection_names"] = empty_subs
    errors.extend(f"Subsection outside parent range: {x}" for x in outside_parent)
    errors.extend(f"Repeated subsection pages: {x}" for x in repeated)
    errors.extend(f"Empty subsection text: {x}" for x in empty_subs)
    return {"status": "passed" if not errors else "failed", "errors": errors, "metrics": metrics}


def build_report(data: dict[str, Any], validation: dict[str, Any], title: str) -> str:
    metrics = validation.get("metrics") or {}
    stats = data.get("extraction", {}).get("statistics", {})
    lines = [
        title,
        "=" * 72,
        f"Generated at: {data.get('extraction', {}).get('generated_at')}",
        f"documentId: {data.get('documentId')}",
        f"document_key: {data.get('document_key')}",
        f"book_title: {data.get('book_title')}",
        f"pdf_page_count: {data.get('pdf_page_count')}",
        f"validation_status: {validation.get('status')}",
        f"text_accuracy_status: {data.get('text_accuracy_status')}",
        f"production_status: {data.get('production_status')}",
        "",
        "Metrics:",
    ]
    wanted = [
        "page_extractions", "unique_page_numbers", "chapters", "sections", "sections_with_subsections", "total_subsections",
        "front_matter_pages", "teaching_pages", "non_teaching_chapter_pages", "safe_lines", "review_required_lines",
        "artifact_lines", "formula_review_lines", "table_diagram_review_lines", "table_diagram_context_lines",
        "discarded_noise_lines", "discarded_non_content_lines", "unresolved_review_items", "reviewed_items_applied",
        "subsections_with_unresolved_review_items", "subsections_with_empty_text", "subsections_outside_parent_range", "subsections_with_repeated_pdf_pages",
    ]
    for key in wanted:
        value = metrics.get(key, stats.get(key))
        if value is not None:
            lines.append(f"- {key}: {value}")
    if data.get("chapters"):
        lines.extend(["", "Chapter ranges:"])
        for chapter in data.get("chapters") or []:
            lines.append(
                f"- {chapter.get('chapter_number')} {chapter.get('chapter_title')}: "
                f"PDF {chapter.get('start_pdf_page')}-{chapter.get('end_pdf_page')}, "
                f"subsections={len(chapter.get('subsections') or [])}, chars={chapter.get('text_length_chars')}, "
                f"unresolved_review_items={chapter.get('unresolved_review_items')}"
            )
            for sub in chapter.get("subsections") or []:
                lines.append(
                    f"  - {sub.get('subsection_title')}: PDF {sub.get('start_pdf_page')}-{sub.get('end_pdf_page')}, "
                    f"chars={sub.get('text_length_chars')}, unresolved_review_items={sub.get('unresolved_review_items')}, "
                    f"flags={sub.get('quality_flags') or []}"
                )
    if validation.get("errors"):
        lines.extend(["", "Errors:"])
        lines.extend(f"- {err}" for err in validation["errors"])
    else:
        lines.extend(["", "No structural validation errors found."])
    if data.get("production_status") != "production_ready":
        lines.extend([
            "",
            "Production note:",
            "- This output is structurally valid but not exact-formula production-ready until unresolved review items are corrected.",
            "- Fill Grade10_Physics_formula_corrections.json reviewed_blocks using the generated review queue and rerun.",
        ])
    return "\n".join(lines) + "\n"
