#!/usr/bin/env python3
"""
Production-safe cleanup/gating pass for Maths_RSAgarwal math-aware JSON.

Purpose:
- Apply one more conservative math/text cleanup layer.
- Keep the JSON structure compatible with earlier extraction output.
- Mark only pages that pass strict QA as include_in_embeddings=true.
- Preserve flagged pages for later Mathpix/vision/manual QA instead of silently embedding bad math.

This does NOT claim to turn scanned textbook OCR into perfect symbolic LaTeX.
It makes the JSON production-safe for RAG ingestion by preventing unreliable pages
from entering the vector index.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(os.environ.get(
    'MATHS_RSAGGARWAL_ROOT',
    Path(__file__).resolve().parents[2],
))
BASE = Path(os.environ.get(
    'MATHS_RSAGGARWAL_OUTPUT_DIR',
    PROJECT_ROOT / 'output' / 'maths_rsagarwal',
))

INPUT_JSON = BASE / 'Maths_RSAgarwal_math_aware_extraction_v3_cleaned.json'
OUTPUT_JSON = BASE / 'Maths_RSAgarwal_math_aware_extraction_v4_production_safe.json'
REPORT_TXT = BASE / 'Maths_RSAgarwal_math_aware_v4_production_validation_report.txt'
QA_CSV = BASE / 'Maths_RSAgarwal_math_aware_v4_pages_requiring_vision_qa.csv'
SCRIPT_COPY = BASE / 'cleanup_maths_rsaggarwal_v4_production_safe.py'



def slugify(value: Any) -> str:
    """Create stable lowercase ids for documentId/document_key."""
    text = str(value or '').strip().lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'-+', '-', text).strip('-')
    return text


def compact_author_slug(value: Any) -> str:
    """R S Aggarwal -> rsaggarwal, while keeping normal names slug-safe."""
    parts = re.findall(r'[A-Za-z0-9]+', str(value or ''))
    if not parts:
        return ''
    if len(parts) >= 2 and all(len(p) == 1 for p in parts[:-1]):
        return ''.join(p.lower() for p in parts)
    return slugify(' '.join(parts))


def normalize_school_slug(value: Any) -> str:
    """Mother Miracle School -> mother-miracle, matching the Poorvi document_key style."""
    slug = slugify(value)
    slug = re.sub(r'(^|-)school($|-)', r'\1', slug).strip('-')
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


def build_document_identity(data: dict[str, Any]) -> tuple[str, str]:
    """Build stable production document ids, with environment overrides."""
    metadata = data.get('metadata') or {}
    extraction = data.get('extraction') or {}

    grade_slug = slugify(metadata.get('grade') or metadata.get('class_name') or 'class-7')
    subject_slug = slugify(extraction.get('subject') or metadata.get('subject') or 'maths')
    publisher_slug = slugify(metadata.get('publisher') or '')
    school_slug = normalize_school_slug(metadata.get('school_name') or 'mother-miracle')
    author_slug = compact_author_slug(extraction.get('author') or '')
    book_slug = slugify(extraction.get('book_title') or Path(str(metadata.get('source_file') or '')).stem)

    # Prefer author slug for this book so we get rsaggarwal rather than only mathematics-for-class-7.
    book_identity_slug = author_slug or book_slug or 'rsaggarwal'

    default_document_id_parts = [subject_slug, book_identity_slug, grade_slug, publisher_slug]
    default_document_id = '-'.join(part for part in default_document_id_parts if part)

    default_document_key_parts = [school_slug, grade_slug, subject_slug, book_identity_slug]
    default_document_key = '-'.join(part for part in default_document_key_parts if part)

    document_id = os.environ.get('MATHS_RSAGGARWAL_DOCUMENT_ID', default_document_id)
    document_key = os.environ.get('MATHS_RSAGGARWAL_DOCUMENT_KEY', default_document_key)
    return document_id, document_key

FAKE_CURRENCY_SYMBOLS = '¥€£®'
MONEY_CHAPTERS = {
    'Percentage', 'Profit and Loss', 'Simple Interest', 'Unitary Method', 'Ratio and Proportion'
}
ANSWER_KEY_TITLES = {'Answers'}

STRICT_BAD_PATTERNS = [
    ('artifact_Ifa', re.compile(r'\bIfa\b')),
    ('artifact_at_plus_b', re.compile(r'\bat\+b\b')),
    ('artifact_b_plus_e', re.compile(r'\bb\+e\b')),
    ('artifact_cent_for_c', re.compile(r'¢')),
    ('artifact_pipe_roman', re.compile(r'(?m)^\s*\|\.\s+')),
    ('artifact_ill_roman_heading', re.compile(r'(?m)^\s*Ill\.\s+')),
    ('artifact_ete', re.compile(r'\bete\.', re.I)),
]

# Symbols that should not remain in production-ready text unless a human/vision pass confirms them.
AMBIGUOUS_SYMBOL_RE = re.compile(r'[¥€£®¢]')

# Lines likely to be formulas/equations where O/I/l cleanup is safer.
MATH_CONTEXT_RE = re.compile(r'[=+\-×x*/÷<>≠√(){}\[\]|]')


def clean_math_context_line(line: str) -> str:
    """Clean OCR confusions only inside math-like lines."""
    if not MATH_CONTEXT_RE.search(line):
        return line

    # Capital O read for zero in isolated formula positions.
    line = re.sub(r'(?<=[=+\-×x*/÷(\[{\s])O(?=[\s=+\-×x*/÷)\]}.,;:])', '0', line)
    line = re.sub(r'(?<=\d)O(?=\d)', '0', line)
    line = re.sub(r'(?<=[=+\-×x*/÷(\[{])\s*O\s*(?=[=+\-×x*/÷)\]}])', '0', line)
    line = re.sub(r'\bO\s*\+\s*', '0 + ', line)
    line = re.sub(r'\+\s*O\b', '+ 0', line)
    line = re.sub(r'=\s*O\b', '= 0', line)

    # I/l read as 1 in numeric expressions only.
    line = re.sub(r'(?<=[(\[\{+\-×x*/÷=\s])I(?=[)\]\}.\s,+\-×x*/÷=])', '1', line)
    line = re.sub(r'(?<=[(\[\{+\-×x*/÷=\s])l(?=[)\]\}.\s,+\-×x*/÷=])', '1', line)
    line = re.sub(r'(?<=\d)l(?=\d)', '1', line)
    line = re.sub(r'(?<=\d)I(?=\d)', '1', line)

    return line


def clean_text(text: str, *, chapter_title: str | None = None) -> tuple[str, list[str]]:
    """Conservative cleanup rules for common OCR artifacts."""
    if not isinstance(text, str):
        return text, []

    original = text
    fixes: list[str] = []

    replacements = [
        (r'\bIfa\b', 'If a', 'Ifa_to_If_a'),
        (r'\bLIfa\b', 'If a', 'LIfa_to_If_a'),
        (r'\bat\+b\b', 'a+b', 'at_plus_b_to_a_plus_b'),
        (r'\bb\+e\b', 'b+c', 'b_plus_e_to_b_plus_c'),
        (r'\bbxe\b', 'bxc', 'bxe_to_bxc'),
        (r'\bb\s*x\s*e\b', 'b x c', 'b_x_e_to_b_x_c'),
        (r'\ba\+\(b\+e\)', 'a+(b+c)', 'a_plus_b_plus_e_to_a_plus_b_plus_c'),
        (r'\bete\.', 'etc.', 'ete_to_etc'),
        (r'\bEte\.', 'Etc.', 'Ete_to_Etc'),
        (r'\bseam-4', 'Thus, ..., -4', 'seam_to_thus_sequence'),
        (r'\|\.\s*Closure', 'I. Closure', 'pipe_to_I_Closure'),
        (r'(?m)^\s*\|\.\s+', 'I. ', 'pipe_roman_to_I'),
        (r'(?m)^\s*Ill\.\s+', 'III. ', 'Ill_to_III_heading'),
        (r'(?m)^\s*Il\.\s+', 'II. ', 'Il_to_II_heading'),
        (r'(?m)^\s*ll\.\s+', 'II. ', 'll_to_II_heading'),
        (r'\bIll\.\s+Associative', 'III. Associative', 'Ill_Associative_to_III'),
        (r'\bIll\.\s+Distributive', 'III. Distributive', 'Ill_Distributive_to_III'),
        (r'\bIll instead of III\b', 'III', 'Ill_phrase_to_III'),
        (r'\bili\)', 'iii)', 'ili_to_iii'),
        (r'\biti\)', 'iii)', 'iti_to_iii'),
        (r'\bti\)', 'ii)', 'ti_to_ii'),
        (r'\¢', 'c', 'cent_to_c'),
        (r'\bFor any integers 4, b, c\b', 'For any integers a, b, c', '4_to_a_in_formula_intro'),
        (r'\bFor any integers a, b, c\.\b', 'For any integers a, b, c, we have:', 'formula_intro_punctuation'),
        (r'\bIf a, b, care\b', 'If a, b, c are', 'care_to_c_are'),
        (r'\bIfa, b, c are\b', 'If a, b, c are', 'Ifa_b_c_to_If_a_b_c'),
        (r'\ba\+b=b\+a\b', 'a + b = b + a', 'space_commutative_addition'),
        (r'\bat\+b=b\+a\b', 'a + b = b + a', 'at_commutative_addition'),
        (r'\(axb\)xc=ax\(bxc\)', '(a x b) x c = a x (b x c)', 'space_associative_multiplication'),
        (r'\(a\+b\)\+c=a\+\(b\+c\)', '(a + b) + c = a + (b + c)', 'space_associative_addition'),
        (r'\ba-b=a\+\(-b\)', 'a - b = a + (-b)', 'space_subtraction_definition'),
        (r'\ba~-b', 'a - b', 'tilde_minus_cleanup'),
        (r'~', '-', 'tilde_to_minus'),
        (r'#', '≠', 'hash_to_not_equal'),
    ]

    for pattern, repl, label in replacements:
        text2, n = re.subn(pattern, repl, text)
        if n:
            fixes.append(f'{label}:{n}')
            text = text2

    # Money symbol cleanup only where context makes rupee likely.
    if chapter_title in MONEY_CHAPTERS or re.search(r'\b(profit|loss|interest|amount|principal|rate|per annum|cost price|selling price|rupees?)\b', text, re.I):
        text2, n = re.subn(r'(?<![A-Za-z])[' + re.escape(FAKE_CURRENCY_SYMBOLS) + r']\s*(?=\d)', '₹', text)
        if n:
            fixes.append(f'fake_currency_to_rupee_before_digits:{n}')
            text = text2
        text2, n = re.subn(r'\b(Rs|rupees|amount|principal|profit|loss|interest)(\s+)[' + re.escape(FAKE_CURRENCY_SYMBOLS) + r']', r'\1\2₹', text, flags=re.I)
        if n:
            fixes.append(f'fake_currency_to_rupee_after_money_word:{n}')
            text = text2

    # Fix common OCR of + as 4 only in equality tails where pattern is obvious.
    text2, n = re.subn(r'=\s*4(?=\d\b)', '= +', text)
    if n:
        fixes.append(f'equal_4_digits_to_equal_plus_digits:{n}')
        text = text2

    # Clean O/I/l inside math-like lines only.
    cleaned_lines = []
    changed_lines = 0
    for line in text.splitlines():
        new_line = clean_math_context_line(line)
        if new_line != line:
            changed_lines += 1
        cleaned_lines.append(new_line)
    if changed_lines:
        fixes.append(f'math_context_O_I_l_cleanup_lines:{changed_lines}')
    text = '\n'.join(cleaned_lines)

    # Whitespace cleanup but preserve paragraph breaks.
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if text != original and not fixes:
        fixes.append('generic_text_changed')
    return text, fixes


def clean_any(value: Any, chapter_title: str | None = None) -> tuple[Any, list[str]]:
    fixes: list[str] = []
    if isinstance(value, str):
        return clean_text(value, chapter_title=chapter_title)
    if isinstance(value, list):
        out = []
        for item in value:
            cleaned, f = clean_any(item, chapter_title=chapter_title)
            out.append(cleaned)
            fixes.extend(f)
        return out, fixes
    if isinstance(value, dict):
        out = {}
        local_chapter = value.get('chapter_title') or chapter_title
        for k, v in value.items():
            if k in {'selectable_text'}:
                # Keep raw selectable text unchanged for audit; it may contain original corruption.
                out[k] = v
                continue
            cleaned, f = clean_any(v, chapter_title=local_chapter)
            out[k] = cleaned
            if k in {'text', 'text_plain', 'ocr_text', 'lesson_text', 'math_lines', 'extracted_blocks'}:
                fixes.extend(f)
        return out, fixes
    return value, fixes


def page_failure_reasons(page: dict[str, Any]) -> list[str]:
    text = page.get('text') or ''
    chapter_title = page.get('chapter_title') or ''
    reasons: list[str] = []

    # Always exclude answer key/dense appendix from production embeddings unless later vision/manual approved.
    if page.get('chapter_type') == 'appendix' or chapter_title in ANSWER_KEY_TITLES or (page.get('page_number') or 0) >= 308:
        reasons.append('answer_key_or_appendix_requires_manual_or_vision_qa')

    # Existing flags from previous pass.
    for flag in page.get('quality_flags') or []:
        if any(key in str(flag) for key in [
            'dense_math_layout_requires_vision_qa',
            'answer_key_dense_requires_manual_or_vision_qa',
            'remaining_ambiguous_symbol_requires_qa',
            'bad_math_ocr_detected',
        ]):
            reasons.append(str(flag))

    # Remaining high-risk artifacts after cleanup.
    remaining_checks = {
        'remaining_Ifa': r'\bIfa\b',
        'remaining_at_plus_b': r'\bat\+b\b',
        'remaining_b_plus_e': r'\bb\+e\b',
        'remaining_pipe_roman': r'(?m)^\s*\|\.\s+',
        'remaining_Ill_heading': r'(?m)^\s*Ill\.\s+',
        'remaining_ete': r'\bete\.',
        'remaining_cent_symbol': r'¢',
    }
    for name, pat in remaining_checks.items():
        if re.search(pat, text, re.I if name == 'remaining_ete' else 0):
            reasons.append(name)

    # Ambiguous symbols remain: do not assume they are rupee; require vision QA.
    if AMBIGUOUS_SYMBOL_RE.search(text):
        syms = ''.join(sorted(set(AMBIGUOUS_SYMBOL_RE.findall(text))))
        reasons.append(f'remaining_ambiguous_symbol_requires_vision_qa:{syms}')

    # Dense formula/fraction-looking pages are risky for exact math. Do not exclude all formulas;
    # only pages with many flattened fraction symbols/numerators or very high equation density.
    equation_markers = len(re.findall(r'[=√/]|\bfrac\b|\d+\s+\d+\s+\d+', text))
    short_math_lines = sum(1 for line in text.splitlines() if MATH_CONTEXT_RE.search(line) and len(line.strip()) < 80)
    if equation_markers >= 45 or short_math_lines >= 45:
        reasons.append('dense_formula_layout_requires_vision_qa')

    # Very low OCR confidence proxies. Count characters that are neither ASCII nor a small allowlist.
    allowed_non_ascii = set('₹≠°²³√×÷—–’“”')
    weird_chars = [ch for ch in text if ord(ch) > 127 and ch not in allowed_non_ascii]
    weird_ratio = len(weird_chars) / max(len(text), 1)
    if weird_ratio > 0.003:
        reasons.append(f'high_non_ascii_artifact_ratio:{weird_ratio:.4f}')


    # De-duplicate preserving order.
    seen = set()
    unique = []
    for r in reasons:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique



# ----------------------------
# Chapter subsection support
# ----------------------------
# Runtime-only subsection detection.
#
# This intentionally does NOT use a book-specific hardcoded subsection-plan
# dictionary. It detects Exercise/Activity headings from the cleaned OCR text in
# page_extractions, then converts those markers into page-level subsection ranges.
#
# Rule used for page ranges:
#   - Subsection 1 starts at the chapter start page and ends on the page where
#     the first Exercise/Activity anchor appears.
#   - Subsection 2 starts on the page after the previous anchor page and ends on
#     the next anchor page, and so on.
#   - The last subsection ends at the chapter end page.
#   - If a trailing Objective/immediate exercise is detected, it is grouped with
#     the previous subsection instead of creating a new subsection.
#
# This matches the desired Chapter 1 behavior:
#   1.1 Exercise 1A: PDF 8-11
#   1.2 Exercise 1B: PDF 12-16
#   1.3 Exercise 1C: PDF 17-22, includes Exercise 1C + Exercise 1D

EXERCISE_TOKEN_RE = re.compile(
    r'\bEXERCISE\s+([0-9A-Za-z¢©€£®]{1,8})\b',
    re.IGNORECASE,
)
ACTIVITY_TOKEN_RE = re.compile(
    r'\bACTIVITY\s*[-:]?\s*([0-9]{1,2})\b',
    re.IGNORECASE,
)

# Headings like "OBJECTIVE QUESTIONS" / "Mark (/) against..." usually introduce
# end-of-chapter immediate exercises. These should normally be grouped with the
# preceding concept subsection.
OBJECTIVE_CONTEXT_RE = re.compile(
    r'\b(OBJECTIVE\s+QUESTIONS?|MULTIPLE\s+CHOICE|MARK\s*\(?/?\)?\s+AGAINST|'
    r'MARK\s+THE\s+CORRECT|CHOOSE\s+THE\s+CORRECT)\b',
    re.IGNORECASE,
)

# OCR sometimes reads exercise suffix letters as symbol-like characters. These
# are only used as hints. Sequence-aware inference below is the main protection
# against bad OCR such as "10B" being read as "108".
OCR_SUFFIX_HINTS = {
    '¢': 'C',
    '©': 'C',
    '€': 'C',
    '£': 'C',
    '®': 'B',
    '8': 'B',
    '6': 'G',
    '0': 'D',
}


def _alpha_next(value: str | None) -> str:
    if not value or not value.isalpha():
        return 'A'
    value = value.upper()
    if value >= 'Z':
        return 'Z'
    return chr(ord(value) + 1)


def _normalize_marker_token(token: str) -> str:
    token = (token or '').strip().upper()
    token = token.replace(' ', '')
    token = token.replace('.', '')
    token = token.replace(':', '')
    token = token.replace('-', '')
    # These replacements are safe in the numeric prefix of exercise labels:
    # EXERCISE I5A -> 15A, EXERCISE l0A -> 10A.
    token = token.replace('I', '1').replace('L', '1').replace('O', '0')
    return token


def _raw_suffix_from_exercise_token(raw_token: str, chapter_number: str) -> tuple[str | None, bool]:
    """
    Return (suffix, explicit_suffix).

    explicit_suffix=True means OCR clearly gave us a letter/symbol suffix.
    explicit_suffix=False means the suffix was missing/ambiguous and may need
    sequence inference.
    """
    token = _normalize_marker_token(raw_token)
    chapter_number = str(chapter_number)

    if not token:
        return None, False

    suffix = ''
    if token.startswith(chapter_number):
        suffix = token[len(chapter_number):]
    elif len(token) == 1 and token.isalpha():
        suffix = token
    elif len(token) == 1 and token in OCR_SUFFIX_HINTS:
        suffix = token
    else:
        # Example: OCR may return "108" for "10B" or "15B".
        # If it starts with the chapter number, the remainder is ambiguous.
        if token.startswith(chapter_number):
            suffix = token[len(chapter_number):]
        else:
            return None, False

    if not suffix:
        return None, False

    # Prefer real alpha suffixes when present.
    alpha = re.sub(r'[^A-Z]', '', suffix)
    if alpha:
        return alpha[0], True

    # Use OCR hint only as explicit if the whole suffix maps to a likely letter.
    if len(suffix) == 1 and suffix in OCR_SUFFIX_HINTS:
        return OCR_SUFFIX_HINTS[suffix], True

    # Ambiguous numeric suffix. Let sequence inference decide.
    return None, False


def _candidate_has_suffix_hint(raw_token: str, chapter_number: str) -> bool:
    suffix, explicit = _raw_suffix_from_exercise_token(raw_token, chapter_number)
    if suffix and explicit:
        return True
    token = _normalize_marker_token(raw_token)
    if token.startswith(str(chapter_number)) and len(token) > len(str(chapter_number)):
        return True
    return False


def _iter_marker_candidates(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract raw Exercise/Activity heading candidates from one page."""
    text = page.get('text') or page.get('text_plain') or page.get('ocr_text') or ''
    if not text:
        return []

    lines = text.splitlines()
    candidates: list[dict[str, Any]] = []
    running_offset = 0

    for line_no, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            running_offset += len(line) + 1
            continue

        # Use only heading-like occurrences. This avoids picking up casual
        # references such as "solve exercise 1A" inside body text.
        headingish = bool(re.match(r'^\s*(?:EXERCISE|Exercise|ACTIVITY|Activity)\b', stripped))
        headingish = headingish or bool(re.search(r'^\s*(?:OBJECTIVE\s+QUESTIONS?\s*)?(?:EXERCISE|ACTIVITY)\b', stripped, re.I))
        if not headingish:
            running_offset += len(line) + 1
            continue

        before_context = '\n'.join(lines[max(0, line_no - 6):line_no])
        after_context = '\n'.join(lines[line_no + 1:line_no + 5])
        local_offset = running_offset

        for m in EXERCISE_TOKEN_RE.finditer(stripped):
            candidates.append({
                'kind': 'exercise',
                'raw_token': m.group(1),
                'raw_heading': stripped,
                'page_number': page.get('page_number'),
                'printed_page_number': page.get('printed_page_number'),
                'order': local_offset + m.start(),
                'line_number': line_no + 1,
                'before_context': before_context,
                'after_context': after_context,
                'objective_context': bool(OBJECTIVE_CONTEXT_RE.search(before_context + '\n' + stripped)),
            })

        for m in ACTIVITY_TOKEN_RE.finditer(stripped):
            candidates.append({
                'kind': 'activity',
                'raw_token': m.group(1),
                'raw_heading': stripped,
                'page_number': page.get('page_number'),
                'printed_page_number': page.get('printed_page_number'),
                'order': local_offset + m.start(),
                'line_number': line_no + 1,
                'before_context': before_context,
                'after_context': after_context,
                'objective_context': False,
            })

        running_offset += len(line) + 1

    return candidates


def _canonicalize_exercise_candidates(
    chapter: dict[str, Any],
    raw_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert noisy OCR exercise candidates into canonical Exercise labels."""
    chapter_number = str(chapter.get('chapter_number') or '')
    if not chapter_number.isdigit():
        return []

    # If any candidate has a probable suffix, then a suffixless/ambiguous first
    # marker is likely OCR dropping A/B/C. Infer it sequence-wise.
    has_suffix_style = any(_candidate_has_suffix_hint(c.get('raw_token', ''), chapter_number) for c in raw_candidates)
    has_multiple = len(raw_candidates) > 1

    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_suffix: str | None = None

    for c in raw_candidates:
        suffix, explicit = _raw_suffix_from_exercise_token(c.get('raw_token', ''), chapter_number)

        if suffix and suffix.isalpha():
            # When OCR suffix is suspiciously ahead of the natural sequence
            # because B/D/G was read as 8/0/6, trust sequence for the first
            # ambiguous-looking marker. Example: "Exercise 108" as first marker
            # should be Exercise 10A, not 10B.
            raw_token = _normalize_marker_token(c.get('raw_token', ''))
            raw_tail = raw_token[len(chapter_number):] if raw_token.startswith(chapter_number) else raw_token
            ambiguous_tail = raw_tail and not re.search(r'[A-Z¢©€£®]', raw_tail)
            expected = _alpha_next(last_suffix)
            if ambiguous_tail and (has_multiple or has_suffix_style):
                suffix = expected
        elif has_multiple or has_suffix_style:
            suffix = _alpha_next(last_suffix)
        else:
            suffix = None

        marker = f'Exercise {chapter_number}{suffix}' if suffix else f'Exercise {chapter_number}'

        if marker in seen:
            continue

        seen.add(marker)
        if suffix:
            last_suffix = suffix

        item = dict(c)
        item['marker'] = marker
        item['suffix'] = suffix
        item['canonical_source'] = 'runtime_ocr_heading_detection'
        canonical.append(item)

    return canonical


def detect_subsection_markers(chapter: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect Exercise/Activity anchors at runtime from OCR text."""
    chapter_number = str(chapter.get('chapter_number') or '')
    chapter_title = chapter.get('chapter_title') or ''
    is_activity_chapter = chapter_number == '24' or str(chapter_title).strip().lower() == 'activities'

    raw: list[dict[str, Any]] = []
    for page in pages:
        for c in _iter_marker_candidates(page):
            if is_activity_chapter and c['kind'] == 'activity':
                raw.append(c)
            elif not is_activity_chapter and c['kind'] == 'exercise':
                raw.append(c)

    raw.sort(key=lambda x: (int(x.get('page_number') or 0), int(x.get('order') or 0), x.get('raw_heading') or ''))

    if is_activity_chapter:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in raw:
            try:
                number = int(c.get('raw_token'))
            except Exception:
                continue
            marker = f'Activity {number}'
            if marker in seen:
                continue
            seen.add(marker)
            item = dict(c)
            item['marker'] = marker
            item['canonical_source'] = 'runtime_ocr_activity_heading_detection'
            out.append(item)
        return out

    return _canonicalize_exercise_candidates(chapter, raw)


def _should_group_with_previous(marker: dict[str, Any], previous_marker: dict[str, Any] | None) -> bool:
    """Return True for objective/immediate trailing exercises."""
    if not previous_marker:
        return False
    if marker.get('kind') != 'exercise':
        return False

    # If the next exercise heading starts on the same PDF page as the previous
    # exercise heading, it is usually an immediate objective exercise in this book.
    if marker.get('page_number') == previous_marker.get('page_number'):
        return True

    # Group objective-question exercises with the previous concept subsection.
    if marker.get('objective_context'):
        return True

    # If the heading/context itself says objective questions, group it.
    context = '\n'.join([
        str(marker.get('before_context') or ''),
        str(marker.get('raw_heading') or ''),
        str(marker.get('after_context') or ''),
    ])
    if OBJECTIVE_CONTEXT_RE.search(context):
        return True

    return False


def group_subsection_markers(chapter: dict[str, Any], markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group runtime-detected markers into subsection anchors.

    Normal exercises create a new subsection. Objective/immediate exercises are
    included in the preceding subsection.
    """
    groups: list[dict[str, Any]] = []
    previous_marker: dict[str, Any] | None = None

    for marker in markers:
        label = marker['marker']
        if _should_group_with_previous(marker, previous_marker) and groups:
            groups[-1]['included_markers'].append(label)
            groups[-1]['included_marker_details'].append(marker)
            note = 'Runtime grouped this immediate/objective exercise with the preceding concept subsection.'
            if note not in groups[-1]['notes']:
                groups[-1]['notes'].append(note)
        else:
            groups.append({
                'anchor': marker,
                'included_markers': [label],
                'included_marker_details': [marker],
                'notes': ['Subsection anchor detected at runtime from OCR Exercise/Activity heading.'],
            })
        previous_marker = marker

    return groups


def _format_page_block_for_subsection(page_record: dict[str, Any]) -> str:
    pp = page_record.get('printed_page_number')
    pp_label = pp if pp is not None else page_record.get('printed_page_label')
    return f"[PDF page {page_record.get('page_number')} / printed page {pp_label}]\n{(page_record.get('text') or '').strip()}".strip()


def _build_subsection_record_from_pages(
    chapter: dict[str, Any],
    group: dict[str, Any],
    idx: int,
    start_page: int,
    end_page: int,
    subsection_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    page_numbers = [p.get('page_number') for p in subsection_pages if p.get('page_number') is not None]
    printed_page_numbers = [p.get('printed_page_number') for p in subsection_pages if p.get('printed_page_number') is not None]
    subsection_text = '\n\n'.join(_format_page_block_for_subsection(p) for p in subsection_pages if (p.get('text') or '').strip())
    production_pages = [p for p in subsection_pages if p.get('include_in_embeddings') is True]
    production_text = '\n\n'.join((p.get('text') or '').strip() for p in production_pages if (p.get('text') or '').strip())
    excluded_pages = [
        {
            'page_number': p.get('page_number'),
            'printed_page_number': p.get('printed_page_number'),
            'reasons': p.get('production_exclusion_reasons', []),
        }
        for p in subsection_pages if p.get('include_in_embeddings') is not True
    ]

    combined_math_lines: list[str] = []
    for p in subsection_pages:
        combined_math_lines.extend(p.get('math_lines', [])[:20])

    anchor = group['anchor']
    included = group.get('included_markers') or [anchor.get('marker')]
    start_printed = printed_page_numbers[0] if printed_page_numbers else None
    end_printed = printed_page_numbers[-1] if printed_page_numbers else None
    flags = {flag for p in subsection_pages for flag in p.get('quality_flags', [])}
    if excluded_pages:
        flags.add('some_pages_excluded_from_production_embeddings')
    if subsection_pages and not production_pages:
        flags.add('subsection_requires_vision_qa_before_embedding')

    notes = ['Page-level range; if an exercise/activity heading starts mid-page, the whole PDF page is assigned to this subsection.']
    notes.extend(group.get('notes') or [])

    return {
        'section_number': chapter.get('chapter_number'),
        'section_title': chapter.get('chapter_title'),
        'unit_number': chapter.get('unit_number'),
        'unit_title': chapter.get('unit_title'),
        'chapter_type': chapter.get('chapter_type'),
        'chapter_number': chapter.get('chapter_number'),
        'chapter_title': chapter.get('chapter_title'),
        'subsection_number': f"{chapter.get('chapter_number')}.{idx}",
        'subsection_title': f"Before/through {anchor.get('marker')}",
        'anchor_marker': anchor.get('marker'),
        'anchor_pdf_page': anchor.get('page_number'),
        'anchor_printed_page': anchor.get('printed_page_number'),
        'anchor_detection_method': anchor.get('canonical_source') or 'runtime_ocr_heading_detection',
        'anchor_raw_heading': anchor.get('raw_heading'),
        'included_exercises_or_activities': included,
        'includes': included,
        'start_page': start_page,
        'end_page': end_page,
        'start_pdf_page': start_page,
        'end_pdf_page': end_page,
        'printed_start_page': start_printed,
        'printed_end_page': end_printed,
        'start_printed_page': start_printed,
        'end_printed_page': end_printed,
        'pdf_pages': {'start': start_page, 'end': end_page},
        'printed_pages': {'start': start_printed, 'end': end_printed},
        'page_count': len(page_numbers),
        'subsection_text': subsection_text,
        'subsection_text_plain': subsection_text,
        'text_plain': subsection_text,
        'production_subsection_text': production_text,
        'production_indexed_page_numbers': [p.get('page_number') for p in production_pages],
        'production_printed_page_numbers': [p.get('printed_page_number') for p in production_pages],
        'production_excluded_pages': excluded_pages,
        'production_page_count': len(production_pages),
        'subsection_math_lines': combined_math_lines[:300],
        'math_lines': combined_math_lines[:300],
        'physical_start_page': start_page,
        'physical_end_page': end_page,
        'physical_printed_start_page': start_printed,
        'physical_printed_end_page': end_printed,
        'physical_page_count': max(0, end_page - start_page + 1),
        'page_numbers': page_numbers,
        'printed_page_numbers': printed_page_numbers,
        'excluded_related_pages': [],
        'text_sources': ['rendered_page_tesseract_ocr'],
        'quality_flags': sorted(flags),
        'include_in_embeddings': len(production_pages) > 0,
        'notes': notes,
    }


def build_subsections_for_chapter(chapter: dict[str, Any], chapter_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chapter_pages = [p for p in chapter_pages if p.get('include_in_lesson_text', True)]
    chapter_pages.sort(key=lambda p: int(p.get('page_number') or 0))

    if not chapter_pages or chapter.get('chapter_type') == 'appendix':
        return []

    markers = detect_subsection_markers(chapter, chapter_pages)

    if not markers:
        start_page = int(chapter_pages[0].get('page_number'))
        end_page = int(chapter_pages[-1].get('page_number'))
        fallback_group = {
            'anchor': {
                'marker': 'Chapter body',
                'page_number': start_page,
                'printed_page_number': chapter_pages[0].get('printed_page_number'),
                'canonical_source': 'runtime_no_marker_fallback',
                'raw_heading': None,
            },
            'included_markers': [],
            'included_marker_details': [],
            'notes': ['No exercise/activity heading was detected by OCR; fallback subsection covers the full chapter.'],
        }
        return [_build_subsection_record_from_pages(chapter, fallback_group, 1, start_page, end_page, chapter_pages)]

    groups = group_subsection_markers(chapter, markers)

    chapter_start = int(chapter_pages[0].get('page_number'))
    chapter_end = int(chapter_pages[-1].get('page_number'))
    subsections: list[dict[str, Any]] = []

    is_activity_chapter = (
        str(chapter.get('chapter_number') or '') == '24'
        or str(chapter.get('chapter_title') or '').strip().lower() == 'activities'
    )

    for idx, group in enumerate(groups, start=1):
        anchor_page = int(group['anchor'].get('page_number') or chapter_start)

        if is_activity_chapter:
            # Activity sections start on their own activity-heading page and run
            # until the page before the next activity heading.
            start_page = anchor_page
            if idx < len(groups):
                next_anchor_page = int(groups[idx]['anchor'].get('page_number') or chapter_end)
                end_page = next_anchor_page - 1
            else:
                end_page = chapter_end
        else:
            # Exercise-based chapters are split as "concept before/through
            # Exercise XA", so the exercise anchor page closes that subsection.
            if idx == 1:
                start_page = chapter_start
            else:
                prev_anchor_page = int(groups[idx - 2]['anchor'].get('page_number') or chapter_start)
                start_page = prev_anchor_page + 1
            end_page = chapter_end if idx == len(groups) else anchor_page

        start_page = max(chapter_start, min(start_page, chapter_end))
        end_page = max(start_page, min(end_page, chapter_end))

        subsection_pages = [
            p for p in chapter_pages
            if start_page <= int(p.get('page_number') or -1) <= end_page
        ]
        subsections.append(_build_subsection_record_from_pages(chapter, group, idx, start_page, end_page, subsection_pages))

    return subsections


def sync_subsections_to_section_index(data: dict[str, Any]) -> None:
    """Add the same runtime subsections to section_index entries without changing existing fields."""
    chapters = data.get('extraction', {}).get('chapters', []) or []
    section_index = data.get('extraction', {}).get('section_index', []) or []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for chapter in chapters:
        key = (str(chapter.get('chapter_number')), str(chapter.get('chapter_title')))
        by_key[key] = copy.deepcopy(chapter.get('subsections') or [])

    for section in section_index:
        key = (str(section.get('chapter_number') or section.get('section_number')), str(section.get('chapter_title') or section.get('section_title')))
        if key in by_key:
            section['subsections'] = copy.deepcopy(by_key[key])

def rebuild_chapters(data: dict[str, Any]) -> None:
    pages = data['extraction']['page_extractions']
    pages_by_chapter: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in pages:
        if p.get('chapter_number') is not None and p.get('chapter_title'):
            pages_by_chapter[(str(p.get('chapter_number')), str(p.get('chapter_title')))].append(p)

    for chapter in data['extraction'].get('chapters', []):
        cnum = str(chapter.get('chapter_number'))
        ctitle = str(chapter.get('chapter_title'))
        cpages = pages_by_chapter.get((cnum, ctitle), [])
        cpages.sort(key=lambda p: int(p.get('page_number') or 0))

        for lesson in chapter.get('lessons', []):
            start = lesson.get('start_page')
            end = lesson.get('end_page')
            lesson_pages = [p for p in cpages if start and end and start <= p.get('page_number', -1) <= end]
            lesson_pages = [p for p in lesson_pages if p.get('include_in_lesson_text', True)]
            production_pages = [p for p in lesson_pages if p.get('include_in_embeddings') is True]
            excluded_pages = [
                {
                    'page_number': p.get('page_number'),
                    'printed_page_number': p.get('printed_page_number'),
                    'reasons': p.get('production_exclusion_reasons', []),
                }
                for p in lesson_pages if p.get('include_in_embeddings') is not True
            ]

            lesson['lesson_text'] = '\n\n'.join((p.get('text') or '').strip() for p in lesson_pages if (p.get('text') or '').strip())
            lesson['text_plain'] = lesson['lesson_text']
            lesson['production_lesson_text'] = '\n\n'.join((p.get('text') or '').strip() for p in production_pages if (p.get('text') or '').strip())
            lesson['production_indexed_page_numbers'] = [p.get('page_number') for p in production_pages]
            lesson['production_printed_page_numbers'] = [p.get('printed_page_number') for p in production_pages]
            lesson['production_excluded_pages'] = excluded_pages
            lesson['production_page_count'] = len(production_pages)
            lesson['include_in_embeddings'] = len(production_pages) > 0
            flags = set(lesson.get('quality_flags') or [])
            if excluded_pages:
                flags.add('some_pages_excluded_from_production_embeddings')
            if len(production_pages) == 0:
                flags.add('lesson_requires_vision_qa_before_embedding')
            lesson['quality_flags'] = sorted(flags)

        # Add/rebuild per-chapter subsections from final cleaned/gated page text.
        # This is additive: existing chapter/lesson/page output remains unchanged.
        chapter['subsections'] = build_subsections_for_chapter(chapter, cpages)

    sync_subsections_to_section_index(data)


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    data = json.loads(INPUT_JSON.read_text(encoding='utf-8'))
    data = copy.deepcopy(data)

    document_id, document_key = build_document_identity(data)
    data['documentId'] = document_id
    data['document_key'] = document_key

    metadata = data.setdefault('metadata', {})
    metadata['document_key'] = document_key
    metadata['source_type'] = metadata.get('source_type') or 'textbook_pdf'

    cleanup_counter = Counter()

    # Clean page fields.
    for page in data['extraction']['page_extractions']:
        chapter_title = page.get('chapter_title')
        page_fixes: list[str] = []
        for field in ['text', 'text_plain', 'ocr_text']:
            if isinstance(page.get(field), str):
                cleaned, fixes = clean_text(page[field], chapter_title=chapter_title)
                page[field] = cleaned
                page_fixes.extend(fixes)
        if isinstance(page.get('math_lines'), list):
            cleaned, fixes = clean_any(page['math_lines'], chapter_title=chapter_title)
            page['math_lines'] = cleaned
            page_fixes.extend(fixes)
        if isinstance(page.get('extracted_blocks'), list):
            cleaned, fixes = clean_any(page['extracted_blocks'], chapter_title=chapter_title)
            page['extracted_blocks'] = cleaned
            page_fixes.extend(fixes)

        for fix in page_fixes:
            cleanup_counter[fix.split(':')[0]] += int(fix.split(':')[-1]) if ':' in fix and fix.split(':')[-1].isdigit() else 1

        if page_fixes:
            stats = page.setdefault('cleanup_stats', {})
            stats['v4_production_cleanup_fixes'] = page_fixes

        reasons = page_failure_reasons(page)
        page['production_exclusion_reasons'] = reasons
        flags = set(page.get('quality_flags') or [])
        if reasons:
            flags.add('production_embedding_excluded_until_vision_qa')
            page['include_in_embeddings'] = False
            page['embedding_readiness'] = 'needs_vision_qa_before_production_embedding'
        else:
            flags.add('production_embedding_ready')
            page['include_in_embeddings'] = True
            page['embedding_readiness'] = 'ready_for_production_embedding'
        page['quality_flags'] = sorted(flags)
        page['production_text'] = page.get('text') if page.get('include_in_embeddings') else ''

    rebuild_chapters(data)

    # Update extraction-level summary.
    pages = data['extraction']['page_extractions']
    ready_pages = [p for p in pages if p.get('include_in_embeddings') is True]
    excluded_pages = [p for p in pages if p.get('include_in_embeddings') is not True]
    reason_counts = Counter()
    for p in excluded_pages:
        reason_counts.update(p.get('production_exclusion_reasons') or [])

    data['extraction']['production_embedding_policy'] = {
        'status': 'production_safe_gated',
        'meaning': 'Only pages with embedding_readiness=ready_for_production_embedding should be embedded. Excluded pages remain in JSON for Mathpix/vision/manual QA.',
        'embed_only_when': {
            'include_in_embeddings': True,
            'embedding_readiness': 'ready_for_production_embedding',
        },
        'do_not_embed_when_flags_include': [
            'production_embedding_excluded_until_vision_qa',
            'dense_formula_layout_requires_vision_qa',
            'answer_key_or_appendix_requires_manual_or_vision_qa',
            'remaining_ambiguous_symbol_requires_vision_qa:*',
        ],
        'note': 'This file is production-safe for RAG ingestion because unreliable math pages are gated out. It is not a substitute for Mathpix/vision extraction for exact symbolic math on excluded pages.',
    }
    data['extraction']['quality_summary']['v4_production_cleanup'] = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'document_id_present': bool(data.get('documentId')),
        'document_key_present': bool(data.get('document_key')),
        'documentId': data.get('documentId'),
        'document_key': data.get('document_key'),
        'safe_for_production_reindex': bool(data.get('document_key')),
        'cleanup_rule_counts': dict(cleanup_counter),
        'total_pages': len(pages),
        'total_subsections': sum(len(ch.get('subsections', [])) for ch in data['extraction'].get('chapters', [])),
        'ready_for_production_embedding_pages': len(ready_pages),
        'excluded_until_vision_qa_pages': len(excluded_pages),
        'exclusion_reason_counts': dict(reason_counts),
    }
    data['extraction']['generated_at_utc'] = datetime.now(timezone.utc).isoformat()

    OUTPUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    # CSV of pages requiring QA.
    with QA_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['page_number', 'printed_page_number', 'chapter_title', 'embedding_readiness', 'reasons', 'sample_text'])
        writer.writeheader()
        for p in excluded_pages:
            sample = re.sub(r'\s+', ' ', (p.get('text') or '')[:300])
            writer.writerow({
                'page_number': p.get('page_number'),
                'printed_page_number': p.get('printed_page_number'),
                'chapter_title': p.get('chapter_title'),
                'embedding_readiness': p.get('embedding_readiness'),
                'reasons': '; '.join(p.get('production_exclusion_reasons') or []),
                'sample_text': sample,
            })

    # Validation checks for listed issues.
    listed_patterns = {
        'Ifa': r'\bIfa\b',
        'at+b': r'\bat\+b\b',
        '|. Closure': r'\|\.\s*Closure',
        'Ill_heading': r'(?m)^\s*Ill\.\s+',
        '¢': r'¢',
        'b+e': r'\bb\+e\b',
        'ete.': r'\bete\.',
    }
    remaining_listed = {}
    for name, pat in listed_patterns.items():
        hits = []
        for p in pages:
            if re.search(pat, p.get('text') or '', re.I if name == 'ete.' else 0):
                hits.append((p.get('page_number'), p.get('printed_page_number'), p.get('chapter_title')))
        remaining_listed[name] = hits

    # Warning on remaining ambiguous currency-like symbols.
    ambig_hits = []
    for p in pages:
        syms = sorted(set(AMBIGUOUS_SYMBOL_RE.findall(p.get('text') or '')))
        if syms:
            ambig_hits.append((p.get('page_number'), p.get('printed_page_number'), p.get('chapter_title'), ''.join(syms)))

    report_lines = []
    report_lines.append('Maths RSAggarwal v4 Production-Safe Validation Report')
    report_lines.append('=' * 60)
    report_lines.append(f'Generated at UTC: {datetime.now(timezone.utc).isoformat()}')
    report_lines.append(f'Input JSON: {INPUT_JSON.name}')
    report_lines.append(f'Output JSON: {OUTPUT_JSON.name}')
    report_lines.append(f"documentId: {data.get('documentId')}")
    report_lines.append(f"document_key: {data.get('document_key')}")
    report_lines.append('')
    report_lines.append('Production status: SAFE TO INGEST ONLY WITH EMBEDDING GATING')
    report_lines.append('This means: embed pages where include_in_embeddings=true and embedding_readiness=ready_for_production_embedding.')
    report_lines.append('Pages excluded from embeddings require Mathpix/vision/manual QA before production embedding.')
    report_lines.append('')
    report_lines.append(f'Total pages: {len(pages)}')
    report_lines.append(f'Total subsections: {sum(len(ch.get("subsections", [])) for ch in data["extraction"].get("chapters", []))}')
    report_lines.append(f'Ready for production embedding: {len(ready_pages)}')
    report_lines.append(f'Excluded until vision QA: {len(excluded_pages)}')
    report_lines.append('')
    report_lines.append('Cleanup rule counts:')
    for k, v in cleanup_counter.most_common():
        report_lines.append(f'  - {k}: {v}')
    report_lines.append('')
    report_lines.append('Remaining occurrences of specifically reported artifacts in final page text:')
    any_remaining = False
    for name, hits in remaining_listed.items():
        if hits:
            any_remaining = True
            report_lines.append(f'  - {name}: {len(hits)} hit(s); examples: {hits[:10]}')
        else:
            report_lines.append(f'  - {name}: 0')
    report_lines.append('')
    report_lines.append('Remaining ambiguous fake-currency/math symbols:')
    if ambig_hits:
        report_lines.append(f'  - {len(ambig_hits)} page(s) still contain ambiguous symbols that require vision QA; examples: {ambig_hits[:20]}')
    else:
        report_lines.append('  - 0')
    report_lines.append('')
    report_lines.append('Top production exclusion reasons:')
    for reason, count in reason_counts.most_common(30):
        report_lines.append(f'  - {reason}: {count}')
    report_lines.append('')
    report_lines.append('Important note:')
    report_lines.append('  v4 is production-safe because it prevents unreliable OCR/math pages from entering embeddings.')
    report_lines.append('  It is not full production-grade exact math extraction for every page. Exact symbolic math still requires Mathpix/vision/manual QA for excluded pages.')

    REPORT_TXT.write_text('\n'.join(report_lines), encoding='utf-8')

    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_TXT}')
    print(f'Wrote {QA_CSV}')
    print(f'Ready pages: {len(ready_pages)} / {len(pages)}')
    print(f'Excluded pages: {len(excluded_pages)} / {len(pages)}')


if __name__ == '__main__':
    main()
