#!/usr/bin/env python3
"""
Step 5: math-precision audit + repair for Grade10_Maths.

Step 4 is a residual OCR-garbage gate. It proves coverage and removes obvious random
OCR noise. It does not prove that mathematical notation was preserved correctly.

This step catches and repairs errors that can still look like readable English but are
not production-grade for a maths textbook, for example:
- CHAPTER LE instead of CHAPTER 1
- /2, ./3, ./5 instead of sqrt/√ terms
- x?, OP?, PN?, cos? instead of powers
- corrupted formula/fraction structures
- corrupted statistics class intervals and mode formulae
- residual currency/symbol OCR artifacts in production-safe fields

It uses the same OpenAI vision infrastructure as Step 3/4, but with a stricter prompt
focused on exact formulas and table values. If a page cannot be repaired, it is gated
out so the final JSON cannot falsely claim production_complete_ready.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from make_grade10_maths_step_2_production_gate import rebuild_chapters_and_sections  # noqa: E402
from make_grade10_maths_step_3_vision_repair import (  # noqa: E402
    call_openai_vision,
    load_manual_overrides,
    mark_page_repaired,
    normalize_repair_payload,
    render_page_to_data_url,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_math_precision_audit_report.txt"
DEFAULT_SUSPICIOUS_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_math_precision_suspicious_pages.csv"
DEFAULT_REMAINING_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_math_precision_remaining_pages.csv"

PROMPT_VERSION = "math_precision_v8_strict_formula_sync_2026_06_16"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {value!r}") from exc


def stable_cache_key(pdf_path: Path, *, model: str, scale: float, prompt_version: str) -> str:
    stat = pdf_path.stat()
    raw = f"{pdf_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|model={model}|scale={scale}|prompt={prompt_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_pages_arg(value: str | None) -> set[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages


def is_lesson_body_page(page: dict[str, Any]) -> bool:
    return page.get("content_type") == "lesson_body" and page.get("include_in_lesson_text") is True


def page_is_ready(page: dict[str, Any]) -> bool:
    return page.get("include_in_embeddings") is True and page.get("embedding_readiness") == "ready_for_production_embedding"


def safe_text(page: dict[str, Any]) -> str:
    return str(page.get("production_safe_text") or page.get("text") or "")


# Specific formula/math corruption patterns. These are deliberately stronger than Step 4.
BAD_CHAPTER_HEADER_RE = re.compile(r"\bCHAPTER\s+(?:LE|Lh|LO|LI|lE|lh|lo|[I|]E|[I|]O)\b", re.I)
BAD_SQRT_SERIES_RE = re.compile(r"(?i)(?:such as|irrationality of many numbers|prove the irrationality)[^\n]{0,160}(?:/2|\./3|\./5|/3|/5)")
BAD_SQRT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(?:\./|/)(?:2|3|5|7|10|11|13)(?![A-Za-z0-9])")
# Do not flag ordinary punctuation questions. This catches OCR power corruption tokens.
POWER_QUESTION_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,5}|[A-Z]{2,}|sin|cos|tan|cot|sec|cosec|OP|PN|PT|AB|BC|AC|PQ|QR|PR|OA|OB|OC)(?:\d+)?\?(?![A-Za-z0-9])"
    r"|(?<=\d)[A-Za-z]{1,4}\?(?![A-Za-z0-9])"
    r"|\)\s*\?(?![A-Za-z0-9])"
    r")"
)
SUSPICIOUS_SYMBOL_RE = re.compile(r"[¥¢£€™®§©¤¦¨¬¯´¸¿¡]")
DEVANAGARI_OR_MOJIBAKE_RE = re.compile(r"[\u0900-\u097F\u0A80-\u0AFF]|[�]|(?:Â|Ã|â€|â€™|â€œ|â€\u009d)")
KNOWN_BAD_SNIPPETS = [
    ("known_bad_chapter1_header", re.compile(r"\bCHAPTER\s+LE\b", re.I)),
    ("known_bad_euclid_proof_corruption", re.compile(r"hf\s+H0S\s+RET|0S\s+1%|a\s*=\s*bq,\s*\+n|q\s+and\s+7,", re.I)),
    ("known_bad_trig_identity_wrong_factor", re.compile(r"x\^2\s*-\s*9\s*=\s*\(x\s*-\s*a\)\s*\(x\s*\+\s*3\)", re.I)),
    ("known_bad_trig_fraction_identity_lost", re.compile(r"\(x\s*-\s*a\)\s*\(x\s*-\s*b\)\s*\+\s*\(x\s*-\s*b\)\s*\(x\s*-\s*c\)", re.I)),
    ("known_bad_statistics_mode_formula", re.compile(r"Mode\s*=\s*1\s*\+\s*-\s*/\s*-\s*fi__y|3454\s+ee|2f\s*-\s*fi", re.I)),
    ("known_bad_statistics_class_intervals", re.compile(r"Age\s*\(\s*in\s*years\s*\)\s*:\s*45-145\s+145-245|445-545", re.I)),
    ("known_bad_probability_outcomes", re.compile(r"\bOutces\b|\bsubsef\b|\bknow1\b|\bs¢\b", re.I)),
    ("known_bad_page12_power_formula", re.compile(r"4m\?|9q\*|a\s*=\s*\(3q\)\s*=|a\s*=\s*4gq\s*\+\s*r", re.I)),
    ("known_bad_page14_odd_square_formula", re.compile(r"y\s+Pp\s+8!|2n4\+1P|ae\s+of\s+af|x\+y\^2\s*=", re.I)),
    ("known_bad_page163_cross_multiplication", re.compile(r"Ga\s+Ty|Dandy\s*=%|x\s+_\s+-y\s+_\s+1", re.I)),
]

# Dense OCR-ish short-token lines like "Wt ot oat oY pg Mg BL".  Step 4's fast mode is
# conservative; this pass treats them as blockers when they are still present after all repair.
def dense_short_token_noise_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        raw = line.strip()
        if len(raw) < 12 or len(raw) > 100:
            continue
        if re.search(r"[=+\-*/^√×÷<>≤≥≠∠⊥∥°]", raw):
            continue
        toks = re.findall(r"[A-Za-z0-9]+", raw)
        if len(toks) < 6:
            continue
        alpha = [t for t in toks if re.search(r"[A-Za-z]", t)]
        if len(alpha) < 6:
            continue
        short_ratio = sum(1 for t in alpha if len(t) <= 3) / max(len(alpha), 1)
        weird_case = sum(1 for t in alpha if not (t.islower() or t.isupper()))
        common = sum(1 for t in alpha if t.lower() in {"a", "an", "as", "at", "be", "by", "if", "in", "is", "it", "of", "on", "or", "so", "to", "we"})
        if short_ratio >= 0.85 and common <= 1 and weird_case >= 1:
            count += 1
    return count


def likely_power_question_count(text: str) -> int:
    count = 0
    for match in POWER_QUESTION_RE.finditer(text):
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        ctx = text[start:end]
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]
        # Formula context only; otherwise it may be a real question mark.
        if (
            re.search(r"[=+\-*/^√×÷<>≤≥≠∠⊥∥]", ctx)
            or re.search(r"\b(?:square|squares|cube|cubes|hypotenuse|area|perimeter|cos|sin|tan|cot|sec|cosec|prove|show|find|identity|theorem)\b", ctx, re.I)
            or re.search(r"[A-Za-z]\s*[+\-*/=]", line)
        ):
            count += 1
    return count


def detect_math_precision_issues(page: dict[str, Any], *, mode: str = "standard") -> list[str]:
    text = safe_text(page)
    stripped = text.strip()
    reasons: list[str] = []
    if not stripped:
        reasons.append("math_precision_empty_text")
        return reasons

    # Known reviewer examples and exact corruption signatures.
    for name, pattern in KNOWN_BAD_SNIPPETS:
        if pattern.search(stripped):
            reasons.append(name)
    reasons.extend(detect_known_broken_formula_fragments(stripped))

    if BAD_CHAPTER_HEADER_RE.search(stripped[:300]):
        reasons.append("math_precision_bad_chapter_header")
    if BAD_SQRT_SERIES_RE.search(stripped) or len(BAD_SQRT_TOKEN_RE.findall(stripped)) >= 2:
        reasons.append("math_precision_broken_square_root_tokens")

    power_q = likely_power_question_count(stripped)
    if power_q >= 1:
        reasons.append(f"math_precision_likely_power_question_mark:{power_q}")

    suspicious_symbols = SUSPICIOUS_SYMBOL_RE.findall(stripped)
    if suspicious_symbols:
        # For a production math JSON, these are almost never legitimate.
        reasons.append(f"math_precision_suspicious_ocr_symbols:{len(suspicious_symbols)}")

    if DEVANAGARI_OR_MOJIBAKE_RE.search(stripped):
        reasons.append("math_precision_remaining_devanagari_or_mojibake")

    dense = dense_short_token_noise_lines(stripped)
    if dense:
        reasons.append(f"math_precision_dense_short_token_noise_lines:{dense}")

    # Strict mode: catch more broken formula/table tokens. Use for QA, not as default if it over-flags.
    if (mode or "standard").lower() == "strict":
        if re.search(r"\b[A-Za-z]{3,}\d[A-Za-z]+\b|\b[A-Za-z]+\d[A-Za-z]{3,}\b", stripped):
            reasons.append("math_precision_broken_alpha_digit_words")
        if re.search(r"\b(?:fi__|__y|ee\b|px\)|\(\s*n\s*\))", stripped):
            reasons.append("math_precision_formula_artifact_tokens")
    return sorted(set(reasons))



def formula_context(line: str) -> bool:
    return bool(
        re.search(r"[=+\-*/^√×÷<>≤≥≠∠⊥∥]", line)
        or re.search(r"\b(?:square|squares|cube|cubes|area|perimeter|cos|sin|tan|cot|sec|cosec|prove|show|identity|theorem|hence|therefore)\b", line, re.I)
    )


def normalize_power_question_marks_in_line(line: str) -> str:
    if "?" not in line or not formula_context(line):
        return line
    # OCR often reads a superscript 2 as '?'. This is conservative because it only runs
    # on formula-like lines, not ordinary prose questions.
    line = re.sub(r"\)(\s*)\?(?![A-Za-z0-9])", r")^2", line)
    line = re.sub(r"(?<=\d)([A-Za-z]{1,4})\?(?![A-Za-z0-9])", r"\1^2", line)
    line = re.sub(r"(?<![A-Za-z0-9])([A-Za-z]{1,5}|[A-Z]{2,}|sin|cos|tan|cot|sec|cosec|OP|PN|PT|AB|BC|AC|PQ|QR|PR|OA|OB|OC)(\d*)\?(?![A-Za-z0-9])", r"\1\2^2", line)
    # Some OCR engines output '*' for superscript 2 on scanned math pages.
    line = re.sub(r"(?<![A-Za-z0-9])([A-Za-z]{1,4}|[A-Z]{2,})(\d*)\*(?![A-Za-z0-9])", r"\1\2^2", line)
    return line


def detect_known_broken_formula_fragments(text: str) -> list[str]:
    reasons: list[str] = []
    patterns = [
        ("math_precision_broken_power_question_after_digit", re.compile(r"\b\d+[A-Za-z]{1,4}\?\b")),
        ("math_precision_broken_parenthesized_power", re.compile(r"\)[ ]*\?")),
        ("math_precision_broken_star_power", re.compile(r"\b(?:[A-Za-z]{1,4}|[A-Z]{2,})\*\b")),
        ("math_precision_bad_algebra_fragment_page14", re.compile(r"\b(?:2n4\+1P|ae\s+of\s+af|y\s+Pp\s+8!)\b", re.I)),
        ("math_precision_bad_cross_multiplication_fragment", re.compile(r"Ga\s+Ty|Dandy\s*=%", re.I)),
    ]
    for name, pattern in patterns:
        if pattern.search(text):
            reasons.append(name)
    return reasons


def sync_ready_page_text_fields_from_production(pages: list[dict[str, Any]]) -> int:
    """Make legacy page text fields match production_safe_text for ready pages.

    DB loaders sometimes use text/text_plain instead of production_safe_text. For a final
    production artifact, ready page legacy fields must not contain the old Tesseract OCR.
    ocr_text/selectable_text are left untouched as raw audit fields.
    """
    changed = 0
    for page in pages:
        if not page_is_ready(page):
            continue
        prod = str(page.get("production_safe_text") or "").strip()
        if not prod:
            continue
        for field in ("text", "text_plain"):
            if page.get(field) != prod:
                page[field] = prod
                changed += 1
        flags = set(page.get("quality_flags") or [])
        flags.add("legacy_text_fields_synced_to_production_safe_text")
        page["quality_flags"] = sorted(flags)
    return changed

def apply_deterministic_safe_math_fixes(page: dict[str, Any]) -> bool:
    """Apply only high-confidence local fixes. Returns True when text changed."""
    text = safe_text(page)
    if not text:
        return False
    original = text

    # Fix broken chapter header only when the page metadata tells us the chapter number.
    chapter_number = page.get("chapter_number") or page.get("chapter_sequence")
    if chapter_number is not None:
        text = re.sub(r"\bCHAPTER\s+(?:LE|Lh|LO|LI|lE|lh|lo|[I|]E|[I|]O)\b", f"CHAPTER {chapter_number}", text, count=1, flags=re.I)

    # Fix the well-known introductory sqrt list without changing arbitrary divisions.
    def _sqrt_context_fix(match: re.Match[str]) -> str:
        s = match.group(0)
        s = re.sub(r"(?<![A-Za-z0-9])(?:\./|/)(2|3|5|7|10|11|13)(?![A-Za-z0-9])", r"√\1", s)
        return s
    text = BAD_SQRT_SERIES_RE.sub(_sqrt_context_fix, text)

    # Fix common power-2 OCR question marks only in formula-like contexts.
    text = "\n".join(normalize_power_question_marks_in_line(line) for line in text.splitlines())

    # Page-local high confidence repairs for repeated known OCR artifacts.
    text = re.sub(r"\ba\s*=\s*4gq\s*\+\s*r", "a = 4q + r", text, flags=re.I)
    text = re.sub(r"\bO\s*<\s*r\s*<\s*4", "0 ≤ r < 4", text)
    text = re.sub(r"x\^2\s*=\s*\(2m\s*\+\s*1\)\s*=", "x^2 = (2m + 1)^2 =", text)
    text = re.sub(r"a\s*=\s*\(3q\)\s*=", "a^2 = (3q)^2 =", text)

    # Remove stray currency/trademark OCR symbols without deleting surrounding formula text.
    text = SUSPICIOUS_SYMBOL_RE.sub("", text)

    if text != original:
        page["production_safe_text"] = text
        flags = set(page.get("quality_flags") or [])
        flags.add("math_precision_deterministic_fixes_applied")
        page["quality_flags"] = sorted(flags)
        return True
    return False


def build_precision_prompt(page: dict[str, Any], reasons: list[str]) -> str:
    previous = safe_text(page)[:5000]
    printed = page.get("printed_page_number")
    chapter = page.get("chapter_title") or ""
    section = page.get("section_title") or ""
    return f"""
You are doing MATH-PRECISION OCR QA for a scanned Class 10 mathematics textbook page.

The current transcription is readable but may be mathematically wrong. It was rejected for:
{'; '.join(reasons)}

Transcribe the page image again. This is NOT a summary task.

Critical rules:
- Trust the image over the current transcription.
- Preserve every visible formula, identity, fraction, table value, class interval, exponent, radical, and trigonometric symbol.
- Use plain text that is safe for JSON and database embeddings.
- Use ^2, ^3 for powers. Do not leave x?, 4m?, q*, cos?, OP?, PN?, PT?, or a parenthesized expression followed by ?.
- Use √2 or sqrt(2) for square roots. Do not write /2 or ./3 when the page shows a radical.
- Preserve fractions using clear linear notation: (numerator)/(denominator). For stacked fractions, write the full numerator and denominator.
- Preserve class intervals exactly, including decimal points such as 4.5-14.5 and 14.5-24.5.
- Preserve algebraic identities exactly. For example, x^2 - 9 should be (x - 3)(x + 3), not (x - a)(x + 3). Do not replace 3 with a, b, or c unless the image shows that.
- If an equation is large, use one full line per equation instead of compressing it incorrectly.
- Remove OCR garbage symbols like ¥, ¢, £, €, ™, ®, § and stray Hindi glyphs if they are not visible content.
- Do not solve exercises. Do not invent content. If a tiny part is genuinely unreadable, write [unreadable] only there.

Context:
PDF page number: {page.get('page_number')}
Printed page number: {printed}
Chapter: {chapter}
Section: {section}

Current transcription, for hints only; it contains mistakes. Trust the image more:
---
{previous}
---

Return JSON only with this exact shape:
{{
  "production_safe_text": "full corrected page transcription",
  "math_lines": ["important equations/formulas/tables from the page"],
  "confidence": 0.95,
  "notes": "brief math precision QA note"
}}
""".strip()


def get_manual_override(page: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    page_number = page.get("page_number")
    printed = page.get("printed_page_number")
    if page_number is not None and f"page:{int(page_number)}" in overrides:
        return overrides[f"page:{int(page_number)}"]
    if printed is not None and f"printed:{int(printed)}" in overrides:
        return overrides[f"printed:{int(printed)}"]
    return None


def override_to_payload(override: dict[str, Any]) -> dict[str, Any]:
    return {
        "production_safe_text": override.get("production_safe_text") or override.get("text") or "",
        "math_lines": override.get("math_lines") or [],
        "confidence": override.get("confidence", 1.0),
        "notes": override.get("notes") or "manual_math_precision_override",
    }


def mark_page_math_precision_approved(page: dict[str, Any], *, reasons_before: list[str], warning_reasons: list[str] | None = None) -> None:
    page["math_precision_audit_status"] = "approved_with_warnings" if warning_reasons else "approved"
    page["math_precision_prompt_version"] = PROMPT_VERSION
    page["math_precision_reasons_before_repair"] = sorted(set(reasons_before))
    if warning_reasons:
        page["math_precision_warning_reasons_after_repair"] = sorted(set(warning_reasons))
    flags = set(page.get("quality_flags") or [])
    flags.add("math_precision_audit_passed")
    flags.discard("production_embedding_excluded_until_math_precision_qa")
    page["quality_flags"] = sorted(flags)


def mark_page_math_precision_failed(page: dict[str, Any], *, reasons: list[str], error: str | None = None) -> None:
    page["include_in_embeddings"] = False
    page["embedding_readiness"] = "needs_math_precision_qa_before_production_embedding"
    page["production_safe_text"] = ""
    page["math_precision_audit_status"] = "failed" if error else "suspicious_gated"
    page["math_precision_prompt_version"] = PROMPT_VERSION
    page["math_precision_audit_reasons"] = sorted(set(reasons))
    if error:
        page["math_precision_audit_error"] = error
    flags = set(page.get("quality_flags") or [])
    flags.add("production_embedding_excluded_until_math_precision_qa")
    flags.discard("production_embedding_ready")
    page["quality_flags"] = sorted(flags)
    merged = set(page.get("production_exclusion_reasons") or [])
    merged.update(reasons or ["math_precision_audit_requires_repair"])
    if error:
        merged.add("math_precision_repair_failed")
    page["production_exclusion_reasons"] = sorted(merged)


def collect_candidates(pages: list[dict[str, Any]], *, selected_pages: set[int] | None, mode: str, force: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for page in pages:
        if not is_lesson_body_page(page):
            continue
        page_number = int(page.get("page_number") or 0)
        if selected_pages is not None and page_number not in selected_pages:
            continue
        if not page_is_ready(page):
            # If Step 4 already gated it, keep it as a blocker for this precision pass.
            candidates.append({"page": page, "candidate_kind": "excluded_lesson_body", "reasons": list(page.get("production_exclusion_reasons") or ["already_excluded_before_math_precision_audit"])})
            continue
        current_version = page.get("math_precision_prompt_version") == PROMPT_VERSION
        previously_approved = page.get("math_precision_audit_status") in {"approved", "approved_with_warnings"}
        apply_deterministic_safe_math_fixes(page)
        reasons = detect_math_precision_issues(page, mode=mode)
        if reasons:
            candidates.append({"page": page, "candidate_kind": "math_precision_suspicious_ready_page", "reasons": reasons})
        elif force or not previously_approved or not current_version:
            mark_page_math_precision_approved(page, reasons_before=[])
    return sorted(candidates, key=lambda row: int(row["page"].get("page_number") or 0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_number", "printed_page_number", "chapter_title", "candidate_kind", "reasons", "sample_text"])
        writer.writeheader()
        for row in rows:
            page = row["page"]
            sample = re.sub(r"\s+", " ", safe_text(page)[:500])
            writer.writerow({
                "page_number": page.get("page_number"),
                "printed_page_number": page.get("printed_page_number"),
                "chapter_title": page.get("chapter_title"),
                "candidate_kind": row.get("candidate_kind"),
                "reasons": "; ".join(row.get("reasons") or []),
                "sample_text": sample,
            })


def remaining_math_precision_blockers(pages: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        if not is_lesson_body_page(page):
            continue
        if not page_is_ready(page):
            rows.append({"page": page, "candidate_kind": "excluded_lesson_body", "reasons": list(page.get("production_exclusion_reasons") or ["excluded_lesson_body"])})
            continue
        reasons = detect_math_precision_issues(page, mode=mode)
        if reasons:
            rows.append({"page": page, "candidate_kind": "remaining_math_precision_issue", "reasons": reasons})
    return sorted(rows, key=lambda row: int(row["page"].get("page_number") or 0))


def count_empty_production_subsections(data: dict[str, Any]) -> list[dict[str, Any]]:
    empty: list[dict[str, Any]] = []
    for chapter in data.get("extraction", {}).get("chapters", []) or []:
        for sub in chapter.get("subsections", []) or []:
            start = sub.get("start_page") or sub.get("start_pdf_page")
            end = sub.get("end_page") or sub.get("end_pdf_page")
            if start is None or end is None:
                continue
            if int(sub.get("production_page_count") or 0) <= 0 or not str(sub.get("production_subsection_text") or "").strip():
                empty.append({
                    "chapter_number": chapter.get("chapter_number"),
                    "chapter_title": chapter.get("chapter_title"),
                    "subsection_number": sub.get("subsection_number"),
                    "subsection_title": sub.get("subsection_title"),
                    "start_page": start,
                    "end_page": end,
                })
    return empty


def refresh_policy(data: dict[str, Any], *, counts: Counter, initial_candidates: list[dict[str, Any]], remaining: list[dict[str, Any]], empty_subsections: list[dict[str, Any]], strict_complete: bool, mode: str, errors: list[str]) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    pages = extraction.get("page_extractions", []) or []
    lesson_body_pages = [p for p in pages if is_lesson_body_page(p)]
    excluded_lesson_body = [p for p in lesson_body_pages if not page_is_ready(p)]
    ready_pages = [p for p in pages if page_is_ready(p)]

    previous_policy = extraction.get("production_embedding_policy") or {}
    run_scope_blockers = previous_policy.get("run_scope_blockers") or []
    if run_scope_blockers:
        status = "smoke_or_partial_not_production_ready"
    elif excluded_lesson_body or remaining or empty_subsections:
        status = "production_safe_gated_needs_math_precision_qa"
    else:
        status = "production_complete_ready"

    previous_policy.update({
        "status": status,
        "production_complete": status == "production_complete_ready",
        "strict_complete": bool(strict_complete),
        "math_precision_audit_applied": True,
        "math_precision_remaining_pages": len(remaining),
        "math_precision_empty_production_subsections": len(empty_subsections),
        "math_precision_errors": errors[:50],
        "do_not_embed_when_flags_include": sorted(set((previous_policy.get("do_not_embed_when_flags_include") or []) + [
            "production_embedding_excluded_until_math_precision_qa",
            "math_precision_audit_failed",
        ])),
    })
    extraction["production_embedding_policy"] = previous_policy

    qs = extraction.setdefault("quality_summary", {})
    summary = {
        "generated_at_utc": now_utc(),
        "audit_mode": mode,
        "gate_status": status,
        "initial_candidates": len(initial_candidates),
        "vision_repaired_pages": int(counts.get("vision_repaired", 0)),
        "manual_override_pages": int(counts.get("manual_override", 0)),
        "cached_pages_used": int(counts.get("cached", 0)),
        "deterministic_fix_pages": int(counts.get("deterministic_fix", 0)),
        "approved_pages": int(counts.get("approved", 0)),
        "gated_failed_pages": int(counts.get("gated_failed", 0)),
        "total_pages_in_json": len(pages),
        "ready_for_production_embedding_pages": len(ready_pages),
        "lesson_body_pages": len(lesson_body_pages),
        "excluded_lesson_body_pages": len(excluded_lesson_body),
        "remaining_math_precision_pages": len(remaining),
        "empty_production_subsections": len(empty_subsections),
        "remaining_reason_counts": dict(Counter(reason for row in remaining for reason in row.get("reasons", []))),
        "empty_subsection_samples": empty_subsections[:30],
    }
    qs["math_precision_audit"] = summary
    extraction["generated_at_utc"] = now_utc()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Math precision audit/repair for Grade10_Maths production JSON.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--suspicious-csv", type=Path, default=DEFAULT_SUSPICIOUS_CSV)
    parser.add_argument("--remaining-csv", type=Path, default=DEFAULT_REMAINING_CSV)
    parser.add_argument("--model", default=os.environ.get("GRADE10_MATHS_MATH_PRECISION_MODEL") or os.environ.get("GRADE10_MATHS_VISION_MODEL", "gpt-4o-mini"))
    parser.add_argument("--scale", type=float, default=env_float("GRADE10_MATHS_VISION_SCALE", 2.5))
    parser.add_argument("--pages", default=None, help="Optional PDF pages to math-audit/repair, e.g. 8,11,503,710")
    parser.add_argument("--max-pages", type=int, default=env_int("GRADE10_MATHS_MATH_AUDIT_MAX_ITEMS", 0))
    parser.add_argument("--audit-mode", choices=["standard", "strict"], default=os.environ.get("GRADE10_MATHS_MATH_AUDIT_MODE", "standard").strip().lower() or "standard")
    parser.add_argument("--repair-with-vision", action="store_true")
    parser.add_argument("--force-vision", action="store_true", help="Ignore math precision repair cache.")
    parser.add_argument("--force-recheck", action="store_true", help="Recheck pages even if they were already marked math_precision_audit_status=approved.")
    parser.add_argument("--manual-overrides", type=Path, default=Path(os.environ.get("GRADE10_MATHS_PAGE_OVERRIDES", "")) if os.environ.get("GRADE10_MATHS_PAGE_OVERRIDES") else None)
    parser.add_argument("--trust-manual-overrides", action="store_true", default=str(os.environ.get("GRADE10_MATHS_TRUST_PAGE_OVERRIDES", "true")).strip().lower() in {"1", "true", "yes", "y", "on"}, help="Treat page overrides as reviewed/trusted and do not re-gate them with heuristic math detectors. Default true.")
    parser.add_argument("--min-confidence", type=float, default=env_float("GRADE10_MATHS_AUTO_REVIEW_THRESHOLD", 0.90))
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--strict-complete", action="store_true")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    pages = data.get("extraction", {}).get("page_extractions", []) or []
    if not pages:
        raise RuntimeError("Input JSON has no extraction.page_extractions")

    selected_pages = parse_pages_arg(args.pages)
    # Apply deterministic fixes before candidate CSV is written.
    for page in pages:
        if is_lesson_body_page(page) and page_is_ready(page):
            if apply_deterministic_safe_math_fixes(page):
                pass

    candidates = collect_candidates(pages, selected_pages=selected_pages, mode=args.audit_mode, force=args.force_recheck)
    if args.max_pages and args.max_pages > 0:
        candidates = candidates[: args.max_pages]
    print(f"Math precision audit candidates: {len(candidates)}")
    write_csv(args.suspicious_csv, candidates)
    print(f"Math precision suspicious CSV: {args.suspicious_csv}")

    cache_dir = args.output.parent / ".vision_qa_cache" / stable_cache_key(args.pdf, model=args.model, scale=args.scale, prompt_version=PROMPT_VERSION)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    overrides = load_manual_overrides(args.manual_overrides) if args.manual_overrides and args.manual_overrides.exists() else {}
    if args.manual_overrides:
        print(f"Manual overrides: {args.manual_overrides} ({len(overrides)} keys loaded)")
    print(f"Repair with vision: {args.repair_with_vision}")
    print(f"Math audit mode: {args.audit_mode}")

    counts: Counter = Counter()
    errors: list[str] = []
    pdf_doc: fitz.Document | None = fitz.open(args.pdf) if args.repair_with_vision and candidates else None

    try:
        for idx, row in enumerate(candidates, start=1):
            page = row["page"]
            page_number = int(page.get("page_number") or 0)
            printed = page.get("printed_page_number")
            reasons_before = list(row.get("reasons") or [])

            override = get_manual_override(page, overrides)
            payload: dict[str, Any] | None = None
            cache_path = cache_dir / f"page_{page_number}.json"
            source = "vision"

            if override:
                payload = override_to_payload(override)
                source = "manual_override"
                counts["manual_override"] += 1
                if args.trust_manual_overrides:
                    repaired_text, math_lines, confidence, notes = normalize_repair_payload(payload, min_chars=args.min_chars, min_confidence=args.min_confidence)
                    mark_page_repaired(page, repaired_text=repaired_text, math_lines=math_lines, confidence=confidence, notes=f"trusted_math_page_override:{notes}", model=args.model)
                    page["math_precision_audit_status"] = "approved_by_trusted_page_override"
                    page["math_precision_prompt_version"] = PROMPT_VERSION
                    page["math_precision_reasons_before_repair"] = sorted(set(reasons_before))
                    flags = set(page.get("quality_flags") or [])
                    flags.add("trusted_page_override_applied")
                    flags.add("math_precision_audit_passed")
                    flags.discard("production_embedding_excluded_until_math_precision_qa")
                    page["quality_flags"] = sorted(flags)
                    counts["approved"] += 1
                    print(f"[{idx}/{len(candidates)}] math-approved-by-trusted-override page={page_number}, printed={printed}, chars={len(repaired_text)}, confidence={confidence}")
                    continue
            elif cache_path.exists() and not args.force_vision:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                source = "cache"
                counts["cached"] += 1
            elif args.repair_with_vision:
                assert pdf_doc is not None
                data_url = render_page_to_data_url(pdf_doc, page_number, scale=args.scale)
                prompt = build_precision_prompt(page, reasons_before)
                last_error: Exception | None = None
                for attempt in range(args.retries + 1):
                    try:
                        payload = call_openai_vision(data_url=data_url, prompt=prompt, model=args.model, timeout=args.timeout)
                        repaired_text, math_lines, confidence, notes = normalize_repair_payload(payload, min_chars=args.min_chars, min_confidence=args.min_confidence)
                        temp_page = dict(page)
                        temp_page["production_safe_text"] = repaired_text
                        remaining_reasons = detect_math_precision_issues(temp_page, mode=args.audit_mode)
                        if remaining_reasons:
                            raise ValueError("math precision audit still suspicious after repair: " + "; ".join(remaining_reasons))
                        payload = {
                            "production_safe_text": repaired_text,
                            "math_lines": math_lines,
                            "confidence": confidence,
                            "notes": notes,
                        }
                        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt >= args.retries:
                            payload = None
                            break
                        wait = args.sleep * (2 ** attempt)
                        print(f"Page {page_number}: retrying math precision repair after error: {exc!r}; wait={wait:.1f}s")
                        time.sleep(wait)
                if payload is None:
                    err = f"math precision repair failed after retries: {last_error!r}"
                    mark_page_math_precision_failed(page, reasons=reasons_before, error=err)
                    counts["gated_failed"] += 1
                    errors.append(f"page {page_number}: {err}")
                    print(f"[{idx}/{len(candidates)}] MATH PRECISION FAILED page={page_number}, printed={printed}: {err}")
                    continue
            else:
                mark_page_math_precision_failed(page, reasons=reasons_before)
                counts["gated_failed"] += 1
                print(f"[{idx}/{len(candidates)}] MATH PRECISION GATED page={page_number}, printed={printed}: {'; '.join(reasons_before)}")
                continue

            try:
                repaired_text, math_lines, confidence, notes = normalize_repair_payload(payload, min_chars=args.min_chars, min_confidence=args.min_confidence)
                mark_page_repaired(page, repaired_text=repaired_text, math_lines=math_lines, confidence=confidence, notes=f"math_precision:{source}:{notes}", model=args.model)
                apply_deterministic_safe_math_fixes(page)
                remaining_reasons = detect_math_precision_issues(page, mode=args.audit_mode)
                if remaining_reasons:
                    mark_page_math_precision_failed(page, reasons=remaining_reasons, error="math precision issues remain after repair")
                    counts["gated_failed"] += 1
                    errors.append(f"page {page_number}: remaining math precision issues: {remaining_reasons}")
                    print(f"[{idx}/{len(candidates)}] MATH PRECISION FAILED page={page_number}, printed={printed}: {'; '.join(remaining_reasons)}")
                    continue
                mark_page_math_precision_approved(page, reasons_before=reasons_before)
                counts["approved"] += 1
                if source == "vision":
                    counts["vision_repaired"] += 1
                print(f"[{idx}/{len(candidates)}] math-approved page={page_number}, printed={printed}, chars={len(repaired_text)}, confidence={confidence}")
            except Exception as exc:
                err = str(exc)
                mark_page_math_precision_failed(page, reasons=reasons_before, error=err)
                counts["gated_failed"] += 1
                errors.append(f"page {page_number}: {err}")
                print(f"[{idx}/{len(candidates)}] MATH PRECISION FAILED page={page_number}, printed={printed}: {err}")
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    synced_legacy_fields = sync_ready_page_text_fields_from_production(pages)
    rebuild_chapters_and_sections(data)
    remaining = remaining_math_precision_blockers(pages, mode=args.audit_mode)
    empty_subsections = count_empty_production_subsections(data)
    write_csv(args.remaining_csv, remaining)
    summary = refresh_policy(
        data,
        counts=counts,
        initial_candidates=candidates,
        remaining=remaining,
        empty_subsections=empty_subsections,
        strict_complete=args.strict_complete,
        mode=args.audit_mode,
        errors=errors,
    )
    summary["legacy_text_fields_synced"] = synced_legacy_fields
    data.setdefault("extraction", {}).setdefault("quality_summary", {}).setdefault("math_precision_audit", {})["legacy_text_fields_synced"] = synced_legacy_fields

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Grade10_Maths Math Precision Audit Report",
        "=" * 60,
        f"Generated at UTC: {now_utc()}",
        f"Input JSON: {args.input}",
        f"Output JSON: {args.output}",
        f"PDF: {args.pdf}",
        f"Suspicious CSV: {args.suspicious_csv}",
        f"Remaining CSV: {args.remaining_csv}",
        f"Model: {args.model}",
        f"Scale: {args.scale}",
        f"Audit mode: {args.audit_mode}",
        "",
        f"Initial candidates: {summary.get('initial_candidates')}",
        f"Vision repaired pages: {summary.get('vision_repaired_pages')}",
        f"Manual override pages: {summary.get('manual_override_pages')}",
        f"Cached pages used: {summary.get('cached_pages_used')}",
        f"Legacy text fields synced: {summary.get('legacy_text_fields_synced')}",
        f"Gated failed pages: {summary.get('gated_failed_pages')}",
        "",
        f"Production status: {summary.get('gate_status')}",
        f"Lesson-body pages: {summary.get('lesson_body_pages')}",
        f"Excluded lesson-body pages: {summary.get('excluded_lesson_body_pages')}",
        f"Remaining math precision pages: {summary.get('remaining_math_precision_pages')}",
        f"Empty production subsections: {summary.get('empty_production_subsections')}",
        f"Ready pages: {summary.get('ready_for_production_embedding_pages')}",
        "",
        "Remaining reason counts:",
    ]
    for reason, count in sorted((summary.get("remaining_reason_counts") or {}).items()):
        lines.append(f"  - {reason}: {count}")
    if errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"  - {e}" for e in errors[:50])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")
    print(f"Wrote: {args.suspicious_csv}")
    print(f"Wrote: {args.remaining_csv}")

    if args.strict_complete and summary.get("gate_status") != "production_complete_ready":
        raise SystemExit(
            "Strict complete failed after math precision audit: "
            f"status={summary.get('gate_status')}, "
            f"excluded_lesson_body_pages={summary.get('excluded_lesson_body_pages')}, "
            f"remaining_math_precision_pages={summary.get('remaining_math_precision_pages')}, "
            f"empty_production_subsections={summary.get('empty_production_subsections')}. "
            f"See {args.report} and {args.remaining_csv}."
        )


if __name__ == "__main__":
    main()
