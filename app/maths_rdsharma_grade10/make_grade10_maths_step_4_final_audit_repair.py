#!/usr/bin/env python3
"""
Step 4: final production audit + repair for Grade10_Maths.

Why this exists:
- Step 2 gates obviously bad raw OCR pages.
- Step 3 repairs pages that were already excluded.
- Some pages can still falsely pass because they have enough text length and no obvious
  Devanagari, but contain residual OCR garbage such as "Wt ot oat oY...", "pte Hy tak...",
  "know1", "Outces", or stray symbols like «/¢.

This step audits every lesson-body page that is currently marked ready for embeddings.
Suspicious ready pages are either repaired with vision or re-gated so the final JSON cannot
claim production_complete_ready while still containing OCR garbage.
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
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

try:
    from dotenv import load_dotenv
except ImportError:  # .env support is optional
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from make_grade10_maths_step_2_production_gate import (  # noqa: E402
    AMBIGUOUS_GARBLED_RE,
    DEVANAGARI_RE,
    SAFE_NON_ASCII,
    clean_text,
    rebuild_chapters_and_sections,
)
from make_grade10_maths_step_3_vision_repair import (  # noqa: E402
    VISION_SAFE_NON_ASCII,
    call_openai_vision,
    load_manual_overrides,
    normalize_repair_payload,
    render_page_to_data_url,
    sanitize_vision_text,
    mark_page_repaired,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_final_audit_report.txt"
DEFAULT_SUSPICIOUS_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_final_audit_suspicious_pages.csv"
DEFAULT_REMAINING_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_final_audit_remaining_suspicious_pages.csv"


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


# Characters that are strong symptoms of bad OCR in production_safe_text.
# Keep this conservative. Math-safe symbols such as √, ×, ÷, ≤, ≥ are allowed elsewhere.
SUSPICIOUS_OCR_CHARS_RE = re.compile(r"[«»¥¢£™®§©¤¦¨¬¯´¸¿¡]")
REPLACEMENT_OR_MOJIBAKE_RE = re.compile(r"[�]|(?:Â|Ã|â€|â€™|â€œ|â€\u009d)")

# Hard garbage tokens seen in the false-ready output. Do NOT put broad two-letter
# tokens here (ot, pg, mg, wt, etc.) because math pages legitimately contain short
# labels/variables and that caused hundreds of unnecessary OpenAI calls.
GARBAGE_TOKEN_RE = re.compile(
    r"\b(?:"
    r"know1|outces|outcmes|subsef|subsef|sdbsciy|waal|eyt|scere|"
    r"thrugh|foem|frst|excericise|possbile|wilh|teo|pte|"
    r"Doo|Gey|Oot|Sst"
    r")\b",
    re.IGNORECASE,
)

BROKEN_ALPHA_DIGIT_RE = re.compile(r"\b[A-Za-z]{3,}\d[A-Za-z]*\b|\b[A-Za-z]*\d[A-Za-z]{3,}\b")
LONG_CONSONANT_WORD_RE = re.compile(r"\b[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]{7,}\b")
REPEATED_NOISE_RUN_RE = re.compile(r"([^\w\s])\1{5,}|([A-Za-z])\2{9,}")

# Lines like "Wt ot oat oY pg Mg BL" should be caught, but normal math lines with
# variables should not. This is intentionally stricter than v4.
SHORT_GIBBERISH_LINE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*[),.:;]?$")
COMMON_SMALL_WORDS = {
    "a", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
    "no", "of", "on", "or", "so", "to", "we", "x", "y", "z", "p", "q", "r", "n", "m",
}

MATHISH_LINE_RE = re.compile(
    r"[=+\-*/^√×÷<>≤≥≠∠⊥∥°]|\b(?:sin|cos|tan|cot|sec|cosec|sqrt|HCF|LCM|cm|m|km|fig|figure)\b",
    re.IGNORECASE,
)


def is_lesson_body_page(page: dict[str, Any]) -> bool:
    return page.get("content_type") == "lesson_body" and page.get("include_in_lesson_text") is True


def safe_text(page: dict[str, Any]) -> str:
    return str(page.get("production_safe_text") or page.get("text") or "")


def count_unexpected_unicode(text: str) -> int:
    safe = set(SAFE_NON_ASCII) | set(VISION_SAFE_NON_ASCII)
    return sum(1 for ch in text if ord(ch) > 127 and ch not in safe)


def looks_like_short_gibberish_line(line: str) -> bool:
    """Conservative detector for dense OCR-noise lines.

    v4 treated many valid math/theorem lines as suspicious. This version only flags
    a line when it is dominated by short random alphabetic tokens and has no math
    operators/formula context.
    """
    raw = line.strip()
    if len(raw) < 10 or len(raw) > 90:
        return False
    if MATHISH_LINE_RE.search(raw):
        return False
    if re.search(r"\b(?:CBSE|NCERT|EXAMPLE|SOLUTION|HINTS|BASIC|HOTS|LOTS|REMARK|THEOREM)\b", raw, re.IGNORECASE):
        return False
    tokens = re.findall(r"\S+", raw)
    if sum(1 for t in tokens if re.search(r"\d", t)) >= 3:
        return False
    if len(tokens) < 7:
        return False
    alpha_tokens = [re.sub(r"[^A-Za-z0-9]", "", t) for t in tokens if re.search(r"[A-Za-z]", t)]
    alpha_tokens = [t for t in alpha_tokens if t]
    if len(alpha_tokens) < 7:
        return False
    short_tokens = [t for t in alpha_tokens if len(t) <= 3]
    very_short = [t for t in alpha_tokens if len(t) <= 2]
    mixed_case = [t for t in alpha_tokens if not (t.islower() or t.isupper())]
    common = [t for t in alpha_tokens if t.lower() in COMMON_SMALL_WORDS]
    common_ratio = len(common) / max(len(alpha_tokens), 1)
    if common_ratio >= 0.30:
        return False
    avg_len = sum(len(t) for t in alpha_tokens) / max(len(alpha_tokens), 1)
    # Example caught: "Wt ot oat oY pg Mg BL". Valid prose usually has longer words;
    # valid math lines usually hit MATHISH_LINE_RE above.
    if avg_len <= 3.2 and len(short_tokens) / len(alpha_tokens) >= 0.85 and common_ratio <= 0.25:
        return True
    if avg_len <= 3.0 and len(very_short) >= 6 and len(mixed_case) >= 2:
        return True
    return False

def audit_text_quality(text: str, *, min_chars: int = 80, mode: str = "fast") -> list[str]:
    """Return reasons showing why production_safe_text is not safe for DB embeddings.

    mode="fast" is the default production mode. It catches high-confidence OCR garbage
    without sending hundreds of valid math pages back to OpenAI.

    mode="strict" is available for a separate slow QA pass, but it may over-flag dense
    equation pages and should not be your default full-book run.
    """
    audit_mode = (mode or "fast").strip().lower()
    reasons: list[str] = []
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        reasons.append(f"final_audit_too_short:{len(stripped)}")

    if DEVANAGARI_RE.search(stripped):
        reasons.append("final_audit_devanagari_or_hindi_ocr_garbage")
    if AMBIGUOUS_GARBLED_RE.search(stripped):
        reasons.append("final_audit_ambiguous_ocr_garbage")
    suspicious_symbols = SUSPICIOUS_OCR_CHARS_RE.findall(stripped)
    # A single odd symbol is common in scanned math text. Treat it as a blocker only
    # when it is repeated enough to indicate OCR noise. Dense gibberish/known-token
    # rules below will still catch pages like "Wt ot oat oY pg Mg BL".
    if len(suspicious_symbols) >= 6:
        reasons.append(f"final_audit_suspicious_ocr_symbols:{len(suspicious_symbols)}")
    if REPLACEMENT_OR_MOJIBAKE_RE.search(stripped):
        reasons.append("final_audit_replacement_or_mojibake")
    if GARBAGE_TOKEN_RE.search(stripped):
        reasons.append("final_audit_known_garbage_tokens")
    repeated_noise = REPEATED_NOISE_RUN_RE.search(stripped)
    if repeated_noise and not set(repeated_noise.group(0)) <= {".", "-"}:
        reasons.append("final_audit_repeated_noise_run")

    unexpected_unicode = count_unexpected_unicode(stripped)
    # Higher threshold than Step 3: clean math pages may contain legitimate unicode.
    if unexpected_unicode >= 20 or unexpected_unicode / max(len(stripped), 1) > 0.006:
        reasons.append(f"final_audit_unexpected_unicode:{unexpected_unicode}")

    dense_gibberish_lines = [line for line in stripped.splitlines() if looks_like_short_gibberish_line(line)]
    if dense_gibberish_lines:
        reasons.append(f"final_audit_dense_gibberish_lines:{len(dense_gibberish_lines)}")

    alpha_digit = BROKEN_ALPHA_DIGIT_RE.findall(stripped)
    suspicious_alpha_digit = [t for t in alpha_digit if not is_math_alpha_digit_token(t)]
    # In fast mode, alpha-digit tokens are usually legitimate linearized math
    # (cos2x, sin2A, tan30, sec2theta). Block only repeated non-math tokens.
    if audit_mode == "strict":
        if len(suspicious_alpha_digit) >= 6 or any(t.lower() in {"know1", "outc0mes", "poss1ble"} for t in alpha_digit):
            reasons.append(f"final_audit_broken_alpha_digit_tokens:{len(suspicious_alpha_digit)}")
    elif len(suspicious_alpha_digit) >= 12 or any(t.lower() in {"know1", "outc0mes", "poss1ble"} for t in alpha_digit):
        reasons.append(f"final_audit_broken_alpha_digit_tokens:{len(suspicious_alpha_digit)}")

    if audit_mode == "strict":
        long_cons = LONG_CONSONANT_WORD_RE.findall(stripped)
        if len(long_cons) >= 6:
            reasons.append(f"final_audit_consonant_garbage_words:{len(long_cons)}")
        words = re.findall(r"\b[A-Za-z]{3,}\b", stripped)
        if len(words) >= 80 and not MATHISH_LINE_RE.search(stripped):
            vowel_words = [w for w in words if re.search(r"[aeiouAEIOU]", w)]
            if len(vowel_words) / max(len(words), 1) < 0.50:
                reasons.append("final_audit_low_vowel_word_ratio")
    return sorted(set(reasons))


LOW_RISK_FINAL_AUDIT_REASON_PREFIXES = (
    "final_audit_dense_gibberish_lines:",
    "final_audit_broken_alpha_digit_tokens:",
)

HIGH_RISK_FINAL_AUDIT_REASON_PREFIXES = (
    "final_audit_too_short:",
    "final_audit_suspicious_ocr_symbols:",
    "final_audit_unexpected_unicode:",
)

HIGH_RISK_FINAL_AUDIT_REASONS = {
    "final_audit_devanagari_or_hindi_ocr_garbage",
    "final_audit_ambiguous_ocr_garbage",
    "final_audit_known_garbage_tokens",
    "final_audit_replacement_or_mojibake",
    "final_audit_repeated_noise_run",
}


def is_low_risk_post_repair_reasons(reasons: list[str]) -> bool:
    """Return True when final audit warnings are likely math/layout artifacts.

    After an OpenAI vision repair has already produced a long clean page, a single
    dense formula/table line or normal trig token such as cos2x/sin2A should not
    exclude the whole page. These reasons are kept as warnings, not blockers.
    """
    if not reasons:
        return True
    for reason in reasons:
        if reason in HIGH_RISK_FINAL_AUDIT_REASONS:
            return False
        if any(reason.startswith(prefix) for prefix in HIGH_RISK_FINAL_AUDIT_REASON_PREFIXES):
            return False
        if not any(reason.startswith(prefix) for prefix in LOW_RISK_FINAL_AUDIT_REASON_PREFIXES):
            return False
    return True


def is_math_alpha_digit_token(token: str) -> bool:
    token_l = token.lower()
    # Common OCR/linearized math tokens that are legitimate in Class 10 maths.
    if re.search(r"(sin|cos|tan|cot|sec|cosec|sqrt|log)", token_l):
        return True
    if re.fullmatch(r"[a-z]{1,3}\d+[a-z]{0,2}", token_l):
        return True
    if re.fullmatch(r"\d+[a-z]{1,3}", token_l):
        return True
    if re.search(r"(fig|example|ex|q|eq|ncert|cbse)\d+", token_l):
        return True
    return False

def audit_page(page: dict[str, Any], *, audit_mode: str = "fast") -> list[str]:
    if not is_lesson_body_page(page):
        return []
    if page.get("include_in_embeddings") is True and page.get("final_audit_status") in {"approved", "approved_with_warnings"}:
        return []
    if page.get("include_in_embeddings") is not True:
        return list(page.get("production_exclusion_reasons") or ["final_audit_already_excluded"])
    if page.get("embedding_readiness") != "ready_for_production_embedding":
        return list(page.get("production_exclusion_reasons") or ["final_audit_not_ready_status"])
    return audit_text_quality(safe_text(page), min_chars=80, mode=audit_mode)


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
        "notes": override.get("notes") or "manual_page_override_reviewed_final_audit",
    }


def build_final_audit_prompt(page: dict[str, Any], reasons: list[str]) -> str:
    printed = page.get("printed_page_number")
    chapter = page.get("chapter_title") or ""
    section = page.get("section_title") or ""
    previous = safe_text(page)[:3500]
    return f"""
You are doing FINAL production OCR audit for a scanned Class 10 mathematics textbook page.

The current transcription was rejected by the final audit for these reasons:
{'; '.join(reasons)}

Task:
- Transcribe the page image into clean production-safe text for database embeddings.
- Preserve textbook text, examples, exercise questions, theorem/proof/solution labels, and important formula lines.
- Preserve mathematical meaning using plain text: x^2 for powers, (a)/(b) for fractions where needed, sqrt(...) or √ for roots, and normal math symbols.
- Remove OCR garbage tokens, random short-letter runs, stray Hindi/Devanagari glyphs, page-border noise, duplicated divider noise, and scanner artifacts.
- Do not summarize. Do not solve questions. Do not add content not visible on the page.
- If a tiny part is truly unreadable, use [unreadable] only for that tiny part.

Context:
PDF page number: {page.get('page_number')}
Printed page number: {printed}
Chapter: {chapter}
Section: {section}

Rejected transcription, only for hints. Trust the image over this text:
---
{previous}
---

Return JSON only:
{{
  "production_safe_text": "full corrected page transcription",
  "math_lines": ["important equations/formulas from the page"],
  "confidence": 0.95,
  "notes": "brief final audit note"
}}
""".strip()


def mark_page_excluded_by_final_audit(page: dict[str, Any], *, reasons: list[str], error: str | None = None) -> None:
    page["include_in_embeddings"] = False
    page["embedding_readiness"] = "needs_final_qa_before_production_embedding"
    page["production_safe_text"] = ""
    merged_reasons = set(page.get("production_exclusion_reasons") or [])
    merged_reasons.update(reasons or ["final_audit_requires_repair"])
    if error:
        merged_reasons.add("final_audit_repair_failed")
        page["final_audit_error"] = error
    page["production_exclusion_reasons"] = sorted(merged_reasons)
    flags = set(page.get("quality_flags") or [])
    flags.add("production_embedding_excluded_until_final_audit_repair")
    flags.discard("production_embedding_ready")
    page["quality_flags"] = sorted(flags)
    page["final_audit_status"] = "failed" if error else "suspicious_gated"
    page["final_audit_reasons"] = sorted(set(reasons))


def mark_page_final_approved(page: dict[str, Any], *, reasons_before: list[str], warning_reasons: list[str] | None = None) -> None:
    warning_reasons = sorted(set(warning_reasons or []))
    page["final_audit_status"] = "approved_with_warnings" if warning_reasons else "approved"
    page["final_audit_reasons_before_repair"] = sorted(set(reasons_before))
    if warning_reasons:
        page["final_audit_warning_reasons_after_repair"] = warning_reasons
    flags = set(page.get("quality_flags") or [])
    flags.add("final_production_audit_passed")
    if warning_reasons:
        flags.add("final_production_audit_passed_with_low_risk_warnings")
    flags.discard("production_embedding_excluded_until_final_audit_repair")
    page["quality_flags"] = sorted(flags)


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page_number",
                "printed_page_number",
                "chapter_title",
                "candidate_kind",
                "reasons",
                "embedding_readiness",
                "sample_text",
            ],
        )
        writer.writeheader()
        for row in rows:
            p = row["page"]
            sample = re.sub(r"\s+", " ", safe_text(p)[:400])
            writer.writerow({
                "page_number": p.get("page_number"),
                "printed_page_number": p.get("printed_page_number"),
                "chapter_title": p.get("chapter_title"),
                "candidate_kind": row.get("candidate_kind"),
                "reasons": "; ".join(row.get("reasons") or []),
                "embedding_readiness": p.get("embedding_readiness"),
                "sample_text": sample,
            })


def collect_candidates(pages: list[dict[str, Any]], *, selected_pages: set[int] | None, audit_mode: str = "fast") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in pages:
        if not is_lesson_body_page(p):
            continue
        page_number = int(p.get("page_number") or 0)
        if selected_pages is not None and page_number not in selected_pages:
            continue
        if p.get("include_in_embeddings") is not True:
            rows.append({
                "page": p,
                "candidate_kind": "excluded_lesson_body",
                "reasons": list(p.get("production_exclusion_reasons") or ["already_excluded_before_final_audit"]),
            })
            continue
        reasons = audit_page(p, audit_mode=audit_mode)
        if reasons:
            rows.append({"page": p, "candidate_kind": "suspicious_ready_page", "reasons": reasons})
    return sorted(rows, key=lambda r: int(r["page"].get("page_number") or 0))


def count_empty_production_subsections(data: dict[str, Any]) -> list[dict[str, Any]]:
    empty: list[dict[str, Any]] = []
    for chapter in data.get("extraction", {}).get("chapters", []) or []:
        for sub in chapter.get("subsections", []) or []:
            # Count only real lesson/day subsections that have a page range.
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


def refresh_final_policy(
    data: dict[str, Any],
    *,
    strict_complete: bool,
    counts: Counter,
    initial_candidates: list[dict[str, Any]],
    errors: list[str],
    audit_mode: str = "fast",
) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    pages = extraction.get("page_extractions", []) or []
    lesson_body_pages = [p for p in pages if is_lesson_body_page(p)]
    excluded_pages = [p for p in pages if p.get("include_in_embeddings") is not True]
    excluded_lesson_body_pages = [p for p in lesson_body_pages if p.get("include_in_embeddings") is not True]
    ready_pages = [p for p in pages if p.get("include_in_embeddings") is True]
    remaining_suspicious_ready = [p for p in lesson_body_pages if p.get("include_in_embeddings") is True and audit_page(p, audit_mode=audit_mode)]
    empty_subsections = count_empty_production_subsections(data)

    previous_policy = extraction.get("production_embedding_policy") or {}
    run_scope_blockers = previous_policy.get("run_scope_blockers") or []

    if run_scope_blockers:
        gate_status = "smoke_or_partial_not_production_ready"
    elif excluded_lesson_body_pages or remaining_suspicious_ready or empty_subsections:
        gate_status = "production_safe_gated_needs_qa"
    else:
        gate_status = "production_complete_ready"

    reason_counts = Counter()
    for p in excluded_pages:
        reason_counts.update(p.get("production_exclusion_reasons") or [])
    for p in remaining_suspicious_ready:
        reason_counts.update(audit_page(p, audit_mode=audit_mode))

    extraction["production_embedding_policy"] = {
        **previous_policy,
        "status": gate_status,
        "meaning": "Use production_safe_text only where include_in_embeddings=true, embedding_readiness=ready_for_production_embedding, and final_production_audit_passed is present for lesson-body pages.",
        "recommended_embedding_text_field": "production_safe_text",
        "embed_only_when": {"include_in_embeddings": True, "embedding_readiness": "ready_for_production_embedding"},
        "do_not_embed_when_flags_include": sorted(set((previous_policy.get("do_not_embed_when_flags_include") or []) + [
            "production_embedding_excluded_until_final_audit_repair",
            "production_embedding_excluded_until_vision_qa",
            "dense_formula_layout_requires_vision_or_mathpix_qa",
            "remaining_garbled_unicode_requires_vision_qa",
            "remaining_devanagari_or_hindi_ocr_garbage_requires_vision_qa",
        ])),
        "run_scope_blockers": run_scope_blockers,
        "strict_complete": bool(strict_complete),
        "production_complete": gate_status == "production_complete_ready",
        "final_production_audit_applied": True,
        "final_audit_remaining_suspicious_ready_pages": len(remaining_suspicious_ready),
        "final_audit_empty_production_subsections": len(empty_subsections),
        "final_audit_errors": errors[:50],
    }

    qs = extraction.setdefault("quality_summary", {})
    qs["final_production_audit"] = {
        "generated_at_utc": now_utc(),
        "audit_mode": audit_mode,
        "gate_status": gate_status,
        "initial_candidates": len(initial_candidates),
        "initial_suspicious_ready_pages": sum(1 for r in initial_candidates if r.get("candidate_kind") == "suspicious_ready_page"),
        "initial_excluded_lesson_body_pages": sum(1 for r in initial_candidates if r.get("candidate_kind") == "excluded_lesson_body"),
        "vision_repaired_pages": int(counts.get("vision_repaired", 0)),
        "manual_override_pages": int(counts.get("manual_override", 0)),
        "cached_pages_used": int(counts.get("cached", 0)),
        "gated_failed_pages": int(counts.get("gated_failed", 0)),
        "approved_with_low_risk_warnings": int(counts.get("approved_with_warnings", 0)),
        "audit_only_gated_pages": int(counts.get("audit_only_gated", 0)),
        "total_pages_in_json": len(pages),
        "ready_for_production_embedding_pages": len(ready_pages),
        "excluded_until_qa_pages": len(excluded_pages),
        "lesson_body_pages": len(lesson_body_pages),
        "excluded_lesson_body_pages": len(excluded_lesson_body_pages),
        "remaining_suspicious_ready_pages": len(remaining_suspicious_ready),
        "empty_production_subsections": len(empty_subsections),
        "exclusion_or_suspicion_reason_counts": dict(reason_counts),
        "empty_subsection_samples": empty_subsections[:30],
    }
    extraction["generated_at_utc"] = now_utc()
    return qs["final_production_audit"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Final production audit/repair for Grade10_Maths JSON.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--suspicious-csv", type=Path, default=DEFAULT_SUSPICIOUS_CSV)
    parser.add_argument("--remaining-suspicious-csv", type=Path, default=DEFAULT_REMAINING_CSV)
    parser.add_argument("--model", default=os.environ.get("GRADE10_MATHS_VISION_MODEL", "gpt-4o-mini"))
    parser.add_argument("--scale", type=float, default=env_float("GRADE10_MATHS_VISION_SCALE", 2.5))
    parser.add_argument("--pages", default=None, help="Optional PDF pages to final-audit/repair, e.g. 52,103,671,750")
    parser.add_argument("--max-pages", type=int, default=env_int("GRADE10_MATHS_FINAL_AUDIT_MAX_ITEMS", 0), help="Optional cap for testing final repair; 0 means all.")
    parser.add_argument("--audit-mode", choices=["fast", "strict"], default=os.environ.get("GRADE10_MATHS_FINAL_AUDIT_MODE", "fast").strip().lower() or "fast", help="Final audit sensitivity. fast is recommended; strict may over-flag dense math pages.")
    parser.add_argument("--repair-with-vision", action="store_true", help="Repair suspicious/excluded lesson pages with OpenAI vision. If omitted, suspicious ready pages are gated out.")
    parser.add_argument("--force-vision", action="store_true", help="Ignore final-audit vision cache.")
    parser.add_argument("--manual-overrides", type=Path, default=Path(os.environ.get("GRADE10_MATHS_PAGE_OVERRIDES", "")) if os.environ.get("GRADE10_MATHS_PAGE_OVERRIDES") else None)
    parser.add_argument("--min-confidence", type=float, default=env_float("GRADE10_MATHS_AUTO_REVIEW_THRESHOLD", 0.90))
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--strict-complete", action="store_true", help="Fail unless final JSON has zero excluded/suspicious lesson pages and zero empty production subsections.")
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
    candidates = collect_candidates(pages, selected_pages=selected_pages, audit_mode=args.audit_mode)
    candidates = sorted(candidates, key=lambda r: int(r["page"].get("page_number") or 0))
    if args.max_pages and args.max_pages > 0:
        candidates = candidates[: args.max_pages]

    write_audit_csv(args.suspicious_csv, candidates)

    manual_overrides = load_manual_overrides(args.manual_overrides)
    prompt_version = "grade10_maths_final_audit_v1_residual_ocr_garbage"
    cache_key = stable_cache_key(args.pdf, model=args.model, scale=args.scale, prompt_version=prompt_version)
    cache_dir = args.output.parent / ".vision_qa_cache" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    errors: list[str] = []

    print(f"Final audit candidates: {len(candidates)}")
    print(f"Suspicious CSV: {args.suspicious_csv}")
    print(f"Cache directory: {cache_dir}")
    if args.manual_overrides:
        print(f"Manual overrides: {args.manual_overrides} ({len(manual_overrides)} keys loaded)")
    print(f"Repair with vision: {args.repair_with_vision}")
    print(f"Audit mode: {args.audit_mode}")

    with fitz.open(args.pdf) as pdf:
        for i, row in enumerate(candidates, start=1):
            page = row["page"]
            page_number = int(page.get("page_number") or 0)
            reasons = list(row.get("reasons") or [])
            cache_file = cache_dir / f"page_{page_number:04d}.json"
            try:
                override = get_manual_override(page, manual_overrides)
                payload = None
                if override is not None:
                    payload = override_to_payload(override)
                    counts["manual_override"] += 1
                elif cache_file.exists() and not args.force_vision:
                    payload = json.loads(cache_file.read_text(encoding="utf-8"))
                    counts["cached"] += 1
                elif args.repair_with_vision:
                    data_url = render_page_to_data_url(pdf, page_number, scale=args.scale)
                    prompt = build_final_audit_prompt(page, reasons)
                    last_error: Exception | None = None
                    repaired_tuple = None
                    for attempt in range(args.retries + 1):
                        try:
                            retry_suffix = ""
                            if attempt > 0 and last_error is not None:
                                retry_suffix = (
                                    "\n\nYour previous response failed final production validation: "
                                    f"{last_error}. Return corrected JSON only. Remove OCR garbage."
                                )
                            payload = call_openai_vision(data_url=data_url, prompt=prompt + retry_suffix, model=args.model, timeout=args.timeout)
                            repaired_tuple = normalize_repair_payload(
                                payload,
                                min_chars=args.min_chars,
                                min_confidence=args.min_confidence,
                            )
                            repaired_text = repaired_tuple[0]
                            final_reasons = audit_text_quality(repaired_text, min_chars=args.min_chars, mode=args.audit_mode)
                            if final_reasons and not is_low_risk_post_repair_reasons(final_reasons):
                                raise ValueError("final audit still suspicious after repair: " + "; ".join(final_reasons))
                            break
                        except Exception as exc:
                            last_error = exc
                            if attempt < args.retries:
                                wait = max(args.sleep, 1.0) * (attempt + 1)
                                print(f"Page {page_number}: retrying final repair after error: {exc!r}; wait={wait}s")
                                time.sleep(wait)
                    if payload is None or repaired_tuple is None:
                        raise RuntimeError(f"final audit vision repair failed after retries: {last_error!r}")
                    payload["_cache_metadata"] = {
                        "pdf_page_number": page_number,
                        "printed_page_number": page.get("printed_page_number"),
                        "model": args.model,
                        "scale": args.scale,
                        "prompt_version": prompt_version,
                        "generated_at_utc": now_utc(),
                        "final_audit_reasons": reasons,
                    }
                    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    mark_page_excluded_by_final_audit(page, reasons=reasons)
                    counts["audit_only_gated"] += 1
                    print(f"[{i}/{len(candidates)}] gated page={page_number}, printed={page.get('printed_page_number')}, reasons={'; '.join(reasons[:3])}")
                    continue

                repaired_text, math_lines, confidence, notes = normalize_repair_payload(
                    payload,
                    min_chars=args.min_chars,
                    min_confidence=args.min_confidence,
                )
                final_reasons = audit_text_quality(repaired_text, min_chars=args.min_chars, mode=args.audit_mode)
                if final_reasons and not is_low_risk_post_repair_reasons(final_reasons):
                    raise ValueError("final audit still suspicious after repair: " + "; ".join(final_reasons))
                mark_page_repaired(page, repaired_text=repaired_text, math_lines=math_lines, confidence=confidence, notes=f"{notes} | final_audit_repaired", model=args.model)
                mark_page_final_approved(page, reasons_before=reasons, warning_reasons=final_reasons)
                if final_reasons:
                    counts["approved_with_warnings"] += 1
                counts["vision_repaired"] += 1
                status_label = "final-approved-with-warnings" if final_reasons else "final-approved"
                print(f"[{i}/{len(candidates)}] {status_label} page={page_number}, printed={page.get('printed_page_number')}, chars={len(repaired_text)}, confidence={confidence}")
                if args.sleep:
                    time.sleep(args.sleep)
            except Exception as exc:
                counts["gated_failed"] += 1
                err = f"page={page_number}, printed={page.get('printed_page_number')}: {exc}"
                errors.append(err)
                mark_page_excluded_by_final_audit(page, reasons=reasons, error=str(exc))
                print(f"[{i}/{len(candidates)}] FINAL AUDIT FAILED {err}", file=sys.stderr)

    # Mark all non-candidate ready lesson pages as audited. This makes the DB ingestion rule explicit.
    for p in pages:
        if is_lesson_body_page(p) and p.get("include_in_embeddings") is True and not audit_page(p, audit_mode=args.audit_mode):
            mark_page_final_approved(p, reasons_before=[])

    rebuild_chapters_and_sections(data)
    summary = refresh_final_policy(
        data,
        strict_complete=args.strict_complete,
        counts=counts,
        initial_candidates=candidates,
        errors=errors,
        audit_mode=args.audit_mode,
    )

    remaining_rows: list[dict[str, Any]] = []
    for p in pages:
        if not is_lesson_body_page(p):
            continue
        reasons = audit_page(p, audit_mode=args.audit_mode)
        if p.get("include_in_embeddings") is not True or reasons:
            remaining_rows.append({
                "page": p,
                "candidate_kind": "remaining_excluded_or_suspicious",
                "reasons": reasons or list(p.get("production_exclusion_reasons") or []),
            })
    write_audit_csv(args.remaining_suspicious_csv, remaining_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "Grade10_Maths Final Production Audit Report",
        "=" * 60,
        f"Generated at UTC: {now_utc()}",
        f"Input JSON: {args.input}",
        f"Output JSON: {args.output}",
        f"PDF: {args.pdf}",
        f"Suspicious CSV: {args.suspicious_csv}",
        f"Remaining suspicious CSV: {args.remaining_suspicious_csv}",
        f"Model: {args.model}",
        f"Scale: {args.scale}",
        f"Audit mode: {args.audit_mode}",
        "",
        f"Initial candidates: {summary.get('initial_candidates')}",
        f"Initial suspicious ready pages: {summary.get('initial_suspicious_ready_pages')}",
        f"Initial excluded lesson-body pages: {summary.get('initial_excluded_lesson_body_pages')}",
        f"Vision repaired/final-approved pages: {summary.get('vision_repaired_pages')}",
        f"Manual override pages: {summary.get('manual_override_pages')}",
        f"Cached pages used: {summary.get('cached_pages_used')}",
        f"Gated failed pages: {summary.get('gated_failed_pages')}",
        f"Approved with low-risk warnings: {summary.get('approved_with_low_risk_warnings')}",
        "",
        f"Production status: {summary.get('gate_status')}",
        f"Lesson-body pages: {summary.get('lesson_body_pages')}",
        f"Excluded lesson-body pages: {summary.get('excluded_lesson_body_pages')}",
        f"Remaining suspicious ready pages: {summary.get('remaining_suspicious_ready_pages')}",
        f"Empty production subsections: {summary.get('empty_production_subsections')}",
        f"Ready pages: {summary.get('ready_for_production_embedding_pages')}",
        f"Excluded pages total: {summary.get('excluded_until_qa_pages')}",
        "",
        "Reason counts:",
    ]
    for k, v in Counter(summary.get("exclusion_or_suspicion_reason_counts") or {}).most_common():
        report_lines.append(f"  - {k}: {v}")
    if errors:
        report_lines.append("")
        report_lines.append("Errors:")
        report_lines.extend(f"  - {e}" for e in errors[:100])
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if args.strict_complete and summary.get("gate_status") != "production_complete_ready":
        raise SystemExit(
            "Strict complete failed after final audit: "
            f"status={summary.get('gate_status')}, "
            f"excluded_lesson_body_pages={summary.get('excluded_lesson_body_pages')}, "
            f"remaining_suspicious_ready_pages={summary.get('remaining_suspicious_ready_pages')}, "
            f"empty_production_subsections={summary.get('empty_production_subsections')}. "
            f"See {args.report} and {args.remaining_suspicious_csv}."
        )

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")
    print(f"Wrote: {args.suspicious_csv}")
    print(f"Wrote: {args.remaining_suspicious_csv}")


if __name__ == "__main__":
    main()
