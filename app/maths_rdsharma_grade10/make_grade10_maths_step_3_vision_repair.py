#!/usr/bin/env python3
"""
Step 3: vision/Math-aware QA repair for Grade10_Maths production JSON.

Why this exists:
- Step 1 uses deterministic rendered-page OCR.
- Step 2 is intentionally conservative and excludes pages where Tesseract OCR is not
  reliable enough for production embeddings.
- Step 3 re-extracts only those excluded lesson-body pages from page images using a
  vision-capable model, then rebuilds chapter/day production text.

This script is resumable. It caches each repaired page in:
  <output-dir>/.vision_qa_cache/<cache-key>/page_XXXX.json

Required:
  pip install openai pymupdf
  set OPENAI_API_KEY=<your key>
  set GRADE10_MATHS_VISION_MODEL=gpt-4o-mini

Example:
  python app/maths_rdsharma_grade10/make_grade10_maths_step_3_vision_repair.py ^
    --pdf input/Grade10_Maths.pdf ^
    --input output/maths_rdsharma_grade10/Grade10_Maths_production_ready.json ^
    --output output/maths_rdsharma_grade10/Grade10_Maths_production_ready.json ^
    --report output/maths_rdsharma_grade10/Grade10_Maths_vision_qa_report.txt ^
    --strict-complete
"""
from __future__ import annotations

import argparse
import base64
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


# Reuse Step 2's conservative text cleanup and chapter/day rebuild logic.
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


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_vision_qa_report.txt"
DEFAULT_REMAINING_QA_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_remaining_pages_requiring_vision_qa.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def render_page_to_data_url(pdf: fitz.Document, page_number: int, *, scale: float) -> str:
    page = pdf.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img_bytes = pix.tobytes("png")
    encoded = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_prompt(page: dict[str, Any]) -> str:
    printed = page.get("printed_page_number")
    chapter = page.get("chapter_title") or ""
    section = page.get("section_title") or ""
    original_ocr = (page.get("text") or page.get("ocr_text") or "")[:2500]
    reasons = "; ".join(page.get("production_exclusion_reasons") or [])
    return f"""
You are doing production OCR QA for a scanned Class 10 mathematics textbook page.

Task:
- Transcribe the page image into clean production-safe text.
- Preserve all mathematical expressions in readable text.
- Do not solve problems, do not summarize, do not add explanations.
- Do not invent missing text. If something is unreadable, write [unreadable].
- Remove OCR garbage such as random Hindi glyphs, broken symbols, or duplicated divider noise.
- Preserve section headings, examples, exercise questions, theorem/proof/solution labels, and figure captions.
- Ignore only repetitive running headers/footers when they do not add lesson content.
- Represent powers as x^2, x^3, etc.
- Represent fractions as (numerator)/(denominator) where needed.
- Represent square roots as sqrt(...), angles as angle ABC, degrees as 30°.
- For diagrams, include a short line like [diagram: description of what is shown] if the diagram matters.

Context:
PDF page number: {page.get("page_number")}
Printed page number: {printed}
Chapter: {chapter}
Section: {section}
Current exclusion reasons: {reasons}

Low-quality OCR reference, only for hints. Trust the image over this OCR:
---
{original_ocr}
---

Return JSON only, with this exact shape:
{{
  "production_safe_text": "full corrected page transcription",
  "math_lines": ["important equations/formulas from the page"],
  "confidence": 0.95,
  "notes": "brief QA note"
}}

Confidence rules:
- Use 0.95 when the page is clearly readable and the transcription is production safe.
- Use 0.90 when mostly readable with minor uncertainty.
- Use below 0.90 only when important content is unreadable or missing.
- Never copy the example blindly; choose the confidence based on the image.
""".strip()


def extract_text_from_response(resp: Any) -> str:
    # Responses API object usually exposes output_text.
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Fallback for chat.completions response.
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        pass
    # Last resort for dict-like objects.
    if isinstance(resp, dict):
        if resp.get("output_text"):
            return str(resp["output_text"])
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return str(resp)
    return str(resp)


def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai_vision(*, data_url: str, prompt: str, model: str, timeout: float) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc

    # OPENAI_API_KEY is used by the OpenAI SDK. All Maths-specific LLM settings use GRADE10_MATHS_* env names.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for vision repair.")
    client = OpenAI(timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a production OCR QA reviewer for a scanned Grade 10 mathematics textbook. "
                    "Transcribe only printed textbook content visible on the page image. Ignore handwriting, stamps, page borders, scanner noise, and decorative marks. "
                    "Preserve math meaning in plain text using ^ for exponents, / for fractions, × for multiplication, ÷ for division, √ for radicals, and standard symbols such as ≤, ≥, ≠. "
                    "Do not solve problems or add explanations. Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return parse_model_json(content)

def is_lesson_body_page(page: dict[str, Any]) -> bool:
    return page.get("content_type") == "lesson_body" and page.get("include_in_lesson_text") is True


def should_repair_page(page: dict[str, Any], *, repair_front_matter: bool) -> bool:
    if page.get("include_in_embeddings") is True and page.get("embedding_readiness") == "ready_for_production_embedding":
        return False
    if is_lesson_body_page(page):
        return True
    return bool(repair_front_matter)



LOW_CONFIDENCE_SCHEMA_COPY_VALUES = {0.0, 0, 0.00}
REFUSAL_OR_SUMMARY_RE = re.compile(
    r"\b(?:i\s+can(?:not|'t)|unable\s+to|cannot\s+transcribe|sorry|summary\s+of\s+the\s+page)\b",
    re.I,
)
REPEATED_NOISE_RE = re.compile(r"([A-Za-z₹°²³√×÷≤≥≠−–—'\"πθαβγ∆Δ∠⊥∥∴±∞∑=+\-*/^])\1{14,}")

# Extra symbols that are valid in Grade 10 maths OCR after vision repair.
# Step 2 is intentionally strict for raw Tesseract OCR, but Step 3 should not reject
# a good vision transcription just because it contains legitimate math symbols.
MATH_SAFE_EXTRA = set(
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    "αβγδεζηθικλμνξοπρστυφχψω"
    "∅∈∉∋∌⊂⊃⊆⊇∪∩∧∨¬⇒⇔→←↔↑↓"
    "≈≃≅≡∝∫∂∇∵∷∎□△▵◁▷○●◦·⋅•"
    "∟⊕⊗⊙∣∤∥∦⊥∡∢⊾⊿⌒"
    "¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞"
)
VISION_SAFE_NON_ASCII = set(SAFE_NON_ASCII) | MATH_SAFE_EXTRA

UNICODE_ASCII_REPLACEMENTS = {
    "\u00a0": " ",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "…": "...",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "⁄": "/",
    "∕": "/",
    "∗": "*",
    "∙": "*",
    "⋅": "*",
    "·": "*",
    " ": " ",
}
SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "⁺": "^+", "⁻": "^-", "⁽": "^(", "⁾": "^)", "ⁿ": "^n",
})
SUBSCRIPT_MAP = str.maketrans({
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
    "₊": "_+", "₋": "_-", "₍": "_(", "₎": "_)",
})


def sanitize_vision_text(text: str) -> tuple[str, list[str]]:
    """Normalize vision text without changing mathematical meaning.

    This is better than hardcoding page text in Python. The model can return otherwise
    correct transcription with curly quotes, invisible characters, or uncommon math glyphs.
    We normalize those deterministically, collapse divider noise, and keep legitimate math
    symbols. Devanagari/Hindi OCR garbage is still rejected later.
    """
    notes: list[str] = []
    if not isinstance(text, str):
        return "", ["non_string_text_coerced"]
    before = text
    for src, dst in UNICODE_ASCII_REPLACEMENTS.items():
        if src in text:
            text = text.replace(src, dst)
            notes.append("unicode_ascii_replacements")
    text2 = text.translate(SUPERSCRIPT_MAP).translate(SUBSCRIPT_MAP)
    if text2 != text:
        notes.append("super_subscript_normalized")
    text = text2
    # Remove combining marks and formatting controls that sometimes come from OCR.
    chars: list[str] = []
    removed = 0
    for ch in unicodedata.normalize("NFKC", text):
        cat = unicodedata.category(ch)
        if cat in {"Mn", "Cf", "Cc"} and ch not in {"\n", "\t"}:
            removed += 1
            continue
        chars.append(ch)
    text = "".join(chars)
    if removed:
        notes.append(f"removed_combining_or_control_chars:{removed}")
    # Collapse long repeated divider/noise runs instead of failing the whole page.
    def _collapse_repeat(match: re.Match[str]) -> str:
        ch = match.group(1)
        return ch * 3
    text2 = REPEATED_NOISE_RE.sub(_collapse_repeat, text)
    if text2 != text:
        notes.append("collapsed_repeated_noise_runs")
    text = text2
    # Drop isolated decorative symbols that are not useful for embeddings.
    cleaned: list[str] = []
    dropped = 0
    for ch in text:
        if ord(ch) <= 127 or ch in VISION_SAFE_NON_ASCII or ch in "\n\t":
            cleaned.append(ch)
            continue
        # Keep broad mathematical/operator unicode symbols; they are better than losing formula meaning.
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, "")
        if cat.startswith("S") and any(key in name for key in ("MATHEMATICAL", "ANGLE", "TRIANGLE", "CIRCLE", "ARROW", "ROOT", "INTEGRAL")):
            cleaned.append(ch)
            continue
        # For letters with accents, use the ASCII base form.
        decomp = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        if decomp:
            cleaned.append(decomp)
            dropped += 1
        else:
            cleaned.append(" ")
            dropped += 1
    if dropped:
        notes.append(f"normalized_or_dropped_unexpected_unicode:{dropped}")
    text = "".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if text != before and not notes:
        notes.append("vision_text_sanitized")
    return text, sorted(set(notes))


def repaired_text_quality_errors(text: str, *, min_chars: int) -> list[str]:
    """Validate the repaired text without rejecting good dense maths pages.

    Step 2 rejects dense equation pages before vision repair because plain Tesseract OCR is
    unsafe there. After a successful vision pass, density alone is not a blocker; we only
    reject text that still looks like garbage, a refusal, or too short to be page text.
    """
    errors: list[str] = []
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        errors.append(f"vision text too short: {len(stripped)} chars, min={min_chars}")
    if REFUSAL_OR_SUMMARY_RE.search(stripped):
        errors.append("vision response looks like refusal/summary instead of transcription")
    if DEVANAGARI_RE.search(stripped):
        errors.append("repaired text still contains Devanagari/Hindi OCR garbage")
    if AMBIGUOUS_GARBLED_RE.search(stripped):
        errors.append("repaired text still contains ambiguous OCR garbage tokens")
    weird = [ch for ch in stripped if ord(ch) > 127 and ch not in VISION_SAFE_NON_ASCII]
    weird_ratio = len(weird) / max(len(stripped), 1)
    # For vision output, a few uncommon unicode characters are not a production blocker.
    # Fail only when the page still looks materially corrupted.
    if weird_ratio > 0.01 or len(weird) >= 40:
        errors.append(f"repaired text still has too many unexpected non-ascii chars: {len(weird)}")
    if REPEATED_NOISE_RE.search(stripped):
        errors.append("repaired text has repeated noise/divider characters")
    if stripped.count("[unreadable]") >= 5:
        errors.append("too many [unreadable] markers for production embedding")
    return errors


def parse_confidence(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def normalize_confidence(confidence: float | None, *, min_confidence: float, text: str, notes: str) -> tuple[float, str]:
    """Handle the common JSON-schema-copy bug safely.

    The previous prompt used `"confidence": 0.0` as the schema example, and GPT sometimes
    copied that literal value even while returning a long, clean transcription. We do not
    blindly trust a model-reported 0.0 if the repaired text passes deterministic quality
    checks. In that case, floor it to the configured threshold and annotate the note.
    """
    if confidence is None:
        return min_confidence, (notes + " | confidence_missing_auto_floored_after_quality_validation").strip(" |")
    if confidence in LOW_CONFIDENCE_SCHEMA_COPY_VALUES and len(text.strip()) >= 250:
        return min_confidence, (notes + " | confidence_zero_auto_floored_after_quality_validation").strip(" |")
    return confidence, notes

def normalize_repair_payload(payload: dict[str, Any], *, min_chars: int, min_confidence: float) -> tuple[str, list[str], float, str]:
    text = str(payload.get("production_safe_text") or "").strip()
    text, sanitize_notes = sanitize_vision_text(text)
    text, clean_fixes = clean_text(text)
    math_lines_raw = payload.get("math_lines") or []
    if not isinstance(math_lines_raw, list):
        math_lines_raw = [str(math_lines_raw)]
    math_lines = [str(x).strip() for x in math_lines_raw if str(x).strip()]
    notes = str(payload.get("notes") or "").strip()
    all_notes = []
    all_notes.extend(sanitize_notes)
    all_notes.extend(clean_fixes)
    if all_notes:
        notes = (notes + " | " + ";".join(sorted(set(all_notes)))).strip(" |")

    quality_errors = repaired_text_quality_errors(text, min_chars=min_chars)
    if quality_errors:
        raise ValueError("; ".join(quality_errors))

    confidence, notes = normalize_confidence(
        parse_confidence(payload.get("confidence")),
        min_confidence=min_confidence,
        text=text,
        notes=notes,
    )
    if confidence < min_confidence:
        raise ValueError(f"vision confidence too low: {confidence}, min={min_confidence}")
    return text, math_lines, confidence, notes

def mark_page_repaired(page: dict[str, Any], *, repaired_text: str, math_lines: list[str], confidence: float, notes: str, model: str) -> None:
    page["vision_qa_status"] = "approved"
    page["vision_qa_model"] = model
    page["vision_qa_confidence"] = confidence
    page["vision_qa_notes"] = notes
    page["vision_qa_text"] = repaired_text

    # Keep original OCR for auditability, but make production/page text use repaired text.
    page["text"] = repaired_text
    page["text_plain"] = repaired_text
    page["production_safe_text"] = repaired_text
    page["math_lines"] = math_lines
    page["include_in_embeddings"] = True
    page["embedding_readiness"] = "ready_for_production_embedding"
    page["production_exclusion_reasons"] = []

    flags = set(page.get("quality_flags") or [])
    flags.discard("production_embedding_excluded_until_vision_qa")
    flags.discard("dense_formula_layout_requires_vision_or_mathpix_qa")
    flags.discard("remaining_garbled_unicode_requires_vision_qa")
    flags.discard("remaining_devanagari_or_hindi_ocr_garbage_requires_vision_qa")
    flags.add("production_embedding_ready")
    flags.add("vision_qa_repaired")
    page["quality_flags"] = sorted(flags)

    sources = list(page.get("text_sources") or [])
    if "openai_vision_math_qa" not in sources:
        sources.append("openai_vision_math_qa")
    page["text_sources"] = sources


def write_remaining_qa_csv(path: Path, pages: list[dict[str, Any]]) -> None:
    remaining = [p for p in pages if p.get("include_in_embeddings") is not True]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_number", "printed_page_number", "chapter_title", "embedding_readiness", "reasons", "sample_text"])
        writer.writeheader()
        for p in remaining:
            sample = re.sub(r"\s+", " ", (p.get("text") or "")[:300])
            writer.writerow({
                "page_number": p.get("page_number"),
                "printed_page_number": p.get("printed_page_number"),
                "chapter_title": p.get("chapter_title"),
                "embedding_readiness": p.get("embedding_readiness"),
                "reasons": "; ".join(p.get("production_exclusion_reasons") or []),
                "sample_text": sample,
            })


def refresh_policy(data: dict[str, Any], *, strict_complete: bool, repair_counts: Counter, report_errors: list[str]) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    pages = extraction.get("page_extractions", []) or []
    lesson_body_pages = [p for p in pages if is_lesson_body_page(p)]
    excluded_pages = [p for p in pages if p.get("include_in_embeddings") is not True]
    excluded_lesson_body_pages = [p for p in lesson_body_pages if p.get("include_in_embeddings") is not True]
    ready_pages = [p for p in pages if p.get("include_in_embeddings") is True]
    reason_counts = Counter()
    for p in excluded_pages:
        reason_counts.update(p.get("production_exclusion_reasons") or [])

    previous_policy = extraction.get("production_embedding_policy") or {}
    run_scope_blockers = previous_policy.get("run_scope_blockers") or []

    if run_scope_blockers:
        gate_status = "smoke_or_partial_not_production_ready"
    elif excluded_lesson_body_pages:
        gate_status = "production_safe_gated_needs_qa"
    else:
        gate_status = "production_complete_ready"

    extraction["production_embedding_policy"] = {
        "status": gate_status,
        "meaning": "Use production_safe_text only where include_in_embeddings=true and embedding_readiness=ready_for_production_embedding.",
        "recommended_embedding_text_field": "production_safe_text",
        "embed_only_when": {"include_in_embeddings": True, "embedding_readiness": "ready_for_production_embedding"},
        "do_not_embed_when_flags_include": [
            "production_embedding_excluded_until_vision_qa",
            "dense_formula_layout_requires_vision_or_mathpix_qa",
            "remaining_garbled_unicode_requires_vision_qa",
            "remaining_devanagari_or_hindi_ocr_garbage_requires_vision_qa",
        ],
        "run_scope_blockers": run_scope_blockers,
        "strict_complete": bool(strict_complete),
        "production_complete": gate_status == "production_complete_ready",
        "vision_qa_repair_applied": True,
        "vision_qa_errors": report_errors[:50],
    }

    qs = extraction.setdefault("quality_summary", {})
    qs["vision_qa_repair"] = {
        "generated_at_utc": now_utc(),
        "repaired_pages": int(repair_counts.get("repaired", 0)),
        "cached_repaired_pages": int(repair_counts.get("cached", 0)),
        "failed_pages": int(repair_counts.get("failed", 0)),
        "skipped_pages": int(repair_counts.get("skipped", 0)),
        "total_pages_in_json": len(pages),
        "ready_for_production_embedding_pages": len(ready_pages),
        "excluded_until_vision_qa_pages": len(excluded_pages),
        "lesson_body_pages": len(lesson_body_pages),
        "excluded_lesson_body_pages": len(excluded_lesson_body_pages),
        "gate_status": gate_status,
        "exclusion_reason_counts": dict(reason_counts),
    }
    extraction["generated_at_utc"] = now_utc()
    return qs["vision_qa_repair"]


def load_manual_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load reviewed page corrections from JSON instead of hardcoding text in code.

    Accepted shapes:
    - [{"page_number": 73, "production_safe_text": "..."}, ...]
    - {"pages": [...]}
    - {"73": {"production_safe_text": "..."}, "printed:66": {...}}
    """
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("pages"), list):
        items = raw["pages"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            if key.isdigit() and not item.get("page_number"):
                item["page_number"] = int(key)
            elif key.startswith("printed:") and not item.get("printed_page_number"):
                item["printed_page_number"] = int(key.split(":", 1)[1])
            items.append(item)
    else:
        raise ValueError(f"Unsupported manual override JSON shape in {path}")

    overrides: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("production_safe_text") or item.get("text") or "").strip()
        if not text:
            continue
        if item.get("page_number") is not None:
            overrides[f"page:{int(item['page_number'])}"] = item
        if item.get("printed_page_number") is not None:
            overrides[f"printed:{int(item['printed_page_number'])}"] = item
    return overrides


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
        "notes": override.get("notes") or "manual_page_override_reviewed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair excluded Grade10_Maths pages using vision OCR QA.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--remaining-qa-csv", type=Path, default=DEFAULT_REMAINING_QA_CSV)
    parser.add_argument("--model", default=os.environ.get("GRADE10_MATHS_VISION_MODEL", "gpt-4o-mini"), help="OpenAI vision model. Defaults from GRADE10_MATHS_VISION_MODEL.")
    parser.add_argument("--scale", type=float, default=env_float("GRADE10_MATHS_VISION_SCALE", 2.5), help="PDF render scale for page images. Defaults from GRADE10_MATHS_VISION_SCALE or 2.5.")
    parser.add_argument("--pages", default=None, help="Optional PDF pages to repair, e.g. 503-543")
    parser.add_argument("--max-pages", type=int, default=env_int("GRADE10_MATHS_AUTO_REVIEW_MAX_ITEMS", 0), help="Optional cap for smoke testing repair. Defaults from GRADE10_MATHS_AUTO_REVIEW_MAX_ITEMS; 0 means all.")
    parser.add_argument("--repair-front-matter", action="store_true", help="Also repair non-lesson/front-matter excluded pages.")
    parser.add_argument("--force-vision", action="store_true", help="Ignore cached page-level vision QA results.")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--min-confidence", type=float, default=env_float("GRADE10_MATHS_AUTO_REVIEW_THRESHOLD", 0.90))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--manual-overrides", type=Path, default=Path(os.environ.get("GRADE10_MATHS_PAGE_OVERRIDES", "")) if os.environ.get("GRADE10_MATHS_PAGE_OVERRIDES") else None, help="Optional JSON file with reviewed page-level production_safe_text overrides. This is the safe alternative to hardcoding page text in Python.")
    parser.add_argument("--strict-complete", action="store_true", help="Fail if any lesson-body page remains excluded after repair.")
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
    candidates = [p for p in pages if should_repair_page(p, repair_front_matter=args.repair_front_matter)]
    if selected_pages is not None:
        candidates = [p for p in candidates if int(p.get("page_number") or 0) in selected_pages]
    candidates = sorted(candidates, key=lambda p: int(p.get("page_number") or 0))
    if args.max_pages and args.max_pages > 0:
        candidates = candidates[: args.max_pages]

    manual_overrides = load_manual_overrides(args.manual_overrides)

    prompt_version = "grade10_maths_math_ocr_v3_unicode_sanitized_manual_overrides"
    cache_key = stable_cache_key(args.pdf, model=args.model, scale=args.scale, prompt_version=prompt_version)
    cache_dir = args.output.parent / ".vision_qa_cache" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    errors: list[str] = []

    print(f"Vision QA candidates: {len(candidates)}")
    print(f"Cache directory: {cache_dir}")
    if args.manual_overrides:
        print(f"Manual overrides: {args.manual_overrides} ({len(manual_overrides)} keys loaded)")

    with fitz.open(args.pdf) as pdf:
        for i, page in enumerate(candidates, start=1):
            page_number = int(page.get("page_number") or 0)
            cache_file = cache_dir / f"page_{page_number:04d}.json"
            try:
                override = get_manual_override(page, manual_overrides)
                if override is not None:
                    payload = override_to_payload(override)
                    counts["manual_override"] += 1
                elif cache_file.exists() and not args.force_vision:
                    payload = json.loads(cache_file.read_text(encoding="utf-8"))
                    counts["cached"] += 1
                else:
                    data_url = render_page_to_data_url(pdf, page_number, scale=args.scale)
                    prompt = build_prompt(page)
                    last_error: Exception | None = None
                    payload = None
                    min_chars_for_page = args.min_chars if is_lesson_body_page(page) else 1
                    min_confidence_for_page = args.min_confidence if is_lesson_body_page(page) else 0.0
                    repaired_tuple = None
                    for attempt in range(args.retries + 1):
                        try:
                            retry_suffix = ""
                            if attempt > 0 and last_error is not None:
                                retry_suffix = (
                                    "\n\nYour previous response failed validation: "
                                    f"{last_error}. Return a corrected JSON object now. "
                                    "If the page is readable, set confidence to 0.90 or higher."
                                )
                            payload = call_openai_vision(data_url=data_url, prompt=prompt + retry_suffix, model=args.model, timeout=args.timeout)
                            repaired_tuple = normalize_repair_payload(
                                payload,
                                min_chars=min_chars_for_page,
                                min_confidence=min_confidence_for_page,
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            if attempt < args.retries:
                                wait = max(args.sleep, 1.0) * (attempt + 1)
                                print(f"Page {page_number}: retrying after validation/API error: {exc!r}; wait={wait}s")
                                time.sleep(wait)
                    if payload is None or repaired_tuple is None:
                        raise RuntimeError(f"vision call/validation failed after retries: {last_error!r}")
                    payload["_cache_metadata"] = {
                        "pdf_page_number": page_number,
                        "printed_page_number": page.get("printed_page_number"),
                        "model": args.model,
                        "scale": args.scale,
                        "prompt_version": prompt_version,
                        "generated_at_utc": now_utc(),
                    }
                    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                repaired_text, math_lines, confidence, notes = normalize_repair_payload(
                    payload,
                    min_chars=args.min_chars if is_lesson_body_page(page) else 1,
                    min_confidence=args.min_confidence if is_lesson_body_page(page) else 0.0,
                )
                mark_page_repaired(page, repaired_text=repaired_text, math_lines=math_lines, confidence=confidence, notes=notes, model=args.model)
                counts["repaired"] += 1
                print(f"[{i}/{len(candidates)}] repaired page={page_number}, printed={page.get('printed_page_number')}, chars={len(repaired_text)}, confidence={confidence}")
                if args.sleep:
                    time.sleep(args.sleep)
            except Exception as exc:
                counts["failed"] += 1
                msg = f"page={page_number}, printed={page.get('printed_page_number')}: {exc}"
                errors.append(msg)
                page["vision_qa_status"] = "failed"
                page["vision_qa_error"] = str(exc)
                print(f"[{i}/{len(candidates)}] FAILED {msg}", file=sys.stderr)

    rebuild_chapters_and_sections(data)
    summary = refresh_policy(data, strict_complete=args.strict_complete, repair_counts=counts, report_errors=errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_remaining_qa_csv(args.remaining_qa_csv, pages)

    report_lines = [
        "Grade10_Maths Vision QA Repair Report",
        "=" * 60,
        f"Generated at UTC: {now_utc()}",
        f"Input JSON: {args.input}",
        f"Output JSON: {args.output}",
        f"Remaining QA CSV: {args.remaining_qa_csv}",
        f"PDF: {args.pdf}",
        f"Model: {args.model}",
        f"Scale: {args.scale}",
        "",
        f"Candidates selected: {len(candidates)}",
        f"Repaired pages: {counts.get('repaired', 0)}",
        f"Cached pages used: {counts.get('cached', 0)}",
        f"Manual override pages used: {counts.get('manual_override', 0)}",
        f"Failed pages: {counts.get('failed', 0)}",
        "",
        f"Production status: {summary.get('gate_status')}",
        f"Lesson-body pages: {summary.get('lesson_body_pages')}",
        f"Excluded lesson-body pages: {summary.get('excluded_lesson_body_pages')}",
        f"Ready pages: {summary.get('ready_for_production_embedding_pages')}",
        f"Excluded pages total: {summary.get('excluded_until_vision_qa_pages')}",
    ]
    if errors:
        report_lines.append("")
        report_lines.append("Errors:")
        report_lines.extend(f"  - {e}" for e in errors[:100])
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if args.strict_complete and summary.get("excluded_lesson_body_pages", 0):
        raise SystemExit(
            "Strict complete failed after vision repair: "
            f"{summary.get('excluded_lesson_body_pages')} lesson-body pages are still excluded. "
            f"See {args.report} and {args.remaining_qa_csv}."
        )

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")
    print(f"Wrote: {args.remaining_qa_csv}")


if __name__ == "__main__":
    main()
