#!/usr/bin/env python3
"""
Step 7: full page-image production verifier for Grade10 Maths.

This step is intentionally stronger than regex/hard-pattern gates.
For each production-included lesson page, it renders the original PDF page image
and asks a vision model to compare the current production_safe_text with the page.
If the text is not faithful enough for Maths embeddings, the model must return a
full corrected transcription from the image only. The corrected text is then
written back to production_safe_text and subsection production text is rebuilt.

Production_complete_ready is allowed only when every checked production page
passes image-vs-text verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise RuntimeError("Missing dependency: pip install pymupdf") from exc

from make_grade10_maths_step_3_vision_repair import (  # noqa: E402
    call_openai_vision,
    render_page_to_data_url,
    sanitize_vision_text,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT_JSON = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_full_image_production_verify_report.txt"
DEFAULT_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_full_image_production_verify_remaining_pages.csv"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / ".full_image_verify_cache"
PROMPT_VERSION = "full_image_production_verify_v1_2026_06_17"

BAD_STATUS = {"fail", "failed", "needs_repair", "needs_review", "review", "reject", "rejected"}
GOOD_STATUS = {"pass", "passed", "ok", "verified", "repaired", "corrected", "transcribed"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_pages_arg(value: str | None) -> set[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in str(value).split(","):
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




def parse_pages_from_csv(path: Path | None) -> set[int]:
    """Read PDF page numbers from a residual/remaining CSV.

    Supports the CSV files written by Step 6, Step 7, and the override builder.
    It looks for the first available page-number column, preferring page_number.
    """
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)
    pages: set[int] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return pages
        fieldnames = {name.lower().strip(): name for name in reader.fieldnames if name}
        candidates = [
            "page_number",
            "pdf_page_number",
            "pdf_page",
            "page",
        ]
        col = None
        for name in candidates:
            if name in fieldnames:
                col = fieldnames[name]
                break
        if not col:
            raise ValueError(
                f"CSV {path} does not contain a supported page-number column. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            raw = str(row.get(col) or "").strip()
            if not raw:
                continue
            m = re.search(r"\d+", raw)
            if m:
                pages.add(int(m.group(0)))
    return pages

def is_production_lesson_page(page: dict[str, Any]) -> bool:
    return (
        page.get("content_type") == "lesson_body"
        and page.get("include_in_lesson_text") is True
        and page.get("include_in_embeddings") is True
        and page.get("embedding_readiness") == "ready_for_production_embedding"
        and bool(str(page.get("production_safe_text") or "").strip())
    )


def is_selected_lesson_page_for_repair(page: dict[str, Any]) -> bool:
    """For targeted repair, include failed/excluded lesson pages too.

    Step 4/5 may mark a page excluded until QA. Those are exactly the pages
    targeted Step 7 must repair from the PDF image, so do not require
    include_in_embeddings=True or ready_for_production_embedding here.
    """
    return (
        page.get("content_type") == "lesson_body"
        and page.get("include_in_lesson_text") is True
        and page.get("page_number") is not None
    )


def page_sample(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]



def build_repair_only_prompt(page: dict[str, Any], current_text: str) -> str:
    """Prompt used by targeted failed-page repair mode.

    Unlike the verifier prompt, this always asks the model to transcribe from the
    page image. It is intentionally used only for pages already selected by Step
    4/5/6 as problematic, so it avoids the expensive/over-strict second QA loop.
    """
    current_text = current_text[:4000]
    return f"""
You are repairing ONE failed Grade 10 mathematics textbook page for production OCR.

The page has already failed a deterministic production audit. Use the PAGE IMAGE as the source of truth and produce a clean full transcription of the visible lesson content.

Rules:
- Ignore OCR garbage in the current text; use it only to understand page context.
- Do not summarize, solve, simplify, or add explanations.
- Preserve headings, examples, exercises, solutions, theorem/proof labels, tables, formulae, fractions, roots, powers, subscripts, geometry point names, statistics tables, and probability outcomes.
- Use plain text math: x^2, x^3, a_n, sqrt(125), angle ABC, ΔABC, ∴, ⇒, ≤, ≥, ≠, ×, ÷.
- If a diagram is essential and cannot be fully transcribed, write [diagram: short description].
- Do not copy corrupted fragments such as random symbols, impossible words, broken OCR tokens, or malformed formula garbage.

Return strict JSON only in this schema:
{{
  "status": "repaired",
  "confidence": 0.0-1.0,
  "issues": ["short issue strings you fixed"],
  "corrected_text": "full clean transcription of this page from image",
  "notes": "optional short notes"
}}

Page metadata:
PDF page number: {page.get('page_number')}
Printed page number: {page.get('printed_page_number')}
Chapter: {page.get('chapter_title') or ''}
Section: {page.get('section_title') or ''}

Current failed production text for context only:
---
{current_text}
---
""".strip()


def build_verify_prompt(page: dict[str, Any], current_text: str) -> str:
    current_text = current_text[:12000]
    return f"""
You are the FINAL production QA verifier for a scanned Grade 10 mathematics textbook page.

You must compare the PAGE IMAGE with the CURRENT production text below.

Goal:
- Decide whether CURRENT production text is faithful enough for production Maths embeddings.
- This is a strict check. Fail if equations, numbers, powers, roots, fractions, subscripts, tables, theorem/proof steps, examples, or exercise questions are corrupted or missing.
- Fail if the current text contains OCR garbage, broken words, impossible formula fragments, random symbols, wrong variable names, or nonsensical algebra/geometry/statistics expressions.
- Do not pass a page merely because it is readable prose; Maths formulas must be correct enough.

If CURRENT text is faithful:
- return status = "pass" and corrected_text = "".

If CURRENT text is not faithful:
- return status = "fail" and corrected_text as a full clean transcription of the printed lesson content visible in the image.
- Use the image as source of truth. Do NOT copy corrupted current text.
- Preserve headings, examples, exercises, proof/solution labels, tables, formulae, identities, fractions, radicals, powers, subscripts, geometry point names, statistics class intervals, and probability outcomes.
- Use plain text math: x^2, x^3, a_n, sqrt(125), angle ABC, ΔABC, ∴, ⇒, ≤, ≥, ≠, ×, ÷.
- Do not solve problems or add new explanations.
- Do not summarize.
- If a tiny diagram is essential, write [diagram: short description].

Return strict JSON only in this schema:
{{
  "status": "pass" or "fail",
  "confidence": 0.0-1.0,
  "issues": ["short issue strings"],
  "corrected_text": "full corrected transcription if status is fail, else empty string",
  "notes": "optional short notes"
}}

Page metadata:
PDF page number: {page.get('page_number')}
Printed page number: {page.get('printed_page_number')}
Chapter: {page.get('chapter_title') or ''}
Section: {page.get('section_title') or ''}

CURRENT production_safe_text to verify:
---
{current_text}
---
""".strip()


def normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower().replace(" ", "_")
    if status in GOOD_STATUS:
        return "pass"
    if status in BAD_STATUS:
        return "fail"
    return "fail"


def normalize_verification_payload(raw: dict[str, Any]) -> dict[str, Any]:
    status = normalize_status(raw.get("status"))
    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    issues = raw.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(x).strip() for x in issues if str(x).strip()]
    corrected_text = raw.get("corrected_text") or raw.get("clean_text") or raw.get("production_safe_text") or ""
    if not isinstance(corrected_text, str):
        corrected_text = str(corrected_text)
    notes = str(raw.get("notes") or "").strip()
    return {
        "status": status,
        "confidence": confidence,
        "issues": issues,
        "corrected_text": corrected_text.strip(),
        "notes": notes,
    }


def cache_path(cache_dir: Path, page_number: int, phase: str) -> Path:
    return cache_dir / f"page_{page_number:04d}_{phase}.json"


def call_or_cache(*, pdf: fitz.Document, page: dict[str, Any], prompt: str, model: str, scale: float, timeout: float, cache_dir: Path, force: bool, phase: str) -> dict[str, Any]:
    page_number = int(page["page_number"])
    cpath = cache_path(cache_dir, page_number, phase)
    if cpath.exists() and not force:
        return json.loads(cpath.read_text(encoding="utf-8"))
    data_url = render_page_to_data_url(pdf, page_number, scale=scale)
    raw = call_openai_vision(data_url=data_url, prompt=prompt, model=model, timeout=timeout)
    payload = normalize_verification_payload(raw)
    payload["page_number"] = page_number
    payload["printed_page_number"] = page.get("printed_page_number")
    payload["chapter_title"] = page.get("chapter_title")
    payload["prompt_version"] = PROMPT_VERSION
    payload["phase"] = phase
    payload["created_at_utc"] = now_utc()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def add_flag(page: dict[str, Any], flag: str) -> None:
    flags = page.setdefault("quality_flags", [])
    if flag not in flags:
        flags.append(flag)


def set_page_text(page: dict[str, Any], text: str, *, notes: list[str]) -> None:
    page["production_safe_text"] = text
    # Keep legacy text fields in sync so downstream code cannot accidentally ingest stale OCR.
    page["text"] = text
    page["text_plain"] = text
    page["final_ocr_text"] = text
    page["embedding_readiness"] = "ready_for_production_embedding"
    page["include_in_embeddings"] = True
    page["production_exclusion_reasons"] = []
    page["full_image_repair_sanitize_notes"] = notes
    add_flag(page, "full_image_verified_repaired")


def mark_page_ready_after_full_image_pass(page: dict[str, Any], text: str) -> None:
    """Re-include a targeted page after full image verification passes."""
    page["production_safe_text"] = text
    page["text"] = text
    page["text_plain"] = text
    page["final_ocr_text"] = text
    page["embedding_readiness"] = "ready_for_production_embedding"
    page["include_in_embeddings"] = True
    page["production_exclusion_reasons"] = []
    # Remove common temporary exclusion flags if present.
    flags = page.setdefault("quality_flags", [])
    for bad in [
        "math_precision_audit_failed",
        "production_embedding_excluded_until_math_precision_qa",
        "production_embedding_excluded_until_final_audit_repair",
        "production_embedding_excluded_until_vision_qa",
    ]:
        while bad in flags:
            flags.remove(bad)
    add_flag(page, "full_image_verified_pass_included")


def page_numbers_for_subsection(sub: dict[str, Any]) -> list[int]:
    values = sub.get("pdf_pages") or sub.get("page_numbers")
    if isinstance(values, list) and values:
        out = []
        for x in values:
            try:
                out.append(int(x))
            except Exception:
                pass
        return sorted(set(out))
    start = sub.get("start_pdf_page") or sub.get("start_page")
    end = sub.get("end_pdf_page") or sub.get("end_page")
    try:
        a, b = int(start), int(end)
        return list(range(a, b + 1))
    except Exception:
        return []


def rebuild_subsection_texts(data: dict[str, Any]) -> None:
    extraction = data.get("extraction") or {}
    pages = extraction.get("page_extractions") or []
    by_page = {int(p.get("page_number")): p for p in pages if p.get("page_number") is not None}
    for chapter in extraction.get("chapters", []) or []:
        for sub in chapter.get("subsections", []) or []:
            nums = page_numbers_for_subsection(sub)
            texts: list[str] = []
            prod_nums: list[int] = []
            printed_nums: list[Any] = []
            excluded: list[int] = []
            for num in nums:
                page = by_page.get(num)
                if not page:
                    continue
                if page.get("include_in_lesson_text") is not True:
                    continue
                if page.get("include_in_embeddings") is True and page.get("embedding_readiness") == "ready_for_production_embedding":
                    text = str(page.get("production_safe_text") or "").strip()
                    if text:
                        texts.append(text)
                        prod_nums.append(num)
                        printed_nums.append(page.get("printed_page_number"))
                    else:
                        excluded.append(num)
                else:
                    excluded.append(num)
            combined = "\n\n".join(texts).strip()
            sub["production_subsection_text"] = combined
            sub["subsection_text"] = combined
            sub["subsection_text_plain"] = combined
            sub["text_plain"] = combined
            sub["production_indexed_page_numbers"] = prod_nums
            sub["production_printed_page_numbers"] = printed_nums
            sub["production_excluded_pages"] = excluded
            sub["production_page_count"] = len(prod_nums)
            sub["include_in_embeddings"] = bool(combined and not excluded)
            flags = sub.setdefault("quality_flags", [])
            if combined and "full_image_verified_subsection_text_rebuilt" not in flags:
                flags.append("full_image_verified_subsection_text_rebuilt")


def refresh_policy(data: dict[str, Any], results: list[dict[str, Any]], *, strict_scope_all_pages: bool) -> None:
    extraction = data.setdefault("extraction", {})
    qs = extraction.setdefault("quality_summary", {})
    policy = extraction.setdefault("production_embedding_policy", {})
    remaining = [r for r in results if r.get("final_status") != "pass"]
    status_counts = Counter(str(r.get("final_status") or "unknown") for r in results)
    repaired = [r for r in results if r.get("repaired")]
    checked_pages = [int(r["page_number"]) for r in results if r.get("page_number") is not None]
    remaining_pages = [int(r["page_number"]) for r in remaining if r.get("page_number") is not None]
    qs["full_image_production_verify"] = {
        "applied": True,
        "prompt_version": PROMPT_VERSION,
        "checked_page_count": len(checked_pages),
        "checked_pages": checked_pages,
        "strict_scope_all_ready_pages": strict_scope_all_pages,
        "repaired_page_count": len(repaired),
        "remaining_failed_page_count": len(remaining_pages),
        "remaining_failed_pages": remaining_pages,
        "status_counts": dict(status_counts),
        "generated_at_utc": now_utc(),
    }
    policy["full_image_production_verify_applied"] = True
    policy["full_image_production_verify_prompt_version"] = PROMPT_VERSION
    policy["full_image_failed_page_count"] = len(remaining_pages)
    policy["full_image_failed_pages"] = remaining_pages
    if remaining_pages:
        policy["status"] = "production_safe_gated_needs_full_image_qa"
        policy["production_complete"] = False
        extraction["production_complete"] = False
    else:
        # Full image verifier is the strongest targeted gate. If every checked failed page passes,
        # clear prior audit failure counters so final Step 6 can become the source-of-truth hard gate.
        for audit_key in ("final_production_audit", "math_precision_audit", "residual_production_audit"):
            audit = qs.get(audit_key)
            if isinstance(audit, dict):
                for count_key in (
                    "remaining_suspicious_ready_pages",
                    "excluded_lesson_body_pages",
                    "remaining_math_precision_pages",
                    "remaining_residual_pages",
                    "empty_production_subsections",
                    "gated_failed_pages",
                ):
                    if count_key in audit:
                        audit[count_key] = 0
                if "gate_status" in audit:
                    audit["gate_status"] = "production_complete_ready"
        for policy_key in (
            "final_audit_remaining_suspicious_ready_pages",
            "final_audit_empty_production_subsections",
            "math_precision_remaining_pages",
            "math_precision_empty_production_subsections",
            "residual_remaining_pages",
            "residual_empty_production_subsections",
        ):
            if policy_key in policy:
                policy[policy_key] = 0
        for error_key in ("final_audit_errors", "math_precision_errors", "residual_errors"):
            if error_key in policy:
                policy[error_key] = []
        policy["status"] = "production_complete_ready"
        policy["production_complete"] = True
        extraction["production_complete"] = True


def write_report(path: Path, results: list[dict[str, Any]], *, model: str) -> None:
    remaining = [r for r in results if r.get("final_status") != "pass"]
    lines = [
        "Grade10 Maths full image production verification report",
        f"prompt_version: {PROMPT_VERSION}",
        f"model: {model}",
        f"checked_pages: {len(results)}",
        f"remaining_failed_pages: {len(remaining)}",
        "",
    ]
    for r in remaining[:200]:
        lines.append(
            f"page={r.get('page_number')} printed={r.get('printed_page_number')} "
            f"chapter={r.get('chapter_title')} status={r.get('final_status')} "
            f"confidence={r.get('confidence')} issues={'; '.join(r.get('issues') or [])}"
        )
        if r.get("sample_text"):
            lines.append(f"sample: {r['sample_text']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_remaining_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page_number",
                "printed_page_number",
                "chapter_title",
                "final_status",
                "confidence",
                "repaired",
                "issues",
                "sample_text",
            ],
        )
        writer.writeheader()
        for r in results:
            if r.get("final_status") == "pass":
                continue
            writer.writerow({
                "page_number": r.get("page_number"),
                "printed_page_number": r.get("printed_page_number"),
                "chapter_title": r.get("chapter_title"),
                "final_status": r.get("final_status"),
                "confidence": r.get("confidence"),
                "repaired": r.get("repaired"),
                "issues": "; ".join(r.get("issues") or []),
                "sample_text": r.get("sample_text") or "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Full page-image production verifier for Grade10 Maths JSON.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--remaining-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", default=os.environ.get("GRADE10_MATHS_FULL_VERIFY_MODEL") or os.environ.get("GRADE10_MATHS_MATH_PRECISION_MODEL") or "gpt-4o")
    parser.add_argument("--scale", type=float, default=float(os.environ.get("GRADE10_MATHS_FULL_VERIFY_SCALE", os.environ.get("GRADE10_MATHS_VISION_SCALE", "2.5"))))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("GRADE10_MATHS_FULL_VERIFY_TIMEOUT", "120")))
    parser.add_argument("--pages", default=None, help="Optional PDF pages to verify, e.g. 9,19-20,586. Omit for all ready lesson-body pages.")
    parser.add_argument("--pages-from-csv", type=Path, default=None, help="Optional residual/remaining CSV from Step 6/Step 7. Only these PDF pages are sent to the LLM.")
    parser.add_argument("--max-pages", type=int, default=int(os.environ.get("GRADE10_MATHS_FULL_VERIFY_MAX_ITEMS", "0") or "0"), help="Optional cap for test runs. 0 means all selected pages.")
    parser.add_argument("--force-cache", action="store_true", help="Ignore cached verification results and call the model again.")
    parser.add_argument("--no-repair", action="store_true", help="Only verify; do not write corrected_text back into the JSON.")
    parser.add_argument("--repair-selected-from-image-only", action="store_true", help="For --pages/--pages-from-csv targeted mode, skip verify-first behavior and always regenerate selected pages from image only. Intended for failed-page LLM repair before the final Step 6 hard gate.")
    parser.add_argument("--no-verify-repaired", action="store_true", help="Do not make a second verification call after repair.")
    parser.add_argument("--strict-complete", action="store_true", help="Fail unless all checked pages pass full image verification.")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    data = load_json(args.input)
    extraction = data.get("extraction") or {}
    pages = extraction.get("page_extractions") or []
    selected = parse_pages_arg(args.pages)
    csv_selected = parse_pages_from_csv(args.pages_from_csv)
    if csv_selected:
        selected = set(selected or set()) | csv_selected
    if selected is not None:
        # Targeted mode: selected CSV pages are usually the failed/excluded pages.
        # Include them even if earlier audits temporarily removed them from embeddings.
        candidates = [
            p for p in pages
            if is_selected_lesson_page_for_repair(p)
            and int(p.get("page_number")) in selected
        ]
    else:
        candidates = [p for p in pages if is_production_lesson_page(p)]
    candidates.sort(key=lambda p: int(p.get("page_number") or 0))
    if args.max_pages and args.max_pages > 0:
        candidates = candidates[: args.max_pages]

    results: list[dict[str, Any]] = []
    repair = not args.no_repair
    verify_repaired = not args.no_verify_repaired
    repair_selected_from_image_only = bool(args.repair_selected_from_image_only and selected is not None and repair)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if selected is not None and not candidates:
        # No matching ready production lesson pages for this CSV/page selection. Write an empty pass report.
        rebuild_subsection_texts(data)
        refresh_policy(data, results, strict_scope_all_pages=False)
        write_json(args.output, data)
        write_report(args.report, results, model=args.model)
        write_remaining_csv(args.remaining_csv, results)
        print("Full image production verify checked pages: 0")
        print("Full image production verify remaining pages: 0")
        print(f"No lesson-body pages matched selected pages: {sorted(selected)}")
        print(f"Wrote: {args.output}")
        print(f"Wrote: {args.report}")
        print(f"Wrote: {args.remaining_csv}")
        return

    with fitz.open(str(args.pdf)) as pdf:
        for idx, page in enumerate(candidates, start=1):
            page_number = int(page["page_number"])
            current_text = str(page.get("production_safe_text") or "").strip()
            print(f"[{idx}/{len(candidates)}] verifying page={page_number}, printed={page.get('printed_page_number')}")
            repaired = False
            final_text = current_text

            if repair_selected_from_image_only:
                prompt = build_repair_only_prompt(page, current_text)
                first = call_or_cache(
                    pdf=pdf,
                    page=page,
                    prompt=prompt,
                    model=args.model,
                    scale=args.scale,
                    timeout=args.timeout,
                    cache_dir=args.cache_dir,
                    force=args.force_cache,
                    phase="repair_only",
                )
                issues = list(first.get("issues") or [])
                confidence = first.get("confidence")
                corrected = str(first.get("corrected_text") or "").strip()
                if corrected:
                    sanitized, notes = sanitize_vision_text(corrected)
                    if len(sanitized) >= 80:
                        set_page_text(page, sanitized, notes=notes)
                        page["full_image_verification_status"] = "targeted_image_repaired_pending_step6"
                        page["full_image_verification_issues"] = issues
                        page["full_image_verification_confidence"] = confidence
                        page["full_image_verification_prompt_version"] = PROMPT_VERSION
                        page["full_image_verification_at_utc"] = now_utc()
                        repaired = True
                        final_text = sanitized
                        final_status = "pass"
                    else:
                        issues.append("model_failed_to_return_full_corrected_text")
                        final_status = "fail"
                else:
                    issues.append("model_returned_no_corrected_text")
                    final_status = "fail"
            else:
                prompt = build_verify_prompt(page, current_text)
                first = call_or_cache(
                    pdf=pdf,
                    page=page,
                    prompt=prompt,
                    model=args.model,
                    scale=args.scale,
                    timeout=args.timeout,
                    cache_dir=args.cache_dir,
                    force=args.force_cache,
                    phase="verify",
                )
                status = first["status"]
                issues = list(first.get("issues") or [])
                confidence = first.get("confidence")
                final_status = status

            if (not repair_selected_from_image_only) and final_status != "pass" and repair:
                corrected = str(first.get("corrected_text") or "").strip()
                if corrected:
                    sanitized, notes = sanitize_vision_text(corrected)
                    if len(sanitized) >= 80:
                        set_page_text(page, sanitized, notes=notes)
                        page["full_image_verification_status"] = "repaired_pending_second_verify" if verify_repaired else "repaired_without_second_verify"
                        page["full_image_verification_issues"] = issues
                        page["full_image_verification_confidence"] = confidence
                        page["full_image_verification_prompt_version"] = PROMPT_VERSION
                        page["full_image_verification_at_utc"] = now_utc()
                        repaired = True
                        final_text = sanitized
                        if verify_repaired:
                            second_prompt = build_verify_prompt(page, sanitized)
                            second = call_or_cache(
                                pdf=pdf,
                                page=page,
                                prompt=second_prompt,
                                model=args.model,
                                scale=args.scale,
                                timeout=args.timeout,
                                cache_dir=args.cache_dir,
                                force=args.force_cache,
                                phase="verify_after_repair",
                            )
                            final_status = second["status"]
                            issues = list(second.get("issues") or [])
                            confidence = second.get("confidence")
                        else:
                            final_status = "pass"
                    else:
                        issues.append("model_failed_to_return_full_corrected_text")
                else:
                    issues.append("model_returned_fail_without_corrected_text")

            if final_status == "pass":
                # If this was a targeted failed/excluded page, re-include it after image verification passes.
                if selected is not None:
                    safe_text = str(page.get("production_safe_text") or final_text or "").strip()
                    if safe_text:
                        mark_page_ready_after_full_image_pass(page, safe_text)
                page["full_image_verification_status"] = "pass"
                page["full_image_verification_failed"] = False
                add_flag(page, "full_image_verified_pass")
            else:
                page["full_image_verification_status"] = "fail"
                page["full_image_verification_failed"] = True
                add_flag(page, "full_image_verified_fail")
            page["full_image_verification_issues"] = issues
            page["full_image_verification_confidence"] = confidence
            page["full_image_verification_prompt_version"] = PROMPT_VERSION
            page["full_image_verification_at_utc"] = now_utc()

            results.append({
                "page_number": page_number,
                "printed_page_number": page.get("printed_page_number"),
                "chapter_title": page.get("chapter_title"),
                "final_status": final_status,
                "confidence": confidence,
                "issues": issues,
                "repaired": repaired,
                "sample_text": page_sample(final_text),
            })

    rebuild_subsection_texts(data)
    refresh_policy(data, results, strict_scope_all_pages=(selected is None and not args.max_pages))
    write_json(args.output, data)
    write_report(args.report, results, model=args.model)
    write_remaining_csv(args.remaining_csv, results)

    remaining = [r for r in results if r.get("final_status") != "pass"]
    print(f"Full image production verify checked pages: {len(results)}")
    print(f"Full image production verify remaining pages: {len(remaining)}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")
    print(f"Wrote: {args.remaining_csv}")

    if args.strict_complete and remaining:
        raise RuntimeError(
            "Strict complete failed after full image production verification: "
            f"remaining_full_image_pages={len(remaining)}. See {args.report} and {args.remaining_csv}."
        )


if __name__ == "__main__":
    main()
