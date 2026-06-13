#!/usr/bin/env python3
"""Auto-review Physics formula/OCR crops and merge results into corrections JSON.

Why this exists:
- PyMuPDF selectable text in this Grade 10 Physics PDF is not trustworthy for formulas.
- Step 2 intentionally isolates risky formula/table/diagram lines instead of putting corrupt text into production.
- This script renders the original PDF crop for each unresolved review_id and asks a visual reviewer
  provider to transcribe the crop exactly. Only accepted items are written to corrections JSON.

Providers:
- openai: uses an image-capable OpenAI chat model. Recommended for production automation.
- tesseract: local OCR only. Useful for candidate text, but not recommended as a final authority for formulas.
- none: only writes crop images + OCR candidates into an audit CSV; does not change corrections JSON unless --accept-tesseract is used.

Production rule:
Final Step 5 must still fail unless unresolved_review_items == 0. Do not use --allow-review-required for final output.
"""
from __future__ import annotations

import argparse
import base64
import csv
import difflib
import json
import os
from dotenv import load_dotenv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageOps

try:
    import pytesseract
except Exception:  # pragma: no cover - handled at runtime
    pytesseract = None  # type: ignore


SUSPICIOUS_RE = re.compile(r"[\u0900-\u097F�￾]|(?:कि|हि|पि)|(?:0\s*[°~]\s*C)|(?:10\s*[°%]\s*C)", re.I)
MATH_SIGNAL_RE = re.compile(r"[=×÷/^]|\b(?:10\^|sin|cos|tan|theta|λ|μ|Ω|ohm|mC|uC|C|V|A|W|J|N|Hz|D)\b", re.I)
TRUTHY = {"1", "y", "yes", "true", "t"}

load_dotenv()


@dataclass
class CropResult:
    image_path: Path | None
    bbox: list[float] | None
    tesseract_text: str
    tesseract_confidence: float
    notes: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\u00a0", " ")).strip()


def normalize_for_match(text: str) -> str:
    text = compact(text).lower()
    text = re.sub(r"[^a-z0-9μΩ]+", "", text)
    return text


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:120]


def parse_tesseract_conf(data: dict[str, list[str]]) -> float:
    vals: list[float] = []
    for raw in data.get("conf", []):
        try:
            v = float(raw)
        except Exception:
            continue
        if v >= 0:
            vals.append(v)
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals) / 100.0, 4)


def clean_ocr_text(text: str) -> str:
    s = compact(text)
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace(" x ", " × ")
    # Keep ASCII-friendly formula output for JSON/search while preserving common symbols.
    s = re.sub(r"\b1uC\b", "1 μC", s)
    s = re.sub(r"\bluC\b", "1 μC", s)
    s = re.sub(r"\buC\b", "μC", s)
    s = re.sub(r"\bQ\s*=\s*l\s*C\b", "Q = 1 C", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"([=+\-×÷/])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_page(page_extractions: list[dict[str, Any]], page_number: int) -> dict[str, Any] | None:
    for page in page_extractions:
        try:
            if int(page.get("page_number") or 0) == int(page_number):
                return page
        except Exception:
            continue
    return None


def find_bbox(page_data: dict[str, Any], item: dict[str, Any]) -> tuple[list[float] | None, list[str]]:
    """Find the best layout bbox for a review item.

    Step 2 review IDs are based on cleaned text line numbers, while layout_lines are raw PDF layout
    line numbers. They are usually exact or off by a few lines. We try exact/nearby line_no first,
    then fuzzy text matching.
    """
    notes: list[str] = []
    layout = page_data.get("layout_lines") or []
    if not layout:
        return None, ["page_has_no_layout_lines"]

    target_line = int(item.get("line_no") or -1)
    target_text = compact(item.get("raw_text") or item.get("text") or "")
    target_key = normalize_for_match(target_text)

    def score(ln: dict[str, Any]) -> float:
        ln_no = int(ln.get("line_no") or -99999)
        raw = compact(ln.get("text") or "")
        raw_key = normalize_for_match(raw)
        if target_key and raw_key:
            sim = difflib.SequenceMatcher(None, target_key, raw_key).ratio()
        else:
            sim = 0.0
        line_score = max(0.0, 1.0 - abs(ln_no - target_line) / 8.0)
        if ln_no == target_line:
            line_score += 0.35
        return sim * 0.75 + line_score * 0.25

    # Prefer exact/nearby lines when the text is tiny formula text such as '= ne'.
    nearby = [ln for ln in layout if abs(int(ln.get("line_no") or -99999) - target_line) <= 4]
    candidates = nearby or list(layout)
    best = max(candidates, key=score, default=None)
    if best and score(best) >= 0.35:
        if int(best.get("line_no") or 0) != target_line:
            notes.append(f"bbox_matched_layout_line_{best.get('line_no')}_for_review_line_{target_line}")
        return list(best.get("bbox") or []), notes

    # Fallback: best fuzzy match across the page.
    best = max(layout, key=score, default=None)
    if best and score(best) >= 0.55:
        notes.append(f"bbox_fuzzy_matched_layout_line_{best.get('line_no')}")
        return list(best.get("bbox") or []), notes

    return None, ["bbox_not_found"]


def expand_bbox(page_rect: fitz.Rect, bbox: list[float], xpad: float, ypad: float, context_y: float) -> fitz.Rect:
    rect = fitz.Rect(bbox)
    rect = fitz.Rect(rect.x0 - xpad, rect.y0 - ypad - context_y, rect.x1 + xpad, rect.y1 + ypad + context_y)
    return rect & page_rect


def render_crop(pdf: fitz.Document, page_number: int, bbox: list[float], out_dir: Path, review_id: str,
                dpi: int = 360, xpad: float = 10, ypad: float = 5, context_y: float = 0) -> Path:
    page = pdf.load_page(page_number - 1)
    rect = expand_bbox(page.rect, bbox, xpad=xpad, ypad=ypad, context_y=context_y)
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_name(review_id)}_p{page_number:03d}.png"
    pix.save(str(path))
    return path


def run_tesseract(image_path: Path, psm: int = 7, lang: str = "eng") -> tuple[str, float]:
    if pytesseract is None:
        return "", 0.0
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(2.2)
    config = f"--psm {psm} -l {lang}"
    text = pytesseract.image_to_string(img, config=config)
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
        conf = parse_tesseract_conf(data)
    except Exception:
        conf = 0.0
    return clean_ocr_text(text), conf


def local_crop_review(pdf: fitz.Document, page_data: dict[str, Any], item: dict[str, Any], crop_dir: Path,
                      dpi: int, lang: str, context_y: float) -> CropResult:
    review_id = str(item.get("review_id"))
    page_number = int(item.get("page_number") or page_data.get("page_number") or 0)
    bbox, notes = find_bbox(page_data, item)
    if not bbox:
        return CropResult(None, None, "", 0.0, notes)
    image_path = render_crop(pdf, page_number, bbox, crop_dir, review_id, dpi=dpi, context_y=context_y)
    t7, c7 = run_tesseract(image_path, psm=7, lang=lang)
    t6, c6 = run_tesseract(image_path, psm=6, lang=lang)
    if c6 > c7 + 0.05 and len(t6) >= len(t7):
        return CropResult(image_path, bbox, t6, c6, notes + ["tesseract_psm6_selected"])
    return CropResult(image_path, bbox, t7, c7, notes + ["tesseract_psm7_selected"])


def data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_provider_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def review_with_openai(image_path: Path, raw_text: str, reasons: list[str], model: str) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is not installed. Install with: pip install openai") from exc

    client = OpenAI()
    prompt = (
        "You are reviewing a crop from a Grade 10 physics textbook. "
        "Transcribe ONLY the printed textbook text visible in the crop. Ignore handwriting, stamps, page numbers, and decorative noise. "
        "For formulas, preserve exact meaning using plain text: use ^ for exponents, / for fractions, × for multiplication, μ for micro, Ω for ohm. "
        "Examples: 10^-3 C, 1.6 × 10^-19 C, Q = ne, n = Q/e. "
        "If the crop is not textbook content or is only OCR garbage/duplicate fragments, action='discard'. "
        "If the crop is genuinely unreadable, action='review'. "
        "Return strict JSON only with keys: action, text, confidence, reason. "
        "action must be one of: correct, discard, review. confidence is 0.0 to 1.0."
    )
    user_text = (
        f"Raw selectable/OCR text from PDF: {raw_text!r}\n"
        f"Risk reasons: {reasons}\n"
        "Return the corrected transcription for the crop."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url(image_path)}},
            ]},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return parse_provider_json(content)


def should_accept_tesseract(item: dict[str, Any], text: str, conf: float, threshold: float) -> bool:
    """Conservative local OCR acceptance.

    Local Tesseract is allowed to auto-fix clear prose/artifact lines, but not formulas with exponents.
    Formulas should go through OpenAI/Mathpix/human review because Tesseract commonly reads 10^-3 as 10°C.
    """
    if conf < threshold or not text.strip():
        return False
    raw = compact(item.get("raw_text") or item.get("text") or "")
    reasons = set(item.get("reasons") or [])
    if SUSPICIOUS_RE.search(text):
        return False
    if "formula_or_numeric_risk" in reasons or MATH_SIGNAL_RE.search(raw) or MATH_SIGNAL_RE.search(text):
        return False
    # Must be a real improvement over suspicious raw text.
    return bool(SUSPICIOUS_RE.search(raw) or len(text) > len(raw) + 3)


def load_or_init_corrections(path: Path) -> dict[str, Any]:
    if path.exists():
        data = read_json(path)
    else:
        data = {
            "schema_version": "2.0",
            "description": "Curated corrections for Grade 10 Physics formula/math review queue.",
            "review_policy": {
                "reviewed_blocks": "Exact text to insert into production text for the matching review_id.",
                "discard_review_ids": "Review IDs visually confirmed as non-content, duplicated fragments, or OCR garbage to omit.",
                "do_not_use_raw_corrupted_text": True,
            },
            "reviewed_blocks": {},
            "discard_review_ids": [],
            "global_replacements": [],
        }
    data.setdefault("reviewed_blocks", {})
    data.setdefault("discard_review_ids", [])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-review formula/OCR crops and merge accepted corrections into corrections JSON.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--step-json", type=Path, required=True, help="Step 2 or Step 3 JSON containing page_extractions and layout_lines.")
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--corrections-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/physics_tenth/auto_formula_review"))
    parser.add_argument("--provider", choices=["none", "tesseract", "openai"], default="none")
    parser.add_argument("--model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"))
    parser.add_argument("--confidence-threshold", type=float, default=float(os.getenv("PHYSICS_AUTO_REVIEW_THRESHOLD", "0.92")))
    parser.add_argument("--tesseract-lang", default="eng")
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument("--context-y", type=float, default=3.0, help="Vertical PDF-point context added above and below the crop.")
    parser.add_argument("--accept-tesseract", action="store_true", help="Allow conservative local OCR auto-acceptance for non-formula artifact-only lines.")
    parser.add_argument("--max-items", type=int, default=int(os.getenv("PHYSICS_AUTO_REVIEW_MAX_ITEMS", "0") or "0"), help="0 means all items.")
    parser.add_argument("--only-unresolved", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output dir already has files: {args.output_dir}. Use --force to overwrite/add.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "crops"

    step = read_json(args.step_json)
    queue = read_json(args.review_queue)
    items = list(queue.get("review_items") or [])
    if args.max_items and args.max_items > 0:
        items = items[: args.max_items]

    corrections = load_or_init_corrections(args.corrections_json)
    reviewed_blocks: dict[str, Any] = corrections.setdefault("reviewed_blocks", {})
    discard_ids = set(str(x) for x in corrections.setdefault("discard_review_ids", []))

    audit_rows: list[dict[str, Any]] = []
    accepted = 0
    discarded = 0
    needs_review = 0
    skipped_existing = 0

    with fitz.open(str(args.pdf)) as pdf:
        for idx, item in enumerate(items, start=1):
            rid = str(item.get("review_id") or "")
            if not rid:
                continue
            if args.only_unresolved and (rid in reviewed_blocks or rid in discard_ids):
                skipped_existing += 1
                continue
            page_number = int(item.get("page_number") or 0)
            page_data = find_page(step.get("page_extractions") or [], page_number)
            if not page_data:
                needs_review += 1
                audit_rows.append({"review_id": rid, "status": "review", "reason": "page_not_found"})
                continue

            crop = local_crop_review(pdf, page_data, item, crop_dir, args.dpi, args.tesseract_lang, args.context_y)
            raw = compact(item.get("raw_text") or "")
            reasons = item.get("reasons") or []
            action = "review"
            final_text = ""
            confidence = 0.0
            reason = "; ".join(crop.notes)

            if args.provider == "openai" and crop.image_path:
                try:
                    result = review_with_openai(crop.image_path, raw, reasons, args.model)
                    action = str(result.get("action") or "review").strip().lower()
                    final_text = clean_ocr_text(str(result.get("text") or ""))
                    confidence = float(result.get("confidence") or 0.0)
                    reason = str(result.get("reason") or reason)
                except Exception as exc:
                    action = "review"
                    reason = f"openai_error: {exc}; {reason}"
            elif args.provider == "tesseract" or args.accept_tesseract:
                if should_accept_tesseract(item, crop.tesseract_text, crop.tesseract_confidence, args.confidence_threshold):
                    action = "correct"
                    final_text = crop.tesseract_text
                    confidence = crop.tesseract_confidence
                    reason = "accepted_conservative_tesseract_non_formula_fix"
                else:
                    action = "review"
                    final_text = crop.tesseract_text
                    confidence = crop.tesseract_confidence
                    reason = "tesseract_candidate_only_not_auto_accepted; " + reason

            if action == "correct" and final_text and confidence >= args.confidence_threshold:
                reviewed_blocks[rid] = final_text
                accepted += 1
                status = "accepted"
            elif action == "discard":
                # For discard, the provider is saying the crop is non-content/duplicate/OCR garbage.
                # Some models return confidence=0.0 for discards because there is no text to transcribe.
                # Do not let the numeric transcription threshold block confirmed discard decisions.
                discard_ids.add(rid)
                discarded += 1
                status = "discarded"
            else:
                needs_review += 1
                status = "review"

            audit_rows.append({
                "review_id": rid,
                "page_number": page_number,
                "raw_text": raw,
                "reasons": "|".join(reasons),
                "crop_image": str(crop.image_path or ""),
                "bbox": json.dumps(crop.bbox or []),
                "tesseract_text": crop.tesseract_text,
                "tesseract_confidence": crop.tesseract_confidence,
                "provider": args.provider,
                "provider_action": action,
                "provider_text": final_text,
                "provider_confidence": confidence,
                "status": status,
                "reason": reason,
            })
            if idx % 25 == 0:
                print(f"Reviewed {idx}/{len(items)} | accepted={accepted} discarded={discarded} still_review={needs_review}")

    corrections["discard_review_ids"] = sorted(discard_ids)
    corrections.setdefault("auto_review_runs", []).append({
        "generated_at": utc_now(),
        "provider": args.provider,
        "model": args.model if args.provider == "openai" else None,
        "source_review_queue": str(args.review_queue),
        "source_step_json": str(args.step_json),
        "accepted_reviewed_blocks": accepted,
        "accepted_discards": discarded,
        "needs_manual_review": needs_review,
        "skipped_existing": skipped_existing,
        "confidence_threshold": args.confidence_threshold,
        "audit_csv": str(args.output_dir / "auto_review_audit.csv"),
    })
    write_json(args.corrections_json, corrections)

    audit_path = args.output_dir / "auto_review_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "review_id", "page_number", "raw_text", "reasons", "crop_image", "bbox",
            "tesseract_text", "tesseract_confidence", "provider", "provider_action",
            "provider_text", "provider_confidence", "status", "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Corrections JSON updated: {args.corrections_json}")
    print(f"Audit CSV: {audit_path}")
    print(f"Accepted reviewed_blocks: {accepted}")
    print(f"Accepted discards: {discarded}")
    print(f"Needs manual review: {needs_review}")
    if args.provider != "openai" and needs_review:
        print("For exact production formulas, rerun with --provider openai or manually fill remaining reviewed_blocks.")


if __name__ == "__main__":
    main()
