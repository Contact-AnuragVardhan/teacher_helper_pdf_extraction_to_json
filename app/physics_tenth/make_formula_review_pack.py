#!/usr/bin/env python3
"""Create a visual review pack for unresolved Physics formula/OCR items.

This utility does not guess formulas. It renders a small PDF crop around each
review_required item so a reviewer can type exact text into
Grade10_Physics_formula_corrections.json.

Example:
python app/physics_tenth/make_formula_review_pack.py \
  --pdf input/Grade10_Physics.pdf \
  --step-json output/physics_tenth/Grade10_Physics_step3_reviewed_text_extraction.json \
  --review-queue output/physics_tenth/Grade10_Physics_step3_remaining_review_queue.json \
  --output-dir output/physics_tenth/formula_review_pack \
  --force
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def find_bbox(page_data: dict[str, Any], item: dict[str, Any]) -> fitz.Rect | None:
    """Best-effort bbox lookup from Step JSON layout_lines.

    line_items are produced after cleanup, so their line_no can sometimes be off
    by one or two compared with raw layout_lines. We therefore try exact line_no,
    nearby line_no, then text match.
    """
    target_line = int(item.get("line_no") or -1)
    raw = str(item.get("raw_text") or item.get("text") or "").strip()
    layout = page_data.get("layout_lines") or []

    candidates: list[dict[str, Any]] = []
    for ln in layout:
        try:
            ln_no = int(ln.get("line_no") or -999999)
        except Exception:
            ln_no = -999999
        if ln_no == target_line:
            candidates.append(ln)
    if not candidates:
        for ln in layout:
            try:
                ln_no = int(ln.get("line_no") or -999999)
            except Exception:
                ln_no = -999999
            if abs(ln_no - target_line) <= 2:
                candidates.append(ln)
    if not candidates and raw:
        raw_norm = re.sub(r"\s+", " ", raw).lower()
        for ln in layout:
            txt = re.sub(r"\s+", " ", str(ln.get("text") or "").strip()).lower()
            if txt and (txt == raw_norm or raw_norm in txt or txt in raw_norm):
                candidates.append(ln)
                break

    if not candidates:
        return None
    rects = [fitz.Rect(c["bbox"]) for c in candidates if c.get("bbox")]
    if not rects:
        return None
    rect = rects[0]
    for r in rects[1:]:
        rect |= r
    return rect


def render_crop(doc: fitz.Document, page_number: int, rect: fitz.Rect | None, out_path: Path, zoom: float = 3.0) -> None:
    page = doc[page_number - 1]
    page_rect = page.rect
    if rect is None:
        clip = page_rect
    else:
        # Expand heavily because formulas often span adjacent visual lines.
        clip = fitz.Rect(rect.x0 - 80, rect.y0 - 45, rect.x1 + 260, rect.y1 + 65)
        clip &= page_rect
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual review pack for Physics formula review IDs.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--step-json", type=Path, required=True, help="Step 2 or Step 3 JSON containing layout_lines/page_extractions")
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional max review items to render")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output folder exists and is not empty: {args.output_dir}. Use --force.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    step_data = read_json(args.step_json)
    queue = read_json(args.review_queue)
    pages_by_num = {int(p["page_number"]): p for p in step_data.get("page_extractions") or []}
    review_items = list(queue.get("review_items") or [])
    if args.limit and args.limit > 0:
        review_items = review_items[: args.limit]

    doc = fitz.open(args.pdf)
    csv_path = args.output_dir / "review_items.csv"
    html_path = args.output_dir / "index.html"

    rows = []
    for item in review_items:
        rid = str(item.get("review_id"))
        page_number = int(item.get("page_number"))
        page_data = pages_by_num.get(page_number, {})
        rect = find_bbox(page_data, item)
        crop_file = crops_dir / f"{safe_name(rid)}_p{page_number:03d}.png"
        render_crop(doc, page_number, rect, crop_file)
        rows.append({
            "review_id": rid,
            "page_number": page_number,
            "printed_page_number": item.get("printed_page_number"),
            "chapter_title": item.get("chapter_title"),
            "line_no": item.get("line_no"),
            "reasons": ";".join(item.get("reasons") or []),
            "raw_text": item.get("raw_text") or "",
            "reviewed_text": "",
            "discard": "",
            "crop_file": str(crop_file.relative_to(args.output_dir)).replace('\\\\', '/'),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["review_id"])
        writer.writeheader()
        writer.writerows(rows)

    parts = [
        "<html><head><meta charset='utf-8'><title>Physics Formula Review Pack</title>",
        "<style>body{font-family:Arial,sans-serif} .item{border:1px solid #ccc;margin:16px;padding:12px} img{max-width:100%;border:1px solid #ddd} code{background:#f6f6f6;padding:2px 4px}</style>",
        "</head><body>",
        f"<h1>Physics Formula Review Pack</h1><p>Total rendered items: {len(rows)}</p>",
        "<p>Type exact corrected text into <code>review_items.csv</code>, then copy values into <code>Grade10_Physics_formula_corrections.json</code> under <code>reviewed_blocks</code>. Mark <code>discard=yes</code> only for confirmed non-content/duplicates/garbage.</p>",
    ]
    for r in rows:
        parts.append("<div class='item'>")
        parts.append(f"<h3>{html.escape(r['review_id'])} — page {r['page_number']} / line {r['line_no']}</h3>")
        parts.append(f"<p><b>Chapter:</b> {html.escape(str(r['chapter_title']))}<br><b>Reasons:</b> {html.escape(r['reasons'])}</p>")
        parts.append(f"<p><b>Raw:</b> <code>{html.escape(r['raw_text'])}</code></p>")
        parts.append(f"<img src='{html.escape(r['crop_file'])}' />")
        parts.append("</div>")
    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")

    print(f"Review pack written: {args.output_dir}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
