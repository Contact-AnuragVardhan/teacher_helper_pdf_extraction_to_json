#!/usr/bin/env python3
"""
make_maths_rsaggarwal_chapters.py

Step 0 for the R. S. Aggarwal maths pipeline.

Purpose:
  Read the book's Contents page from the scanned PDF and generate the CHAPTERS
  array that make_maths_rsaggarwal_step_1.py currently keeps as a static list.

Outputs:
  1) JSON config with:
       - printed_page_offset
       - toc_pdf_page
       - chapters
  2) Optional Python file containing:
       PRINTED_OFFSET = ...
       CHAPTERS = [...]
  3) Prints the CHAPTERS array to console.

Run from project root:
  python app/maths_rsaggarwal/make_maths_rsaggarwal_chapters.py \
    --pdf input/Maths_RSAgarwal.pdf \
    --output-json output/maths_rsagarwal/Maths_RSAgarwal_chapters.json \
    --output-py output/maths_rsagarwal/Maths_RSAgarwal_chapters.py

Dependencies:
  pip install pymupdf
  Tesseract must be installed and available on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF


@dataclass
class PageOcr:
    page_number: int  # 1-based PDF page number
    text: str


CHAPTER_LINE_RE = re.compile(
    r"^\s*(?P<num>\d{1,2})\s*[\.)\],:]?\s+"
    r"(?P<title>.+?)"
    r"(?:\s|\.)+"
    r"(?P<page>\d{1,4})\s*$"
)

ANSWERS_LINE_RE = re.compile(r"^\s*Answers\s+(?P<page>\d{1,4})\s*$", re.IGNORECASE)

NOISE_LINE_RE = re.compile(r"^\s*(?:contents|\([ivxlcdm]+\)|page|chapter)\s*$", re.IGNORECASE)


def run_tesseract_on_page(
    doc: fitz.Document,
    page_number: int,
    *,
    scale: float = 3.0,
    psm: str = "6",
    lang: str = "eng",
    crop_top_ratio: float | None = None,
) -> PageOcr:
    """Render a 1-based PDF page and OCR it with Tesseract."""
    page = doc.load_page(page_number - 1)
    rect = page.rect
    clip = None
    if crop_top_ratio is not None:
        crop_top_ratio = max(0.05, min(1.0, crop_top_ratio))
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * crop_top_ratio)

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, clip=clip)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_path = Path(tmp.name)

    try:
        pix.save(str(img_path))
        env = os.environ.copy()
        env["OMP_THREAD_LIMIT"] = "1"
        cmd = [
            "tesseract",
            str(img_path),
            "stdout",
            "-l",
            lang,
            "--psm",
            psm,
            "-c",
            "preserve_interword_spaces=1",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
            timeout=45,
        )
        stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Tesseract failed on PDF page {page_number} with exit code {proc.returncode}: "
                f"{stderr_text[:1000]}"
            )
        return PageOcr(page_number=page_number, text=normalize_ocr_text(stdout_text))
    finally:
        try:
            img_path.unlink(missing_ok=True)
        except OSError:
            pass


def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    lines = []
    for line in text.split("\n"):
        s = re.sub(r"\s+", " ", line).strip()
        if s:
            lines.append(s)
    return "\n".join(lines).strip()


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    title = title.strip(" .·•:-")

    # Conservative cleanups for common TOC OCR punctuation only.
    title = title.replace("Three Dimensional", "Three-Dimensional")
    title = title.replace("Three -Dimensional", "Three-Dimensional")
    title = title.replace("One variable", "One Variable")
    title = title.replace("Mean, Median And Mode", "Mean, Median and Mode")

    # Fix occasional OCR spacing around parentheses.
    title = re.sub(r"\s+\)", ")", title)
    title = re.sub(r"\(\s+", "(", title)
    return title


def parse_chapters_from_toc_text(toc_text: str) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    seen_numbers: set[str] = set()

    for raw_line in toc_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or NOISE_LINE_RE.match(line):
            continue

        line = line.replace("।", ".")
        line = re.sub(r"^(\d{1,2})\s*,\s+", r"\1. ", line)  # OCR: "4, Rational Numbers 54"
        line = re.sub(r"^(\d{1,2})\s+", r"\1. ", line)    # OCR: "1 Integers 1"

        m = CHAPTER_LINE_RE.match(line)
        if m:
            num = m.group("num")
            title = clean_title(m.group("title"))
            page = int(m.group("page"))

            if num in seen_numbers:
                continue
            if not title or page <= 0:
                continue

            chapters.append(
                {
                    "chapter_number": num,
                    "chapter_title": title,
                    "printed_start_page": page,
                }
            )
            seen_numbers.add(num)
            continue

        m = ANSWERS_LINE_RE.match(line)
        if m:
            if "A" not in seen_numbers:
                chapters.append(
                    {
                        "chapter_number": "A",
                        "chapter_title": "Answers",
                        "printed_start_page": int(m.group("page")),
                        "chapter_type": "appendix",
                    }
                )
                seen_numbers.add("A")

    chapters = sorted(
        chapters,
        key=lambda ch: (9999 if ch["chapter_number"] == "A" else int(ch["chapter_number"])),
    )
    validate_chapters(chapters)
    return chapters


def validate_chapters(chapters: list[dict[str, Any]]) -> None:
    if len(chapters) < 5:
        raise ValueError(f"Only {len(chapters)} chapter rows were detected from TOC; OCR/parse likely failed.")

    numeric = [ch for ch in chapters if str(ch.get("chapter_number", "")).isdigit()]
    expected = list(range(1, len(numeric) + 1))
    got = [int(ch["chapter_number"]) for ch in numeric]
    if got != expected:
        raise ValueError(f"Chapter numbers are not consecutive. Expected {expected}, got {got}")

    starts = [int(ch["printed_start_page"]) for ch in chapters]
    if starts != sorted(starts):
        raise ValueError(f"Printed start pages are not increasing: {starts}")

    if chapters[-1].get("chapter_title") != "Answers":
        raise ValueError("Could not detect final Answers appendix row from TOC.")


def toc_score(text: str) -> int:
    lower = text.lower()
    score = 0
    if "contents" in lower:
        score += 5
    for token in [
        "integers",
        "fractions",
        "decimals",
        "rational numbers",
        "algebraic expressions",
        "linear equations",
        "mensuration",
        "answers",
    ]:
        if token in lower:
            score += 1
    # TOC should have many numbered rows.
    score += len(re.findall(r"(?m)^\s*\d{1,2}[\.)\],:]?\s+", text))
    return score


def find_toc_page(doc: fitz.Document, max_front_pages: int, *, lang: str) -> tuple[int, str]:
    """Find and OCR the table-of-contents page."""
    max_front_pages = min(max_front_pages, doc.page_count)
    candidates: list[tuple[int, int, str]] = []

    for page_number in range(1, max_front_pages + 1):
        # Selectable text may be corrupt, but it is cheap and good enough to shortlist.
        selectable = normalize_ocr_text(doc.load_page(page_number - 1).get_text("text") or "")
        score = toc_score(selectable)
        candidates.append((score, page_number, selectable))

    candidates.sort(reverse=True)
    shortlisted = [p for _, p, _ in candidates[:4]]

    best: tuple[int, int, str] | None = None
    for page_number in shortlisted:
        ocr = run_tesseract_on_page(doc, page_number, scale=3.0, psm="6", lang=lang).text
        score = toc_score(ocr)
        if best is None or score > best[0]:
            best = (score, page_number, ocr)

    if not best or best[0] < 8:
        raise ValueError("Could not confidently find the Contents page in the PDF front matter.")

    return best[1], best[2]


def normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_printed_page_offset(
    doc: fitz.Document,
    *,
    toc_pdf_page: int,
    first_chapter_title: str,
    first_printed_page: int,
    lang: str,
) -> int:
    """Find physical PDF page for first printed page and return physical - printed offset."""
    needle = normalize_for_match(first_chapter_title)
    best: tuple[int, int] | None = None

    start = toc_pdf_page + 1
    end = min(doc.page_count, toc_pdf_page + 12)
    for page_number in range(start, end + 1):
        selectable = normalize_ocr_text(doc.load_page(page_number - 1).get_text("text") or "")
        top_text = selectable[:1000]
        score = 0
        if needle and needle in normalize_for_match(top_text):
            score += 5

        # OCR only the top half; the first chapter heading is usually near top of first content page.
        if score < 5:
            try:
                ocr_top = run_tesseract_on_page(
                    doc,
                    page_number,
                    scale=2.5,
                    psm="6",
                    lang=lang,
                    crop_top_ratio=0.45,
                ).text
                lines = [ln.strip() for ln in ocr_top.splitlines() if ln.strip()]
                top_joined = normalize_for_match(" ".join(lines[:8]))
                if needle and needle in top_joined:
                    score += 10
            except Exception:
                pass

        if best is None or score > best[0]:
            best = (score, page_number)

    if not best or best[0] < 5:
        # Conservative fallback for books where printed page 1 starts immediately after TOC/front matter.
        # We still fail loudly unless caller passes --allow-offset-fallback.
        raise ValueError(
            f"Could not locate first chapter page for title {first_chapter_title!r}. "
            "Pass --printed-page-offset manually if needed."
        )

    first_pdf_page = best[1]
    return first_pdf_page - first_printed_page


def write_python_config(path: Path, *, printed_offset: int, chapters: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Auto-generated by make_maths_rsaggarwal_chapters.py\n"
        f"# Generated at UTC: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"PRINTED_OFFSET = {printed_offset}\n\n"
        "CHAPTERS = "
        + json.dumps(chapters, ensure_ascii=False, indent=4)
        + "\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHAPTERS array from Maths RSAggarwal PDF Contents page.")
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON config path")
    parser.add_argument("--output-py", type=Path, default=None, help="Optional Python file containing PRINTED_OFFSET and CHAPTERS")
    parser.add_argument("--front-pages", type=int, default=12, help="How many front pages to scan for Contents")
    parser.add_argument("--lang", default="eng", help="Tesseract language; default eng")
    parser.add_argument("--printed-page-offset", type=int, default=None, help="Override printed-page offset instead of auto-detecting it")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    try:
        subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as exc:
        raise SystemExit("Tesseract is required on PATH. Confirm `tesseract --version` works.") from exc

    doc = fitz.open(str(args.pdf))
    toc_pdf_page, toc_text = find_toc_page(doc, args.front_pages, lang=args.lang)
    chapters = parse_chapters_from_toc_text(toc_text)

    if args.printed_page_offset is None:
        printed_offset = find_printed_page_offset(
            doc,
            toc_pdf_page=toc_pdf_page,
            first_chapter_title=chapters[0]["chapter_title"],
            first_printed_page=int(chapters[0]["printed_start_page"]),
            lang=args.lang,
        )
    else:
        printed_offset = int(args.printed_page_offset)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pdf": str(args.pdf),
        "total_pdf_pages": doc.page_count,
        "toc_pdf_page": toc_pdf_page,
        "printed_page_offset": printed_offset,
        "content_start_page": chapters[0]["printed_start_page"] + printed_offset,
        "answers_start_page": chapters[-1]["printed_start_page"] + printed_offset,
        "chapters": chapters,
        "toc_ocr_text": toc_text,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_py:
        write_python_config(args.output_py, printed_offset=printed_offset, chapters=chapters)

    print("Detected TOC PDF page:", toc_pdf_page)
    print("Detected printed-page offset:", printed_offset)
    print("Detected chapters:", len(chapters))
    print("\nCHAPTERS = ")
    print(json.dumps(chapters, ensure_ascii=False, indent=4))
    print(f"\nWrote: {args.output_json}")
    if args.output_py:
        print(f"Wrote: {args.output_py}")


if __name__ == "__main__":
    main()
