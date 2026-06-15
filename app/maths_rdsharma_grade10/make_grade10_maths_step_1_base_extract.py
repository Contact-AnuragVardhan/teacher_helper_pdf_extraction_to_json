#!/usr/bin/env python3
"""
Step 1: rendered-page OCR extraction for Grade10_Maths / R.D. Sharma Class X.

Design goals:
- Never use the corrupt selectable PDF text as production text.
- Render pages deterministically at high DPI and OCR the image.
- Cache page OCR so repeated runs are fast and reproducible.
- Build page, chapter, lesson, and TOC-section/day structures compatible with Teacher Helper ingestion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF

DEFAULT_ROOT = Path(os.environ.get("GRADE10_MATHS_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("GRADE10_MATHS_OUTPUT_DIR", DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"))
DEFAULT_PDF = Path(os.environ.get("GRADE10_MATHS_PDF", DEFAULT_ROOT / "input" / "Grade10_Maths.pdf"))
DEFAULT_CHAPTERS_JSON = Path(os.environ.get("GRADE10_MATHS_CHAPTERS_JSON", DEFAULT_OUTPUT_DIR / "Grade10_Maths_chapters.json"))
DEFAULT_SUBSECTIONS_JSON = Path(os.environ.get("GRADE10_MATHS_SUBSECTIONS_JSON", Path(__file__).resolve().parent / "Grade10_Maths_static_subsection_ranges.json"))

SAFE_NON_ASCII = set("₹°²³√×÷≤≥≠−–—’‘“”πθαβγ∆Δ∠⊥∥∴±∞∑")
SUSPICIOUS_UNICODE_RE = re.compile(r"[\u0900-\u097F\uFFFD\u0080-\u009F]")
MATH_LINE_RE = re.compile(r"(?=.*[0-9A-Za-z])(?=.*[=+\-−—*/×÷<>^|(){}\[\]√%°])")
HEADING_RE = re.compile(
    r"^(CHAPTER\s+\d+|\d{1,2}\.\d+\s+|EXAMPLE\s*\d+|ILLUSTRATION\s*\d+|EXERCISE\s+\d|BASED\s+ON|HINTS\s+TO|THEOREM|PROOF|SOLUTION|DEFINITION|REMARK)",
    re.IGNORECASE,
)


def stable_file_key(pdf_path: Path, *, scale: float, psm: str, lang: str) -> str:
    stat = pdf_path.stat()
    raw = f"{pdf_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|scale={scale}|psm={psm}|lang={lang}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("−", "-").replace("—", "-").replace("–", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    lines: list[str] = []
    for line in text.split("\n"):
        s = line.rstrip()
        # Drop long fake divider/border noise only.
        if len(s.strip()) > 25 and set(s.strip()) <= set("-_~=Ss. *|"):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalize_static_marker(label: Any) -> str:
    label = str(label or "").strip()
    if re.match(r"^\d{1,2}\.\d+", label):
        return label
    return label


def load_chapters(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("chapters"), list) or not data["chapters"]:
        raise ValueError(f"Invalid chapters JSON: {path}")
    return data


def load_static_subsection_plan(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    plan: dict[str, list[dict[str, Any]]] = {}
    for chapter in data.get("chapters") or []:
        cnum = str(chapter.get("chapter_number") or chapter.get("sequence") or "").strip()
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(chapter.get("days") or [], start=1):
            day_no = int(item.get("day") or idx)
            section_number = str(item.get("section_number") or "").strip()
            section_title = str(item.get("section_title") or item.get("includes") or "").strip()
            included = [normalize_static_marker(x) for x in (item.get("exercises") or [section_number]) if str(x).strip()]
            rows.append({
                "day": day_no,
                "day_title": item.get("day_title") or f"{section_number} {section_title}".strip() or f"Day {day_no}",
                "day_type": item.get("day_type") or "toc_section",
                "section_number": section_number,
                "section_title": section_title,
                "anchor_marker": section_number,
                "included_exercises_or_activities": included,
                "includes_text": item.get("includes") or section_title,
                "start_page": int(item["start_pdf_page"]),
                "end_page": int(item["end_pdf_page"]),
                "printed_start_page": int(item["start_book_page"]),
                "printed_end_page": int(item["end_book_page"]),
                "range_source": item.get("range_source") or data.get("subsection_policy") or "maintained_static_json",
                "boundary_overlap_with_previous_day": bool(item.get("boundary_overlap_with_previous_day")),
                "notes": [
                    "Subsection/day range loaded from Grade10_Maths_static_subsection_ranges.json; no runtime OCR section detection used.",
                    "Page-level range: if a section starts mid-page, the whole page is assigned to that section range.",
                ],
            })
        if rows:
            plan[cnum] = rows
    return data, plan


def ocr_page_worker(args: tuple[str, int, str, float, str, str, bool]) -> dict[str, Any]:
    pdf_path, page_number, cache_dir, scale, psm, lang, force = args
    pdf = Path(pdf_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"page_{page_number:04d}.json"
    if cache_file.exists() and not force:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["from_cache"] = True
            return cached
        except Exception:
            pass

    tmp_path: Optional[Path] = None
    try:
        doc = fitz.open(str(pdf))
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        pix.save(str(tmp_path))
        env = os.environ.copy()
        env["OMP_THREAD_LIMIT"] = "1"
        proc = subprocess.run(
            ["tesseract", str(tmp_path), "stdout", "-l", lang, "--psm", psm, "-c", "preserve_interword_spaces=1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
            timeout=90,
        )
        stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(stderr_text[:1000])
        result = {
            "page_number": page_number,
            "ocr_text_raw": clean_text(stdout_text),
            "ocr_error": None,
            "from_cache": False,
            "ocr_stderr_sample": stderr_text[:300],
        }
    except Exception as exc:
        result = {
            "page_number": page_number,
            "ocr_text_raw": "",
            "ocr_error": repr(exc),
            "from_cache": False,
        }
    finally:
        if tmp_path:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def extract_math_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if len(line) >= 3 and MATH_LINE_RE.search(line):
            out.append(line)
    return out[:100]


def extract_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for line in (text or "").split("\n"):
        s = line.strip()
        if not s:
            continue
        if HEADING_RE.search(s):
            if current:
                blocks.append(current)
            current = {"heading": s, "text": ""}
        elif current:
            current["text"] = (current["text"] + "\n" + s).strip()
    if current:
        blocks.append(current)
    return blocks[:80]


def ocr_quality_flags(text: str, selectable_text: str, error: Optional[str]) -> tuple[list[str], dict[str, Any]]:
    flags: list[str] = ["rendered_page_ocr_used_for_math_text"]
    suspicious = SUSPICIOUS_UNICODE_RE.findall(text or "")
    weird = [ch for ch in (text or "") if ord(ch) > 127 and ch not in SAFE_NON_ASCII]
    math_lines = extract_math_lines(text or "")
    metrics = {
        "text_length": len(text or ""),
        "suspicious_unicode_count": len(suspicious),
        "weird_non_ascii_count": len(weird),
        "weird_non_ascii_ratio": round(len(weird) / max(len(text or ""), 1), 6),
        "math_line_count": len(math_lines),
        "selectable_text_length": len(selectable_text or ""),
    }
    if error:
        flags.append("tesseract_ocr_error")
        metrics["ocr_error"] = error
    if not (text or "").strip():
        flags.append("blank_or_no_ocr_text_detected")
    if suspicious:
        flags.append("suspicious_unicode_in_ocr_text")
    if metrics["math_line_count"] > 25:
        flags.append("math_dense_page")
    return sorted(set(flags)), metrics


def printed_page_from_pdf(pdf_page: int, offset: int, content_start_pdf: int) -> Optional[int]:
    if pdf_page >= content_start_pdf:
        return pdf_page - offset
    return None


def classify_front_matter(pdf_page: int, text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "blank"
    if pdf_page == 1:
        return "cover"
    if "preface" in t:
        return "preface"
    if "contents" in t:
        return "toc"
    return "front_matter"


def find_chapter_for_page(pdf_page: int, chapters: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for ch in chapters:
        if int(ch["start_pdf_page"]) <= pdf_page <= int(ch["end_pdf_page"]):
            return ch
    return None


def format_page_block(page: dict[str, Any]) -> str:
    pp = page.get("printed_page_number") or page.get("printed_page_label")
    return f"[PDF page {page.get('page_number')} / printed page {pp}]\n{(page.get('text') or '').strip()}".strip()


def build_subsections_for_chapter(chapter: dict[str, Any], pages: list[dict[str, Any]], plan_by_chapter: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cnum = str(chapter.get("chapter_number"))
    plan = plan_by_chapter.get(cnum) or []
    page_by_number = {int(p["page_number"]): p for p in pages}
    if not plan:
        return []
    subsections: list[dict[str, Any]] = []
    for item in plan:
        start_page = int(item["start_page"])
        end_page = int(item["end_page"])
        subsection_pages = [page_by_number[p] for p in range(start_page, end_page + 1) if p in page_by_number]
        page_numbers = [p["page_number"] for p in subsection_pages]
        printed_pages = [p["printed_page_number"] for p in subsection_pages if p.get("printed_page_number") is not None]
        text = "\n\n".join(format_page_block(p) for p in subsection_pages if (p.get("text") or "").strip())
        math_lines: list[str] = []
        for p in subsection_pages:
            math_lines.extend((p.get("math_lines") or [])[:25])
        day_no = int(item["day"])
        subsection_title = item.get("day_title") or f"Day {day_no}"
        subsections.append({
            "section_number": cnum,
            "section_title": chapter.get("chapter_title"),
            "chapter_type": "chapter",
            "chapter_number": cnum,
            "chapter_title": chapter.get("chapter_title"),
            "subsection_number": f"{cnum}.{day_no}",
            "subsection_title": subsection_title,
            "day": day_no,
            "day_title": subsection_title,
            "day_type": item.get("day_type"),
            "toc_section_number": item.get("section_number"),
            "toc_section_title": item.get("section_title"),
            "anchor_marker": item.get("anchor_marker"),
            "anchor_pdf_page": start_page,
            "anchor_printed_page": item.get("printed_start_page"),
            "anchor_detection_method": "maintained_static_json_toc_ranges",
            "included_exercises_or_activities": item.get("included_exercises_or_activities") or [],
            "includes": item.get("included_exercises_or_activities") or [],
            "day_includes": item.get("includes_text"),
            "start_page": start_page,
            "end_page": end_page,
            "start_pdf_page": start_page,
            "end_pdf_page": end_page,
            "printed_start_page": printed_pages[0] if printed_pages else item.get("printed_start_page"),
            "printed_end_page": printed_pages[-1] if printed_pages else item.get("printed_end_page"),
            "pdf_pages": {"start": start_page, "end": end_page},
            "printed_pages": {"start": item.get("printed_start_page"), "end": item.get("printed_end_page")},
            "page_count": len(page_numbers),
            "physical_start_page": start_page,
            "physical_end_page": end_page,
            "physical_printed_start_page": item.get("printed_start_page"),
            "physical_printed_end_page": item.get("printed_end_page"),
            "physical_page_count": end_page - start_page + 1,
            "subsection_text": text,
            "subsection_text_plain": text,
            "text_plain": text,
            "subsection_math_lines": math_lines[:300],
            "math_lines": math_lines[:300],
            "page_numbers": page_numbers,
            "printed_page_numbers": printed_pages,
            "range_source": item.get("range_source"),
            "notes": item.get("notes") or [],
            "include_in_embeddings": bool(text),
        })
    return subsections


def parse_pages_arg(value: Optional[str], total_pages: int) -> list[int]:
    if not value:
        return list(range(1, total_pages + 1))
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(max(1, int(a)), min(total_pages, int(b)) + 1))
        else:
            n = int(part)
            if 1 <= n <= total_pages:
                pages.add(n)
    return sorted(pages)


def build_run_scope(*, total_pages: int, selected_pages: list[int], content_start_pdf: int, content_end_pdf: int, page_extractions: list[dict[str, Any]], chapters: list[dict[str, Any]], plan_by_chapter: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected = set(selected_pages)
    extracted = {int(p.get("page_number")) for p in page_extractions if p.get("page_number") is not None}
    expected_all_pages = set(range(1, total_pages + 1))
    expected_content_pages = set(range(content_start_pdf, content_end_pdf + 1))
    missing_selected = sorted(selected - extracted)
    missing_content = sorted(expected_content_pages - extracted)

    chapter_coverage_issues: list[dict[str, Any]] = []
    for chapter in chapters:
        cpages = set(range(int(chapter["start_pdf_page"]), int(chapter["end_pdf_page"]) + 1))
        missing = sorted(cpages - extracted)
        if missing:
            chapter_coverage_issues.append({
                "chapter_number": str(chapter.get("chapter_number")),
                "chapter_title": chapter.get("chapter_title"),
                "missing_pdf_pages": missing[:100],
                "missing_page_count": len(missing),
            })

    subsection_coverage_issues: list[dict[str, Any]] = []
    for chapter_number, rows in plan_by_chapter.items():
        for row in rows:
            spages = set(range(int(row["start_page"]), int(row["end_page"]) + 1))
            missing = sorted(spages - extracted)
            if missing:
                subsection_coverage_issues.append({
                    "chapter_number": str(chapter_number),
                    "day": int(row.get("day") or 0),
                    "day_title": row.get("day_title"),
                    "missing_pdf_pages": missing[:100],
                    "missing_page_count": len(missing),
                })

    is_full_book_run = selected == expected_all_pages and not missing_selected
    is_full_content_run = expected_content_pages.issubset(extracted)
    return {
        "requested_pages_arg": None,
        "selected_page_count": len(selected_pages),
        "selected_pages_preview": selected_pages[:20] + (["..."] if len(selected_pages) > 40 else []) + (selected_pages[-20:] if len(selected_pages) > 40 else []),
        "total_pdf_pages": total_pages,
        "content_start_page": content_start_pdf,
        "content_end_page": content_end_pdf,
        "is_full_book_run": is_full_book_run,
        "is_full_content_run": is_full_content_run,
        "missing_selected_pages": missing_selected,
        "missing_content_pages": missing_content[:200],
        "missing_content_page_count": len(missing_content),
        "chapter_coverage_issues": chapter_coverage_issues,
        "subsection_coverage_issues": subsection_coverage_issues,
        "production_candidate": bool(is_full_book_run and is_full_content_run and not chapter_coverage_issues and not subsection_coverage_issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rendered-page OCR extraction for Grade10_Maths.pdf")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--chapters-json", type=Path, default=DEFAULT_CHAPTERS_JSON)
    parser.add_argument("--subsections-json", type=Path, default=DEFAULT_SUBSECTIONS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=float(os.environ.get("GRADE10_MATHS_OCR_SCALE", "3.0")))
    parser.add_argument("--psm", default=os.environ.get("GRADE10_MATHS_TESSERACT_PSM", "4"))
    parser.add_argument("--lang", default=os.environ.get("GRADE10_MATHS_TESSERACT_LANG", "eng"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("GRADE10_MATHS_OCR_WORKERS", "4")))
    parser.add_argument("--pages", default=None, help="Optional page subset for smoke tests, e.g. 1-20,500")
    parser.add_argument("--force-ocr", action="store_true", help="Ignore OCR cache and OCR pages again")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    if not args.chapters_json.exists():
        raise FileNotFoundError(args.chapters_json)
    if not args.subsections_json.exists():
        raise FileNotFoundError(args.subsections_json)
    try:
        subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as exc:
        raise SystemExit("Tesseract is required on PATH. Confirm `tesseract --version` works.") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "Grade10_Maths_step1_base_extraction.json")
    report = args.report or (args.output_dir / "Grade10_Maths_step1_validation_report.txt")

    chapters_payload = load_chapters(args.chapters_json)
    static_config, plan_by_chapter = load_static_subsection_plan(args.subsections_json)
    offset = int(chapters_payload["printed_page_offset"])
    content_start_pdf = int(chapters_payload["content_start_page"])
    content_end_pdf = int(chapters_payload["content_end_page"])
    chapters = chapters_payload["chapters"]

    doc = fitz.open(str(args.pdf))
    total_pages = doc.page_count
    selected_pages = parse_pages_arg(args.pages, total_pages)
    cache_key = stable_file_key(args.pdf, scale=args.scale, psm=args.psm, lang=args.lang)
    cache_dir = args.output_dir / ".ocr_cache" / f"grade10_maths_scale_{str(args.scale).replace('.', '_')}_{cache_key}"

    selectable_by_page: dict[int, str] = {}
    for p in selected_pages:
        selectable_by_page[p] = clean_text(doc.load_page(p - 1).get_text("text") or "")

    print(f"PDF pages: {total_pages}")
    print(f"Selected pages: {len(selected_pages)}")
    print(f"Running OCR scale={args.scale}, estimated_dpi={int(args.scale*72)}, psm={args.psm}, workers={args.workers}")
    tasks = [(str(args.pdf), p, str(cache_dir), args.scale, args.psm, args.lang, args.force_ocr) for p in selected_pages]
    ocr_by_page: dict[int, dict[str, Any]] = {}
    if args.workers <= 1:
        for i, task in enumerate(tasks, start=1):
            result = ocr_page_worker(task)
            ocr_by_page[int(result["page_number"])] = result
            if i % 25 == 0 or i == len(tasks):
                print(f"OCR completed {i}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(ocr_page_worker, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                ocr_by_page[int(result["page_number"])] = result
                if i % 25 == 0 or i == len(futures):
                    print(f"OCR completed {i}/{len(futures)}")

    page_extractions: list[dict[str, Any]] = []
    front_matter_pages: list[dict[str, Any]] = []
    pages_by_chapter: dict[str, list[dict[str, Any]]] = {str(ch["chapter_number"]): [] for ch in chapters}

    for p in selected_pages:
        selectable_text = selectable_by_page.get(p, "")
        ocr_result = ocr_by_page.get(p, {})
        text = clean_text(ocr_result.get("ocr_text_raw") or "")
        assigned = find_chapter_for_page(p, chapters)
        flags, metrics = ocr_quality_flags(text, selectable_text, ocr_result.get("ocr_error"))
        printed = printed_page_from_pdf(p, offset, content_start_pdf)
        base = {
            "page_number": p,
            "printed_page_number": printed,
            "printed_page_label": str(printed) if printed is not None else None,
            "text": text,
            "text_plain": text,
            "ocr_text": text,
            "selectable_text": selectable_text,
            "math_lines": extract_math_lines(text),
            "extracted_blocks": extract_blocks(text),
            "ocr_engine": "tesseract",
            "ocr_profile": {
                "method": "rendered_page_tesseract_ocr",
                "render_scale": args.scale,
                "estimated_dpi": int(args.scale * 72),
                "language": args.lang,
                "psm": args.psm,
                "preserve_interword_spaces": True,
                "cache_dir": str(cache_dir),
                "from_cache": bool(ocr_result.get("from_cache")),
            },
            "text_sources": ["rendered_page_tesseract_ocr", "pdf_text_layer_reference_only"],
            "quality_flags": flags,
            "quality_metrics": metrics,
        }
        if assigned:
            rec = {
                **base,
                "chapter_type": "chapter",
                "chapter_number": str(assigned["chapter_number"]),
                "chapter_title": assigned["chapter_title"],
                "section_number": str(assigned["chapter_number"]),
                "section_title": assigned["chapter_title"],
                "content_type": "lesson_body",
                "assignment_status": "assigned_to_chapter",
                "include_in_lesson_text": True,
                "include_in_embeddings": bool(text),
                "linked_section_title": None,
                "linked_section_number": None,
                "unit_number": None,
                "unit_title": None,
            }
            pages_by_chapter[str(assigned["chapter_number"])].append(rec)
        else:
            ctype = classify_front_matter(p, text)
            rec = {
                **base,
                "chapter_type": None,
                "chapter_number": None,
                "chapter_title": None,
                "section_number": None,
                "section_title": None,
                "content_type": ctype,
                "assignment_status": "front_matter" if ctype != "blank" else "blank",
                "include_in_lesson_text": False,
                "include_in_embeddings": False,
                "linked_section_title": None,
                "linked_section_number": None,
                "unit_number": None,
                "unit_title": None,
            }
            front_matter_pages.append(rec)
        page_extractions.append(rec)

    chapters_out: list[dict[str, Any]] = []
    section_index: list[dict[str, Any]] = []
    for ch in chapters:
        cnum = str(ch["chapter_number"])
        cpages = sorted(pages_by_chapter.get(cnum) or [], key=lambda x: int(x["page_number"]))
        indexed_pages = [p for p in cpages if p.get("include_in_lesson_text")]
        page_numbers = [p["page_number"] for p in indexed_pages]
        printed_numbers = [p["printed_page_number"] for p in indexed_pages if p.get("printed_page_number") is not None]
        lesson_text = "\n\n".join(format_page_block(p) for p in indexed_pages if (p.get("text") or "").strip())
        math_lines: list[str] = []
        for p in indexed_pages:
            math_lines.extend((p.get("math_lines") or [])[:25])
        lesson = {
            "section_number": cnum,
            "section_title": ch["chapter_title"],
            "chapter_type": "chapter",
            "chapter_number": cnum,
            "chapter_title": ch["chapter_title"],
            "unit_number": None,
            "unit_title": None,
            "start_page": page_numbers[0] if page_numbers else ch["start_pdf_page"],
            "end_page": page_numbers[-1] if page_numbers else ch["end_pdf_page"],
            "printed_start_page": printed_numbers[0] if printed_numbers else ch["printed_start_page"],
            "printed_end_page": printed_numbers[-1] if printed_numbers else ch["printed_end_page"],
            "page_count": len(page_numbers),
            "lesson_text": lesson_text,
            "text_plain": lesson_text,
            "math_lines": math_lines[:500],
            "physical_start_page": ch["start_pdf_page"],
            "physical_end_page": ch["end_pdf_page"],
            "physical_printed_start_page": ch["printed_start_page"],
            "physical_printed_end_page": ch["printed_end_page"],
            "physical_page_count": int(ch["end_pdf_page"]) - int(ch["start_pdf_page"]) + 1,
            "page_numbers": page_numbers,
            "printed_page_numbers": printed_numbers,
            "text_sources": ["rendered_page_tesseract_ocr"],
            "quality_flags": sorted({flag for p in indexed_pages for flag in p.get("quality_flags", [])}),
            "include_in_embeddings": bool(lesson_text),
        }
        subsections = build_subsections_for_chapter(ch, indexed_pages, plan_by_chapter)
        chapter_obj = {
            "chapter_type": "chapter",
            "chapter_number": cnum,
            "chapter_title": ch["chapter_title"],
            "unit_number": None,
            "unit_title": None,
            "start_page": ch["start_pdf_page"],
            "end_page": ch["end_pdf_page"],
            "printed_start_page": ch["printed_start_page"],
            "printed_end_page": ch["printed_end_page"],
            "lessons": [lesson],
            "subsections": subsections,
        }
        chapters_out.append(chapter_obj)
        section_index.append({
            "section_number": cnum,
            "section_title": ch["chapter_title"],
            "chapter_type": "chapter",
            "chapter_number": cnum,
            "chapter_title": ch["chapter_title"],
            "start_page": lesson["start_page"],
            "end_page": lesson["end_page"],
            "printed_start_page": lesson["printed_start_page"],
            "printed_end_page": lesson["printed_end_page"],
            "page_count": lesson["page_count"],
            "text_length_chars": len(lesson_text),
            "physical_start_page": lesson["physical_start_page"],
            "physical_end_page": lesson["physical_end_page"],
            "physical_printed_start_page": lesson["physical_printed_start_page"],
            "physical_printed_end_page": lesson["physical_printed_end_page"],
            "physical_page_count": lesson["physical_page_count"],
            "indexed_page_count": lesson["page_count"],
            "indexed_page_numbers": lesson["page_numbers"],
            "indexed_printed_page_numbers": lesson["printed_page_numbers"],
            "text_sources": lesson["text_sources"],
            "quality_flags": lesson["quality_flags"],
            "subsections": subsections,
        })

    total_text_chars = sum(len(p.get("text") or "") for p in page_extractions)
    run_scope = build_run_scope(
        total_pages=total_pages,
        selected_pages=selected_pages,
        content_start_pdf=content_start_pdf,
        content_end_pdf=content_end_pdf,
        page_extractions=page_extractions,
        chapters=chapters,
        plan_by_chapter=plan_by_chapter,
    )
    run_scope["requested_pages_arg"] = args.pages
    data = {
        "metadata": {
            "school_name": os.environ.get("GRADE10_MATHS_SCHOOL_NAME", "Mother Miracle School"),
            "class_name": "Class-10",
            "grade": "Class-10",
            "board": "CBSE",
            "medium": "English",
            "publisher": "Dhanpat Rai Publications",
            "copyright_status": "copyrighted",
            "source_file": args.pdf.name,
        },
        "extraction": {
            "book_title": "Mathematics for Class X",
            "subject": "Maths",
            "language": "English",
            "content_profile": "math_textbook",
            "structure_type": "chapters_with_toc_sections",
            "author": "R.D. Sharma",
            "total_pdf_pages": total_pages,
            "content_start_page": content_start_pdf,
            "content_end_page": content_end_pdf,
            "printed_page_offset": offset,
            "printed_start_page": chapters_payload.get("printed_start_page"),
            "printed_end_page": chapters_payload.get("printed_end_page"),
            "subsection_policy": static_config.get("subsection_policy"),
            "day_split_policy": static_config.get("day_split_policy"),
            "subsections_json_source": str(args.subsections_json),
            "math_ocr_profile": {
                "method": "deterministic_high_resolution_rendered_page_ocr_before_embedding",
                "engine": "tesseract",
                "language": args.lang,
                "psm": args.psm,
                "render_scale": args.scale,
                "estimated_dpi": int(args.scale * 72),
                "preserve_interword_spaces": True,
                "text_layer_policy": "selectable PDF text is kept only for audit; final text uses rendered-page OCR",
            },
            "notes": [
                "This JSON regenerates page text from rendered page images instead of relying on the corrupt selectable PDF/OCR layer.",
                "Printed page 1 maps to PDF page 8, so printed_page_offset is 7.",
                "Subsection/day ranges are loaded from Grade10_Maths_static_subsection_ranges.json using the book contents-in-detail section starts.",
                "For exact symbolic math on dense formula pages, use the Step 2 production gate and send excluded pages to vision/Mathpix/manual QA before embedding.",
            ],
            "section_index": section_index,
            "chapters": chapters_out,
            "front_matter_pages": front_matter_pages,
            "page_extractions": page_extractions,
            "transcripts": [],
            "unit_level_pages": [],
            "detected_transcript_pages": [],
            "quality_summary": {
                "total_pages": total_pages,
                "pages_in_this_run": len(selected_pages),
                "assigned_content_pages": sum(1 for p in page_extractions if p.get("include_in_lesson_text")),
                "total_chapters": len(chapters_out),
                "total_subsections": sum(len(ch.get("subsections", [])) for ch in chapters_out),
                "subsections_json_source": str(args.subsections_json),
                "front_matter_page_count": len(front_matter_pages),
                "empty_or_no_text_pages": [p["page_number"] for p in page_extractions if not (p.get("text") or "").strip()],
                "pages_with_suspicious_unicode_in_final_ocr_text": [p["page_number"] for p in page_extractions if p.get("quality_metrics", {}).get("suspicious_unicode_count", 0) > 0],
                "total_extracted_text_chars": total_text_chars,
                "recommended_embedding_text_field_after_step2": "production_safe_text",
                "recommended_embedding_filter_after_step2": "include_in_embeddings == true and embedding_readiness == 'ready_for_production_embedding'",
                "run_scope": run_scope,
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "Grade10_Maths Step 1 Base OCR Extraction Report",
        "=" * 60,
        f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"PDF: {args.pdf}",
        f"Total PDF pages: {total_pages}",
        f"Pages processed in this run: {len(selected_pages)}",
        f"Chapters: {len(chapters_out)}",
        f"Subsections/days: {sum(len(ch.get('subsections', [])) for ch in chapters_out)}",
        f"Full-book run: {run_scope['is_full_book_run']}",
        f"Full-content run: {run_scope['is_full_content_run']}",
        f"Missing selected pages: {len(run_scope['missing_selected_pages'])}",
        f"Missing content pages: {run_scope['missing_content_page_count']}",
        f"OCR scale / estimated DPI: {args.scale} / {int(args.scale*72)}",
        f"OCR cache: {cache_dir}",
        f"Empty/no-text pages: {data['extraction']['quality_summary']['empty_or_no_text_pages'][:50]}",
        f"Suspicious unicode pages: {data['extraction']['quality_summary']['pages_with_suspicious_unicode_in_final_ocr_text'][:50]}",
        "",
        "Step 1 is not the production embedding artifact. Run Step 2 to clean and gate pages before vector ingestion.",
    ]
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {output_json}")
    print(f"Wrote: {report}")


if __name__ == "__main__":
    main()
