#!/usr/bin/env python3
"""
make_english_poorvi.py

Orchestrates the full English Poorvi extraction pipeline from the project root.

Runs:
  1) poorvi_step_1_base_extract.py
  2) poorvi_step_2_hybrid_correct.py
  3) poorvi_step_3_publish_production.py

Usage from project root:
  python app/make_english_poorvi.py --force

Optional:
  python app/make_english_poorvi.py --pdf input/English_Poorvi.pdf --output-dir output/english_poorvi --force
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("english_poorvi.pipeline")


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging for command-line debugging."""
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )



# Default Poorvi lesson ranges used to generate production-safe day/subsections.
# This makes `python app/make_english_poorvi.py --force` always publish subsections
# without requiring a separate --subsections-json argument.
#
# IMPORTANT:
# - These ranges are lesson-level semantic/indexed ranges, not broad physical ranges.
# - Transcript pages and unit-level activity pages are intentionally excluded.
# - subsection_count is intentionally variable. Short lessons should NOT be forced
#   into five repeated Day ranges. Example: National War Memorial is only two
#   lesson pages, so it gets two subsections only.
# - The helper below clamps every generated range inside the parent lesson range,
#   so a subsection can never swallow another lesson.
PRINTED_OFFSET = 16
DEFAULT_MAX_DAY_COUNT_PER_LESSON = 5

DEFAULT_POORVI_LESSON_RANGES = [
    {"sequence": 1, "chapter_name": "A Bottle of Dew", "start_book_page": 1, "end_book_page": 12, "start_pdf_page": 17, "end_pdf_page": 28, "subsection_count": 5},
    {"sequence": 2, "chapter_name": "The Raven and the Fox", "start_book_page": 13, "end_book_page": 19, "start_pdf_page": 29, "end_pdf_page": 35, "subsection_count": 5},
    {"sequence": 3, "chapter_name": "Rama to the Rescue", "start_book_page": 20, "end_book_page": 35, "start_pdf_page": 36, "end_pdf_page": 51, "subsection_count": 5},
    {"sequence": 4, "chapter_name": "The Unlikely Best Friends", "start_book_page": 39, "end_book_page": 51, "start_pdf_page": 55, "end_pdf_page": 67, "subsection_count": 5},
    {"sequence": 5, "chapter_name": "A Friend’s Prayer", "start_book_page": 52, "end_book_page": 58, "start_pdf_page": 68, "end_pdf_page": 74, "subsection_count": 5},
    {"sequence": 6, "chapter_name": "The Chair", "start_book_page": 59, "end_book_page": 70, "start_pdf_page": 75, "end_pdf_page": 86, "subsection_count": 5},
    {"sequence": 7, "chapter_name": "Neem Baba", "start_book_page": 75, "end_book_page": 84, "start_pdf_page": 91, "end_pdf_page": 100, "subsection_count": 5},
    {"sequence": 8, "chapter_name": "What a Bird Thought", "start_book_page": 85, "end_book_page": 92, "start_pdf_page": 101, "end_pdf_page": 108, "subsection_count": 5},
    {"sequence": 9, "chapter_name": "Spices that Heal Us", "start_book_page": 93, "end_book_page": 100, "start_pdf_page": 109, "end_pdf_page": 116, "subsection_count": 5},
    {"sequence": 10, "chapter_name": "Change of Heart", "start_book_page": 103, "end_book_page": 114, "start_pdf_page": 119, "end_pdf_page": 130, "subsection_count": 5},
    {"sequence": 11, "chapter_name": "The Winner", "start_book_page": 115, "end_book_page": 121, "start_pdf_page": 131, "end_pdf_page": 137, "subsection_count": 5},
    {"sequence": 12, "chapter_name": "Yoga—A Way of Life", "start_book_page": 122, "end_book_page": 127, "start_pdf_page": 138, "end_pdf_page": 143, "subsection_count": 5},
    {"sequence": 13, "chapter_name": "Hamara Bharat—Incredible India!", "start_book_page": 131, "end_book_page": 140, "start_pdf_page": 147, "end_pdf_page": 156, "subsection_count": 5},
    {"sequence": 14, "chapter_name": "The Kites", "start_book_page": 141, "end_book_page": 150, "start_pdf_page": 157, "end_pdf_page": 166, "subsection_count": 5},
    {"sequence": 15, "chapter_name": "Ila Sachani: Embroidering Dreams with her Feet", "start_book_page": 151, "end_book_page": 159, "start_pdf_page": 167, "end_pdf_page": 175, "subsection_count": 5},
    {"sequence": 16, "chapter_name": "National War Memorial", "start_book_page": 160, "end_book_page": 161, "start_pdf_page": 176, "end_pdf_page": 177, "subsection_count": 2},
]


def split_inclusive_range(start: int, end: int, parts: int = DEFAULT_MAX_DAY_COUNT_PER_LESSON) -> list[tuple[int, int]]:
    """Split an inclusive range into `parts` safe Day ranges.

    Split into non-overlapping inclusive ranges.

    If a lesson has fewer pages than the requested parts, the part count is
    reduced to the available page count so pages are not repeated.
    """
    if start > end:
        raise ValueError(f"Invalid inclusive range: {start}-{end}")
    count = end - start + 1
    parts = max(1, min(parts, count))
    base = count // parts
    remainder = count % parts
    ranges: list[tuple[int, int]] = []
    cursor = start
    for idx in range(parts):
        size = base + (1 if idx < remainder else 0)
        r_start = cursor
        r_end = cursor + size - 1
        ranges.append((r_start, r_end))
        cursor = r_end + 1
    return ranges


def build_default_poorvi_subsections_map() -> dict:
    """Build the embedded production-safe Poorvi subsection/day map."""
    chapters = []
    for lesson in DEFAULT_POORVI_LESSON_RANGES:
        requested_count = int(lesson.get("subsection_count", DEFAULT_MAX_DAY_COUNT_PER_LESSON))
        pdf_ranges = split_inclusive_range(lesson["start_pdf_page"], lesson["end_pdf_page"], requested_count)
        book_ranges = split_inclusive_range(lesson["start_book_page"], lesson["end_book_page"], requested_count)
        days = []
        for idx, ((pdf_start, pdf_end), (book_start, book_end)) in enumerate(zip(pdf_ranges, book_ranges), start=1):
            days.append({
                "day": idx,
                "start_book_page": book_start,
                "end_book_page": book_end,
                "start_pdf_page": pdf_start,
                "end_pdf_page": pdf_end,
                "range_source": "embedded_production_safe_variable_split",
            })
        chapters.append({
            "sequence": lesson["sequence"],
            "chapter_name": lesson["chapter_name"],
            "book_page": lesson["start_book_page"],
            "start_pdf_page": lesson["start_pdf_page"],
            "end_pdf_page": lesson["end_pdf_page"],
            "subsection_count": len(days),
            "days": days,
        })
    return {
        "book_name": "poorvi_grade6_english",
        "book_title": "Poorvi Grade 6 English",
        "grade": 6,
        "subject": "English",
        "pdf_offset": PRINTED_OFFSET,
        "pdf_page_count": 180,
        "subsection_policy": "variable_day_ranges_clamped_inside_each_parent_lesson_no_repeated_pages",
        "chapters": chapters,
    }


DEFAULT_POORVI_SUBSECTIONS_MAP = build_default_poorvi_subsections_map()

DEFAULT_SUBSECTIONS_JSON = "English_Poorvi_default_subsections_embedded.json"

STEP1_JSON = "English_Poorvi_section_extraction.json"
STEP1_REPORT = "English_Poorvi_step1_validation_report.txt"
STEP2_JSON = "English_Poorvi_hybrid_corrected_extraction_v2.json"
STEP2_REPORT = "English_Poorvi_hybrid_corrected_extraction_v2_validation_report.txt"
PRODUCTION_JSON = "English_Poorvi_production_ready.json"
PRODUCTION_REPORT = "English_Poorvi_production_validation_report.txt"


def find_step_script(app_dir: Path, script_name: str) -> Path:
    """Find step scripts whether they are directly under app/ or app/english_poorvi/."""
    candidates = [
        app_dir / script_name,
        app_dir / "english_poorvi" / script_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {script_name}. Checked:\n{checked}")


def run_cmd(cmd: list[str]) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    LOGGER.info("Running command: %s", printable)
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(printable)
    print("=" * 100)
    subprocess.run(cmd, check=True)
    LOGGER.info("Command completed successfully")



def write_embedded_subsections_json(output_dir: Path) -> Path:
    """Write the embedded Poorvi subsection/day map to the output folder and return its path."""
    path = output_dir / DEFAULT_SUBSECTIONS_JSON
    path.write_text(json.dumps(DEFAULT_POORVI_SUBSECTIONS_MAP, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote embedded/default subsections JSON: %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full English Poorvi JSON extraction pipeline.")
    parser.add_argument("--pdf", type=Path, default=Path("input/English_Poorvi.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/english_poorvi"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--skip-tesseract-audit", action="store_true", help="Skip optional Step 2 Tesseract OCR audit.")
    parser.add_argument("--document-id", default="english-poorvi-class-6-ncert-2026-27")
    parser.add_argument("--document-key", default="mother-miracle-class-6-english-poorvi")
    parser.add_argument(
        "--subsections-json",
        type=Path,
        default=None,
        help=(
            "Optional override days/chapter-map JSON. If omitted, this script writes and uses "
            "the embedded DEFAULT_POORVI_SUBSECTIONS_MAP so subsections are always included."
        ),
    )
    parser.add_argument(
        "--no-subsections",
        action="store_true",
        help="Disable default embedded subsection generation. Normally do not use this.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    project_root = Path.cwd()
    app_dir = Path(__file__).resolve().parent

    pdf_path = args.pdf if args.pdf.is_absolute() else project_root / args.pdf
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_subsections:
        subsections_json = None
    elif args.subsections_json is not None:
        subsections_json = args.subsections_json if args.subsections_json.is_absolute() else project_root / args.subsections_json
    else:
        subsections_json = write_embedded_subsections_json(output_dir)

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("PDF path: %s", pdf_path)
    LOGGER.info("Output dir: %s", output_dir)
    if subsections_json:
        LOGGER.info("Subsections/days JSON: %s", subsections_json)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Put the PDF at input/English_Poorvi.pdf or pass --pdf with the correct path."
        )
    if subsections_json and not subsections_json.exists():
        raise FileNotFoundError(
            f"Subsections/days JSON not found: {subsections_json}\n"
            "Pass --subsections-json with the correct chapter_map_final.json path, or omit it to use the embedded default map."
        )

    step1_script = find_step_script(app_dir, "poorvi_step_1_base_extract.py")
    step2_script = find_step_script(app_dir, "poorvi_step_2_hybrid_correct.py")
    step3_script = find_step_script(app_dir, "poorvi_step_3_publish_production.py")

    step1_json = output_dir / STEP1_JSON
    step1_report = output_dir / STEP1_REPORT
    step2_json = output_dir / STEP2_JSON
    step2_report = output_dir / STEP2_REPORT
    production_json = output_dir / PRODUCTION_JSON
    production_report = output_dir / PRODUCTION_REPORT

    outputs = [step1_json, step1_report, step2_json, step2_report, production_json, production_report]
    if not args.force:
        existing = [p for p in outputs if p.exists()]
        if existing:
            existing_text = "\n".join(f"- {p}" for p in existing)
            raise FileExistsError(
                "Output files already exist. Use --force to overwrite them:\n" + existing_text
            )

    run_cmd([
        sys.executable,
        str(step1_script),
        "--pdf", str(pdf_path),
        "--output", str(step1_json),
        "--report", str(step1_report),
        "--log-level", args.log_level,
    ])

    step2_cmd = [
        sys.executable,
        str(step2_script),
        "--pdf", str(pdf_path),
        "--input-json", str(step1_json),
        "--output-json", str(step2_json),
        "--report", str(step2_report),
        "--log-level", args.log_level,
    ]
    if args.skip_tesseract_audit:
        step2_cmd.append("--skip-tesseract-audit")
    run_cmd(step2_cmd)

    step3_cmd = [
        sys.executable,
        str(step3_script),
        "--input", str(step2_json),
        "--output", str(production_json),
        "--report", str(production_report),
        "--document-id", args.document_id,
        "--document-key", args.document_key,
        "--log-level", args.log_level,
    ]
    if subsections_json:
        step3_cmd.extend(["--subsections-json", str(subsections_json)])
    run_cmd(step3_cmd)

    LOGGER.info("English Poorvi pipeline completed")
    print("\nDONE")
    print(f"Step 1 JSON:        {step1_json}")
    print(f"Step 2 JSON:        {step2_json}")
    print(f"Production JSON:    {production_json}")
    print(f"Production report:  {production_report}")


if __name__ == "__main__":
    main()
