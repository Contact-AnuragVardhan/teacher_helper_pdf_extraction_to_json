#!/usr/bin/env python3
"""Step 4: build chapters, section_index, and subsection/day text from reviewed-safe page text."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from physics_common import (
    add_physical_ranges, as_int, build_chapter_and_section_index, load_static_chapters,
    read_json, setup_logging, utc_now, validate_output, write_json, build_report,
)


def build_step4_json(input_json: Path, subsections_json: Path) -> tuple[dict[str, Any], str]:
    data = read_json(input_json)
    spec, chapters_spec = load_static_chapters(subsections_json)
    pdf_page_count = int(data.get("pdf_page_count") or spec.get("pdf_page_count") or 0)
    add_physical_ranges(chapters_spec, pdf_page_count)
    pages = data.get("page_extractions") or []
    pages_by_number = {int(p["page_number"]): p for p in pages}
    stats = Counter(data.get("extraction", {}).get("statistics") or {})
    chapters, section_index, errors = build_chapter_and_section_index(chapters_spec, pages_by_number, pdf_page_count, stats)
    data["static_chapters"] = chapters_spec
    data["chapters"] = chapters
    data["section_index"] = section_index
    validation = validate_output(data, stats, errors)
    data["extraction"] = {
        "step": 4,
        "status": "step4_structured_subsections_ready" if validation["status"] == "passed" else "step4_validation_failed",
        "generated_at": utc_now(),
        "generator": "physics_step_4_build_subsections.py",
        "method": "build_chapters_sections_subsections_from_reviewed_safe_page_text_and_static_day_ranges",
        "source_step3_json": str(input_json),
        "source_subsections_json": str(subsections_json),
        "statistics": dict(stats),
        "validation": validation,
    }
    return data, build_report(data, validation, "Grade 10 Physics Step 4 structured subsection report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: build Physics chapters/sections/subsections from safe page text.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--subsections-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)
    for path in [args.output_json, args.report]:
        if path and path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")
    data, report = build_step4_json(args.input_json.resolve(), args.subsections_json.resolve())
    write_json(args.output_json.resolve(), data)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(f"Step 4 JSON: {args.output_json.resolve()}")
    if args.report:
        print(f"Step 4 report: {args.report.resolve()}")
    print(f"Step 4 validation status: {data['extraction']['validation']['status']}")


if __name__ == "__main__":
    main()
