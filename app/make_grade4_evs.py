#!/usr/bin/env python3
"""
make_grade4_evs.py

Production JSON builder for NCERT Our Wondrous World: The World Around Us
for Grade 4 / Class 4 Environmental Studies (EVS).

It uses the maintained static Our Wondrous World subsection/day JSON as the
source of truth for unit/chapter/day boundaries and extracts selectable PDF
text into the same schema shape used by the Santoor Grade 4 English builder.

Usage from project root:
  python app/make_grade4_evs.py \
    --pdf input/Our_Wondrous_World_Grade4_EVS.pdf \
    --output-dir output/evs_wondrous_grade4 \
    --force

Override the day range JSON:
  python app/make_grade4_evs.py \
    --pdf input/Our_Wondrous_World_Grade4_EVS.pdf \
    --subsections-json app/evs_wondrous_grade4/Our_Wondrous_World_Grade4_EVS_static_subsection_ranges.json \
    --output-dir output/evs_wondrous_grade4 \
    --force
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from evs_wondrous_grade4.wondrous_common import build_production_json, write_json

LOGGER = logging.getLogger("evs_wondrous_grade4.pipeline")

DEFAULT_SUBSECTIONS_JSON = "Our_Wondrous_World_Grade4_EVS_static_subsection_ranges.json"
PRODUCTION_JSON = "Our_Wondrous_World_Grade4_EVS_production_ready.json"
PRODUCTION_REPORT = "Our_Wondrous_World_Grade4_EVS_production_validation_report.txt"


def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def find_default_subsections_json(project_root: Path, app_dir: Path, output_dir: Path) -> Path:
    candidates = [
        project_root / DEFAULT_SUBSECTIONS_JSON,
        app_dir / DEFAULT_SUBSECTIONS_JSON,
        app_dir / "evs_wondrous_grade4" / DEFAULT_SUBSECTIONS_JSON,
        output_dir / DEFAULT_SUBSECTIONS_JSON,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Default Our Wondrous World subsections/day JSON not found: {DEFAULT_SUBSECTIONS_JSON}\n"
        "Place it in one of these paths or pass --subsections-json explicitly:\n"
        f"{checked}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build production-ready Our Wondrous World Grade 4 EVS JSON from PDF.")
    parser.add_argument("--pdf", type=Path, default=Path("input/Our_Wondrous_World_Grade4_EVS.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/evs_wondrous_grade4"))
    parser.add_argument("--subsections-json", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite output files if they already exist.")
    parser.add_argument("--document-id", default="evs-our-wondrous-world-class-4-ncert-2026-27")
    parser.add_argument("--document-key", default="mother-miracle-class-4-evs-our-wondrous-world")
    parser.add_argument("--school-name", default="Mother Miracle School")
    parser.add_argument("--board", default="CBSE")
    parser.add_argument("--medium", default="English")
    parser.add_argument("--strict", action="store_true", help="Fail if production validation has warnings.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    project_root = Path.cwd()
    app_dir = Path(__file__).resolve().parent
    pdf_path = args.pdf if args.pdf.is_absolute() else project_root / args.pdf
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subsections_json is not None:
        subsections_json = args.subsections_json if args.subsections_json.is_absolute() else project_root / args.subsections_json
    else:
        subsections_json = find_default_subsections_json(project_root, app_dir, output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not subsections_json.exists():
        raise FileNotFoundError(f"Subsections/day JSON not found: {subsections_json}")

    production_json = output_dir / PRODUCTION_JSON
    production_report = output_dir / PRODUCTION_REPORT
    static_copy = output_dir / DEFAULT_SUBSECTIONS_JSON
    outputs = [production_json, production_report, static_copy]
    if not args.force:
        existing = [p for p in outputs if p.exists()]
        if existing:
            raise FileExistsError("Output files already exist. Use --force to overwrite:\n" + "\n".join(map(str, existing)))

    LOGGER.info("PDF path: %s", pdf_path)
    LOGGER.info("Subsections JSON: %s", subsections_json)
    LOGGER.info("Output dir: %s", output_dir)

    data, report, errors, warnings = build_production_json(
        pdf_path=pdf_path,
        static_json_path=subsections_json,
        document_id=args.document_id,
        document_key=args.document_key,
        school_name=args.school_name,
        board=args.board,
        medium=args.medium,
    )
    write_json(production_json, data)
    production_report.write_text(report, encoding="utf-8")
    shutil.copy2(subsections_json, static_copy)

    if errors:
        raise SystemExit("Production validation failed. See report: " + str(production_report))
    if args.strict and warnings:
        raise SystemExit("Production validation has warnings and --strict was set. See report: " + str(production_report))

    print("\nDONE")
    print(f"Production JSON:   {production_json}")
    print(f"Validation report: {production_report}")
    print(f"Static day JSON:   {static_copy}")


if __name__ == "__main__":
    main()
