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



# Default Poorvi subsection/day ranges now live in a JSON file instead of
# being generated from a Python static array. This makes the ranges easy to
# maintain without changing code.
#
# Put this file next to make_english_poorvi.py, under app/english_poorvi/,
# or in the project root. You can still override it with --subsections-json.
DEFAULT_SUBSECTIONS_JSON = "English_Poorvi_static_subsection_ranges.json"


def find_default_subsections_json(project_root: Path, app_dir: Path, output_dir: Path) -> Path:
    """Find the maintained static Poorvi subsection/day range JSON."""
    candidates = [
        project_root / DEFAULT_SUBSECTIONS_JSON,
        app_dir / DEFAULT_SUBSECTIONS_JSON,
        app_dir / "english_poorvi" / DEFAULT_SUBSECTIONS_JSON,
        output_dir / DEFAULT_SUBSECTIONS_JSON,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Default subsections/day ranges JSON not found: {DEFAULT_SUBSECTIONS_JSON}\n"
        "Create the JSON file from the maintained static map, place it in one of these locations, "
        "or pass --subsections-json explicitly:\n"
        f"{checked}"
    )


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
            "Optional override days/chapter-map JSON. If omitted, this script reads "
            f"{DEFAULT_SUBSECTIONS_JSON} from the project/app folder."
        ),
    )
    parser.add_argument(
        "--no-subsections",
        action="store_true",
        help="Disable subsection generation entirely. Normally do not use this.",
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
        subsections_json = find_default_subsections_json(project_root, app_dir, output_dir)

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
            f"Pass --subsections-json with the correct JSON path, or create {DEFAULT_SUBSECTIONS_JSON} in the project/app folder."
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
