#!/usr/bin/env python3
"""
make_maths_rsaggarwal.py

Orchestrates the full R. S. Aggarwal Class 7 Maths extraction pipeline.

Important:
  Day/subsection ranges are NOT detected dynamically. They are read from the
  standalone Maths_RSAgarwal_static_subsection_ranges.json file.

Usage from project root:
  python app/maths_rsaggarwal/make_maths_rsaggarwal.py --force

Optional:
  python app/maths_rsaggarwal/make_maths_rsaggarwal.py \
    --pdf input/Maths_RSAgarwal.pdf \
    --output-dir output/maths_rsagarwal \
    --subsections-json app/maths_rsaggarwal/Maths_RSAgarwal_static_subsection_ranges.json \
    --force
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger("maths_rsaggarwal.pipeline")

DEFAULT_SUBSECTIONS_JSON = "Maths_RSAgarwal_static_subsection_ranges.json"

STEP0_CHAPTERS_JSON = "Maths_RSAgarwal_chapters.json"
STEP0_CHAPTERS_PY = "Maths_RSAgarwal_chapters.py"
STEP1_JSON = "Maths_RSAgarwal_math_aware_extraction.json"
STEP1_REPORT = "Maths_RSAgarwal_math_aware_validation_report.txt"
STEP2_JSON = "Maths_RSAgarwal_math_aware_extraction_v2_cleaned.json"
STEP2_REPORT = "Maths_RSAgarwal_math_aware_v2_validation_report.txt"
STEP3_JSON = "Maths_RSAgarwal_math_aware_extraction_v3_cleaned.json"
STEP3_REPORT = "Maths_RSAgarwal_math_aware_v3_validation_report.txt"
STEP4_JSON = "Maths_RSAgarwal_math_aware_extraction_v4_production_safe.json"
STEP4_REPORT = "Maths_RSAgarwal_math_aware_v4_production_validation_report.txt"


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def find_step_script(app_dir: Path, script_name: str) -> Path:
    candidates = [
        app_dir / script_name,
        app_dir / "maths_rsaggarwal" / script_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {script_name}. Checked:\n{checked}")


def find_default_subsections_json(project_root: Path, app_dir: Path, output_dir: Path) -> Path:
    candidates = [
        project_root / DEFAULT_SUBSECTIONS_JSON,
        app_dir / DEFAULT_SUBSECTIONS_JSON,
        app_dir / "maths_rsaggarwal" / DEFAULT_SUBSECTIONS_JSON,
        output_dir / DEFAULT_SUBSECTIONS_JSON,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Default subsections/day ranges JSON not found: {DEFAULT_SUBSECTIONS_JSON}\n"
        "Place the standalone JSON in one of these locations, or pass --subsections-json explicitly:\n"
        f"{checked}"
    )


def run_cmd(cmd: list[str], *, env: dict[str, str]) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    LOGGER.info("Running command: %s", printable)
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(printable)
    print("=" * 100)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Maths RSAggarwal extraction pipeline with static day ranges JSON.")
    parser.add_argument("--pdf", type=Path, default=Path("input/Maths_RSAgarwal.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/maths_rsagarwal"))
    parser.add_argument("--subsections-json", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--skip-step0", action="store_true", help="Use an existing Maths_RSAgarwal_chapters.json.")
    parser.add_argument("--printed-page-offset", type=int, default=7)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    project_root = Path.cwd()
    app_dir = Path(__file__).resolve().parent

    pdf_path = args.pdf if args.pdf.is_absolute() else project_root / args.pdf
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subsections_json:
        subsections_json = args.subsections_json if args.subsections_json.is_absolute() else project_root / args.subsections_json
    else:
        subsections_json = find_default_subsections_json(project_root, app_dir, output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not subsections_json.exists():
        raise FileNotFoundError(f"Subsections/day ranges JSON not found: {subsections_json}")

    output_files = [
        output_dir / STEP0_CHAPTERS_JSON,
        output_dir / STEP0_CHAPTERS_PY,
        output_dir / STEP1_JSON,
        output_dir / STEP1_REPORT,
        output_dir / STEP2_JSON,
        output_dir / STEP2_REPORT,
        output_dir / STEP3_JSON,
        output_dir / STEP3_REPORT,
        output_dir / STEP4_JSON,
        output_dir / STEP4_REPORT,
    ]
    if not args.force:
        existing = [p for p in output_files if p.exists()]
        if existing:
            existing_text = "\n".join(f"- {p}" for p in existing)
            raise FileExistsError("Output files already exist. Use --force to overwrite them:\n" + existing_text)

    step0 = find_step_script(app_dir, "make_maths_rsaggarwal_step_0.py")
    step1 = find_step_script(app_dir, "make_maths_rsaggarwal_step_1.py")
    step2 = find_step_script(app_dir, "make_maths_rsaggarwal_step_2.py")
    step3 = find_step_script(app_dir, "make_maths_rsaggarwal_step_3.py")
    step4 = find_step_script(app_dir, "make_maths_rsaggarwal_step_4.py")

    chapters_json = output_dir / STEP0_CHAPTERS_JSON
    chapters_py = output_dir / STEP0_CHAPTERS_PY

    env = os.environ.copy()
    env.update({
        "MATHS_RSAGGARWAL_ROOT": str(project_root),
        "MATHS_RSAGGARWAL_PDF": str(pdf_path),
        "MATHS_RSAGGARWAL_OUTPUT_DIR": str(output_dir),
        "MATHS_RSAGGARWAL_CHAPTERS_JSON": str(chapters_json),
        "MATHS_RSAGGARWAL_SUBSECTIONS_JSON": str(subsections_json),
    })

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("PDF path: %s", pdf_path)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Static subsections/day ranges JSON: %s", subsections_json)

    if not args.skip_step0:
        run_cmd([
            sys.executable,
            str(step0),
            "--pdf", str(pdf_path),
            "--output-json", str(chapters_json),
            "--output-py", str(chapters_py),
            "--printed-page-offset", str(args.printed_page_offset),
        ], env=env)

    run_cmd([sys.executable, str(step1)], env=env)
    run_cmd([
        sys.executable,
        str(step2),
        "--input", str(output_dir / STEP1_JSON),
        "--output", str(output_dir / STEP2_JSON),
        "--report", str(output_dir / STEP2_REPORT),
    ], env=env)
    run_cmd([
        sys.executable,
        str(step3),
        "--input", str(output_dir / STEP2_JSON),
        "--output", str(output_dir / STEP3_JSON),
        "--report", str(output_dir / STEP3_REPORT),
    ], env=env)
    run_cmd([sys.executable, str(step4)], env=env)

    print("\nDONE")
    print(f"Production JSON:   {output_dir / STEP4_JSON}")
    print(f"Production report: {output_dir / STEP4_REPORT}")
    print(f"Subsections JSON:  {subsections_json}")


if __name__ == "__main__":
    main()
