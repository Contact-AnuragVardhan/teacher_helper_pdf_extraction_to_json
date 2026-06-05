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
import subprocess
import sys
from pathlib import Path


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
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full English Poorvi JSON extraction pipeline.")
    parser.add_argument("--pdf", type=Path, default=Path("input/English_Poorvi.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/english_poorvi"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--skip-tesseract-audit", action="store_true", help="Skip optional Step 2 Tesseract OCR audit.")
    parser.add_argument("--document-id", default="english-poorvi-class-6-ncert-2026-27")
    parser.add_argument("--document-key", default="mother-miracle-class-6-english-poorvi")
    args = parser.parse_args()

    project_root = Path.cwd()
    app_dir = Path(__file__).resolve().parent

    pdf_path = args.pdf if args.pdf.is_absolute() else project_root / args.pdf
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Put the PDF at input/English_Poorvi.pdf or pass --pdf with the correct path."
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
    ])

    step2_cmd = [
        sys.executable,
        str(step2_script),
        "--pdf", str(pdf_path),
        "--input-json", str(step1_json),
        "--output-json", str(step2_json),
        "--report", str(step2_report),
    ]
    if args.skip_tesseract_audit:
        step2_cmd.append("--skip-tesseract-audit")
    run_cmd(step2_cmd)

    run_cmd([
        sys.executable,
        str(step3_script),
        "--input", str(step2_json),
        "--output", str(production_json),
        "--report", str(production_report),
        "--document-id", args.document_id,
        "--document-key", args.document_key,
    ])

    print("\nDONE")
    print(f"Step 1 JSON:        {step1_json}")
    print(f"Step 2 JSON:        {step2_json}")
    print(f"Production JSON:    {production_json}")
    print(f"Production report:  {production_report}")


if __name__ == "__main__":
    main()
