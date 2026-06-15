#!/usr/bin/env python3
"""
Orchestrates the Grade10_Maths / R.D. Sharma Class X extraction pipeline.

Usage from project root:
  python app/make_grade10_maths.py --pdf input/Grade10_Maths.pdf --force

For a smoke test on selected pages only. This intentionally writes smoke-gate files,
not Grade10_Maths_production_ready.json:
  python app/make_grade10_maths.py --pdf input/Grade10_Maths.pdf --pages 1-20,500 --force
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger("grade10_maths.pipeline")

STEP0_JSON = "Grade10_Maths_chapters.json"
STEP0_PY = "Grade10_Maths_chapters.py"
STEP1_JSON = "Grade10_Maths_step1_base_extraction.json"
STEP1_REPORT = "Grade10_Maths_step1_validation_report.txt"
STEP2_JSON = "Grade10_Maths_production_ready.json"
STEP2_REPORT = "Grade10_Maths_production_validation_report.txt"
STEP2_QA_CSV = "Grade10_Maths_pages_requiring_vision_qa.csv"
DEFAULT_SUBSECTIONS_JSON = "Grade10_Maths_static_subsection_ranges.json"


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    LOGGER.info("Running command: %s", printable)
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(printable)
    print("=" * 100)
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict:
    import json
    if not path.exists():
        raise FileNotFoundError(f"Expected output was not created: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_step1_complete(path: Path, *, smoke_run: bool) -> None:
    data = load_json(path)
    extraction = data.get("extraction") or {}
    qs = extraction.get("quality_summary") or {}
    run_scope = qs.get("run_scope") or {}
    page_extractions = extraction.get("page_extractions") or []
    if not page_extractions:
        raise RuntimeError(f"Step 1 produced no page_extractions: {path}")
    missing = run_scope.get("missing_selected_pages") or []
    if missing:
        raise RuntimeError(f"Step 1 did not finish all selected pages. Missing: {missing[:20]}")
    if not smoke_run and not run_scope.get("is_full_book_run"):
        raise RuntimeError("Step 1 was not a full-book run. Do not call this production.")


def assert_step2_complete(path: Path, *, smoke_run: bool) -> None:
    data = load_json(path)
    extraction = data.get("extraction") or {}
    policy = extraction.get("production_embedding_policy") or {}
    status = str(policy.get("status") or "")
    if smoke_run:
        if "smoke" not in status and "partial" not in status:
            raise RuntimeError(f"Smoke run did not mark output as partial/smoke. Status={status}")
        return
    if status not in {"production_complete_ready", "production_safe_gated_needs_qa"}:
        raise RuntimeError(f"Production gate did not pass as a full-book artifact. Status={status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grade10_Maths production-safe OCR extraction pipeline.")
    parser.add_argument("--pdf", type=Path, default=Path("input/Grade10_Maths.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/maths_rdsharma_grade10"))
    parser.add_argument("--subsections-json", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--skip-step0", action="store_true", help="Use an existing Grade10_Maths_chapters.json in the output dir.")
    parser.add_argument("--pages", default=None, help="Optional page subset for smoke tests, e.g. 1-20,500")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("GRADE10_MATHS_OCR_WORKERS", "4")))
    parser.add_argument("--scale", type=float, default=float(os.environ.get("GRADE10_MATHS_OCR_SCALE", "3.0")))
    parser.add_argument("--psm", default=os.environ.get("GRADE10_MATHS_TESSERACT_PSM", "4"))
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--strict-complete", action="store_true", help="Fail unless full-book run has zero pages excluded by the production gate.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    project_root = Path.cwd()
    app_dir = Path(__file__).resolve().parent
    package_dir = app_dir / "maths_rdsharma_grade10"

    pdf_path = resolve(args.pdf, project_root)
    output_dir = resolve(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    subsections_json = resolve(args.subsections_json, project_root) if args.subsections_json else package_dir / DEFAULT_SUBSECTIONS_JSON

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    if not subsections_json.exists():
        raise FileNotFoundError(subsections_json)

    step0 = package_dir / "make_grade10_maths_step_0_structure.py"
    step1 = package_dir / "make_grade10_maths_step_1_base_extract.py"
    step2 = package_dir / "make_grade10_maths_step_2_production_gate.py"
    for script in [step0, step1, step2]:
        if not script.exists():
            raise FileNotFoundError(script)

    output_files = [
        output_dir / STEP0_JSON,
        output_dir / STEP0_PY,
        output_dir / STEP1_JSON,
        output_dir / STEP1_REPORT,
        output_dir / STEP2_JSON,
        output_dir / STEP2_REPORT,
        output_dir / STEP2_QA_CSV,
    ]
    if not args.force:
        existing = [p for p in output_files if p.exists()]
        if existing:
            raise FileExistsError("Output files already exist. Use --force to overwrite:\n" + "\n".join(f"- {p}" for p in existing))

    env = os.environ.copy()
    env.update({
        "GRADE10_MATHS_ROOT": str(project_root),
        "GRADE10_MATHS_PDF": str(pdf_path),
        "GRADE10_MATHS_OUTPUT_DIR": str(output_dir),
        "GRADE10_MATHS_CHAPTERS_JSON": str(output_dir / STEP0_JSON),
        "GRADE10_MATHS_SUBSECTIONS_JSON": str(subsections_json),
    })

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("PDF path: %s", pdf_path)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Static subsection ranges: %s", subsections_json)

    smoke_run = bool(args.pages)
    step1_json = output_dir / ("Grade10_Maths_step1_smoke_extraction.json" if smoke_run else STEP1_JSON)
    step1_report = output_dir / ("Grade10_Maths_step1_smoke_validation_report.txt" if smoke_run else STEP1_REPORT)
    step2_json = output_dir / ("Grade10_Maths_smoke_gate.json" if smoke_run else STEP2_JSON)
    step2_report = output_dir / ("Grade10_Maths_smoke_gate_report.txt" if smoke_run else STEP2_REPORT)
    step2_qa_csv = output_dir / ("Grade10_Maths_smoke_pages_requiring_vision_qa.csv" if smoke_run else STEP2_QA_CSV)
    if smoke_run:
        LOGGER.warning("--pages was supplied, so this is a smoke/partial run and will NOT write %s", STEP2_JSON)

    if not args.skip_step0:
        run_cmd([
            sys.executable,
            str(step0),
            "--pdf", str(pdf_path),
            "--static-json", str(subsections_json),
            "--output-json", str(output_dir / STEP0_JSON),
            "--output-py", str(output_dir / STEP0_PY),
        ], env)

    step1_cmd = [
        sys.executable,
        str(step1),
        "--pdf", str(pdf_path),
        "--chapters-json", str(output_dir / STEP0_JSON),
        "--subsections-json", str(subsections_json),
        "--output-dir", str(output_dir),
        "--output-json", str(step1_json),
        "--report", str(step1_report),
        "--workers", str(args.workers),
        "--scale", str(args.scale),
        "--psm", str(args.psm),
    ]
    if args.pages:
        step1_cmd.extend(["--pages", args.pages])
    if args.force_ocr:
        step1_cmd.append("--force-ocr")
    run_cmd(step1_cmd, env)
    assert_step1_complete(step1_json, smoke_run=smoke_run)

    step2_cmd = [
        sys.executable,
        str(step2),
        "--input", str(step1_json),
        "--output", str(step2_json),
        "--report", str(step2_report),
        "--qa-csv", str(step2_qa_csv),
    ]
    if smoke_run:
        step2_cmd.append("--allow-partial")
    if args.strict_complete:
        step2_cmd.append("--strict-complete")
    run_cmd(step2_cmd, env)
    assert_step2_complete(step2_json, smoke_run=smoke_run)

    print("\nDONE")
    if smoke_run:
        print("Smoke/partial JSON:   " + str(step2_json))
        print("Smoke/partial report: " + str(step2_report))
        print("Smoke/partial QA CSV: " + str(step2_qa_csv))
        print("This is NOT a production artifact. Re-run without --pages for the full book.")
    else:
        print(f"Production JSON:   {step2_json}")
        print(f"Production report: {step2_report}")
        print(f"Vision QA CSV:     {step2_qa_csv}")
    print(f"Subsections JSON:  {subsections_json}")


if __name__ == "__main__":
    main()
