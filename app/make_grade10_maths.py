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

try:
    from dotenv import load_dotenv
except ImportError:  # .env support is optional
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {value!r}") from exc


STEP0_JSON = "Grade10_Maths_chapters.json"
STEP0_PY = "Grade10_Maths_chapters.py"
STEP1_JSON = "Grade10_Maths_step1_base_extraction.json"
STEP1_REPORT = "Grade10_Maths_step1_validation_report.txt"
STEP2_JSON = "Grade10_Maths_production_ready.json"
STEP2_REPORT = "Grade10_Maths_production_validation_report.txt"
STEP2_QA_CSV = "Grade10_Maths_pages_requiring_vision_qa.csv"
STEP3_REPORT = "Grade10_Maths_vision_qa_report.txt"
STEP3_REMAINING_QA_CSV = "Grade10_Maths_remaining_pages_requiring_vision_qa.csv"
STEP4_REPORT = "Grade10_Maths_final_audit_report.txt"
STEP4_SUSPICIOUS_CSV = "Grade10_Maths_final_audit_suspicious_pages.csv"
STEP4_REMAINING_SUSPICIOUS_CSV = "Grade10_Maths_final_audit_remaining_suspicious_pages.csv"
STEP5_REPORT = "Grade10_Maths_math_precision_audit_report.txt"
STEP5_SUSPICIOUS_CSV = "Grade10_Maths_math_precision_suspicious_pages.csv"
STEP5_REMAINING_CSV = "Grade10_Maths_math_precision_remaining_pages.csv"
STEP6_REPORT = "Grade10_Maths_residual_production_audit_report.txt"
STEP6_REMAINING_CSV = "Grade10_Maths_residual_production_remaining_pages.csv"
STEP7_REPORT = "Grade10_Maths_full_image_production_verify_report.txt"
STEP7_REMAINING_CSV = "Grade10_Maths_full_image_production_verify_remaining_pages.csv"
STEP7_CACHE_DIR = ".full_image_verify_cache"
STEP_TARGETED_LLM_REPAIR_CSV = "Grade10_Maths_targeted_llm_repair_pages.csv"
KNOWN_FIXES_REPORT = "Grade10_Maths_known_page_fixes_report.json"
DEFAULT_SUBSECTIONS_JSON = "Grade10_Maths_static_subsection_ranges.json"
DEFAULT_PAGE_OVERRIDES_JSON = "Grade10_Maths_page_overrides.json"


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def run_cmd(cmd: list[str]) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    LOGGER.info("Running command: %s", printable)
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(printable)
    print("=" * 100)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    import json
    if not path.exists():
        raise FileNotFoundError(f"Expected output was not created: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_page_numbers_from_csv(path: Path) -> list[int]:
    """Return sorted PDF page numbers from a Step 6/Step 7 remaining CSV."""
    import csv
    import re
    if not path.exists():
        return []
    pages: set[int] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        fieldnames = {name.lower().strip(): name for name in reader.fieldnames if name}
        col = None
        for candidate in ("page_number", "pdf_page_number", "pdf_page", "page"):
            if candidate in fieldnames:
                col = fieldnames[candidate]
                break
        if not col:
            return []
        for row in reader:
            raw = str(row.get(col) or "").strip()
            m = re.search(r"\d+", raw)
            if m:
                pages.add(int(m.group(0)))
    return sorted(pages)



def write_page_numbers_to_csv(path: Path, pages: list[int], *, source_csvs: list[Path]) -> None:
    """Write a simple page-number CSV that Step 7 can consume with --pages-from-csv."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = ";".join(str(p) for p in source_csvs if p.exists())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_number", "source_csvs"])
        writer.writeheader()
        for page in sorted(set(int(p) for p in pages)):
            writer.writerow({"page_number": page, "source_csvs": sources})


def collect_page_numbers_from_csvs(paths: list[Path]) -> list[int]:
    """Collect unique page_number values from Step 4, Step 5, Step 6 and Step 7 remaining CSVs."""
    pages: set[int] = set()
    for path in paths:
        pages.update(read_page_numbers_from_csv(path))
    return sorted(pages)


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


def assert_final_production_quality(path: Path, *, strict_complete: bool) -> None:
    data = load_json(path)
    extraction = data.get("extraction") or {}
    policy = extraction.get("production_embedding_policy") or {}
    status = str(policy.get("status") or "")
    if not policy.get("final_production_audit_applied"):
        raise RuntimeError("Final production audit did not run. Do not ingest this JSON into DB.")
    final_audit = (extraction.get("quality_summary") or {}).get("final_production_audit") or {}
    remaining_suspicious = int(final_audit.get("remaining_suspicious_ready_pages") or 0)
    excluded_lesson_body = int(final_audit.get("excluded_lesson_body_pages") or 0)
    empty_subsections = int(final_audit.get("empty_production_subsections") or 0)
    if remaining_suspicious or excluded_lesson_body or empty_subsections:
        msg = (
            "Final production audit still has blockers: "
            f"status={status}, excluded_lesson_body_pages={excluded_lesson_body}, "
            f"remaining_suspicious_ready_pages={remaining_suspicious}, "
            f"empty_production_subsections={empty_subsections}"
        )
        if strict_complete:
            raise RuntimeError(msg)
        LOGGER.warning(msg)
    math_audit = (extraction.get("quality_summary") or {}).get("math_precision_audit") or {}
    if policy.get("math_precision_audit_applied"):
        remaining_math = int(math_audit.get("remaining_math_precision_pages") or 0)
        math_excluded = int(math_audit.get("excluded_lesson_body_pages") or 0)
        math_empty = int(math_audit.get("empty_production_subsections") or 0)
        if remaining_math or math_excluded or math_empty:
            msg = (
                "Math precision audit still has blockers: "
                f"status={status}, excluded_lesson_body_pages={math_excluded}, "
                f"remaining_math_precision_pages={remaining_math}, "
                f"empty_production_subsections={math_empty}"
            )
            if strict_complete:
                raise RuntimeError(msg)
            LOGGER.warning(msg)
    residual = (extraction.get("quality_summary") or {}).get("residual_production_audit") or {}
    if policy.get("residual_production_audit_applied"):
        remaining_residual = int(residual.get("remaining_residual_pages") or 0)
        residual_excluded = int(residual.get("excluded_lesson_body_pages") or 0)
        residual_empty = int(residual.get("empty_production_subsections") or 0)
        if remaining_residual or residual_excluded or residual_empty:
            msg = (
                "Residual production audit still has blockers: "
                f"status={status}, excluded_lesson_body_pages={residual_excluded}, "
                f"remaining_residual_pages={remaining_residual}, "
                f"empty_production_subsections={residual_empty}"
            )
            if strict_complete:
                raise RuntimeError(msg)
            LOGGER.warning(msg)
    if strict_complete and status != "production_complete_ready":
        raise RuntimeError(f"Strict final production gate did not pass. Status={status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grade10_Maths production-safe OCR extraction pipeline.")
    parser.add_argument("--pdf", type=Path, default=Path("input/Grade10_Maths.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/maths_rdsharma_grade10"))
    parser.add_argument("--subsections-json", type=Path, default=None)
    parser.add_argument("--page-overrides-json", type=Path, default=None, help="Optional reviewed page-level overrides JSON. Safe alternative to hardcoding page text in Python. Defaults to app/maths_rdsharma_grade10/Grade10_Maths_page_overrides.json if it exists, or GRADE10_MATHS_PAGE_OVERRIDES.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--skip-step0", action="store_true", help="Use an existing Grade10_Maths_chapters.json in the output dir.")
    parser.add_argument("--pages", default=None, help="Optional page subset for smoke tests, e.g. 1-20,500")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--psm", default="4")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--strict-complete", action="store_true", help="Fail unless the final full-book JSON has zero lesson-body pages excluded. Also enabled by GRADE10_MATHS_FAIL_ON_REVIEW_REQUIRED=true.")
    parser.add_argument("--repair-with-vision", action="store_true", help="After Step 2, repair excluded lesson-body pages with OpenAI vision and rebuild final production JSON. Also enabled by GRADE10_MATHS_AUTO_REVIEW_PROVIDER=openai.")
    parser.add_argument("--vision-model", default=None, help="Optional override for the OpenAI vision model. If omitted, uses GRADE10_MATHS_VISION_MODEL, defaulting to gpt-4o-mini.")
    parser.add_argument("--vision-scale", type=float, default=env_float("GRADE10_MATHS_VISION_SCALE", 2.5), help="PDF render scale for page images. Defaults from GRADE10_MATHS_VISION_SCALE or 2.5.")
    parser.add_argument("--vision-pages", default=None, help="Optional PDF pages to repair with vision, e.g. 503-543.")
    parser.add_argument("--vision-max-pages", type=int, default=env_int("GRADE10_MATHS_AUTO_REVIEW_MAX_ITEMS", 0), help="Optional cap for testing vision repair; defaults from GRADE10_MATHS_AUTO_REVIEW_MAX_ITEMS. 0 means all.")
    parser.add_argument("--vision-min-confidence", type=float, default=env_float("GRADE10_MATHS_AUTO_REVIEW_THRESHOLD", 0.90), help="Minimum confidence accepted from OpenAI. Defaults from GRADE10_MATHS_AUTO_REVIEW_THRESHOLD.")
    parser.add_argument("--force-vision", action="store_true", help="Ignore cached vision QA page results.")
    parser.add_argument("--skip-final-audit", action="store_true", help="Skip Step 4 residual-OCR final audit. Not recommended for production DB ingestion.")
    parser.add_argument("--final-audit-pages", default=None, help="Optional PDF pages to final-audit/repair, e.g. 52,103,671,750.")
    parser.add_argument("--final-audit-max-pages", type=int, default=env_int("GRADE10_MATHS_FINAL_AUDIT_MAX_ITEMS", 0), help="Optional cap for testing final audit repair. 0 means all.")
    parser.add_argument("--final-audit-mode", choices=["fast", "strict"], default=os.environ.get("GRADE10_MATHS_FINAL_AUDIT_MODE", "fast").strip().lower() or "fast", help="Final audit sensitivity. Use fast for production runs; strict is a slow QA pass and can over-flag math pages.")
    parser.add_argument("--force-final-audit", action="store_true", help="Run final audit in the pipeline. Does not ignore final-audit cache.")
    parser.add_argument("--force-final-audit-cache", action="store_true", help="Ignore cached final-audit vision repair results and call OpenAI again.")
    parser.add_argument("--skip-math-precision-audit", action="store_true", help="Skip Step 5 math/formula precision audit. Not recommended for production maths DB ingestion.")
    parser.add_argument("--math-audit-pages", default=None, help="Optional PDF pages to math-precision audit/repair, e.g. 8,11,503,710.")
    parser.add_argument("--math-audit-max-pages", type=int, default=env_int("GRADE10_MATHS_MATH_AUDIT_MAX_ITEMS", 0), help="Optional cap for testing math precision audit. 0 means all.")
    parser.add_argument("--math-audit-mode", choices=["standard", "strict"], default=os.environ.get("GRADE10_MATHS_MATH_AUDIT_MODE", "standard").strip().lower() or "standard", help="Math precision audit sensitivity. standard is recommended; strict may over-flag.")
    parser.add_argument("--force-math-audit-cache", action="store_true", help="Ignore cached math precision vision repair results and call OpenAI again.")
    parser.add_argument("--skip-residual-production-audit", action="store_true", help="Skip Step 6 final residual production text audit. Not recommended for production DB ingestion.")
    parser.add_argument("--residual-audit-pages", default=None, help="Optional PDF pages to residual-audit only, e.g. 128,503,710.")
    parser.add_argument("--full-image-verify", action="store_true", help="Run Step 7 full image-vs-production-text verifier. Also enabled by GRADE10_MATHS_FULL_IMAGE_VERIFY=true. This can call the vision model for every production page.")
    parser.add_argument("--llm-repair-failed-pages", action="store_true", help="Cheap production mode: collect failed pages from Step 4/5/6, send only those pages to Step 7 LLM/image repair, then re-run Step 6 strictly.")
    parser.add_argument("--llm-repair-max-rounds", type=int, default=env_int("GRADE10_MATHS_LLM_REPAIR_MAX_ROUNDS", 2), help="Maximum targeted LLM repair rounds. Each round repairs remaining Step 6/Step 7 pages only. Default 2.")
    parser.add_argument("--full-verify-pages", default=None, help="Optional PDF pages for Step 7 full image verification, e.g. 9,19-20,586. Omit for all ready lesson-body pages.")
    parser.add_argument("--full-verify-max-pages", type=int, default=env_int("GRADE10_MATHS_FULL_VERIFY_MAX_ITEMS", 0), help="Optional cap for Step 7 test runs. 0 means all selected pages.")
    parser.add_argument("--full-verify-model", default=os.environ.get("GRADE10_MATHS_FULL_VERIFY_MODEL", "").strip() or None, help="Vision model for Step 7. Defaults to GRADE10_MATHS_FULL_VERIFY_MODEL or math precision model or gpt-4o.")
    parser.add_argument("--force-full-verify-cache", action="store_true", help="Ignore cached Step 7 full image verification results.")
    parser.add_argument("--full-verify-no-repair", action="store_true", help="Only verify in Step 7; do not write corrected text back into the JSON.")
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
    env_overrides = os.environ.get("GRADE10_MATHS_PAGE_OVERRIDES", "").strip()
    if args.page_overrides_json:
        page_overrides_json = resolve(args.page_overrides_json, project_root)
    elif env_overrides:
        page_overrides_json = resolve(Path(env_overrides), project_root)
    else:
        page_overrides_json = package_dir / DEFAULT_PAGE_OVERRIDES_JSON

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    if not subsections_json.exists():
        raise FileNotFoundError(subsections_json)

    step0 = package_dir / "make_grade10_maths_step_0_structure.py"
    step1 = package_dir / "make_grade10_maths_step_1_base_extract.py"
    step2 = package_dir / "make_grade10_maths_step_2_production_gate.py"
    step3 = package_dir / "make_grade10_maths_step_3_vision_repair.py"
    step4 = package_dir / "make_grade10_maths_step_4_final_audit_repair.py"
    step5 = package_dir / "make_grade10_maths_step_5_math_precision_audit_repair.py"
    step6 = package_dir / "make_grade10_maths_step_6_residual_production_audit.py"
    step7 = package_dir / "make_grade10_maths_step_7_full_image_production_verify.py"
    known_fixes = package_dir / "make_grade10_maths_known_page_fixes.py"

    auto_review_provider = os.environ.get("GRADE10_MATHS_AUTO_REVIEW_PROVIDER", "").strip().lower()
    env_repair_with_vision = auto_review_provider in {"openai", "openai_vision", "vision"}
    if auto_review_provider and auto_review_provider not in {"none", "off", "disabled", "openai", "openai_vision", "vision"}:
        raise ValueError(
            "Unsupported GRADE10_MATHS_AUTO_REVIEW_PROVIDER="
            f"{auto_review_provider!r}. Use openai or none."
        )
    repair_with_vision = bool(args.repair_with_vision or env_repair_with_vision)
    strict_complete = bool(args.strict_complete or env_bool("GRADE10_MATHS_FAIL_ON_REVIEW_REQUIRED", False))
    vision_model = args.vision_model or os.environ.get("GRADE10_MATHS_VISION_MODEL", "gpt-4o-mini")
    math_precision_model = os.environ.get("GRADE10_MATHS_MATH_PRECISION_MODEL", "").strip() or vision_model
    final_audit_enabled = (not args.skip_final_audit) and env_bool("GRADE10_MATHS_FINAL_AUDIT", True)
    math_precision_enabled = (not args.skip_math_precision_audit) and env_bool("GRADE10_MATHS_MATH_PRECISION_AUDIT", True)
    residual_audit_enabled = (not args.skip_residual_production_audit) and env_bool("GRADE10_MATHS_RESIDUAL_PRODUCTION_AUDIT", True)
    full_image_verify_enabled = bool(args.full_image_verify or env_bool("GRADE10_MATHS_FULL_IMAGE_VERIFY", False))
    llm_repair_failed_pages_enabled = bool(args.llm_repair_failed_pages or env_bool("GRADE10_MATHS_LLM_REPAIR_FAILED_PAGES", False))
    # When any Step 7 repair mode is enabled, earlier audits should collect failed pages but not stop the pipeline.
    # Step 7 repairs the collected pages from the PDF image, then Step 6 runs strictly as the final hard gate.
    defer_strict_to_step7 = bool(full_image_verify_enabled or llm_repair_failed_pages_enabled)
    run_full_scope_step7 = bool(full_image_verify_enabled and not llm_repair_failed_pages_enabled)

    scripts_to_check = [step0, step1, step2, known_fixes]
    if repair_with_vision:
        scripts_to_check.append(step3)
    if final_audit_enabled:
        scripts_to_check.append(step4)
    if math_precision_enabled:
        scripts_to_check.append(step5)
    if residual_audit_enabled:
        scripts_to_check.append(step6)
    if full_image_verify_enabled or llm_repair_failed_pages_enabled:
        scripts_to_check.append(step7)
    for script in scripts_to_check:
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

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("PDF path: %s", pdf_path)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Static subsection ranges: %s", subsections_json)
    LOGGER.info("Page overrides JSON: %s%s", page_overrides_json, "" if page_overrides_json.exists() else " (not found; skipped)")
    LOGGER.info("Grade10 Maths auto review provider: %s", auto_review_provider or "<not set>")
    LOGGER.info("Repair with vision: %s", repair_with_vision)
    LOGGER.info("Strict complete: %s", strict_complete)
    LOGGER.info("Final production audit: %s", final_audit_enabled)
    LOGGER.info("Final audit mode: %s", args.final_audit_mode)
    LOGGER.info("Math precision audit: %s", math_precision_enabled)
    LOGGER.info("Math precision audit mode: %s", args.math_audit_mode)
    LOGGER.info("Residual production audit: %s", residual_audit_enabled)
    LOGGER.info("Full image production verify: %s", full_image_verify_enabled)
    LOGGER.info("LLM repair failed/remaining pages only: %s", llm_repair_failed_pages_enabled)
    if llm_repair_failed_pages_enabled:
        LOGGER.info("LLM repair max rounds: %s", args.llm_repair_max_rounds)
    LOGGER.info("Step 7 full-scope mode after target override: %s", run_full_scope_step7)
    if math_precision_enabled:
        LOGGER.info("Math precision model: %s", math_precision_model)
    if repair_with_vision:
        LOGGER.info("Vision model: %s", vision_model)
        LOGGER.info("Vision min confidence: %s", args.vision_min_confidence)
        LOGGER.info("Vision max pages: %s", args.vision_max_pages)

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
        ])

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
    run_cmd(step1_cmd)
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
    # If we are repairing with vision, let Step 2 produce the QA list first.
    # Strict completeness is enforced after Step 3.
    if strict_complete and not repair_with_vision:
        step2_cmd.append("--strict-complete")
    run_cmd(step2_cmd)
    assert_step2_complete(step2_json, smoke_run=smoke_run)

    if repair_with_vision:
        if smoke_run:
            raise RuntimeError("--repair-with-vision is for full-book production runs. Do not combine it with --pages smoke runs.")
        step3_report = output_dir / STEP3_REPORT
        remaining_qa_csv = output_dir / STEP3_REMAINING_QA_CSV
        step3_cmd = [
            sys.executable,
            str(step3),
            "--pdf", str(pdf_path),
            "--input", str(step2_json),
            "--output", str(step2_json),
            "--report", str(step3_report),
            "--remaining-qa-csv", str(remaining_qa_csv),
            "--scale", str(args.vision_scale),
            "--min-confidence", str(args.vision_min_confidence),
        ]
        if page_overrides_json.exists():
            step3_cmd.extend(["--manual-overrides", str(page_overrides_json)])
        if vision_model:
            step3_cmd.extend(["--model", str(vision_model)])
        if args.vision_pages:
            step3_cmd.extend(["--pages", args.vision_pages])
        if args.vision_max_pages and args.vision_max_pages > 0:
            step3_cmd.extend(["--max-pages", str(args.vision_max_pages)])
        if args.force_vision:
            step3_cmd.append("--force-vision")
        # Do not stop at Step 3 when Step 4 is enabled. Step 4 is the final strict gate:
        # it repairs both Step-3 failures and false-ready pages with residual OCR garbage.
        if strict_complete and not final_audit_enabled:
            step3_cmd.append("--strict-complete")
        run_cmd(step3_cmd)
        assert_step2_complete(step2_json, smoke_run=False)

    if final_audit_enabled and not smoke_run:
        step4_report = output_dir / STEP4_REPORT
        suspicious_csv = output_dir / STEP4_SUSPICIOUS_CSV
        remaining_suspicious_csv = output_dir / STEP4_REMAINING_SUSPICIOUS_CSV
        step4_cmd = [
            sys.executable,
            str(step4),
            "--pdf", str(pdf_path),
            "--input", str(step2_json),
            "--output", str(step2_json),
            "--report", str(step4_report),
            "--suspicious-csv", str(suspicious_csv),
            "--remaining-suspicious-csv", str(remaining_suspicious_csv),
            "--scale", str(args.vision_scale),
            "--min-confidence", str(args.vision_min_confidence),
            "--audit-mode", str(args.final_audit_mode),
        ]
        if page_overrides_json.exists():
            step4_cmd.extend(["--manual-overrides", str(page_overrides_json)])
        if vision_model:
            step4_cmd.extend(["--model", str(vision_model)])
        if args.final_audit_pages:
            step4_cmd.extend(["--pages", args.final_audit_pages])
        if args.final_audit_max_pages and args.final_audit_max_pages > 0:
            step4_cmd.extend(["--max-pages", str(args.final_audit_max_pages)])
        if repair_with_vision:
            step4_cmd.append("--repair-with-vision")
        if args.force_vision or args.force_final_audit_cache:
            step4_cmd.append("--force-vision")
        if strict_complete and not defer_strict_to_step7:
            step4_cmd.append("--strict-complete")
        run_cmd(step4_cmd)
        assert_final_production_quality(step2_json, strict_complete=(strict_complete and not defer_strict_to_step7))


    if math_precision_enabled and not smoke_run:
        step5_report = output_dir / STEP5_REPORT
        math_suspicious_csv = output_dir / STEP5_SUSPICIOUS_CSV
        math_remaining_csv = output_dir / STEP5_REMAINING_CSV
        step5_cmd = [
            sys.executable,
            str(step5),
            "--pdf", str(pdf_path),
            "--input", str(step2_json),
            "--output", str(step2_json),
            "--report", str(step5_report),
            "--suspicious-csv", str(math_suspicious_csv),
            "--remaining-csv", str(math_remaining_csv),
            "--scale", str(args.vision_scale),
            "--min-confidence", str(args.vision_min_confidence),
            "--audit-mode", str(args.math_audit_mode),
        ]
        if page_overrides_json.exists():
            step5_cmd.extend(["--manual-overrides", str(page_overrides_json)])
        if math_precision_model:
            step5_cmd.extend(["--model", str(math_precision_model)])
        if args.math_audit_pages:
            step5_cmd.extend(["--pages", args.math_audit_pages])
        if args.math_audit_max_pages and args.math_audit_max_pages > 0:
            step5_cmd.extend(["--max-pages", str(args.math_audit_max_pages)])
        if repair_with_vision:
            step5_cmd.append("--repair-with-vision")
        if args.force_vision or args.force_math_audit_cache:
            step5_cmd.append("--force-vision")
        if strict_complete and not defer_strict_to_step7:
            step5_cmd.append("--strict-complete")
        run_cmd(step5_cmd)
        assert_final_production_quality(step2_json, strict_complete=(strict_complete and not defer_strict_to_step7))
    if residual_audit_enabled and not smoke_run:
        # v20: run deterministic reviewer-confirmed fixes immediately before the final
        # hard production gate. Step 6 still remains the source of truth and fails if
        # any hard OCR/math pattern survives in production_safe_text or
        # production_subsection_text.
        known_report = output_dir / KNOWN_FIXES_REPORT
        run_cmd([
            sys.executable,
            str(known_fixes),
            "--input", str(step2_json),
            "--output", str(step2_json),
            "--report", str(known_report),
        ])
        step6_report = output_dir / STEP6_REPORT
        residual_remaining_csv = output_dir / STEP6_REMAINING_CSV
        step6_cmd = [
            sys.executable,
            str(step6),
            "--input", str(step2_json),
            "--output", str(step2_json),
            "--report", str(step6_report),
            "--remaining-csv", str(residual_remaining_csv),
        ]
        if args.residual_audit_pages:
            step6_cmd.extend(["--pages", args.residual_audit_pages])
        # If Step 7 full image verification or targeted LLM repair is enabled, do not stop here on known
        # hard-gate failures. Step 7 is allowed to repair them from the page image.
        if strict_complete and not defer_strict_to_step7:
            step6_cmd.append("--strict-complete")
        run_cmd(step6_cmd)
        assert_final_production_quality(step2_json, strict_complete=(strict_complete and not defer_strict_to_step7))

        if run_full_scope_step7 or llm_repair_failed_pages_enabled:
            step7_report = output_dir / STEP7_REPORT
            step7_remaining_csv = output_dir / STEP7_REMAINING_CSV
            step7_cache_dir = output_dir / STEP7_CACHE_DIR
            targeted_repair_csv = output_dir / STEP_TARGETED_LLM_REPAIR_CSV
            full_model = args.full_verify_model or math_precision_model or "gpt-4o"

            if run_full_scope_step7 and not llm_repair_failed_pages_enabled:
                # True full image verification mode: keep strict Step 7 semantics.
                step7_cmd = [
                    sys.executable,
                    str(step7),
                    "--pdf", str(pdf_path),
                    "--input", str(step2_json),
                    "--output", str(step2_json),
                    "--report", str(step7_report),
                    "--remaining-csv", str(step7_remaining_csv),
                    "--cache-dir", str(step7_cache_dir),
                    "--scale", str(args.vision_scale),
                ]
                if full_model:
                    step7_cmd.extend(["--model", str(full_model)])
                if args.full_verify_pages:
                    step7_cmd.extend(["--pages", args.full_verify_pages])
                if args.full_verify_max_pages and args.full_verify_max_pages > 0:
                    step7_cmd.extend(["--max-pages", str(args.full_verify_max_pages)])
                if args.force_full_verify_cache:
                    step7_cmd.append("--force-cache")
                if args.full_verify_no_repair:
                    step7_cmd.append("--no-repair")
                if strict_complete:
                    step7_cmd.append("--strict-complete")
                run_cmd(step7_cmd)

                final_step6_cmd = [
                    sys.executable,
                    str(step6),
                    "--input", str(step2_json),
                    "--output", str(step2_json),
                    "--report", str(step6_report),
                    "--remaining-csv", str(residual_remaining_csv),
                ]
                if strict_complete:
                    final_step6_cmd.append("--strict-complete")
                run_cmd(final_step6_cmd)
                assert_final_production_quality(step2_json, strict_complete=strict_complete)

            elif llm_repair_failed_pages_enabled:
                # Targeted economical mode: do not use Step 7's strict full-image gate.
                # Step 7 regenerates selected failed pages from the image only, then
                # Step 6 is the final strict production hard gate. This avoids the
                # over-strict verifier loop that can fail after a good transcription
                # because of formatting/detail nitpicks.
                max_rounds = max(1, int(args.llm_repair_max_rounds or 1))
                last_step6_strict_run = False
                for round_no in range(1, max_rounds + 1):
                    if round_no == 1:
                        target_source_csvs = [
                            output_dir / STEP4_REMAINING_SUSPICIOUS_CSV,
                            output_dir / STEP5_REMAINING_CSV,
                            residual_remaining_csv,
                            step7_remaining_csv,
                        ]
                    else:
                        # After the first image-only repair, use only fresh failures.
                        target_source_csvs = [residual_remaining_csv, step7_remaining_csv]
                    targeted_pages = collect_page_numbers_from_csvs(target_source_csvs)
                    if not targeted_pages:
                        LOGGER.info("No failed/remaining pages found for targeted LLM repair round %s.", round_no)
                        break
                    existing_sources = [p for p in target_source_csvs if p.exists()]
                    write_page_numbers_to_csv(targeted_repair_csv, targeted_pages, source_csvs=existing_sources)
                    LOGGER.info("Targeted LLM repair round %s/%s pages: %s", round_no, max_rounds, targeted_pages)
                    LOGGER.info("Targeted LLM repair CSV: %s", targeted_repair_csv)

                    step7_cmd = [
                        sys.executable,
                        str(step7),
                        "--pdf", str(pdf_path),
                        "--input", str(step2_json),
                        "--output", str(step2_json),
                        "--report", str(step7_report),
                        "--remaining-csv", str(step7_remaining_csv),
                        "--cache-dir", str(step7_cache_dir),
                        "--scale", str(args.vision_scale),
                        "--pages-from-csv", str(targeted_repair_csv),
                        "--repair-selected-from-image-only",
                        "--no-verify-repaired",
                    ]
                    if full_model:
                        step7_cmd.extend(["--model", str(full_model)])
                    if args.full_verify_max_pages and args.full_verify_max_pages > 0:
                        step7_cmd.extend(["--max-pages", str(args.full_verify_max_pages)])
                    if args.force_full_verify_cache:
                        step7_cmd.append("--force-cache")
                    if args.full_verify_no_repair:
                        step7_cmd.append("--no-repair")
                    run_cmd(step7_cmd)

                    # Re-run deterministic known fixes after image text replacement.
                    run_cmd([
                        sys.executable,
                        str(known_fixes),
                        "--input", str(step2_json),
                        "--output", str(step2_json),
                        "--report", str(known_report),
                    ])

                    # Run Step 6. On intermediate rounds, collect failures but do not stop.
                    round_step6_cmd = [
                        sys.executable,
                        str(step6),
                        "--input", str(step2_json),
                        "--output", str(step2_json),
                        "--report", str(step6_report),
                        "--remaining-csv", str(residual_remaining_csv),
                    ]
                    if strict_complete and round_no == max_rounds:
                        round_step6_cmd.append("--strict-complete")
                        last_step6_strict_run = True
                    run_cmd(round_step6_cmd)
                    remaining_after_round = read_page_numbers_from_csv(residual_remaining_csv)
                    if not remaining_after_round:
                        LOGGER.info("Targeted LLM repair succeeded after round %s; no Step 6 residual pages remain.", round_no)
                        break
                    LOGGER.warning("Targeted LLM repair round %s left %s Step 6 residual pages: %s", round_no, len(remaining_after_round), remaining_after_round)

                if strict_complete and not last_step6_strict_run:
                    final_step6_cmd = [
                        sys.executable,
                        str(step6),
                        "--input", str(step2_json),
                        "--output", str(step2_json),
                        "--report", str(step6_report),
                        "--remaining-csv", str(residual_remaining_csv),
                        "--strict-complete",
                    ]
                    run_cmd(final_step6_cmd)
                assert_final_production_quality(step2_json, strict_complete=strict_complete)
    elif residual_audit_enabled and smoke_run:
        LOGGER.warning("Skipping residual production audit for --pages smoke/partial run.")

    elif math_precision_enabled and smoke_run:
        LOGGER.warning("Skipping math precision audit for --pages smoke/partial run.")
    elif final_audit_enabled and smoke_run:
        LOGGER.warning("Skipping final production audit for --pages smoke/partial run.")

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
        if repair_with_vision:
            print(f"Vision repair report: {output_dir / STEP3_REPORT}")
            print(f"Remaining QA CSV:     {output_dir / STEP3_REMAINING_QA_CSV}")
        if final_audit_enabled:
            print(f"Final audit report:   {output_dir / STEP4_REPORT}")
            print(f"Final suspicious CSV: {output_dir / STEP4_SUSPICIOUS_CSV}")
            print(f"Final remaining CSV:  {output_dir / STEP4_REMAINING_SUSPICIOUS_CSV}")
        if math_precision_enabled:
            print(f"Math precision report:   {output_dir / STEP5_REPORT}")
            print(f"Math suspicious CSV:     {output_dir / STEP5_SUSPICIOUS_CSV}")
            print(f"Math remaining CSV:      {output_dir / STEP5_REMAINING_CSV}")
        if residual_audit_enabled:
            print(f"Known fixes report: {output_dir / KNOWN_FIXES_REPORT}")
            print(f"Residual production report: {output_dir / STEP6_REPORT}")
            print(f"Residual remaining CSV:    {output_dir / STEP6_REMAINING_CSV}")
        if full_image_verify_enabled:
            print(f"Full image verify report: {output_dir / STEP7_REPORT}")
            print(f"Full image verify remaining CSV: {output_dir / STEP7_REMAINING_CSV}")
        if llm_repair_failed_pages_enabled:
            print(f"Targeted LLM repair CSV: {output_dir / STEP_TARGETED_LLM_REPAIR_CSV}")
    print(f"Subsections JSON:  {subsections_json}")


if __name__ == "__main__":
    main()
