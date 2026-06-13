#!/usr/bin/env python3
"""
Orchestrates the production-safe Grade 10 Physics extraction pipeline.

Default project structure:
app/
  make_physics_grade10.py
  physics_tenth/
    Grade10_Physics_static_subsection_ranges.json
    Grade10_Physics_formula_corrections.json
    physics_common.py
    physics_step_1_base_extract.py
    physics_step_2_production_text.py
    physics_step_3_apply_corrections.py
    physics_step_4_build_subsections.py
    physics_step_5_publish_production.py
input/
  Grade10_Physics.pdf
output/
  physics_tenth/
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # keep .env support optional
    def load_dotenv(*_args, **_kwargs):
        return False

LOGGER = logging.getLogger("physics_tenth.pipeline")

load_dotenv()

DEFAULT_SUBSECTIONS_JSON = "Grade10_Physics_static_subsection_ranges.json"
DEFAULT_CORRECTIONS_JSON = "Grade10_Physics_formula_corrections.json"

STEP1_JSON = "Grade10_Physics_step1_base_extraction.json"
STEP1_REPORT = "Grade10_Physics_step1_validation_report.txt"
STEP2_JSON = "Grade10_Physics_step2_safe_text_extraction.json"
STEP2_REPORT = "Grade10_Physics_step2_safe_text_report.txt"
STEP2_REVIEW_QUEUE = "Grade10_Physics_step2_review_queue.json"
STEP3_JSON = "Grade10_Physics_step3_reviewed_text_extraction.json"
STEP3_REPORT = "Grade10_Physics_step3_corrections_report.txt"
STEP3_REVIEW_QUEUE = "Grade10_Physics_step3_remaining_review_queue.json"
STEP4_JSON = "Grade10_Physics_step4_structured_subsections.json"
STEP4_REPORT = "Grade10_Physics_step4_validation_report.txt"
STEP4_GATED_JSON = "Grade10_Physics_step4_final_artifact_gated.json"
FINAL_ARTIFACT_REPORT = "Grade10_Physics_final_artifact_gate_report.txt"
FINAL_ARTIFACT_REVIEW_QUEUE = "Grade10_Physics_final_artifact_review_queue.json"
FINAL_JSON = "Grade10_Physics_production_ready.json"
FINAL_REPORT = "Grade10_Physics_production_validation_report.txt"


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def find_file(project_root: Path, app_dir: Path, output_dir: Path, filename: str) -> Path:
    candidates = [
        project_root / filename,
        app_dir / filename,
        app_dir / "physics_tenth" / filename,
        output_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n{checked}")


def find_step_script(app_dir: Path, script_name: str) -> Path:
    candidates = [app_dir / script_name, app_dir / "physics_tenth" / script_name]
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


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {name}: {value!r}") from exc


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {name}: {value!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production-safe Grade 10 Physics JSON extraction pipeline.")
    parser.add_argument("--pdf", type=Path, default=Path("input/Grade10_Physics.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/physics_tenth"))
    parser.add_argument("--subsections-json", type=Path, default=None)
    parser.add_argument("--corrections-json", type=Path, default=None)
    parser.add_argument("--document-id", default="modern-abc-physics-class-10-2022-23")
    parser.add_argument("--document-key", default="mother-miracle-class-10-physics-modern-abc")
    parser.add_argument("--allow-review-required", action="store_true", default=env_truthy("PHYSICS_ALLOW_REVIEW_REQUIRED", False), help="Generate final output even if formula review remains. Output status will be review_required_not_formula_safe.")
    parser.add_argument("--fail-on-review-required", action="store_true", default=env_truthy("PHYSICS_FAIL_ON_REVIEW_REQUIRED", False), help="Exit non-zero if unresolved formula/table/diagram review items remain.")
    parser.add_argument("--placeholder-unreviewed", action="store_true", default=env_truthy("PHYSICS_PLACEHOLDER_UNREVIEWED", False), help="Keep placeholders where unreviewed formula/table/diagram text was removed.")
    parser.add_argument("--skip-final-artifact-gate", action="store_true", default=env_truthy("PHYSICS_SKIP_FINAL_ARTIFACT_GATE", False), help="Skip the final production-text artifact scanner. Not recommended for Physics production.")
    parser.add_argument("--allow-final-artifacts", action="store_true", default=env_truthy("PHYSICS_ALLOW_FINAL_ARTIFACTS", False), help="Write gated Step 4 output even if leftover artifacts remain. Not for final production.")
    parser.add_argument("--auto-review-provider", choices=["none", "tesseract", "openai"], default=os.getenv("PHYSICS_AUTO_REVIEW_PROVIDER", "none"), help="Optionally auto-fill corrections JSON from PDF crops before Step 3. Use openai for production-grade formula transcription.")
    parser.add_argument("--auto-review-model", default=os.getenv("OPENAI_VISION_MODEL"), help="Vision model name used when --auto-review-provider openai. Defaults to OPENAI_VISION_MODEL or script default.")
    parser.add_argument("--auto-review-threshold", type=float, default=env_float("PHYSICS_AUTO_REVIEW_THRESHOLD", 0.92), help="Minimum provider confidence accepted into corrections JSON.")
    parser.add_argument("--auto-review-max-items", type=int, default=env_int("PHYSICS_AUTO_REVIEW_MAX_ITEMS", 0), help="Limit auto-review items for testing. 0 means all.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    project_root = Path.cwd()
    app_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.pdf if args.pdf.is_absolute() else project_root / args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    subsections_json = args.subsections_json if args.subsections_json else find_file(project_root, app_dir, output_dir, DEFAULT_SUBSECTIONS_JSON)
    if not subsections_json.is_absolute():
        subsections_json = project_root / subsections_json
    corrections_json = args.corrections_json if args.corrections_json else find_file(project_root, app_dir, output_dir, DEFAULT_CORRECTIONS_JSON)
    if not corrections_json.is_absolute():
        corrections_json = project_root / corrections_json

    scripts = {
        "step1": find_step_script(app_dir, "physics_step_1_base_extract.py"),
        "step2": find_step_script(app_dir, "physics_step_2_production_text.py"),
        "step3": find_step_script(app_dir, "physics_step_3_apply_corrections.py"),
        "step4": find_step_script(app_dir, "physics_step_4_build_subsections.py"),
        "step5a": find_step_script(app_dir, "physics_step_5a_final_artifact_gate.py"),
        "step5": find_step_script(app_dir, "physics_step_5_publish_production.py"),
    }
    auto_review_script = find_step_script(app_dir, "auto_review_formula_corrections.py") if args.auto_review_provider != "none" else None

    outputs = [
        output_dir / STEP1_JSON, output_dir / STEP1_REPORT,
        output_dir / STEP2_JSON, output_dir / STEP2_REPORT, output_dir / STEP2_REVIEW_QUEUE,
        output_dir / STEP3_JSON, output_dir / STEP3_REPORT, output_dir / STEP3_REVIEW_QUEUE,
        output_dir / STEP4_JSON, output_dir / STEP4_REPORT,
        output_dir / STEP4_GATED_JSON, output_dir / FINAL_ARTIFACT_REPORT, output_dir / FINAL_ARTIFACT_REVIEW_QUEUE,
        output_dir / FINAL_JSON, output_dir / FINAL_REPORT,
    ]
    if not args.force:
        existing = [p for p in outputs if p.exists()]
        if existing:
            raise FileExistsError("Output files already exist. Use --force to overwrite:\n" + "\n".join(f"- {p}" for p in existing))

    run_cmd([sys.executable, str(scripts["step1"]), "--pdf", str(pdf_path), "--subsections-json", str(subsections_json), "--output", str(output_dir / STEP1_JSON), "--report", str(output_dir / STEP1_REPORT), "--force", "--log-level", args.log_level])
    step2_cmd = [sys.executable, str(scripts["step2"]), "--input-json", str(output_dir / STEP1_JSON), "--output-json", str(output_dir / STEP2_JSON), "--report", str(output_dir / STEP2_REPORT), "--review-queue", str(output_dir / STEP2_REVIEW_QUEUE), "--force", "--log-level", args.log_level]
    if args.placeholder_unreviewed:
        step2_cmd.append("--placeholder-unreviewed")
    run_cmd(step2_cmd)
    if args.auto_review_provider != "none":
        auto_cmd = [
            sys.executable, str(auto_review_script),
            "--pdf", str(pdf_path),
            "--step-json", str(output_dir / STEP2_JSON),
            "--review-queue", str(output_dir / STEP2_REVIEW_QUEUE),
            "--corrections-json", str(corrections_json),
            "--output-dir", str(output_dir / "auto_formula_review"),
            "--provider", args.auto_review_provider,
            "--confidence-threshold", str(args.auto_review_threshold),
            "--force",
        ]
        if args.auto_review_model:
            auto_cmd.extend(["--model", args.auto_review_model])
        if args.auto_review_max_items and args.auto_review_max_items > 0:
            auto_cmd.extend(["--max-items", str(args.auto_review_max_items)])
        if args.auto_review_provider == "tesseract":
            auto_cmd.append("--accept-tesseract")
        run_cmd(auto_cmd)
    step3_cmd = [sys.executable, str(scripts["step3"]), "--input-json", str(output_dir / STEP2_JSON), "--corrections-json", str(corrections_json), "--output-json", str(output_dir / STEP3_JSON), "--report", str(output_dir / STEP3_REPORT), "--review-queue", str(output_dir / STEP3_REVIEW_QUEUE), "--force", "--log-level", args.log_level]
    if args.placeholder_unreviewed:
        step3_cmd.append("--placeholder-unreviewed")
    run_cmd(step3_cmd)
    run_cmd([sys.executable, str(scripts["step4"]), "--input-json", str(output_dir / STEP3_JSON), "--subsections-json", str(subsections_json), "--output-json", str(output_dir / STEP4_JSON), "--report", str(output_dir / STEP4_REPORT), "--force", "--log-level", args.log_level])

    step5_input_json = output_dir / STEP4_JSON
    if not args.skip_final_artifact_gate:
        final_gate_cmd = [
            sys.executable, str(scripts["step5a"]),
            "--input-json", str(output_dir / STEP4_JSON),
            "--output-json", str(output_dir / STEP4_GATED_JSON),
            "--review-queue", str(output_dir / FINAL_ARTIFACT_REVIEW_QUEUE),
            "--report", str(output_dir / FINAL_ARTIFACT_REPORT),
            "--force",
            "--log-level", args.log_level,
        ]
        if args.allow_final_artifacts or args.allow_review_required:
            final_gate_cmd.append("--allow-artifacts")
        run_cmd(final_gate_cmd)
        step5_input_json = output_dir / STEP4_GATED_JSON

    step5_cmd = [sys.executable, str(scripts["step5"]), "--input-json", str(step5_input_json), "--output-json", str(output_dir / FINAL_JSON), "--report", str(output_dir / FINAL_REPORT), "--document-id", args.document_id, "--document-key", args.document_key, "--force", "--log-level", args.log_level]
    if args.allow_review_required:
        step5_cmd.append("--allow-review-required")
    if args.fail_on_review_required:
        step5_cmd.append("--fail-on-review-required")
    run_cmd(step5_cmd)

    print("\nDone.")
    print(f"Output dir: {output_dir}")
    print(f"Final JSON: {output_dir / FINAL_JSON}")
    print(f"Final report: {output_dir / FINAL_REPORT}")
    print(f"Formula review queue: {output_dir / STEP3_REVIEW_QUEUE}")
    if not args.skip_final_artifact_gate:
        print(f"Final artifact review queue: {output_dir / FINAL_ARTIFACT_REVIEW_QUEUE}")


if __name__ == "__main__":
    main()
