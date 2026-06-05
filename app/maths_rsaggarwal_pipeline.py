#!/usr/bin/env python3
"""
run_maths_rsaggarwal_pipeline.py

Place this file at:
  pdf_extraction_to_json/app/run_maths_rsaggarwal_pipeline.py

Expected project layout:
  pdf_extraction_to_json/
    app/
      run_maths_rsaggarwal_pipeline.py
      maths_rsaggarwal/
        make_maths_rsaggarwal_step_1.py
        make_maths_rsaggarwal_step_2.py
        make_maths_rsaggarwal_step_3.py
        make_maths_rsaggarwal_step_4.py
        # or make_maths_rsaggarwal_step_4(1).py
    input/
      Maths_RSAgarwal.pdf
    output/
      maths_rsagarwal/

Run from project root:
  python app/run_maths_rsaggarwal_pipeline.py

Optional:
  python app/run_maths_rsaggarwal_pipeline.py --force
  python app/run_maths_rsaggarwal_pipeline.py --python .venv/Scripts/python.exe
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


STEP0_NAME = "make_maths_rsaggarwal_step_0.py"
STEP1_NAME = "make_maths_rsaggarwal_step_1.py"
STEP2_NAME = "make_maths_rsaggarwal_step_2.py"
STEP3_NAME = "make_maths_rsaggarwal_step_3.py"
STEP4_NAME = "make_maths_rsaggarwal_step_4.py"

OUT_CHAPTERS_JSON = "Maths_RSAgarwal_chapters.json"
OUT_CHAPTERS_PY = "Maths_RSAgarwal_chapters.py"

OUT_STEP1_JSON = "Maths_RSAgarwal_math_aware_extraction.json"
OUT_STEP1_REPORT = "Maths_RSAgarwal_math_aware_validation_report.txt"

OUT_STEP2_JSON = "Maths_RSAgarwal_math_aware_extraction_v2_cleaned.json"
OUT_STEP2_REPORT = "Maths_RSAgarwal_math_aware_v2_validation_report.txt"

OUT_STEP3_JSON = "Maths_RSAgarwal_math_aware_extraction_v3_cleaned.json"
OUT_STEP3_REPORT = "Maths_RSAgarwal_math_aware_v3_validation_report.txt"

OUT_STEP4_JSON = "Maths_RSAgarwal_math_aware_extraction_v4_production_safe.json"
OUT_STEP4_REPORT = "Maths_RSAgarwal_math_aware_v4_production_validation_report.txt"
OUT_STEP4_CSV = "Maths_RSAgarwal_math_aware_v4_pages_requiring_vision_qa.csv"


def q(path: Path) -> str:
    """Return a raw string literal path for patching Python code."""
    return "Path(r" + repr(str(path)) + ")"


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("\n" + "=" * 90)
    print("RUNNING:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print("=" * 90, flush=True)

    child_env = os.environ.copy()
    if env:
        child_env.update({k: str(v) for k, v in env.items()})

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=child_env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"\nCommand failed with exit code {proc.returncode}: {' '.join(cmd)}")


def check_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def check_tesseract() -> None:
    try:
        proc = subprocess.run(
            ["tesseract", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        raise SystemExit(
            "Tesseract is not found on PATH.\n"
            "Install Tesseract OCR and confirm this works:\n"
            "  tesseract --version"
        )

    if proc.returncode != 0:
        raise SystemExit("Tesseract exists but failed to run: tesseract --version")

    first_line = (proc.stdout or proc.stderr or "").splitlines()[0]
    print(f"Tesseract OK: {first_line}")


def check_python_dependency() -> None:
    try:
        import fitz  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing Python dependency: PyMuPDF.\n"
            "Install it with:\n"
            "  pip install pymupdf\n"
            f"Original error: {exc}"
        )


def patch_step1(src: Path, dst: Path, *, pdf_path: Path, output_dir: Path, chapters_json: Path) -> None:
    """
    Step 1 has hardcoded /mnt/data paths and a Windows cp1252 subprocess issue.
    This creates a patched copy under output/.runner_patched/ without editing your original file.
    """
    text = src.read_text(encoding="utf-8", errors="replace")

    replacements = {
        "PDF_PATH = Path('/mnt/data/Maths_RSAgarwal(2).pdf')":
            f"PDF_PATH = {q(pdf_path)}",
        "OUTPUT_JSON = Path('/mnt/data/Maths_RSAgarwal_math_aware_extraction.json')":
            f"OUTPUT_JSON = {q(output_dir / OUT_STEP1_JSON)}",
        "VALIDATION_REPORT = Path('/mnt/data/Maths_RSAgarwal_math_aware_validation_report.txt')":
            f"VALIDATION_REPORT = {q(output_dir / OUT_STEP1_REPORT)}",
        "OCR_CACHE_DIR = Path('/mnt/data/maths_rsaggarwal_ocr_cache_fast')":
            f"OCR_CACHE_DIR = {q(output_dir / '.ocr_cache' / 'maths_rsaggarwal_ocr_cache_fast')}",
        "HIGH_RES_OCR_CACHE_DIR = Path('/mnt/data/maths_rsaggarwal_ocr_cache')":
            f"HIGH_RES_OCR_CACHE_DIR = {q(output_dir / '.ocr_cache' / 'maths_rsaggarwal_ocr_cache')}",
    }

    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
        elif "MATHS_RSAGGARWAL_OUTPUT_DIR" in text and "PROJECT_ROOT" in text:
            # Newer Step 1 already uses project-relative paths/environment variables.
            # Keep it unchanged; the default output folder is output/maths_rsagarwal.
            pass
        else:
            print(f"WARNING: expected Step 1 path line not found, skipping patch: {old}")

    # Replace the static CHAPTERS array with the Step 0 JSON output.
    # The original Step 1 code already imports json and Path, so this injected block is enough.
    auto_chapters_block = f"""AUTO_CHAPTERS_JSON = {q(chapters_json)}
if not AUTO_CHAPTERS_JSON.exists():
    raise FileNotFoundError(f'Auto-generated chapters file not found: {{AUTO_CHAPTERS_JSON}}')
_auto_chapter_config = json.loads(AUTO_CHAPTERS_JSON.read_text(encoding='utf-8'))
CHAPTERS = _auto_chapter_config['chapters']
PRINTED_OFFSET = int(_auto_chapter_config.get('printed_page_offset', PRINTED_OFFSET))
"""
    text, n = re.subn(
        r"CHAPTERS\s*=\s*\[\n.*?\n\]\n(?=\nROMAN_LABELS)",
        auto_chapters_block,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        if (
            "AUTO_CHAPTERS_JSON" in text
            and "_auto_chapter_config" in text
            and "CHAPTERS = _auto_chapter_config" in text
        ):
            print("Step 1 already reads chapters from Step 0 JSON; skipping CHAPTERS patch.")
        else:
            raise SystemExit("Could not replace static CHAPTERS block in Step 1 script.")

    # Fix Windows UnicodeDecodeError from subprocess.run(..., text=True).
    old_block = """proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=20)
        if proc.returncode != 0:
            ocr = ''
            err = proc.stderr.strip()[:1000]
        else:
            ocr = proc.stdout
            err = ''
        cache_file.write_text(ocr, encoding='utf-8')"""

    new_block = """proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env=env, timeout=60)
        stdout_text = (proc.stdout or b'').decode('utf-8', errors='replace')
        stderr_text = (proc.stderr or b'').decode('utf-8', errors='replace')
        if proc.returncode != 0:
            ocr = ''
            err = stderr_text.strip()[:1000]
        else:
            ocr = stdout_text or ''
            err = ''
        cache_file.write_text(ocr, encoding='utf-8', errors='replace')"""

    if old_block in text:
        text = text.replace(old_block, new_block)
    else:
        # Fallback patch for slightly different formatting.
        text2 = text.replace("text=True, env=env, timeout=20", "text=False, env=env, timeout=60")
        if text2 != text:
            text = text2
            # Add safety net after subprocess if exact block was not found.
            text = text.replace("ocr = proc.stdout", "ocr = (proc.stdout or b'').decode('utf-8', errors='replace')")
            text = text.replace("proc.stderr.strip()[:1000]", "(proc.stderr or b'').decode('utf-8', errors='replace').strip()[:1000]")
            text = text.replace("cache_file.write_text(ocr, encoding='utf-8')", "cache_file.write_text(ocr or '', encoding='utf-8', errors='replace')")
        else:
            print("WARNING: could not find Step 1 subprocess text=True block to patch.")

    dst.write_text(text, encoding="utf-8")


def patch_step4(src: Path, dst: Path, *, output_dir: Path) -> None:
    """
    Step 4 has hardcoded BASE = Path('/'), so patch it to output_dir.
    This creates a patched copy under output/.runner_patched/ without editing your original file.
    """
    text = src.read_text(encoding="utf-8", errors="replace")

    if "BASE = Path('/')" in text:
        text = text.replace("BASE = Path('/')", f"BASE = {q(output_dir)}")
    else:
        print("WARNING: expected Step 4 BASE = Path('/') line not found.")

    dst.write_text(text, encoding="utf-8")


def print_final_summary(output_dir: Path) -> None:
    final_json = output_dir / OUT_STEP4_JSON
    final_report = output_dir / OUT_STEP4_REPORT
    qa_csv = output_dir / OUT_STEP4_CSV

    print("\n" + "=" * 90)
    print("PIPELINE FINISHED")
    print("=" * 90)
    print(f"Generated chapters: {output_dir / OUT_CHAPTERS_JSON}")
    print(f"Final JSON:        {final_json}")
    print(f"Validation report: {final_report}")
    print(f"Vision QA CSV:     {qa_csv}")
    print("")
    print("For embeddings, use the final JSON and include only pages where:")
    print('  include_in_embeddings == true')
    print('  embedding_readiness == "ready_for_production_embedding"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 4-step Maths RSAggarwal PDF-to-JSON pipeline.")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to Maths_RSAgarwal.pdf")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--scripts-dir", type=Path, default=None, help="Directory containing step scripts")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts")
    parser.add_argument("--force", action="store_true", help="Delete existing output JSON/report files before running")
    args = parser.parse_args()

    # This file is expected at root/app/run_maths_rsaggarwal_pipeline.py
    app_dir = Path(__file__).resolve().parent
    root_dir = app_dir.parent

    scripts_dir = (args.scripts_dir or (app_dir / "maths_rsaggarwal")).resolve()
    pdf_path = (args.pdf or (root_dir / "input" / "Maths_RSAgarwal.pdf")).resolve()
    output_dir = (args.output_dir or (root_dir / "output" / "maths_rsagarwal")).resolve()
    patched_dir = output_dir / ".runner_patched"

    output_dir.mkdir(parents=True, exist_ok=True)
    patched_dir.mkdir(parents=True, exist_ok=True)

    step0 = scripts_dir / STEP0_NAME
    step1 = scripts_dir / STEP1_NAME
    step2 = scripts_dir / STEP2_NAME
    step3 = scripts_dir / STEP3_NAME
    step4 = scripts_dir / STEP4_NAME

    check_file(pdf_path, "input PDF")
    check_file(step0, "Step 0 chapter extraction script")
    check_file(step1, "Step 1 script")
    check_file(step2, "Step 2 script")
    check_file(step3, "Step 3 script")
    check_file(step4, "Step 4 script")

    check_python_dependency()
    check_tesseract()

    if args.force:
        for name in [
            OUT_CHAPTERS_JSON, OUT_CHAPTERS_PY,
            OUT_STEP1_JSON, OUT_STEP1_REPORT,
            OUT_STEP2_JSON, OUT_STEP2_REPORT,
            OUT_STEP3_JSON, OUT_STEP3_REPORT,
            OUT_STEP4_JSON, OUT_STEP4_REPORT, OUT_STEP4_CSV,
        ]:
            p = output_dir / name
            if p.exists():
                p.unlink()

    patched_step1 = patched_dir / STEP1_NAME
    patched_step4 = patched_dir / step4.name

    chapters_json = output_dir / OUT_CHAPTERS_JSON
    chapters_py = output_dir / OUT_CHAPTERS_PY

    # Important: patched scripts live under output\maths_rsagarwal\.runner_patched.
    # Without these env vars, Path(__file__).resolve().parents[2] inside the patched
    # scripts points to ...\output instead of the real project root, causing paths like
    # ...\output\output\maths_rsagarwal.
    script_env = {
        "MATHS_RSAGGARWAL_ROOT": str(root_dir),
        "MATHS_RSAGGARWAL_PDF": str(pdf_path),
        "MATHS_RSAGGARWAL_OUTPUT_DIR": str(output_dir),
        "MATHS_RSAGGARWAL_CHAPTERS_JSON": str(chapters_json),
    }

    # Step 0: read the PDF Contents page and generate the CHAPTERS array used by Step 1.
    run_cmd([
        args.python, str(step0),
        "--pdf", str(pdf_path),
        "--output-json", str(chapters_json),
        "--output-py", str(chapters_py),
    ], env=script_env)
    check_file(chapters_json, "Step 0 chapters JSON output")

    patch_step1(step1, patched_step1, pdf_path=pdf_path, output_dir=output_dir, chapters_json=chapters_json)
    patch_step4(step4, patched_step4, output_dir=output_dir)

    # Step 1: PDF -> rendered-page OCR JSON.
    run_cmd([args.python, str(patched_step1)], env=script_env)

    check_file(output_dir / OUT_STEP1_JSON, "Step 1 JSON output")

    # Step 2: first cleanup pass.
    run_cmd([
        args.python, str(step2),
        "--input", str(output_dir / OUT_STEP1_JSON),
        "--output", str(output_dir / OUT_STEP2_JSON),
        "--report", str(output_dir / OUT_STEP2_REPORT),
    ], env=script_env)

    check_file(output_dir / OUT_STEP2_JSON, "Step 2 JSON output")

    # Step 3: deeper cleanup and dense math QA flags.
    run_cmd([
        args.python, str(step3),
        "--input", str(output_dir / OUT_STEP2_JSON),
        "--output", str(output_dir / OUT_STEP3_JSON),
        "--report", str(output_dir / OUT_STEP3_REPORT),
    ], env=script_env)

    check_file(output_dir / OUT_STEP3_JSON, "Step 3 JSON output")

    # Step 4: production-safe gating. Step 4 is patched to use output_dir as BASE.
    run_cmd([args.python, str(patched_step4)], env=script_env)

    check_file(output_dir / OUT_STEP4_JSON, "final Step 4 JSON output")
    check_file(output_dir / OUT_STEP4_REPORT, "final Step 4 validation report")
    check_file(output_dir / OUT_STEP4_CSV, "final Step 4 vision QA CSV")

    print_final_summary(output_dir)


if __name__ == "__main__":
    main()
