#!/usr/bin/env python3
"""
Apply deterministic reviewer-confirmed fixes to final production text.

This script does not replace full copyrighted pages. It only repairs narrow OCR/math
fragments that have been repeatedly confirmed by PDF-page review and should never
survive in production fields. It also normalizes OCR question-mark powers such as OP?
into OP^2 where the final hard validator has confirmed them as production blockers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from make_grade10_maths_step_2_production_gate import rebuild_chapters_and_sections  # noqa: E402
from make_grade10_maths_step_6_residual_production_audit import sync_legacy_page_fields, sync_legacy_sections  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10" / "Grade10_Maths_production_ready.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def page_number(page: dict[str, Any]) -> int:
    try:
        return int(page.get("page_number") or 0)
    except Exception:
        return 0


GEOMETRY_POWER_SYMBOLS = (
    "OP", "PN", "PT", "AB", "BC", "AC", "PQ", "QR", "PR",
    "OA", "OB", "OC", "OT", "PA", "PB", "PC", "TP", "TQ",
    "QS", "RT", "RU", "PS", "PU", "OR", "TR", "BD", "AD",
    "CD", "DE", "AE", "BE", "CE", "AP", "BP", "CP", "CQ", "BR",
)


def apply_common_hard_fixes(text: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    new = text
    # A repeated vision/transcription artifact: (4q)^3^3 or (x)^2^2.
    new2 = re.sub(r"\)\s*\^\s*([23])\s*\^\s*\1", lambda m: f")^{m.group(1)}", new)
    if new2 != new:
        fixes.append("collapse_duplicate_power_suffix")
        new = new2

    # Geometry/Pythagoras OCR commonly turns the superscript 2 into a question mark:
    # OP? = OR? + PR? should be OP^2 = OR^2 + PR^2. The residual validator
    # hard-blocks these exact tokens, so this repair keeps production fields clean.
    sym_alt = "|".join(map(re.escape, GEOMETRY_POWER_SYMBOLS))
    geom_re = re.compile(rf"(?<![A-Za-z0-9])({sym_alt})\s*\?(?![A-Za-z0-9])", re.I)
    new2 = geom_re.sub(lambda m: f"{m.group(1)}^2", new)
    if new2 != new:
        fixes.append("geometry_segment_question_mark_power_to_square")
        new = new2

    # Same corruption sometimes appears after a closing parenthesis, e.g. (x + 3) =?.
    new2 = re.sub(r"(\([^\n]{1,80}\))\s*=\s*\?", r"\1^2 =", new)
    if new2 != new:
        fixes.append("paren_expression_question_mark_power_to_square")
        new = new2

    return new, fixes


def apply_page13_fixes(text: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    new, common = apply_common_hard_fixes(text)
    fixes.extend(common)

    # Page 13 contains Example 7 (cube proof) followed by Examples 8/9, which are
    # square identities. Vision repairs repeatedly mixed these and produced invalid
    # forms like n^3 - n = (2q)^2 and n^3 - 1 = ...16q^2. Keep the cube proof
    # intact, but force Examples 8/9 back to n^2.
    replacements = [
        # Example 9 is n^2 - 1. Reject/fix the recurring bad override where the
        # base was transcribed with cube powers.
        (r"n\s*\^\s*2\s*[-–—]\s*1\s*=\s*\(\s*4q\s*\+\s*1\s*\)\s*\^\s*3\s*[-–—]\s*1", "n^2 - 1 = (4q + 1)^2 - 1"),
        (r"n\s*\^\s*2\s*[-–—]\s*1\s*=\s*\(\s*4q\s*\+\s*3\s*\)\s*\^\s*3\s*[-–—]\s*1", "n^2 - 1 = (4q + 3)^2 - 1"),
        (r"n\s*\^\s*2\s*[-–—]\s*1\s*=\s*\(\s*4q\s*\+\s*1\s*\)\s*\^\s*3", "n^2 - 1 = (4q + 1)^2"),
        (r"n\s*\^\s*2\s*[-–—]\s*1\s*=\s*\(\s*4q\s*\+\s*3\s*\)\s*\^\s*3", "n^2 - 1 = (4q + 3)^2"),
        (r"n\s*(?:\^\s*3|³)\s*[-–—]\s*n", "n^2 - n"),
        (r"n\s*(?:\^\s*3|³)\s*[-–—]\s*1", "n^2 - 1"),
        (r"n\s*(?:\^\s*3|³)\s*minus\s*n", "n^2 - n"),
        (r"n\s*(?:\^\s*3|³)\s*minus\s*1", "n^2 - 1"),
        (r"\(2q\)\^2\s*[-–—]\s*2q", "(2q)^2 - 2q"),
        (r"\(2q\s*\+\s*1\)\^2\s*[-–—]\s*\(2q\s*\+\s*1\)", "(2q + 1)^2 - (2q + 1)"),
        (r"\(4q\)\^3\^3", "(4q)^3"),
        (r"\(4q\s*\+\s*1\)\^3\^3", "(4q + 1)^3"),
        (r"\(4q\s*\+\s*2\)\^3\^3", "(4q + 2)^3"),
        (r"\(4q\s*\+\s*3\)\^3\^3", "(4q + 3)^3"),
        (r"64q°", "64q^3"),
        (r"64q\?", "64q^3"),
        (r"649°", "64q^3"),
        (r"169°", "16q^3"),
        (r"144q7", "144q^2"),
        (r"48q7", "48q^2"),
        (r"16q°", "16q^3"),
        (r"2q¢\s*\+\s*1", "2q + 1"),
        (r"4g\s*\+\s*3", "4q + 3"),
        (r"n>\s*=", "n^3 ="),
    ]
    for pat, repl in replacements:
        new2 = re.sub(pat, repl, new, flags=re.I)
        if new2 != new:
            fixes.append(f"page13:{pat}->{repl}")
            new = new2
    return new, fixes

def apply_page15_fixes(text: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    new, common = apply_common_hard_fixes(text)
    fixes.extend(common)
    replacements = [
        (r"\b3g\s*\+\s*1\b", "3q + 1"),
        (r"\b39\s*\+\s*1\s*\+\s*4\b", "3q + 1 + 4"),
        (r"\b39\s*\+\s*2\s*\+\s*2\b", "3q + 2 + 2"),
        (r"3\(9\s*\+\s*1\)", "3(q + 1)"),
        (r"Thus,\s*1\s*\+\s*4\b", "Thus, n + 4"),
        (r"\bWe know that 77 is\b", "We know that n is"),
        (r"\bnis\b", "n is"),
        (r"\bisnot\b", "is not"),
        (r"\bnand\b", "n and"),
        (r"\barenot\b", "are not"),
        (r"but and n \+ 4", "but n and n + 4"),
    ]
    for pat, repl in replacements:
        new2 = re.sub(pat, repl, new, flags=re.I)
        if new2 != new:
            fixes.append(f"page15:{pat}->{repl}")
            new = new2
    return new, fixes


def apply_known_fixes(data: dict[str, Any]) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    applied: list[dict[str, Any]] = []
    for page in extraction.get("page_extractions", []) or []:
        pno = page_number(page)
        text = str(page.get("production_safe_text") or "")
        if not text:
            continue
        if pno == 13:
            fixed, fixes = apply_page13_fixes(text)
        elif pno == 15:
            fixed, fixes = apply_page15_fixes(text)
        else:
            fixed, fixes = apply_common_hard_fixes(text)
        if fixes and fixed != text:
            page["production_safe_text"] = fixed.strip()
            page["text"] = fixed.strip()
            page["text_plain"] = fixed.strip()
            flags = set(page.get("quality_flags") or [])
            flags.add("known_reviewer_math_fix_applied")
            page["quality_flags"] = sorted(flags)
            page["known_reviewer_math_fixes"] = sorted(set((page.get("known_reviewer_math_fixes") or []) + fixes))
            applied.append({
                "page_number": pno,
                "printed_page_number": page.get("printed_page_number"),
                "fixes": fixes,
            })
    rebuild_chapters_and_sections(data)
    sync_legacy_page_fields(extraction.get("page_extractions", []) or [])
    sync_legacy_sections(data)
    policy = extraction.setdefault("production_embedding_policy", {})
    policy["known_reviewer_math_fixes_applied"] = True
    policy["known_reviewer_math_fixes_applied_at_utc"] = now_utc()
    qs = extraction.setdefault("quality_summary", {})
    qs["known_reviewer_math_fixes"] = {
        "generated_at_utc": now_utc(),
        "applied_page_count": len(applied),
        "applied_pages": applied,
    }
    return {"applied": applied}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic known reviewer math fixes to Grade10_Maths production JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    data = load_json(args.input)
    result = apply_known_fixes(data)
    write_json(args.output, data)
    print(f"Applied known reviewer math fixes: {len(result['applied'])} pages")
    for item in result["applied"]:
        print(f"- page={item['page_number']}, printed={item.get('printed_page_number')}, fixes={len(item['fixes'])}")
    print(f"Wrote: {args.output}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
