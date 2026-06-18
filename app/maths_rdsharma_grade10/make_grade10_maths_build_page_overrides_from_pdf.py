#!/usr/bin/env python3
"""
Build reviewed page-level override JSON from the PDF page image.

This is the safe replacement for hardcoding page text inside Python source code.
It writes Grade10_Maths_page_overrides.json entries that can be applied by Step 3/4/5
or directly by make_grade10_maths_apply_page_overrides.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from make_grade10_maths_step_3_vision_repair import (  # noqa: E402
    call_openai_vision,
    normalize_repair_payload,
    render_page_to_data_url,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT_JSON = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REMAINING_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_math_precision_remaining_pages.csv"
DEFAULT_OVERRIDES_JSON = THIS_DIR / "Grade10_Maths_page_overrides.json"
PROMPT_VERSION = "trusted_page_override_from_pdf_v7_circle_geometry_guardrails_2026_06_16"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_pages_arg(value: str | None) -> set[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def page_map_from_json(input_json: Path) -> dict[int, dict[str, Any]]:
    if not input_json.exists():
        return {}
    data = load_json(input_json)
    pages = data.get("extraction", {}).get("page_extractions", []) or []
    return {int(p.get("page_number")): p for p in pages if p.get("page_number") is not None}


def pages_from_remaining_csv(path: Path) -> set[int]:
    pages: set[int] = set()
    if not path.exists():
        return pages
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("page_number") or row.get("pdf_page") or row.get("page")
            if value and str(value).strip().isdigit():
                pages.add(int(str(value).strip()))
    return pages


def pages_from_input_json_blockers(input_json: Path) -> set[int]:
    if not input_json.exists():
        return set()
    data = load_json(input_json)
    pages: set[int] = set()
    for page in data.get("extraction", {}).get("page_extractions", []) or []:
        if page.get("content_type") != "lesson_body" or page.get("include_in_lesson_text") is not True:
            continue
        if page.get("embedding_readiness") != "ready_for_production_embedding" or not str(page.get("production_safe_text") or "").strip():
            if page.get("page_number") is not None:
                pages.add(int(page["page_number"]))
    return pages


SURFACE_AREA_CORRUPTION_RE = re.compile(
    r"(?:\bpar\s+thy\b|\bfan\s*\?|\bfan\s*\^\s*2\b|"
    r"(?<![A-Za-z0-9])2an\s+hy(?![A-Za-z0-9])|\bwryly\b|"
    r"\bm\s*:\s*2\s*%|\b21xJ2m\b|\b72,3980\b|(?<![A-Za-z0-9])m>(?![A-Za-z0-9])|"
    r"\bPax\b|21%\s*21x|21x\s*/3|\+\s*20\?|"
    r"1\s*%\s*1700\s*%\s*3060|\bJom\s*\^\s*2\b|\bhem\s*\^\s*2\b|\bLO\s*%\s*S0|\bhris\b|"
    r"\bZom\b|\bJom\s*=\s*rem\b|19\s*-\s*2\s*%\s*2|\(\s*19\s*-\s*2\s*%\s*2\s*\)\s*om|\bxem\s*\^\s*2\b|\b36xhem\s*\^?\s*2?\b|\b6\?\s*xh\b|\bVe\s+nr\?h\b|\bnr\?h\b|36h\s*=\s*3627|3627\s*>\s*h\s*=\s*1\s*cm|\bPn\s*x2nrand\b|\b550m\b|\bbem\?|\bagate\?|\bxcent\b|25\s*%\s*9|%\s*9\s*mm|\bCo\]\s*em\b|\badmin\b|\b2225x9442\b|\b2\s*%\s*25\b|«\s*25x|Volume\?\s*theenne|<-+\s*somn\s*-+>|\bBx\s*SxS\b)",
    re.I,
)


def has_surface_area_corruption(text: str) -> bool:
    return bool(SURFACE_AREA_CORRUPTION_RE.search(text or ""))


def is_surface_area_page(page: dict[str, Any] | None, page_number: int) -> bool:
    page = page or {}
    title = str(page.get("chapter_title") or "")
    printed = str(page.get("printed_page_number") or "")
    return (
        "surface areas" in title.lower()
        or page_number in {586, 628, 633, 642, 644, 652, 663}
        or printed in {"579", "621", "626", "635", "637", "645", "656"}
        or has_surface_area_corruption(str(page.get("production_safe_text") or ""))
    )


FOUNDATIONAL_PROBABILITY_CORRUPTION_RE = re.compile(
    r"(?:\bqandrsuch\b|\ba\s*-\s*bg\s*=\s*r\b|\bIfaisanon\b|\bal\{bandc\b|\bIf@and\s+c\b|\bbla\s*>\s*a\s*=\s*\+?b\b|0O\s*<\s*r\s*<\s*b|"
    r"0\s*<\s*52\s*<\s*3|20\s*=\s*3x64\s*\+\s*2|"
    r"\bHCE\b|\bHICK\b|\bWyo\b|(?:\b[23]\?\s*x\s*[235]?|x\s*[23]\?|\b[23]\?\s*=)|"
    r"\bb\s*\+\s*gt\s*\+\s*w\s*=\s*54\b|\bual\)|24x\s*==\s*16)",
    re.I,
)


def has_foundational_probability_corruption(text: str) -> bool:
    return bool(FOUNDATIONAL_PROBABILITY_CORRUPTION_RE.search(text or ""))


def is_foundational_probability_risk_page(page: dict[str, Any] | None, page_number: int) -> bool:
    page = page or {}
    title = str(page.get("chapter_title") or "").lower()
    text = str(page.get("production_safe_text") or "")
    printed = str(page.get("printed_page_number") or "")
    return (
        has_foundational_probability_corruption(text)
        or page_number in {9, 10, 23, 30, 31, 33, 34, 736}
        or printed in {"2", "3", "16", "23", "24", "26", "27", "729"}
        or ("real numbers" in title and page_number <= 40)
        or ("probability" in title and has_foundational_probability_corruption(text))
    )



ALGEBRA_QUADRATIC_CORRUPTION_RE = re.compile(
    r"(?:\bAg\s*\^\s*2\s*99\?|\bBy\s+au\b|(?:\\-octing|(?<![A-Za-z])octing\b)|\bgots\s+t\s+eS\b|\bx\s*\$=|4x\*\s*\+\s*8x\^?2\s*-\s*12x\^?2\s*==|"
    r"\b4g\s*=\s*f\b|\bD\s*=\s*\?\s*-\s*4ac\b|\(\s*-\s*6\s*\)°|\banon\s+cgahomamal\b|"
    r"\bBoa\s+os\b|\bDix\b|X_?3\s*\(\s*g\s*-\s*2\s*\)|\bseins\s+orp\s+pee\b|u#\s*0\s*,\s*040|\by\s*=\s*--\s*==|\bxty\s*=\s*ot4\b|\b2\s+eel\b|Substituting\s+y\s*=\s*or\s+he|ax\s*\+\s*4\s*\(\s*2528\s*\)|"
    r"@\s*eo|\bre1e0\b|a\?b\?x|#4\s*\(|x\s*\+\s*224h)",
    re.I,
)


def has_algebra_quadratic_corruption(text: str) -> bool:
    return bool(ALGEBRA_QUADRATIC_CORRUPTION_RE.search(text or ""))


def is_algebra_quadratic_risk_page(page: dict[str, Any] | None, page_number: int) -> bool:
    page = page or {}
    title = str(page.get("chapter_title") or "").lower()
    text = str(page.get("production_safe_text") or "")
    printed = str(page.get("printed_page_number") or "")
    return (
        has_algebra_quadratic_corruption(text)
        or page_number in {88, 106, 119, 122, 123, 124, 184, 196}
        or printed in {"81", "99", "112", "115", "116", "117", "177", "189"}
        or (("polynomial" in title or "quadratic" in title) and has_algebra_quadratic_corruption(text))
    )


CIRCLE_GEOMETRY_CORRUPTION_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])(?:AT|ET)\?(?![A-Za-z0-9])|"
    r"\(\s*12\s*-\s*x\s*\)\s*=\s*\?|"
    r"\b2dv\b|\bX\s*=\s*OM\b|\bem\s*=\s*2om\b|"
    r"\bBint\b|\biets\b|\bearigent\b|90°\s*%\s*a2|\bJ125\b"
    r")",
    re.I,
)


def has_circle_geometry_corruption(text: str) -> bool:
    return bool(CIRCLE_GEOMETRY_CORRUPTION_RE.search(text or ""))


def is_circle_geometry_risk_page(page: dict[str, Any] | None, page_number: int) -> bool:
    page = page or {}
    title = str(page.get("chapter_title") or "").lower()
    text = str(page.get("production_safe_text") or "")
    printed = str(page.get("printed_page_number") or "")
    return (
        has_circle_geometry_corruption(text)
        or page_number in {439, 454}
        or printed in {"432", "447"}
        or (("circle" in title or "geometry" in title or "triangle" in title) and has_circle_geometry_corruption(text))
    )


GENERAL_FORMULA_CORRUPTION_RE = re.compile(r"(?:\b14g\b|%\s*12\s*#\s*%\s*2)", re.I)


def has_general_formula_corruption(text: str) -> bool:
    return bool(GENERAL_FORMULA_CORRUPTION_RE.search(text or ""))


def is_general_formula_risk_page(page: dict[str, Any] | None, page_number: int) -> bool:
    page = page or {}
    text = str(page.get("production_safe_text") or "")
    printed = str(page.get("printed_page_number") or "")
    return has_general_formula_corruption(text) or page_number in {680, 704} or printed in {"673", "697"}

def load_existing_overrides(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "description": "Reviewed page-level corrections for Grade10 Maths. These entries are trusted page transcriptions generated from the PDF page image and kept outside Python source code.",
            "usage": "Each page has page_number, printed_page_number, production_safe_text, optional math_lines, confidence, and notes.",
            "pages": [],
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("pages"), list):
        return raw
    if isinstance(raw, list):
        return {"description": "Reviewed page-level corrections for Grade10 Maths.", "pages": raw}
    if isinstance(raw, dict):
        pages = []
        for key, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                if key.isdigit() and not item.get("page_number"):
                    item["page_number"] = int(key)
                pages.append(item)
        return {"description": "Reviewed page-level corrections for Grade10 Maths.", "pages": pages}
    raise ValueError(f"Unsupported overrides JSON shape: {path}")


def merge_override(existing: dict[str, Any], entry: dict[str, Any]) -> None:
    pages = existing.setdefault("pages", [])
    page_number = int(entry["page_number"])
    printed = entry.get("printed_page_number")
    replaced = False
    for idx, item in enumerate(list(pages)):
        if not isinstance(item, dict):
            continue
        if item.get("page_number") is not None and int(item["page_number"]) == page_number:
            pages[idx] = entry
            replaced = True
            break
        if printed is not None and item.get("printed_page_number") is not None and int(item["printed_page_number"]) == int(printed):
            pages[idx] = entry
            replaced = True
            break
    if not replaced:
        pages.append(entry)
    pages.sort(key=lambda p: int(p.get("page_number") or 999999))


def page_specific_hints(page: dict[str, Any], page_number: int) -> str:
    """Return extra OCR guardrails for reviewer-confirmed difficult pages."""
    printed = page.get("printed_page_number")
    hints: list[str] = []
    if page_number == 13 or str(printed) == "6":
        hints.append("""
SPECIAL PAGE 13 / PRINTED PAGE 6 INSTRUCTIONS:
- This page contains EXAMPLE 7 in Real Numbers about the CUBE of a positive integer.
- Read all powers in that proof as cube powers where the page discusses cube: n^3, (4q)^3, (4q + 1)^3, (4q + 2)^3, (4q + 3)^3.
- Do NOT output n^2, n>, n°, (4q)°, 64q°, 16q°, 144q7, 649, 4g + 3, or similar OCR fragments in this cube proof.
- If the image shows q, keep q. Do not replace q with g.
- Expand cube expressions carefully line by line; preserve cases exactly as printed.
""".strip())
    if page_number == 233 or str(printed) == "226":
        hints.append("""
SPECIAL PAGE 233 / PRINTED PAGE 226 INSTRUCTIONS:
- This page contains arithmetic progression / sequence notation.
- Carefully transcribe subscripted terms as a_n, a_{n+1}, a_{n+2}, etc.
- Do NOT output OCR fragments like a,,, - 4,, Any,, Qni1, yay, fs Beer, or 2n? + 1.
- Preserve a_{n+1} - a_n and any formula for the nth term exactly.
""".strip())
    if is_surface_area_page(page, page_number):
        hints.append("""
SPECIAL SURFACE AREAS AND VOLUMES INSTRUCTIONS:
- This chapter contains formula-heavy text for cylinders, cones, spheres, frustums, surface area, volume and slant height.
- Use π, r, h, l, r_1, r_2 clearly. Do not turn π into x, n, P, or random letters when the printed formula means pi.
- Carefully transcribe formulae such as l^2 = r^2 + h^2, S = 2πrh + πrl, V = πr^2h + (1/3)πr^2h.
- Do NOT output OCR fragments such as fan?, fan^2, h?, m?, 2an, wryly, m:2%, 21xJ2m, 72,3980, m>, Pax, 21% 21x, 21x /3, 1%1700%3060, Jom^2, hem^2, LO%S0, hris, Zom, Jom =rem, 19-2%2, xem^2, 36xhem, 6? xh, Ve nr?h, nr?h, 3627, Pn x2nrand, 550m, bem?, agate?, xcent, %9, Co] em, admin, 2225x9442, 2%25, «25x, Volume? theenne, <-somn->, or Bx SxS, Sth., ue hot, n(n+n)l, r,?, mr +172, @ eo, re1e0, a?b?x, #4(, x+224h, earigent, 90° % a2, or J125.
- For water-flow/tank problems, carefully transcribe units and multiplication: cm^3, cm^2, m^3, and hours. Do not output percent signs as multiplication signs.
- If the image contains dimensions like 2.1 m, 4.2 m, 52.5 m, preserve the decimal point and unit exactly.
- For formula lines, prefer readable mathematical notation over noisy visual layout.
""".strip())
    if is_foundational_probability_risk_page(page, page_number):
        hints.append("""
SPECIAL FOUNDATIONAL MATH / PROBABILITY OCR INSTRUCTIONS:
- Read Euclid's division lemma carefully: there exist unique integers q and r such that a = bq + r, 0 ≤ r < b.
- Do NOT output OCR fragments like qandrsuch, a-bg=r, 0O<r<b, 0<52<3, 20 =3x64+2, Ifaisanon, al{bandc, If@and c, bla >a=+b, or hf H0S RET.
- In HCF/LCM pages, write HCF, not HCE or HICK. Write HCF × LCM = a × b, and preserve prime powers as ^2, ^3, etc.; do not output 2?, 3?, or Wyo when the image means powers.
- In probability marble problems, transcribe b + g + w = 54, not b+gt+w=54. Do not output ual), 24x==16, or broken arrow/fraction residue.
- When a number followed by ? is a normal question ending, keep it; when it is a power, use ^2 or ^3 according to the image.
""".strip())

    if is_algebra_quadratic_risk_page(page, page_number):
        hints.append("""
SPECIAL POLYNOMIAL / QUADRATIC EQUATION OCR INSTRUCTIONS:
- This page contains algebraic expressions, polynomial identities or quadratic-equation discriminant formulae.
- Carefully distinguish x, q, k, a, b, c and powers such as x^2, x^3.
- For quadratic equations, transcribe D = b^2 - 4ac exactly when the image shows the discriminant formula.
- Do NOT output OCR fragments like Ag^2 99?, By au, \-octing, gots t eS, x $=, 4x* + 8x^2 - 12x^2 ==, 4g =f, D =? - 4ac, (-6)°, anon cgahomamal, Boa os, Dix, X_3(g-2), seins orp pee, u#0,040, y=--==, xty=ot4, or 2 eel.
- For systems of linear equations in two variables, carefully transcribe fractions, substitution steps, and conditions for parallel/coincident/intersecting lines. Do not output broken fragments such as "Substituting y = or he", "ax+4(2528)", "2 eel", or "xty=ot4".

- If the image contains a normal question ending with a number, keep the question mark only when it is truly punctuation, not a power symbol.
""".strip())

    if is_circle_geometry_risk_page(page, page_number):
        hints.append("""
SPECIAL CIRCLE / GEOMETRY OCR INSTRUCTIONS:
- This page contains tangent/circle/geometry formulae. Carefully transcribe point names and squared lengths.
- If the image shows AT^2, ET^2, AE^2, BE^2, AB, etc., write the power explicitly with ^2. Do not write AT? or ET?.
- Carefully transcribe equations involving (12 - x)^2, (3 - x)^2 or other squared terms. Do not output (12-x) =?.
- Do NOT output OCR fragments like AT?, ET?, 14g, 2dv, X= OM, em=2om, Bint, or iets.
- Preserve cm, cm^2, triangle/circle notation, and tangent lengths exactly from the image.
""".strip())

    if is_general_formula_risk_page(page, page_number):
        hints.append("""
SPECIAL FORMULA/TABLE OCR INSTRUCTIONS:
- This page contains formula or table calculations. Transcribe numbers, variables, totals and equations exactly from the image.
- Do NOT output OCR fragments like 14g. If the image means 146, 140, f_1, f_2, or another expression, transcribe that expression clearly.
- Preserve table rows/columns line by line and keep equations readable.
""".strip())
    return "\n\n".join(hints)


def is_page13(page: dict[str, Any] | None, page_number: int) -> bool:
    page = page or {}
    return page_number == 13 or str(page.get("printed_page_number")) == "6"


def is_page233(page: dict[str, Any] | None, page_number: int) -> bool:
    page = page or {}
    return page_number == 233 or str(page.get("printed_page_number")) == "226"


def should_hide_previous_ocr(page: dict[str, Any] | None, page_number: int) -> bool:
    """For reviewer-confirmed bad pages, do not include old OCR in the prompt.

    Including the previous OCR made the vision model copy corrupted strings such as
    n^2, n>, (4q)°, 649°, a,,, and Qni1 back into the trusted override. For these
    pages the image must be the only source of truth.
    """
    return (
        is_page13(page, page_number)
        or is_page233(page, page_number)
        or is_surface_area_page(page, page_number)
        or is_foundational_probability_risk_page(page, page_number)
        or is_algebra_quadratic_risk_page(page, page_number)
        or is_circle_geometry_risk_page(page, page_number)
        or is_general_formula_risk_page(page, page_number)
    )


def clean_page13_cube_corruption(text: str) -> str:
    """Deterministically clean common OCR variants in page 13's cube proof.

    This is not a substitute for the page image transcription. It only fixes the
    exact reviewer-confirmed OCR failure family after the model has read the page:
    cube powers misread as squares/degrees/greater-than signs and q misread as g.
    """
    if not text:
        return text
    t = text
    # Keep q as q in the Euclid division cases.
    t = re.sub(r"\b4g\s*\+\s*([0-3])\b", r"4q + \1", t)
    t = re.sub(r"\b4g\b", "4q", t)

    # Normalize corrupt cube powers in the cube-proof page.
    t = re.sub(r"\bn\s*(?:>|°|º)\b", "n^3", t)
    t = re.sub(r"\bn\^\s*2\b", "n^3", t)
    t = re.sub(r"\bn\s*2\b", "n^3", t)

    # Parenthesized cases: (4q), (4q+1), (4q+2), (4q+3) raised to the cube.
    t = re.sub(r"\((4q(?:\s*\+\s*[0-3])?)\)\s*(?:°|º|>|\^\s*2|2|3)?", lambda m: f"({m.group(1)})^3", t)

    # Common coefficient/power OCR fragments in the standard expansions.
    replacements = {
        "64q°": "64q^3", "64qº": "64q^3", "649°": "64q^3", "649º": "64q^3",
        "16q°": "16q^3", "16qº": "16q^3", "16q9": "16q^3",
        "48q7": "48q^2", "48q²": "48q^2", "48q?": "48q^2",
        "96q7": "96q^2", "96q²": "96q^2", "96q?": "96q^2",
        "144q7": "144q^2", "144q²": "144q^2", "144q?": "144q^2",
        "36q7": "36q^2", "36q²": "36q^2", "36q?": "36q^2",
        "27q7": "27q^2", "27q²": "27q^2", "27q?": "27q^2",
        "12q7": "12q^2", "12q²": "12q^2", "12q?": "12q^2",
    }
    for bad, good in replacements.items():
        t = t.replace(bad, good)

    # Highly specific page-13 fragments seen in reviewer/user logs.
    t = re.sub(r"\b649\b", "64q^3", t)
    t = re.sub(r"\b49\b(?=\s*[+)=])", "4q", t)
    t = re.sub(r"n\s*(?:°|º|0)\s*-\s*n\s*is", "n^3 - n is", t, flags=re.I)
    t = re.sub(r"n\s*(?:°|º|0)\s*-\s*nis", "n^3 - n is", t, flags=re.I)
    return t


def clean_trusted_override_text(page: dict[str, Any] | None, page_number: int, text: str) -> str:
    if is_page13(page, page_number):
        return clean_page13_cube_corruption(text)
    return text


OVERRIDE_HARD_BLOCKERS = [
    (
        "page13_cube_proof_still_corrupt",
        re.compile(
            r"cube\s+of\s+any\s+positive\s+integer[\s\S]{0,3200}"
            r"(?:\bn\^2\b|\bn>\b|\bn°\b|\(4q\)°|64q°|16q°|144q7|48q7|36q7|4g\s*\+\s*3|\b649\b|\b49\b)",
            re.I,
        ),
    ),
    (
        "ap_subscript_formula_still_corrupt",
        re.compile(r"(?:\ba\s*,\s*,\s*,?\s*-\s*4\s*,|\bAny\s*,|\bQni1\b|\byay\s*=|\bfs\s+Beer\b|2n\?\s*\+\s*1)", re.I),
    ),
    (
        "surface_area_volume_formula_still_corrupt",
        SURFACE_AREA_CORRUPTION_RE,
    ),
    (
        "foundational_probability_formula_still_corrupt",
        FOUNDATIONAL_PROBABILITY_CORRUPTION_RE,
    ),
    (
        "algebra_quadratic_formula_still_corrupt",
        ALGEBRA_QUADRATIC_CORRUPTION_RE,
    ),
    (
        "circle_geometry_formula_still_corrupt",
        CIRCLE_GEOMETRY_CORRUPTION_RE,
    ),
    (
        "general_formula_text_still_corrupt",
        GENERAL_FORMULA_CORRUPTION_RE,
    ),
]


def validate_override_text(page_number: int, printed_page_number: Any, text: str) -> list[str]:
    """Reject overrides that still contain reviewer-confirmed corruption."""
    reasons: list[str] = []
    check_text = text or ""
    for name, regex in OVERRIDE_HARD_BLOCKERS:
        if name.startswith("page13") and not (page_number == 13 or str(printed_page_number) == "6"):
            continue
        if name.startswith("ap_") and not (page_number == 233 or str(printed_page_number) == "226"):
            continue
        if regex.search(check_text):
            reasons.append(name)
    return reasons


def build_override_prompt(page: dict[str, Any] | None, page_number: int, *, retry_feedback: str = "") -> str:
    page = page or {}
    previous = "" if should_hide_previous_ocr(page, page_number) else str(page.get("production_safe_text") or page.get("text") or page.get("ocr_text") or "")[:6000]
    specific = page_specific_hints(page, page_number)
    retry_block = f"""
IMPORTANT RETRY FEEDBACK:
Your previous transcription was rejected for these reasons:
{retry_feedback}
Re-read the image and correct those exact math/OCR errors. Do not copy the bad previous OCR.
""".strip() if retry_feedback else ""
    return f"""
You are creating a TRUSTED PAGE OVERRIDE for a scanned Class 10 mathematics textbook page.

Task:
Transcribe the visible page image into production-safe text. This output will replace the old OCR for this page.

Rules:
- Trust the image, not the previous OCR.
- Do not summarize. Do not solve questions. Do not add extra explanation.
- Preserve all headings, example labels, exercise labels, proof/solution labels, tables, formulae, identities, fractions, radicals, powers, subscripts like a_n and a_{{n+1}}, geometry point names, and statistics class intervals.
- Use ^2, ^3, etc. for powers. Carefully distinguish squares from cubes; if the image says cube, transcribe n^3 and (4q)^3, not n^2.
- Use √2, √3, √5, or sqrt(...) for square roots. Never write /2 or ./3 when the image shows a radical.
- Use clear fraction notation: (numerator)/(denominator).
- For arithmetic progression formulae, write a_n, a_{{n+1}}, and a_{{n+1}} - a_n clearly. Never write OCR fragments like a,,, - 4,, Any,, Qni1, yay, or fs Beer.
- For stacked formulae/tables, keep them line-by-line.
- Remove OCR garbage symbols only when they are not real content.
- For diagrams, include a concise line like [diagram: ...] if the diagram matters to surrounding text.
- If a tiny part is genuinely unreadable, write [unreadable] only for that part.

{specific}

{retry_block}

Context:
PDF page number: {page_number}
Printed page number: {page.get('printed_page_number')}
Chapter: {page.get('chapter_title') or ''}
Section: {page.get('section_title') or ''}

Previous OCR, for hints only; it may contain serious mistakes. If blank, ignore old OCR entirely and use only the image:
---
{previous}
---

Return JSON only:
{{
  "production_safe_text": "full corrected page transcription",
  "math_lines": ["important equations/formulas/tables from the page"],
  "confidence": 0.95,
  "notes": "trusted page override generated from PDF image"
}}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trusted page override JSON from PDF page images.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--remaining-csv", type=Path, default=DEFAULT_REMAINING_CSV, help="CSV containing page_number values to override. Used when --pages is omitted.")
    parser.add_argument("--output-overrides", type=Path, default=DEFAULT_OVERRIDES_JSON)
    parser.add_argument("--pages", default=None, help="PDF pages to override, e.g. 12,14,24,503,710 or 24-35")
    parser.add_argument("--model", default=os.environ.get("GRADE10_MATHS_MATH_PRECISION_MODEL") or os.environ.get("GRADE10_MATHS_VISION_MODEL", "gpt-4o"))
    parser.add_argument("--scale", type=float, default=float(os.environ.get("GRADE10_MATHS_VISION_SCALE", "2.5")))
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--min-confidence", type=float, default=float(os.environ.get("GRADE10_MATHS_AUTO_REVIEW_THRESHOLD", "0.90")))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-pages", type=int, default=0)
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    selected = parse_pages_arg(args.pages)
    if selected is None:
        selected = pages_from_remaining_csv(args.remaining_csv)
    if not selected:
        selected = pages_from_input_json_blockers(args.input_json)
    if not selected:
        raise RuntimeError("No pages selected. Pass --pages or provide a remaining CSV/input JSON with blockers.")
    if args.max_pages and args.max_pages > 0:
        selected = set(sorted(selected)[: args.max_pages])

    meta = page_map_from_json(args.input_json)
    overrides = load_existing_overrides(args.output_overrides)
    overrides["generated_or_updated_at_utc"] = now_utc()
    overrides["prompt_version"] = PROMPT_VERSION

    import fitz
    doc = fitz.open(args.pdf)
    try:
        for idx, page_number in enumerate(sorted(selected), start=1):
            page_meta = meta.get(page_number, {"page_number": page_number})
            data_url = render_page_to_data_url(doc, page_number, scale=args.scale)
            retry_feedback = ""
            last_error: str | None = None
            text = ""
            math_lines: list[str] = []
            confidence = 0.0
            notes = ""
            for attempt in range(1, 4):
                prompt = build_override_prompt(page_meta, page_number, retry_feedback=retry_feedback)
                payload = call_openai_vision(data_url=data_url, prompt=prompt, model=args.model, timeout=args.timeout)
                text, math_lines, confidence, notes = normalize_repair_payload(payload, min_chars=args.min_chars, min_confidence=args.min_confidence)
                text = clean_trusted_override_text(page_meta, page_number, text)
                math_lines = [clean_trusted_override_text(page_meta, page_number, str(line)) for line in (math_lines or [])]
                blockers = validate_override_text(page_number, page_meta.get("printed_page_number"), text)
                if not blockers:
                    last_error = None
                    break
                last_error = "; ".join(blockers)
                retry_feedback = last_error
                print(f"Page {page_number}: retrying trusted override after validation error: {last_error}")
            if last_error:
                raise RuntimeError(f"Trusted override for page {page_number} still failed validation after retries: {last_error}")
            entry = {
                "page_number": page_number,
                "printed_page_number": page_meta.get("printed_page_number"),
                "production_safe_text": text,
                "math_lines": math_lines,
                "confidence": confidence,
                "notes": notes or "trusted page override generated from PDF image",
                "source": "openai_vision_pdf_page_override",
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "generated_at_utc": now_utc(),
                "trusted_override": True,
            }
            merge_override(overrides, entry)
            args.output_overrides.parent.mkdir(parents=True, exist_ok=True)
            args.output_overrides.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{idx}/{len(selected)}] wrote trusted override page={page_number}, printed={entry.get('printed_page_number')}, chars={len(text)}, confidence={confidence}")
    finally:
        doc.close()

    print(f"Wrote overrides: {args.output_overrides}")


if __name__ == "__main__":
    main()
