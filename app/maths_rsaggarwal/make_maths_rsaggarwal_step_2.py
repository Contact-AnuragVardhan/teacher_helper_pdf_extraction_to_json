#!/usr/bin/env python3
"""
Cleanup pass for Maths_RSAgarwal_math_aware_extraction.json.

This script keeps the chapter/page structure from the math-aware OCR JSON and applies
conservative rule-based cleanup before embeddings:
- O -> 0 when isolated in math/number contexts
- I/l -> 1 when used as a numeric token
- ~ -> - and a--b -> a-b for OCR-minus errors
- # -> ≠
- = 4NN -> = +NN when OCR read plus as 4 in equation-result context
- common fake rupee symbols before amounts -> ₹

It does not claim perfect symbolic math/LaTeX extraction. Stacked fractions, powers,
answer pages, and multi-column math layouts can still need Mathpix or a vision LLM.
"""

from __future__ import annotations

import argparse, copy, json, os, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(os.environ.get(
    "MATHS_RSAGGARWAL_ROOT",
    Path(__file__).resolve().parents[2],
))
OUTPUT_DIR = Path(os.environ.get(
    "MATHS_RSAGGARWAL_OUTPUT_DIR",
    PROJECT_ROOT / "output" / "maths_rsagarwal",
))

DEFAULT_INPUT = OUTPUT_DIR / "Maths_RSAgarwal_math_aware_extraction.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "Maths_RSAgarwal_math_aware_extraction_v2_cleaned.json"
DEFAULT_REPORT = OUTPUT_DIR / "Maths_RSAgarwal_math_aware_v2_validation_report.txt"

MATH_SYMBOL_RE = re.compile(r"[=+\-−~#≠×x*/÷()\[\]{}]|\d")
CURRENCY_CHAPTERS = {"Profit and Loss", "Simple Interest"}
CURRENCY_KEYWORDS_RE = re.compile(
    r"\b(CP|SP|cost price|selling price|principal|amount|sum|loss|gain|SI|interest|article|bought|sold|borrowed|borrows|paid|will|rupees?|paise|money|price|cost)\b",
    re.IGNORECASE,
)
FAKE_RUPEE_SYMBOLS = "¥€£®@&"


def is_math_like(line: str) -> bool:
    return bool(line and MATH_SYMBOL_RE.search(line))


def is_currency_context(line: str, chapter_title: str | None) -> bool:
    return chapter_title in CURRENCY_CHAPTERS or bool(CURRENCY_KEYWORDS_RE.search(line or ""))


def sub_count(pattern: str, repl: str, s: str, flags: int = 0) -> tuple[str, int]:
    return re.subn(pattern, repl, s, flags=flags)


def cleanup_currency_line(s: str, chapter_title: str | None, stats: Counter) -> str:
    """Careful rupee cleanup without changing percentage results like gain% = 12.5%."""
    if not is_currency_context(s, chapter_title):
        return s

    before_all = s

    # Direct fake rupee glyph before amount: ¥ 130, €4250, etc.
    s, n = re.subn(rf"(?<![A-Za-z0-9])([{re.escape(FAKE_RUPEE_SYMBOLS)}])\s*(?=\d)", "₹ ", s)
    stats["fake_currency_symbol_to_rupee"] += n

    # Direct fake rupee glyph before parenthesized amount: ¥(20-15) -> ₹ (20-15).
    s, n = re.subn(rf"(?<![A-Za-z0-9])([{re.escape(FAKE_RUPEE_SYMBOLS)}])\s*(?=\()", "₹ ", s)
    stats["fake_currency_symbol_before_paren_to_rupee"] += n

    # Standalone lowercase f is sometimes a broken rupee sign in the answers/exercises.
    s, n = re.subn(r"(?<![A-Za-z0-9])f\s+(?=\d{2,7}(?:\b|[.,]))", "₹ ", s)
    stats["lowercase_f_before_amount_to_rupee"] += n

    # Patterns like '= % 400', '=F 1215', '=? 64' mean '= ₹ 400/1215/64'.
    s, n = re.subn(r"=\s*[%?F]\s*(?=\d{1,7}(?:\b|[.,]))", "= ₹ ", s)
    stats["equals_fake_currency_to_rupee"] += n

    # Patterns like 'Principal = = 4500' mean 'Principal = ₹ 4500'.
    # Keep this tied to a preceding money keyword to avoid changing equations.
    money_lhs = r"(CP|SP|Principal|principal|amount|Amount|sum|Sum|SI|Loss|loss|Gain|gain|P)"
    s, n = re.subn(rf"\b{money_lhs}\s*=\s*=\s*(?=\d{{1,7}}(?:\b|[.,]))", lambda m: f"{m.group(1)} = ₹ ", s)
    stats["double_equals_money_to_rupee"] += n

    # Double equals anywhere in a currency line: SP of article = = 336 -> SP of article = ₹ 336.
    s, n = re.subn(r"=\s*=\s*(?=\d{1,7}(?:\b|[.,]))", "= ₹ ", s)
    stats["double_equals_in_currency_line_to_rupee"] += n

    # Patterns like 'SI = 7 720' or 'SP = 2 15', where 7/2 is a broken rupee sign.
    s, n = re.subn(rf"\b{money_lhs}\s*=\s*[72]\s+(?=\d{{1,7}}(?:\b|[.,]))", lambda m: f"{m.group(1)} = ₹ ", s)
    stats["digit_prefix_money_to_rupee"] += n

    # Natural-language money phrases: bought for = 400, borrowed = 6000, paid = 7070, be = 1.
    s, n = re.subn(r"\b(for|to|by|at|of|be|is|was|will|borrowed|borrows|paid|amount to)\s*=\s*(?=\d{1,7}(?:\b|[.,]))", lambda m: f"{m.group(1)} ₹ ", s, flags=re.IGNORECASE)
    stats["phrase_equals_amount_to_rupee"] += n

    # '% 400' or '%5' alone is often broken rupee in this OCR, but only in money context.
    # Do not touch real percentages like '12.5%' because the % is preceded by a digit.
    s, n = re.subn(r"(?<!\d)%\s*(?=\d{1,7}(?:\b|[.,]))", "₹ ", s)
    stats["percent_before_amount_to_rupee"] += n

    # Options in SI/Profit chapters: (a) = 724 -> (a) ₹ 724.
    s, n = re.subn(r"(\([a-d]\)\s*)=\s*(?=\d{1,7}(?:\b|[.,]))", r"\1₹ ", s, flags=re.IGNORECASE)
    stats["mcq_equals_amount_to_rupee"] += n

    # Common attached fake rupee after words: If® 640 / be® 210 -> If ₹ 640 / be ₹ 210.
    s, n = re.subn(rf"\b(If|be|is|was|for|to|by|at|of|will|paid|borrowed|borrows)([{re.escape(FAKE_RUPEE_SYMBOLS)}])\s*(?=\d)", r"\1 ₹ ", s, flags=re.IGNORECASE)
    stats["word_attached_fake_currency_to_rupee"] += n

    if s != before_all:
        stats["currency_lines_changed"] += 1
    return s


def normalize_math_ocr_line(line: str, chapter_title: str | None, stats: Counter) -> str:
    s = line

    if "−" in s:
        stats["unicode_minus_to_hyphen"] += s.count("−")
        s = s.replace("−", "-")

    if "#" in s:
        stats["hash_to_not_equal"] += s.count("#")
        s = s.replace("#", "≠")

    if "~" in s:
        stats["tilde_to_minus"] += s.count("~")
        s = s.replace("~", "-")

    # a-~b becomes a--b after tilde conversion; collapse only variable-variable form.
    s, n = re.subn(r"\b([A-Za-z])--([A-Za-z])\b", r"\1-\2", s)
    stats["double_minus_variable_collapse"] += n

    # Isolated O as zero in math text.
    s, n = re.subn(r"(?<![A-Za-z])O(?![A-Za-z])", "0", s)
    stats["isolated_O_to_zero"] += n

    # l/I as digit 1 when clearly in numeric contexts.
    before = s
    s = re.sub(r"(?<![A-Za-z])[lI](?=\d)", "1", s)       # l005 -> 1005
    s = re.sub(r"(?<=\d)[lI](?![A-Za-z])", "1", s)       # 2l -> 21
    s = re.sub(r"(?<=[(\[{+\-=/xX×÷*\s])([lI])(?=[)\]}+\-=/xX×÷*\s,.;:])", "1", s)  # (-I) -> (-1)
    if s != before:
        stats["I_or_l_to_one_in_math_context"] += 1

    # + read as 4 after equals, e.g. = 423 -> = +23. Restrict to exactly 3 digits.
    if is_math_like(s) and "=" in s:
        before = s
        s = re.sub(r"=\s*4(\d{2})(?=\s*[.,;)]?\s*$)", r"= +\1", s)
        s = re.sub(r"=\s*4(\d{2})(?=[.,;])", r"= +\1", s)
        if s != before:
            stats["four_to_plus_after_equals"] += 1

    s = cleanup_currency_line(s, chapter_title, stats)
    return s


def cleanup_text(text: str, chapter_title: str | None, stats: Counter) -> str:
    return "\n".join(normalize_math_ocr_line(line, chapter_title, stats) for line in (text or "").splitlines())


def extract_math_lines(text: str) -> List[str]:
    out: List[str] = []
    for line in (text or "").splitlines():
        t = line.strip()
        if not t:
            continue
        if re.search(r"\d", t) and re.search(r"[=+\-≠×x*/÷^()\[\]{}]", t):
            out.append(t)
    return out


def cleanup_blocks(blocks: List[Dict[str, Any]], chapter_title: str | None, stats: Counter) -> List[Dict[str, Any]]:
    cleaned = []
    for block in blocks or []:
        b = dict(block)
        for field in ["text", "problem", "solution"]:
            if isinstance(b.get(field), str):
                b[field] = cleanup_text(b[field], chapter_title, stats)
        if isinstance(b.get("solution_latex"), list):
            b["solution_latex"] = [cleanup_text(str(x), chapter_title, stats) for x in b["solution_latex"]]
        cleaned.append(b)
    return cleaned


def rebuild_lesson_text_from_pages(data: Dict[str, Any]) -> None:
    pages_by_number = {p.get("page_number"): p for p in data["extraction"].get("page_extractions", [])}
    for chapter in data["extraction"].get("chapters", []):
        for lesson in chapter.get("lessons", []):
            page_numbers = lesson.get("page_numbers") or []
            lesson_pages = [pages_by_number[n] for n in page_numbers if n in pages_by_number]
            lesson_pages = [p for p in lesson_pages if p.get("include_in_lesson_text", True)]
            lesson_pages.sort(key=lambda p: int(p.get("page_number") or 0))
            lesson_text = "\n\n".join((p.get("text") or "").strip() for p in lesson_pages if (p.get("text") or "").strip())
            lesson["lesson_text"] = lesson_text
            lesson["text_plain"] = lesson_text
            lesson["math_lines"] = extract_math_lines(lesson_text)
            lesson.setdefault("quality_flags", [])
            if "rule_based_math_cleanup_applied" not in lesson["quality_flags"]:
                lesson["quality_flags"].append("rule_based_math_cleanup_applied")


def validate(data: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, int]]:
    errors: List[str] = []
    warnings: List[str] = []
    metrics = Counter()
    for p in data["extraction"].get("page_extractions", []):
        text = p.get("text") or ""
        pn = p.get("page_number")
        chap = p.get("chapter_title")
        if "#" in text:
            metrics["pages_with_hash"] += 1
            warnings.append(f"PDF page {pn}: still contains # after cleanup")
        if "~" in text:
            metrics["pages_with_tilde"] += 1
            warnings.append(f"PDF page {pn}: still contains ~ after cleanup")
        if re.search(r"(?<![A-Za-z])O(?![A-Za-z])", text):
            metrics["pages_with_isolated_O"] += 1
        if re.search(r"\(-[lI]\)|(?<![A-Za-z])[lI]\d|\d[lI](?![A-Za-z])", text):
            metrics["pages_with_l_or_I_numeric_context"] += 1
            warnings.append(f"PDF page {pn}: may still contain l/I used as 1 in numeric context")
        if re.search(r"=\s*4\d{2}\b", text):
            metrics["pages_with_possible_plus_as_4"] += 1
            warnings.append(f"PDF page {pn}: may still contain + read as 4 after equals")
        if chap in CURRENCY_CHAPTERS and re.search(r"[¥€£®]\s*\d", text):
            metrics["pages_with_currency_glyphs"] += 1
            warnings.append(f"PDF page {pn}: may still contain fake currency glyph before amount")
        # Catch over-aggressive currency cleanup introduced by bad rules.
        if re.search(r"(?:gain|loss|rate)%\s*₹", text, re.IGNORECASE):
            metrics["pages_with_rupee_after_percent_label"] += 1
            warnings.append(f"PDF page {pn}: possible bad rupee replacement after percent label")
        if re.search(r"years\s*₹\s*\d", text, re.IGNORECASE):
            metrics["pages_with_rupee_in_year_expression"] += 1
            warnings.append(f"PDF page {pn}: possible bad rupee replacement in year expression")
    for chapter in data["extraction"].get("chapters", []):
        for lesson in chapter.get("lessons", []):
            if lesson.get("page_count") != len(lesson.get("page_numbers") or []):
                errors.append(f"{lesson.get('section_title')}: page_count mismatch")
            if lesson.get("include_in_embeddings", True) and not (lesson.get("lesson_text") or "").strip():
                warnings.append(f"{lesson.get('section_title')}: empty lesson_text")
    return errors, warnings, dict(metrics)


def make_report(data: Dict[str, Any], errors: List[str], warnings: List[str], metrics: Dict[str, int], stats: Counter) -> str:
    lines = []
    lines.append("Maths RSAggarwal math-aware OCR cleanup v2 validation report")
    lines.append(f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("Input profile")
    lines.append(f"- Book title: {data['extraction'].get('book_title')}")
    lines.append(f"- Total PDF pages: {data['extraction'].get('total_pdf_pages')}")
    lines.append(f"- Chapters/sections: {len(data['extraction'].get('chapters', []))}")
    lines.append(f"- Page extractions: {len(data['extraction'].get('page_extractions', []))}")
    lines.append("")
    lines.append("Cleanup replacements applied")
    for k, v in sorted(stats.items()):
        if v:
            lines.append(f"- {k}: {v}")
    if not any(stats.values()):
        lines.append("- None")
    lines.append("")
    lines.append("Post-cleanup quality metrics")
    if metrics:
        for k, v in sorted(metrics.items()):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No remaining tracked OCR artifact patterns found")
    lines.append("")
    lines.append("Errors")
    lines.extend([f"- {e}" for e in errors] or ["- None"])
    lines.append("")
    lines.append("Warnings")
    if warnings:
        lines.extend(f"- {w}" for w in warnings[:250])
        if len(warnings) > 250:
            lines.append(f"- ... {len(warnings)-250} more")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Known limitation")
    lines.append("- This v2 file applies rule-based cleanup after rendered-page OCR. It improves common corrupted characters before embeddings, but it is still not a perfect LaTeX/symbolic math extraction. Stacked fractions, powers, aligned tables, and answer pages may still need a Mathpix/vision-LLM pass for exact symbolic fidelity.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()
    data = copy.deepcopy(json.loads(args.input.read_text(encoding="utf-8")))
    total_stats = Counter()

    for page in data["extraction"].get("page_extractions", []):
        chapter_title = page.get("chapter_title")
        page_stats = Counter()
        for field in ["text", "text_plain", "ocr_text", "selectable_text"]:
            if isinstance(page.get(field), str):
                page[field] = cleanup_text(page[field], chapter_title, page_stats)
        page["math_lines"] = extract_math_lines(page.get("text") or "")
        page["extracted_blocks"] = cleanup_blocks(page.get("extracted_blocks") or [], chapter_title, page_stats)
        if page_stats:
            page.setdefault("quality_flags", [])
            if "rule_based_math_cleanup_applied" not in page["quality_flags"]:
                page["quality_flags"].append("rule_based_math_cleanup_applied")
            page["cleanup_stats"] = {k: v for k, v in page_stats.items() if v}
        total_stats.update(page_stats)

    rebuild_lesson_text_from_pages(data)

    for section in data["extraction"].get("section_index", []):
        section.setdefault("quality_flags", [])
        if "rule_based_math_cleanup_applied" not in section["quality_flags"]:
            section["quality_flags"].append("rule_based_math_cleanup_applied")

    data["extraction"]["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    data["extraction"]["math_ocr_profile"] = "rendered_page_ocr_plus_rule_based_math_cleanup_v2"
    data["extraction"].setdefault("notes", []).append(
        "v2 cleanup applied: O/0, I/l/1, tilde/minus, hash/not-equal, plus-as-4, and rupee-symbol OCR cleanup rules."
    )
    data["extraction"].setdefault("quality_summary", {})
    errors, warnings, metrics = validate(data)
    data["extraction"]["quality_summary"].update({
        "cleanup_version": "v2_rule_based_math_cleanup",
        "cleanup_replacement_stats": {k: v for k, v in total_stats.items() if v},
        "post_cleanup_validation_errors": len(errors),
        "post_cleanup_validation_warnings": len(warnings),
        "post_cleanup_metrics": metrics,
    })

    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(make_report(data, errors, warnings, metrics, total_stats), encoding="utf-8")
    print(f"Wrote JSON: {args.output}")
    print(f"Wrote report: {args.report}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print("Stats:", {k: v for k, v in total_stats.items() if v})

if __name__ == "__main__":
    main()
