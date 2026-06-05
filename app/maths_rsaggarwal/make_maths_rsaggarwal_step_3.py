#!/usr/bin/env python3
"""
cleanup_maths_rsaggarwal_v3.py

Post-OCR cleanup/QA pass for the R. S. Aggarwal Class 7 Maths extraction JSON.

This script is intentionally conservative: it fixes high-confidence OCR/math artifacts
and flags dense math / answer-key pages for vision QA instead of pretending that
plain OCR has perfectly reconstructed stacked fractions, powers, and answer pages.

Input default:
  /mnt/data/Maths_RSAgarwal_math_aware_extraction_v2_cleaned.json

Outputs default:
  /mnt/data/Maths_RSAgarwal_math_aware_extraction_v3_cleaned.json
  /mnt/data/Maths_RSAgarwal_math_aware_v3_validation_report.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import Counter, defaultdict
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

DEFAULT_INPUT = OUTPUT_DIR / "Maths_RSAgarwal_math_aware_extraction_v2_cleaned.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "Maths_RSAgarwal_math_aware_extraction_v3_cleaned.json"
DEFAULT_REPORT = OUTPUT_DIR / "Maths_RSAgarwal_math_aware_v3_validation_report.txt"

TEXT_FIELDS_TO_CLEAN = {
    "text",
    "text_plain",
    "lesson_text",
    "problem",
    "solution",
    "solution_text",
    "solution_latex",
    "title",
}

# Do not mutate the selectable PDF layer; it is useful as a raw-audit source.
RAW_SOURCE_FIELDS_TO_SKIP = {"selectable_text", "raw_text", "raw_ocr_text"}

MONEY_CONTEXT_WORDS = {
    "rupee", "rupees", "paise", "cost", "costs", "rate", "price", "amount",
    "money", "gain", "gains", "loss", "loses", "profit", "interest", "simple interest",
    "principal", "annum", "per annum", "loan", "borrowed", "lent", "sells", "sell",
    "selling", "sold", "bought", "purchased", "shopkeeper", "expenditure", "savings",
    "income", "salary", "fencing", "constructing", "cultivating", "worth", "rent",
}

MONEY_CHAPTERS = {
    "Decimals",
    "Ratio and Proportion",
    "Unitary Method",
    "Percentage",
    "Profit and Loss",
    "Simple Interest",
    "Mensuration",
    "Collection and Organisation of Data (Mean, Median and Mode)",
}

GEOMETRY_CHAPTERS = {
    "Lines and Angles",
    "Properties of Parallel Lines",
    "Properties of Triangles",
    "Congruence",
    "Constructions",
    "Reflection and Rotational Symmetry",
    "Mensuration",
    "Activities",
}

DENSE_MATH_CHAPTERS = {
    "Fractions",
    "Rational Numbers",
    "Exponents",
    "Algebraic Expressions",
    "Linear Equations in One Variable",
    "Mensuration",
    "Answers",
}

SUSPICIOUS_PATTERNS = {
    "seam_integer_list": re.compile(r"\bseam\s*-?\s*4", re.IGNORECASE),
    "ete_typo": re.compile(r"\bete\.\b", re.IGNORECASE),
    "formula_b_plus_e": re.compile(r"a\s*\+\s*\(\s*b\s*\+\s*e\s*\)", re.IGNORECASE),
    "subtraction_ne_artifact": re.compile(r"\(-4\)\s*-\s*2\s+42\s*-\s*\(-4\)"),
    "multiply_25_19_wrong": re.compile(r"\(\s*25\s*x\s*19\s*\)\s*=\s*\+\s*75"),
    "hash_not_equal": re.compile(r"#"),
    "tilde_minus": re.compile(r"~"),
}

FAKE_CURRENCY_SYMBOLS = "¥€£®"


def add_flag(page: Dict[str, Any], flag: str) -> None:
    flags = page.setdefault("quality_flags", [])
    if flag not in flags:
        flags.append(flag)


def has_money_context(line: str, chapter_title: str | None = None) -> bool:
    lower = line.lower()
    if any(w in lower for w in MONEY_CONTEXT_WORDS):
        return True
    # Existing rupee symbol strongly indicates the nearby fake symbol is also money.
    if "₹" in line and re.search(r"[¥€£®%]\s*\d", line):
        return True
    # In finance-oriented chapters, fake symbols followed by numbers are more likely rupee amounts.
    if chapter_title in MONEY_CHAPTERS and re.search(r"[¥€£®]\s*\d", line):
        return True
    return False


def is_square_root_context(line: str, chapter_title: str | None = None) -> bool:
    lower = line.lower()
    if chapter_title == "Exponents":
        return True
    if "square root" in lower or "sqrt" in lower:
        return True
    if re.search(r"\bfind\s*\(i\)\s*[¥√]\s*\d", lower):
        return True
    if "v¥" in line.lower() or "v3" in line.lower() or "hero" in lower:
        return True
    return False


def count_change(before: str, after: str, stats: Counter, key: str) -> None:
    if before != after:
        stats[key] += 1


def normalize_numeric_ocr(text: str, stats: Counter) -> str:
    """Fix O/I/l confusions only in numeric/math contexts."""
    before = text

    # l or I read instead of 1 immediately before a digit or after math punctuation.
    text = re.sub(r"(?<=[(\[\{\s+\-*/=])l(?=\d)", "1", text)
    text = re.sub(r"(?<=[(\[\{\s+\-*/=])I(?=\d)", "1", text)
    text = re.sub(r"(?<=\d)l(?=\d|\b)", "1", text)
    text = re.sub(r"(?<=\d)I(?=\d|\b)", "1", text)

    # O read instead of 0 in math expressions, but avoid words like "Ois".
    text = re.sub(r"(?<=[(\[\{\s+\-*/=])O(?=[)\]\}\s+\-*/=.,;:])", "0", text)
    text = re.sub(r"(?<=\d)O(?=\d|\b)", "0", text)

    count_change(before, text, stats, "numeric_ocr_context_fixes")
    return text


def cleanup_line(line: str, chapter_title: str | None, stats: Counter) -> str:
    original = line
    s = line

    # Common word OCR typos.
    s2 = re.sub(r"\bete\.", "etc.", s, flags=re.IGNORECASE)
    count_change(s, s2, stats, "ete_to_etc")
    s = s2

    s2 = re.sub(r"\bSirmplify\b", "Simplify", s, flags=re.IGNORECASE)
    s2 = re.sub(r"\bEaluate\b", "Evaluate", s2, flags=re.IGNORECASE)
    count_change(s, s2, stats, "common_word_typos")
    s = s2

    # High-confidence exact repair for integer list on printed page 1.
    s2 = re.sub(
        r"\bseam\s*-?\s*4\s*,\s*-3\s*,\s*-2\s*,\s*-1\s*,\s*0\s*,\s*1\s*,\s*2\s*,\s*3\s*,\s*4\s*,\s*\.\.\.\s*,\s*etc\.",
        "Thus, ..., -4, -3, -2, -1, 0, 1, 2, 3, 4, ..., etc.",
        s,
        flags=re.IGNORECASE,
    )
    count_change(s, s2, stats, "seam_integer_list_fixed")
    s = s2

    # Formula artifact: c read as e.
    s2 = re.sub(r"a\s*\+\s*\(\s*b\s*\+\s*e\s*\)", "a+(b+c)", s, flags=re.IGNORECASE)
    count_change(s, s2, stats, "b_plus_e_to_b_plus_c")
    s = s2

    # Not-equal artifacts.
    s2 = s.replace("#", "≠")
    s2 = re.sub(r"\(-4\)\s*-\s*2\s+42\s*-\s*\(-4\)", "(-4)-2 ≠ 2-(-4)", s2)
    s2 = re.sub(r"\(\s*3\s*-\s*5\s*\)\s*4\s*\(\s*5\s*-\s*3\s*\)", "(3-5) ≠ (5-3)", s2)
    count_change(s, s2, stats, "not_equal_artifacts_fixed")
    s = s2

    # Minus symbol artifact in formulas like a~b.
    s2 = re.sub(r"(?<=[A-Za-z0-9)\]])\s*~\s*(?=[A-Za-z0-9(\[])", " - ", s)
    s2 = s2.replace("a - b", "a-b") if "a - b" in s2 and "a+" in s2 else s2
    count_change(s, s2, stats, "tilde_to_minus")
    s = s2

    # Specific multiplication OCR error from page 13 / printed 6.
    s2 = re.sub(r"\(\s*25\s*x\s*19\s*\)\s*=\s*\+\s*75", "(25 x 19) = 475", s)
    count_change(s, s2, stats, "25x19_equals_475_fixed")
    s = s2

    # Square-root artifacts. Do this before money cleanup to avoid turning sqrt signs into rupees.
    s2 = s.replace("V¥", "√")
    if is_square_root_context(s2, chapter_title):
        # ¥196, ¥64, ¥3 are usually √196, √64, √3 in exponent/mensuration formula contexts.
        s2 = re.sub(r"¥(?=\s*\d)", "√", s2)
    count_change(s, s2, stats, "sqrt_symbol_repairs")
    s = s2

    # Geometry angle artifacts: £3, £4, £ABC should be ∠3, ∠4, ∠ABC.
    if chapter_title in GEOMETRY_CHAPTERS or re.search(r"\b(angle|triangle|parallel|line|congruence)\b", s, re.IGNORECASE):
        s2 = re.sub(r"£(?=\s*\d|\s*[A-Z]{2,3}\b)", "∠", s)
        count_change(s, s2, stats, "angle_symbol_repairs")
        s = s2

    # Money/currency artifacts. Conservative for normal pages, but replace all
    # fake currency glyphs on a line once money context is established.
    if has_money_context(s, chapter_title):
        s2 = re.sub(r"[¥€£®]", "₹", s)
        # Percent-as-rupee only if the line has money context and it is not a compact real percent like 20%.
        # Examples: "for % 23.75", "= % 6000", "cost = % 15200".
        s2 = re.sub(r"(?<!\d)%\s+(?=\d)", "₹ ", s2)
        count_change(s, s2, stats, "money_symbol_repairs")
        s = s2

    # Targeted money-fragment repairs when OCR split the money word across lines.
    if chapter_title in {"Decimals", "Simple Interest", "Profit and Loss", "Unitary Method", "Mensuration"}:
        s2 = re.sub(r"=\s*¥\s*--", "= ₹ --", s)
        s2 = re.sub(r"(?<=SI on )€(?=\s*x)", "₹", s2)
        s2 = re.sub(r"(?<=be )%\s*x\b", "₹ x", s2)
        s2 = re.sub(r"(?<=SI on )%\s*(?=\(12000)", "₹ ", s2)
        count_change(s, s2, stats, "targeted_money_fragment_repairs")
        s = s2

    # Answer pages are dense and ambiguous, but large leading %/¥/€ amounts are usually rupees.
    # Do not change compact true percentages like 20% or angle artifacts such as £3=75°.
    if chapter_title == "Answers":
        s2 = re.sub(r"(?<!\d)%\s*(?=\d{3,})", "₹ ", s)
        s2 = re.sub(r"[¥€]\s*(?=\d{3,})", "₹ ", s2)
        s2 = re.sub(r"£(?=\s*\d+\s*[=°])", "∠", s2)
        count_change(s, s2, stats, "answer_page_money_angle_repairs")
        s = s2

    # Normalize rupee spacing after repairs.
    s2 = re.sub(r"₹\s+", "₹ ", s)
    count_change(s, s2, stats, "rupee_spacing_normalized")
    s = s2

    # Numeric OCR cleanup after symbol fixes.
    s = normalize_numeric_ocr(s, stats)

    if original != s:
        stats["lines_changed"] += 1
    return s


def cleanup_text(text: str, chapter_title: str | None, stats: Counter) -> str:
    # Preserve page-level line breaks as much as possible.
    lines = text.splitlines()
    cleaned = [cleanup_line(line, chapter_title, stats) for line in lines]
    out = "\n".join(cleaned)

    # A few cross-line/global spacing fixes.
    before = out
    out = re.sub(r" {2,}", " ", out)
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    if before != out:
        stats["spacing_normalized"] += 1
    return out.strip() if text.strip() else out


def clean_value(value: Any, chapter_title: str | None, stats: Counter, key: str | None = None) -> Any:
    if isinstance(value, str):
        if key in RAW_SOURCE_FIELDS_TO_SKIP:
            return value
        if key in TEXT_FIELDS_TO_CLEAN or key is None:
            return cleanup_text(value, chapter_title, stats)
        return value
    if isinstance(value, list):
        return [clean_value(v, chapter_title, stats, key=None) for v in value]
    if isinstance(value, dict):
        return {
            k: clean_value(v, chapter_title, stats, key=k)
            for k, v in value.items()
        }
    return value


def math_density(text: str) -> Tuple[int, int, float]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0, 0, 0.0
    math_chars = set("0123456789+-*/=()[]{}<>≠≤≥√^_|:,.%₹∠")
    math_like = 0
    for ln in lines:
        no_space = ln.replace(" ", "")
        if not no_space:
            continue
        m = sum(1 for ch in no_space if ch in math_chars)
        letters = sum(1 for ch in no_space if ch.isalpha())
        # Dense formula/answer lines often have mostly symbols/numbers or many separators.
        if (m >= max(4, letters) and len(no_space) <= 120) or re.search(r"\d\s*/\s*\d|\d+\s+[ivx]+\)", ln, re.I):
            math_like += 1
    return math_like, len(lines), math_like / max(1, len(lines))


def extract_math_lines(text: str) -> List[str]:
    out = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        math_like, total, ratio = math_density(stripped)
        if math_like > 0 and (re.search(r"[=+\-*/≠√₹∠]", stripped) or re.search(r"\d", stripped)):
            out.append(stripped)
    return out[:300]


def update_page_flags(page: Dict[str, Any], stats: Counter) -> None:
    chapter_title = page.get("chapter_title")
    text = page.get("text") or page.get("text_plain") or ""
    math_like, total_lines, ratio = math_density(text)

    page.setdefault("quality_metrics", {})["v3_math_like_lines"] = math_like
    page.setdefault("quality_metrics", {})["v3_total_nonblank_lines"] = total_lines
    page.setdefault("quality_metrics", {})["v3_math_line_ratio"] = round(ratio, 4)

    # Dense math/fraction/answers pages should be QA'd by vision before production embeddings.
    if chapter_title == "Answers":
        add_flag(page, "answer_key_dense_requires_manual_or_vision_qa")
        page["embedding_readiness"] = "needs_vision_qa_before_production_embedding"
        # Answer pages are especially dense and low value for lesson-body RAG; skip by default.
        page["include_in_embeddings"] = False
        stats["answer_pages_marked_not_ready"] += 1
    elif chapter_title in DENSE_MATH_CHAPTERS and ratio >= 0.42 and total_lines >= 12:
        add_flag(page, "dense_math_layout_requires_vision_qa")
        page["embedding_readiness"] = "caution_dense_math_layout"
        # Keep regular lesson pages embeddable, but make the warning explicit.
        page.setdefault("include_in_embeddings", True)
        stats["dense_math_pages_flagged"] += 1
    else:
        page.setdefault("embedding_readiness", "ready_with_ocr_cleanup")

    # Remaining unusual symbols are not automatically changed if context is ambiguous.
    rem = sorted(set(ch for ch in text if ch in FAKE_CURRENCY_SYMBOLS))
    if rem:
        add_flag(page, "remaining_ambiguous_symbol_requires_qa:" + "".join(rem))
        stats["pages_with_remaining_ambiguous_symbols"] += 1


def page_by_number(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(p["page_number"]): p for p in data["extraction"].get("page_extractions", [])}


def rebuild_chapter_lesson_text(data: Dict[str, Any]) -> None:
    pages = page_by_number(data)
    for chapter in data["extraction"].get("chapters", []):
        for lesson in chapter.get("lessons", []):
            nums = lesson.get("page_numbers") or list(range(int(lesson.get("start_page", 0)), int(lesson.get("end_page", -1)) + 1))
            lesson_pages = [pages[n] for n in nums if n in pages and pages[n].get("include_in_lesson_text", True)]
            lesson_text = "\n\n".join((p.get("text") or p.get("text_plain") or "").strip() for p in lesson_pages if (p.get("text") or p.get("text_plain") or "").strip())
            lesson["lesson_text"] = lesson_text
            lesson["text_plain"] = lesson_text
            lesson["math_lines"] = extract_math_lines(lesson_text)
            # propagate flags from pages
            flags = set(lesson.get("quality_flags", []))
            for p in lesson_pages:
                for f in p.get("quality_flags", []):
                    if "requires" in f or "ambiguous" in f:
                        flags.add(f)
            lesson["quality_flags"] = sorted(flags)
            # For answer appendix lesson, default not ready for embedding.
            if chapter.get("chapter_title") == "Answers":
                lesson["include_in_embeddings"] = False
                if "answer_key_dense_requires_manual_or_vision_qa" not in lesson["quality_flags"]:
                    lesson["quality_flags"].append("answer_key_dense_requires_manual_or_vision_qa")


def validate(data: Dict[str, Any], cleanup_stats: Counter) -> Tuple[List[str], List[str], Dict[str, Any]]:
    pages = data["extraction"].get("page_extractions", [])
    all_text = "\n".join(p.get("text", "") for p in pages)
    errors: List[str] = []
    warnings: List[str] = []

    pattern_counts = {}
    for name, pat in SUSPICIOUS_PATTERNS.items():
        pattern_counts[name] = len(pat.findall(all_text))

    if pattern_counts["seam_integer_list"]:
        errors.append(f"Remaining seam integer-list artifact count: {pattern_counts['seam_integer_list']}")
    if pattern_counts["formula_b_plus_e"]:
        warnings.append(f"Remaining a+(b+e) style formula artifact count: {pattern_counts['formula_b_plus_e']}")
    if pattern_counts["subtraction_ne_artifact"]:
        warnings.append(f"Remaining (-4)-2 42 -(-4) artifact count: {pattern_counts['subtraction_ne_artifact']}")
    if pattern_counts["multiply_25_19_wrong"]:
        warnings.append(f"Remaining (25 x 19) = +75 artifact count: {pattern_counts['multiply_25_19_wrong']}")
    if pattern_counts["ete_typo"]:
        warnings.append(f"Remaining ete. typo count: {pattern_counts['ete_typo']}")
    if pattern_counts["hash_not_equal"]:
        warnings.append(f"Remaining # not-equal artifact count: {pattern_counts['hash_not_equal']}")
    if pattern_counts["tilde_minus"]:
        warnings.append(f"Remaining tilde-minus artifact count: {pattern_counts['tilde_minus']}")

    remaining_symbol_pages = []
    for p in pages:
        text = p.get("text", "")
        chars = sorted(set(ch for ch in text if ch in FAKE_CURRENCY_SYMBOLS))
        if chars:
            remaining_symbol_pages.append((p.get("page_number"), p.get("printed_page_number"), p.get("chapter_title"), "".join(chars)))

    answer_not_ready = [p for p in pages if p.get("chapter_title") == "Answers" and not p.get("include_in_embeddings", True)]
    dense_flags = [p for p in pages if any(f.startswith("dense_math_layout") for f in p.get("quality_flags", []))]

    if remaining_symbol_pages:
        warnings.append(f"Remaining ambiguous symbol pages: {len(remaining_symbol_pages)}. These are flagged for QA instead of blindly converted.")
    if dense_flags:
        warnings.append(f"Dense math layout pages flagged for vision QA: {len(dense_flags)}")
    if answer_not_ready:
        warnings.append(f"Answer-key pages marked include_in_embeddings=false: {len(answer_not_ready)}")

    summary = {
        "pattern_counts": pattern_counts,
        "remaining_ambiguous_symbol_pages_count": len(remaining_symbol_pages),
        "remaining_ambiguous_symbol_pages_sample": remaining_symbol_pages[:50],
        "dense_math_layout_pages_count": len(dense_flags),
        "dense_math_layout_pages_sample": [
            (p.get("page_number"), p.get("printed_page_number"), p.get("chapter_title")) for p in dense_flags[:50]
        ],
        "answer_pages_marked_not_ready_count": len(answer_not_ready),
        "cleanup_stats": dict(cleanup_stats),
    }
    return errors, warnings, summary


def write_report(path: Path, input_path: Path, output_path: Path, errors: List[str], warnings: List[str], summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("Maths RSAggarwal math-aware OCR cleanup v3 validation report")
    lines.append("=" * 72)
    lines.append(f"Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Input JSON: {input_path}")
    lines.append(f"Output JSON: {output_path}")
    lines.append("")
    lines.append(f"Errors: {len(errors)}")
    if errors:
        lines.extend(f"  - {e}" for e in errors)
    else:
        lines.append("  None")
    lines.append("")
    lines.append(f"Warnings: {len(warnings)}")
    if warnings:
        lines.extend(f"  - {w}" for w in warnings)
    else:
        lines.append("  None")
    lines.append("")
    lines.append("Cleanup statistics")
    lines.append("-" * 72)
    for k, v in sorted(summary.get("cleanup_stats", {}).items()):
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Validation pattern counts")
    lines.append("-" * 72)
    for k, v in summary.get("pattern_counts", {}).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append(f"Remaining ambiguous symbol pages: {summary.get('remaining_ambiguous_symbol_pages_count')}")
    for pno, pp, title, chars in summary.get("remaining_ambiguous_symbol_pages_sample", []):
        lines.append(f"  - PDF page {pno}, printed {pp}, chapter={title}, symbols={chars}")
    lines.append("")
    lines.append(f"Dense math layout pages flagged: {summary.get('dense_math_layout_pages_count')}")
    for pno, pp, title in summary.get("dense_math_layout_pages_sample", []):
        lines.append(f"  - PDF page {pno}, printed {pp}, chapter={title}")
    lines.append("")
    lines.append(f"Answer-key pages marked include_in_embeddings=false: {summary.get('answer_pages_marked_not_ready_count')}")
    lines.append("")
    lines.append("Notes")
    lines.append("-" * 72)
    lines.append("This v3 pass fixes high-confidence OCR artifacts and flags pages requiring vision QA.")
    lines.append("It does not claim perfect reconstruction of stacked fractions, powers, or dense answer pages.")
    lines.append("For production-grade symbolic math, run a page-image vision/Mathpix pass for flagged pages.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    data = copy.deepcopy(data)

    cleanup_stats: Counter = Counter()

    # Clean page-level fields first.
    for page in data["extraction"].get("page_extractions", []):
        chapter_title = page.get("chapter_title")
        before_page = json.dumps(page, ensure_ascii=False, sort_keys=True)
        for key in list(page.keys()):
            if key in RAW_SOURCE_FIELDS_TO_SKIP:
                continue
            if key in TEXT_FIELDS_TO_CLEAN or key in {"math_lines", "extracted_blocks"}:
                page[key] = clean_value(page[key], chapter_title, cleanup_stats, key=key)
        # Always recompute clean math_lines from final text.
        page["math_lines"] = extract_math_lines(page.get("text") or page.get("text_plain") or "")
        update_page_flags(page, cleanup_stats)
        after_page = json.dumps(page, ensure_ascii=False, sort_keys=True)
        if before_page != after_page:
            cleanup_stats["pages_changed_or_flagged"] += 1

    # Clean section index and chapter titles/text metadata where needed.
    for section in data["extraction"].get("section_index", []):
        chapter_title = section.get("chapter_title") or section.get("section_title")
        for key in list(section.keys()):
            if isinstance(section[key], str) and key not in RAW_SOURCE_FIELDS_TO_SKIP:
                section[key] = clean_value(section[key], chapter_title, cleanup_stats, key=key if key in TEXT_FIELDS_TO_CLEAN else None)

    # Rebuild lesson text from cleaned page text; then clean any nested blocks left in chapters.
    rebuild_chapter_lesson_text(data)
    for chapter in data["extraction"].get("chapters", []):
        chapter_title = chapter.get("chapter_title")
        for lesson in chapter.get("lessons", []):
            for key in list(lesson.keys()):
                if key in {"lesson_text", "text_plain", "math_lines", "extracted_blocks"}:
                    lesson[key] = clean_value(lesson[key], chapter_title, cleanup_stats, key=key)

    previous_profile = data["extraction"].get("math_ocr_profile")
    data["extraction"]["math_ocr_profile"] = {
        "previous_profile": previous_profile,
        "current_profile": "rendered_page_ocr_plus_rule_based_math_cleanup_v3",
        "post_cleanup_v3": {
            "description": "Rule-based cleanup for remaining OCR/math artifacts plus QA flags for dense math/answer pages.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cleanup_stats": dict(cleanup_stats),
        },
    }

    errors, warnings, summary = validate(data, cleanup_stats)
    data["extraction"]["quality_summary"]["v3_cleanup_validation"] = {
        "errors": errors,
        "warnings": warnings,
        **summary,
    }
    data["extraction"]["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, input_path, output_path, errors, warnings, summary)

    print(f"Wrote JSON: {output_path}")
    print(f"Wrote report: {report_path}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    for w in warnings[:10]:
        print("WARN:", w)


if __name__ == "__main__":
    main()
