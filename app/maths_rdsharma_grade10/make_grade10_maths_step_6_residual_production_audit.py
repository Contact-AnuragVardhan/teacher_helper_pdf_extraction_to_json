#!/usr/bin/env python3
"""
Step 6: final hard production-text gate for Grade10_Maths.

This version intentionally does NOT trust any approval flag, including
approved_by_trusted_page_override. A page is production-ready only if the final
fields that DB/embeddings use are clean:

  - page_extractions[].production_safe_text
  - chapters[].subsections[].production_subsection_text

The gate applies a small set of deterministic, reviewer-confirmed text fixes
where the correction is unambiguous, rebuilds subsection text, then fails the
artifact if any hard OCR/math corruption remains.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from make_grade10_maths_step_2_production_gate import rebuild_chapters_and_sections  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "maths_rdsharma_grade10"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_production_ready.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "Grade10_Maths_residual_production_audit_report.txt"
DEFAULT_REMAINING_CSV = DEFAULT_OUTPUT_DIR / "Grade10_Maths_residual_production_remaining_pages.csv"
PROMPT_VERSION = "residual_production_audit_v31_frustum_quadratic_circle_hard_blockers_2026_06_16"


GEOMETRY_POWER_SYMBOLS = (
    "OP", "PN", "PT", "AB", "BC", "AC", "PQ", "QR", "PR",
    "OA", "OB", "OC", "OT", "PA", "PB", "PC", "TP", "TQ",
    "QS", "RT", "RU", "PS", "PU", "OR", "TR", "BD", "AD",
    "CD", "DE", "AE", "BE", "CE", "AP", "BP", "CP", "CQ", "BR", "AT", "ET",
)

# Hard patterns. These are intentionally conservative production blockers. They
# are only scanned in final production fields, never in raw OCR fields.
HARD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # reviewer supplied blockers
    ("hard_chapter_ocr_garbage", re.compile(r"CHAPTER\s*[-=S|Ey_]{3,}", re.I)),
    ("hard_page13_square_example_uses_cube_4q1", re.compile(r"n\s*\^\s*2\s*-\s*1\s*=\s*\(\s*4q\s*\+\s*1\s*\)\s*\^\s*3", re.I)),
    ("hard_page13_square_example_uses_cube_4q3", re.compile(r"n\s*\^\s*2\s*-\s*1\s*=\s*\(\s*4q\s*\+\s*3\s*\)\s*\^\s*3", re.I)),
    # Do NOT hard-block every literal "8?" or "0?". In exercise text these are often
    # correct question endings, e.g. "point at 8?" or "which term ... is 0?".
    # Block only reviewer-confirmed OCR-corruption contexts around numeric question
    # marks. Generic literal question marks are allowed.
    ("hard_token_8_question_corrupt_context", re.compile(r"(?:8\?\s*[,;]\s*(?:rua|att|ae\s+of\s+af)|(?:\^|=|\+|[-])\s*8\?\s*(?:\^|=|\+|[-]))", re.I)),
    ("hard_token_0_question_corrupt_context", re.compile(r"(?:0\?\s*\+\s*4V?30\s*-\s*15|(?:\^|=|\+|[-])\s*0\?\s*(?:\^|=|\+|[-]))", re.I)),
    ("hard_token_rua", re.compile(r"\brua\b", re.I)),
    ("hard_token_att_squared", re.compile(r"\batt\s*\^\s*2\b", re.I)),
    ("hard_circle_noise_as_ela", re.compile(r"aS\s*\(\s*ela", re.I)),
    ("hard_token_eet", re.compile(r"(?<![A-Za-z])eet(?![A-Za-z])", re.I)),
    ("hard_token_rll", re.compile(r"\brll\b", re.I)),
    ("hard_token_KOI", re.compile(r"\bKOI\b", re.I)),


    # Reviewer-confirmed remaining blockers in foundational Real Numbers / HCF-LCM
    # and Probability pages. These strings are not valid textbook math/prose and
    # must never pass in production_safe_text or production_subsection_text.
    ("hard_euclid_qandr_token", re.compile(r"\bqandrsuch\b", re.I)),
    ("hard_euclid_a_bg_r_token", re.compile(r"\ba\s*-\s*bg\s*=\s*r\b", re.I)),
    ("hard_euclid_zero_letter_o_bound", re.compile(r"0O\s*<\s*r\s*<\s*b", re.I)),
    ("hard_euclid_bad_0523_bound", re.compile(r"0\s*<\s*52\s*<\s*3", re.I)),
    ("hard_euclid_bad_20_division", re.compile(r"20\s*=\s*3x64\s*\+\s*2", re.I)),
    ("hard_hcf_hce_token", re.compile(r"\bHCE\b", re.I)),
    ("hard_hcf_hick_token", re.compile(r"\bHICK\b", re.I)),
    ("hard_hcf_wyo_token", re.compile(r"\bWyo\b", re.I)),
    ("hard_hcf_power_question_context", re.compile(r"(?:\b[23]\?\s*x\s*[235]?|x\s*[23]\?|\b[23]\?\s*=)", re.I)),
    ("hard_probability_marble_sum_corrupt", re.compile(r"\bb\s*\+\s*gt\s*\+\s*w\s*=\s*54\b", re.I)),
    ("hard_probability_ual_token", re.compile(r"\bual\)", re.I)),
    ("hard_probability_bad_24x_equals", re.compile(r"24x\s*==\s*16", re.I)),

    # Surface Areas and Volumes reviewer-confirmed blockers. These are not
    # valid mathematical notation in production text; they are OCR residue from
    # formula lines such as slant height, curved surface area and volume.
    ("hard_surface_par_thy", re.compile(r"\bpar\s+thy\b", re.I)),
    ("hard_surface_fan_question", re.compile(r"\bfan\s*\?", re.I)),
    ("hard_surface_fan_power", re.compile(r"\bfan\s*\^\s*2\b", re.I)),
    ("hard_surface_2an_token", re.compile(r"(?<![A-Za-z0-9])2an\s+hy(?![A-Za-z0-9])", re.I)),
    ("hard_surface_wryly_token", re.compile(r"\bwryly\b", re.I)),
    ("hard_surface_m_colon_percent", re.compile(r"\bm\s*:\s*2\s*%", re.I)),
    ("hard_surface_21xJ2m_token", re.compile(r"\b21xJ2m\b", re.I)),
    ("hard_surface_723980_token", re.compile(r"\b72,3980\b", re.I)),
    ("hard_surface_m_greater_token", re.compile(r"(?<![A-Za-z0-9])m>(?![A-Za-z0-9])", re.I)),
    ("hard_surface_pax_token", re.compile(r"\bPax\b", re.I)),
    ("hard_surface_percent_formula_noise", re.compile(r"21%\s*21x|21x\s*/3", re.I)),
    ("hard_surface_plus_20_question", re.compile(r"\+\s*20\?", re.I)),

    # Surface Areas/Volumes water-column reviewer blockers (PDF page 633 / printed page 626).
    ("hard_surface_water_column_11700_3060", re.compile(r"1\s*%\s*1700\s*%\s*3060", re.I)),
    ("hard_surface_jom_squared", re.compile(r"\bJom\s*\^\s*2\b", re.I)),
    ("hard_surface_hem_squared", re.compile(r"\bhem\s*\^\s*2\b", re.I)),
    ("hard_surface_lo_s0_percent_noise", re.compile(r"\bLO\s*%\s*S0", re.I)),
    ("hard_surface_hris_token", re.compile(r"\bhris\b", re.I)),
    ("hard_surface_water_column_corrupt_sentence", re.compile(r"Length\s+of\s+the\s+water\s+column\s+in\s+2\s+hours[\s\S]{0,900}(?:1\s*%\s*1700\s*%\s*3060|Jom\s*\^\s*2|hem\s*\^\s*2|LO\s*%\s*S0|hris)", re.I)),

    # Surface Areas/Volumes sphere/tank reviewer blockers (PDF page 628 / printed page 621).
    ("hard_surface_xem_squared", re.compile(r"\bxem\s*\^\s*2\b", re.I)),
    ("hard_surface_36xhem_token", re.compile(r"\b36xhem\s*\^?\s*2?\b", re.I)),
    ("hard_surface_6_question_xh", re.compile(r"\b6\?\s*xh\b", re.I)),
    ("hard_surface_ve_nr_h_question", re.compile(r"\bVe\s+nr\?h\b|\bnr\?h\b", re.I)),
    ("hard_surface_bad_36h_3627_line", re.compile(r"36h\s*=\s*3627|3627\s*>\s*h\s*=\s*1\s*cm", re.I)),
    ("hard_surface_sphere_tank_corrupt_block", re.compile(r"Volume\s+of\s+the\s+sphere[\s\S]{0,900}(?:xem\s*\^\s*2|6\?\s*xh|36xhem|Ve\s+nr\?h|3627)", re.I)),

    # Statistics table reviewer blocker (PDF page 704 / printed page 697).
    ("hard_stats_percent_12_hash_percent_2", re.compile(r"%\s*12\s*#\s*%\s*2", re.I)),


    # Reviewer-confirmed algebra/quadratic/surface remnants found after v24.
    ("hard_surface_zom_token", re.compile(r"\bZom\b", re.I)),
    ("hard_surface_jom_equals_rem", re.compile(r"\bJom\s*=\s*rem\b", re.I)),
    ("hard_surface_19_minus_2_percent_2", re.compile(r"19\s*-\s*2\s*%\s*2", re.I)),
    ("hard_surface_parenthesized_19_2_percent_om", re.compile(r"\(\s*19\s*-\s*2\s*%\s*2\s*\)\s*om", re.I)),
    ("hard_polynomial_ag2_99_question", re.compile(r"\bAg\s*\^\s*2\s*99\?", re.I)),
    ("hard_polynomial_x_dollar_equals", re.compile(r"\bx\s*\$=", re.I)),
    ("hard_polynomial_4x_star_bad_line", re.compile(r"4x\*\s*\+\s*8x\^?2\s*-\s*12x\^?2\s*==", re.I)),
    ("hard_polynomial_4g_equals_f", re.compile(r"\b4g\s*=\s*f\b", re.I)),
    ("hard_quadratic_discriminant_question", re.compile(r"\bD\s*=\s*\?\s*-\s*4ac\b", re.I)),
    ("hard_quadratic_minus6_degree", re.compile(r"\(\s*-\s*6\s*\)°", re.I)),
    ("hard_quadratic_anon_cgahomamal", re.compile(r"\banon\s+cgahomamal\b", re.I)),
    ("hard_probability_ual_standalone", re.compile(r"\bual(?![A-Za-z])", re.I)),

    # Reviewer-confirmed circle/geometry corruption found after v25. These are
    # OCR remnants in tangent/circle formula lines, not valid production maths.
    ("hard_circle_at_et_question_power", re.compile(r"(?<![A-Za-z0-9])(?:AT|ET)\?(?![A-Za-z0-9])")),
    ("hard_circle_12_minus_x_question_power", re.compile(r"\(\s*12\s*-\s*x\s*\)\s*=\s*\?", re.I)),
    ("hard_general_bad_14g_token", re.compile(r"\b14g\b")),
    ("hard_circle_bad_2dv_token", re.compile(r"\b2dv\b", re.I)),
    ("hard_circle_bad_x_equals_om", re.compile(r"\bX\s*=\s*OM\b")),
    ("hard_circle_bad_em_2om", re.compile(r"\bem\s*=\s*2om\b", re.I)),
    ("hard_circle_bint_token", re.compile(r"\bBint\b", re.I)),
    ("hard_circle_iets_token", re.compile(r"\biets\b", re.I)),


    # Reviewer-confirmed blockers found after v27 (Real Numbers, algebra, and Surface Areas/Volumes).
    ("hard_surface_pn_x2nrand", re.compile(r"\bPn\s*x2nrand\b", re.I)),
    ("hard_surface_550m_token", re.compile(r"\b550m\b", re.I)),
    ("hard_surface_bem_question", re.compile(r"\bbem\?", re.I)),
    ("hard_surface_agate_question", re.compile(r"\bagate\?", re.I)),
    ("hard_surface_xcent_token", re.compile(r"\bxcent\b", re.I)),
    ("hard_algebra_by_au_token", re.compile(r"\bBy\s+au\b", re.I)),
    ("hard_algebra_octing_token", re.compile(r"(?:\\-octing|(?<![A-Za-z])octing\b)", re.I)),
    ("hard_algebra_gots_t_es", re.compile(r"\bgots\s+t\s+eS\b", re.I)),
    ("hard_real_ifaisanon_token", re.compile(r"\bIfaisanon\b", re.I)),
    ("hard_real_al_bandc_token", re.compile(r"\bal\{bandc\b", re.I)),
    ("hard_real_if_at_and_c", re.compile(r"\bIf@and\s+c\b", re.I)),
    ("hard_real_bla_gt_a_eq_b", re.compile(r"\bbla\s*>\s*a\s*=\s*\+?b\b", re.I)),

    # Reviewer-confirmed linear-equations/algebra blockers found after v28
    # (PDF pages around 106, 119 and 123). These are OCR residues in
    # production fields, not valid textbook algebra.
    ("hard_linear_boa_os", re.compile(r"\bBoa\s+os\b", re.I)),
    ("hard_linear_dix_token", re.compile(r"\bDix\b", re.I)),
    ("hard_linear_x3_g_minus_2", re.compile(r"X_?3\s*\(\s*g\s*-\s*2\s*\)", re.I)),
    ("hard_linear_seins_orp_pee", re.compile(r"\bseins\s+orp\s+pee\b", re.I)),
    ("hard_linear_u_hash_0040", re.compile(r"u#\s*0\s*,\s*040", re.I)),
    ("hard_linear_y_double_equals", re.compile(r"\by\s*=\s*--\s*==", re.I)),
    ("hard_linear_xty_ot4", re.compile(r"\bxty\s*=\s*ot4\b", re.I)),
    ("hard_linear_2_eel", re.compile(r"\b2\s+eel\b", re.I)),
    ("hard_linear_substituting_y_or_he", re.compile(r"Substituting\s+y\s*=\s*or\s+he", re.I)),
    ("hard_linear_bad_2528_formula", re.compile(r"ax\s*\+\s*4\s*\(\s*2528\s*\)", re.I)),



    # Surface Areas/Volumes hemispherical toy reviewer blockers (PDF page 652 / printed page 645).
    # These tokens are OCR residue in formula lines and must not pass production fields.
    ("hard_surface_percent_9_mm_noise", re.compile(r"(?:25\s*%\s*9|%\s*9\s*mm)", re.I)),
    ("hard_surface_co_bracket_em", re.compile(r"\bCo\]\s*em\b", re.I)),
    ("hard_surface_admin_token", re.compile(r"\badmin\b", re.I)),
    ("hard_surface_2225x9442", re.compile(r"\b2225x9442\b", re.I)),
    ("hard_surface_2_percent_25", re.compile(r"\b2\s*%\s*25\b", re.I)),
    ("hard_surface_guillemet_25x", re.compile(r"«\s*25x", re.I)),
    ("hard_surface_volume_theenne", re.compile(r"Volume\?\s*theenne", re.I)),
    ("hard_surface_somn_arrow", re.compile(r"<-+\s*somn\s*-+>", re.I)),
    ("hard_surface_bx_sxs", re.compile(r"\bBx\s*SxS\b", re.I)),
    ("hard_surface_hemisphere_corrupt_block", re.compile(r"For\s+two\s+hemispherical\s+parts[\s\S]{0,1000}(?:25\s*%\s*9|Co\]\s*em|admin|2225x9442|2\s*%\s*25|«\s*25x|Volume\?\s*theenne|<-+\s*somn\s*-+>|Bx\s*SxS)", re.I)),


    # Reviewer-confirmed frustum/quadratic/circle blockers found after v30.
    # These are corrupted production OCR/math fragments and must never pass final gate.
    ("hard_frustum_sth_token", re.compile(r"\bSth\.", re.I)),
    ("hard_frustum_ue_hot", re.compile(r"\bue\s+hot\b", re.I)),
    ("hard_frustum_n_n_plus_n_l", re.compile(r"n\s*\(\s*n\s*\+\s*n\s*\)\s*l", re.I)),
    ("hard_frustum_r_comma_question", re.compile(r"r\s*,\s*\?", re.I)),
    ("hard_frustum_mr_plus_172", re.compile(r"\bmr\s*\+\s*172\b", re.I)),
    ("hard_quadratic_at_eo", re.compile(r"@\s*eo", re.I)),
    ("hard_quadratic_re1e0", re.compile(r"\bre1e0\b", re.I)),
    ("hard_quadratic_a_question_b_question_x", re.compile(r"a\?b\?x", re.I)),
    ("hard_quadratic_hash4_paren", re.compile(r"#4\s*\(", re.I)),
    ("hard_quadratic_x_plus_224h", re.compile(r"x\s*\+\s*224h", re.I)),
    ("hard_circle_earigent", re.compile(r"\bearigent\b", re.I)),
    ("hard_circle_90_percent_a2", re.compile(r"90°\s*%\s*a2", re.I)),
    ("hard_circle_j125", re.compile(r"\bJ125\b", re.I)),

    # previous hard blockers retained
    ("hard_duplicate_cube_power", re.compile(r"\)\s*\^\s*3\s*\^\s*3")),
    ("hard_duplicate_square_power", re.compile(r"\)\s*\^\s*2\s*\^\s*2")),
    ("hard_cube_proof_square_expansion", re.compile(r"n\s*\^\s*3\s*-\s*n\s*=\s*\(\s*2q\s*\)\s*\^\s*2", re.I)),
    ("hard_cube_minus_one_square_expansion", re.compile(r"n\s*\^\s*3\s*-\s*1[\s\S]{0,260}16q\s*\^\s*2", re.I)),
    ("hard_bad_3g_plus_one", re.compile(r"\b3g\s*\+\s*1\b", re.I)),
    ("hard_bad_39_plus", re.compile(r"\b39\s*\+\s*(?:1|2)\s*\+\s*(?:2|4)\b", re.I)),
    ("hard_bad_thus_1_plus_4", re.compile(r"Thus,?\s*1\s*\+\s*4\b", re.I)),
    ("hard_geometry_power_question", re.compile(r"(?<![A-Za-z0-9])(?:OP|PN|PT|AB|BC|AC|PQ|QR|PR|OA|OB|OC|OT|PA|PB|PC|TP|TQ|QS|RT|RU|PS|PU|OR|TR|BD|AD|CD|DE|AE|BE|CE|AP|BP|CP|CQ|BR)\?(?![A-Za-z0-9])", re.I)),
    ("hard_numeric_question_power", re.compile(r"\b\d+\?\s*-\s*\d+\?\b")),
    ("hard_y_plus_3_question_power", re.compile(r"\(\s*y\s*\+\s*3\s*\)\s*=\s*\?", re.I)),
    ("hard_T_question_star_equals", re.compile(r"\bT\?\s*\*?\s*=", re.I)),
    ("hard_at_equals_4_noise", re.compile(r"@\s*=\s*4")),
    ("hard_stats_fiat_token", re.compile(r"\bfiat\s*;", re.I)),
    ("hard_stats_arn_txfu_token", re.compile(r"Arn\s*\{\s*txfu", re.I)),
    ("hard_stats_ps_eee_token", re.compile(r"\bPS\s+eee\b", re.I)),
    ("hard_bad_n_question_token", re.compile(r"(?<![A-Za-z0-9])n\?(?![A-Za-z0-9])", re.I)),
    ("hard_bad_q7_token", re.compile(r"\b(?:\d+q7|q7)\b", re.I)),
    ("hard_bad_degree_power_token", re.compile(r"\b(?:649|169)°")),
    ("hard_bad_2q_cent_token", re.compile(r"2q¢\s*\+\s*1", re.I)),
    ("hard_bad_plus_i_2n4_token", re.compile(r"\+i\s*=\s*\(\s*2n4\s*\+\s*1P", re.I)),
    ("hard_bad_ae_of_af", re.compile(r"\bae\s+of\s+af\b", re.I)),
    ("hard_malformed_sec_tan_fraction", re.compile(r"sec\s*θ\s*\+\s*tan\s*θ\s*=\s*p[\s\S]{0,500}1\s*/\s*\(\s*p\s*\^\s*2\s*\+\s*1\s*\)\s*/\s*\(\s*2p\s*\+\s*1\s*\)", re.I)),

    # general obvious OCR residue in math/formula pages
    ("hard_known_ocr_garbage_tokens", re.compile(r"\b(?:Dandy|1andy|Iwa|1Ps|rng\s+ere|Outces|subsef|know1|H0S\s+RET)\b", re.I)),
    ("hard_suspicious_ocr_symbols", re.compile(r"[¥¢£€™®§©¤¦¨¬¯´¸¿¡�]")),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def is_lesson_body_page(page: dict[str, Any]) -> bool:
    return page.get("content_type") == "lesson_body" and page.get("include_in_lesson_text") is True


def page_is_ready(page: dict[str, Any]) -> bool:
    return (
        page.get("include_in_embeddings") is True
        and page.get("embedding_readiness") == "ready_for_production_embedding"
        and bool(str(page.get("production_safe_text") or "").strip())
    )


def safe_text(page: dict[str, Any]) -> str:
    return str(page.get("production_safe_text") or "")


def sync_legacy_page_fields(pages: list[dict[str, Any]]) -> int:
    changed = 0
    for page in pages:
        prod = str(page.get("production_safe_text") or "").strip()
        if not prod:
            continue
        for field in ("text", "text_plain"):
            if page.get(field) != prod:
                page[field] = prod
                changed += 1
        flags = set(page.get("quality_flags") or [])
        flags.add("legacy_text_fields_synced_to_production_safe_text")
        page["quality_flags"] = sorted(flags)
    return changed


def sync_legacy_sections(data: dict[str, Any]) -> int:
    changed = 0
    extraction = data.get("extraction", {})
    for chapter in extraction.get("chapters", []) or []:
        prod_ch = chapter.get("production_chapter_text") or ""
        if prod_ch:
            for field in ("chapter_text", "chapter_text_plain", "text_plain"):
                if chapter.get(field) != prod_ch:
                    chapter[field] = prod_ch
                    changed += 1
        for sub in chapter.get("subsections", []) or []:
            prod = sub.get("production_subsection_text") or ""
            if prod:
                for field in ("subsection_text", "subsection_text_plain", "text_plain"):
                    if sub.get(field) != prod:
                        sub[field] = prod
                        changed += 1
    for key in ("lessons", "sections", "subsections"):
        for item in extraction.get(key, []) or []:
            prod = item.get("production_subsection_text") or item.get("production_lesson_text") or item.get("production_text") or ""
            if prod:
                for field in ("subsection_text", "subsection_text_plain", "lesson_text", "text_plain"):
                    if field in item and item.get(field) != prod:
                        item[field] = prod
                        changed += 1
    return changed


def page_number(page: dict[str, Any]) -> int:
    try:
        return int(page.get("page_number") or 0)
    except Exception:
        return 0


def apply_common_deterministic_fixes(text: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    new = text

    new2 = re.sub(r"\)\s*\^\s*([23])\s*\^\s*\1", lambda m: f")^{m.group(1)}", new)
    if new2 != new:
        fixes.append("collapse_duplicate_power_suffix")
        new = new2

    sym_alt = "|".join(map(re.escape, GEOMETRY_POWER_SYMBOLS))
    geom_re = re.compile(rf"(?<![A-Za-z0-9])({sym_alt})\s*\?(?![A-Za-z0-9])", re.I)
    new2 = geom_re.sub(lambda m: f"{m.group(1)}^2", new)
    if new2 != new:
        fixes.append("geometry_segment_question_mark_power_to_square")
        new = new2

    new2 = re.sub(r"(\([^\n]{1,80}\))\s*=\s*\?", r"\1^2 =", new)
    if new2 != new:
        fixes.append("parenthesized_expression_question_mark_power_to_square")
        new = new2

    return new, fixes


def apply_page13_deterministic_fixes(text: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    new, common = apply_common_deterministic_fixes(text)
    fixes.extend(common)

    replacements = [
        # Example 9 is n^2 - 1. Vision sometimes keeps cube powers in the base.
        (r"n\s*\^\s*2\s*-\s*1\s*=\s*\(\s*4q\s*\+\s*1\s*\)\s*\^\s*3\s*-\s*1", "n^2 - 1 = (4q + 1)^2 - 1"),
        (r"n\s*\^\s*2\s*-\s*1\s*=\s*\(\s*4q\s*\+\s*3\s*\)\s*\^\s*3\s*-\s*1", "n^2 - 1 = (4q + 3)^2 - 1"),
        (r"n\s*\^\s*2\s*-\s*1\s*=\s*\(\s*4q\s*\+\s*1\s*\)\s*\^\s*3", "n^2 - 1 = (4q + 1)^2"),
        (r"n\s*\^\s*2\s*-\s*1\s*=\s*\(\s*4q\s*\+\s*3\s*\)\s*\^\s*3", "n^2 - 1 = (4q + 3)^2"),
        # Old OCR forms on this page.
        (r"\(4q\)\^3\^3", "(4q)^3"),
        (r"\(4q\s*\+\s*1\)\^3\^3", "(4q + 1)^3"),
        (r"\(4q\s*\+\s*2\)\^3\^3", "(4q + 2)^3"),
        (r"\(4q\s*\+\s*3\)\^3\^3", "(4q + 3)^3"),
        (r"64q°|64q\?|649°", "64q^3"),
        (r"169°", "16q^3"),
        (r"144q7", "144q^2"),
        (r"48q7", "48q^2"),
        (r"16q°", "16q^3"),
        (r"2q¢\s*\+\s*1", "2q + 1"),
        (r"4g\s*\+\s*3", "4q + 3"),
        (r"n>\s*=", "n^3 ="),
        # Only when the left side already says n^3: remove square/cube mix.
        (r"n\s*\^\s*3\s*-\s*n\s*=\s*\(\s*2q\s*\)\s*\^\s*2", "n^2 - n = (2q)^2"),
        (r"n\s*\^\s*3\s*-\s*1", "n^2 - 1"),
    ]
    for pat, repl in replacements:
        new2 = re.sub(pat, repl, new, flags=re.I)
        if new2 != new:
            fixes.append(f"page13:{pat}->{repl}")
            new = new2
    return new, fixes


def apply_page15_deterministic_fixes(text: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    new, common = apply_common_deterministic_fixes(text)
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
    ]
    for pat, repl in replacements:
        new2 = re.sub(pat, repl, new, flags=re.I)
        if new2 != new:
            fixes.append(f"page15:{pat}->{repl}")
            new = new2
    return new, fixes


def apply_deterministic_known_fixes(data: dict[str, Any]) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    pages = data.setdefault("extraction", {}).get("page_extractions", []) or []
    for page in pages:
        pno = page_number(page)
        text = str(page.get("production_safe_text") or "")
        if not text:
            continue
        if pno == 13:
            fixed, fixes = apply_page13_deterministic_fixes(text)
        elif pno == 15:
            fixed, fixes = apply_page15_deterministic_fixes(text)
        else:
            fixed, fixes = apply_common_deterministic_fixes(text)
        fixed = fixed.strip()
        if fixes and fixed and fixed != text:
            page["production_safe_text"] = fixed
            page["text"] = fixed
            page["text_plain"] = fixed
            flags = set(page.get("quality_flags") or [])
            flags.add("hard_gate_deterministic_fix_applied")
            page["quality_flags"] = sorted(flags)
            page["hard_gate_deterministic_fixes"] = sorted(set((page.get("hard_gate_deterministic_fixes") or []) + fixes))
            applied.append({
                "page_number": pno,
                "printed_page_number": page.get("printed_page_number"),
                "fixes": fixes,
            })
    if applied:
        rebuild_chapters_and_sections(data)
    return applied


def detect_hard_patterns(text: str) -> list[str]:
    if not text:
        return []
    reasons: list[str] = []
    for name, pattern in HARD_PATTERNS:
        if pattern.search(text):
            reasons.append(name)
    return sorted(set(reasons))


def detect_residual_issues(page: dict[str, Any]) -> list[str]:
    text = safe_text(page)
    if not text.strip():
        return ["residual_empty_production_safe_text"]
    return detect_hard_patterns(text)


def collect_residual_blockers(data: dict[str, Any], *, selected_pages: set[int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in data.get("extraction", {}).get("page_extractions", []) or []:
        if not is_lesson_body_page(page):
            continue
        pno = page_number(page)
        if selected_pages is not None and pno not in selected_pages:
            continue
        if not page_is_ready(page):
            rows.append({"page": page, "candidate_kind": "excluded_lesson_body", "reasons": list(page.get("production_exclusion_reasons") or ["excluded_lesson_body"])})
            continue
        reasons = detect_residual_issues(page)
        if reasons:
            rows.append({"page": page, "candidate_kind": "residual_production_text_issue", "reasons": reasons})
    return sorted(rows, key=lambda row: page_number(row["page"]))


def collect_subsection_residual_blockers(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chapter in data.get("extraction", {}).get("chapters", []) or []:
        for sub in chapter.get("subsections", []) or []:
            text = str(sub.get("production_subsection_text") or "")
            reasons = detect_hard_patterns(text)
            if not reasons:
                continue
            rows.append({
                "page_number": sub.get("start_page") or sub.get("start_pdf_page"),
                "printed_page_number": sub.get("printed_start_page") or sub.get("start_book_page"),
                "chapter_title": chapter.get("chapter_title"),
                "candidate_kind": "residual_production_subsection_text_issue",
                "reasons": reasons,
                "sample_text": re.sub(r"\s+", " ", text[:700]),
                "chapter_number": chapter.get("chapter_number"),
                "subsection_number": sub.get("subsection_number"),
                "subsection_title": sub.get("subsection_title"),
            })
    return rows


def count_empty_production_subsections(data: dict[str, Any]) -> list[dict[str, Any]]:
    empty: list[dict[str, Any]] = []
    for chapter in data.get("extraction", {}).get("chapters", []) or []:
        for sub in chapter.get("subsections", []) or []:
            if int(sub.get("production_page_count") or 0) <= 0 or not str(sub.get("production_subsection_text") or "").strip():
                empty.append({
                    "chapter_number": chapter.get("chapter_number"),
                    "chapter_title": chapter.get("chapter_title"),
                    "subsection_number": sub.get("subsection_number"),
                    "subsection_title": sub.get("subsection_title"),
                    "start_page": sub.get("start_page") or sub.get("start_pdf_page"),
                    "end_page": sub.get("end_page") or sub.get("end_pdf_page"),
                })
    return empty


def row_sample(row: dict[str, Any]) -> str:
    if "page" in row:
        return re.sub(r"\s+", " ", safe_text(row["page"])[:700])
    return str(row.get("sample_text") or "")[:700]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["page_number", "printed_page_number", "chapter_title", "candidate_kind", "reasons", "sample_text"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if "page" in row:
                page = row["page"]
                out = {
                    "page_number": page.get("page_number"),
                    "printed_page_number": page.get("printed_page_number"),
                    "chapter_title": page.get("chapter_title"),
                    "candidate_kind": row.get("candidate_kind"),
                    "reasons": "; ".join(row.get("reasons") or []),
                    "sample_text": row_sample(row),
                }
            else:
                out = {
                    "page_number": row.get("page_number"),
                    "printed_page_number": row.get("printed_page_number"),
                    "chapter_title": row.get("chapter_title"),
                    "candidate_kind": row.get("candidate_kind"),
                    "reasons": "; ".join(row.get("reasons") or []),
                    "sample_text": row_sample(row),
                }
            writer.writerow(out)


def refresh_policy(
    data: dict[str, Any], *, blockers: list[dict[str, Any]], subsection_blockers: list[dict[str, Any]],
    empty_subsections: list[dict[str, Any]], page_sync_count: int, section_sync_count: int,
    deterministic_fixes: list[dict[str, Any]], selected_pages: set[int] | None = None,
) -> dict[str, Any]:
    extraction = data.setdefault("extraction", {})
    pages = extraction.get("page_extractions", []) or []
    lesson_body = [p for p in pages if is_lesson_body_page(p)]
    ready = [p for p in lesson_body if page_is_ready(p)]
    excluded = [p for p in lesson_body if not page_is_ready(p)]
    blocker_page_numbers = sorted({page_number(row["page"]) for row in blockers if "page" in row})
    subsection_page_numbers = sorted({int(row.get("page_number") or 0) for row in subsection_blockers if row.get("page_number")})
    hard_blocker_count = len(blockers) + len(subsection_blockers) + len(empty_subsections) + len(excluded)
    status = "production_complete_ready" if hard_blocker_count == 0 else "production_safe_gated_needs_residual_production_qa"

    qs = extraction.setdefault("quality_summary", {})
    reason_counts = Counter(
        [reason for row in blockers for reason in row.get("reasons", [])]
        + [reason for row in subsection_blockers for reason in row.get("reasons", [])]
    )
    summary = {
        "generated_at_utc": now_utc(),
        "prompt_version": PROMPT_VERSION,
        "selected_pages_only": sorted(selected_pages) if selected_pages else None,
        "gate_status": status,
        "hard_production_gate_applied": True,
        "hard_production_blocker_count": hard_blocker_count,
        "lesson_body_pages": len(lesson_body),
        "ready_lesson_body_pages": len(ready),
        "excluded_lesson_body_pages": len(excluded),
        "remaining_residual_pages": len(blockers),
        "remaining_residual_page_numbers": blocker_page_numbers[:500],
        "remaining_residual_subsection_count": len(subsection_blockers),
        "remaining_residual_subsection_start_pages": subsection_page_numbers[:500],
        "empty_production_subsections": len(empty_subsections),
        "page_legacy_fields_synced": page_sync_count,
        "section_legacy_fields_synced": section_sync_count,
        "deterministic_fix_pages": [item.get("page_number") for item in deterministic_fixes],
        "deterministic_fix_count": len(deterministic_fixes),
        "remaining_reason_counts": dict(reason_counts),
        "empty_subsection_samples": empty_subsections[:30],
    }
    qs["residual_production_audit"] = summary
    qs["hard_production_gate"] = summary

    policy = extraction.setdefault("production_embedding_policy", {})
    policy.update({
        "status": status,
        "production_complete": status == "production_complete_ready",
        "residual_production_audit_applied": True,
        "residual_production_audit_version": PROMPT_VERSION,
        "hard_production_gate_applied": True,
        "hard_production_blocker_count": hard_blocker_count,
        "residual_production_remaining_pages": len(blockers),
        "residual_production_subsection_blockers": len(subsection_blockers),
        "residual_production_empty_subsections": len(empty_subsections),
        "excluded_lesson_body_pages": len(excluded),
        "ready_lesson_body_pages": len(ready),
    })
    if status == "production_complete_ready":
        policy["math_precision_remaining_pages"] = 0
        policy["math_precision_empty_production_subsections"] = 0
        policy["math_precision_errors"] = []
        math = qs.setdefault("math_precision_audit", {})
        math["superseded_by_hard_production_gate"] = True
        math["remaining_math_precision_pages"] = 0
        math["excluded_lesson_body_pages"] = 0
        math["empty_production_subsections"] = 0
        math["remaining_reason_counts"] = {}
    extraction["generated_at_utc"] = now_utc()
    return summary


def run_residual_audit(data: dict[str, Any], *, selected_pages: set[int] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    deterministic_fixes = apply_deterministic_known_fixes(data)
    rebuild_chapters_and_sections(data)
    page_sync = sync_legacy_page_fields(data.get("extraction", {}).get("page_extractions", []) or [])
    section_sync = sync_legacy_sections(data)
    blockers = collect_residual_blockers(data, selected_pages=selected_pages)
    subsection_blockers = collect_subsection_residual_blockers(data)
    empty = count_empty_production_subsections(data)
    summary = refresh_policy(
        data,
        blockers=blockers,
        subsection_blockers=subsection_blockers,
        empty_subsections=empty,
        page_sync_count=page_sync,
        section_sync_count=section_sync,
        deterministic_fixes=deterministic_fixes,
        selected_pages=selected_pages,
    )
    return summary, blockers, subsection_blockers


def write_report(path: Path, *, input_path: Path, output_path: Path, remaining_csv: Path, summary: dict[str, Any]) -> None:
    lines = [
        "Grade10_Maths Final Hard Production Gate Report",
        "=" * 60,
        f"Generated at UTC: {now_utc()}",
        f"Input JSON: {input_path}",
        f"Output JSON: {output_path}",
        f"Remaining CSV: {remaining_csv}",
        f"Audit version: {PROMPT_VERSION}",
        "",
        f"Production status: {summary.get('gate_status')}",
        f"Hard production blocker count: {summary.get('hard_production_blocker_count')}",
        f"Lesson-body pages: {summary.get('lesson_body_pages')}",
        f"Ready lesson-body pages: {summary.get('ready_lesson_body_pages')}",
        f"Excluded lesson-body pages: {summary.get('excluded_lesson_body_pages')}",
        f"Remaining residual pages: {summary.get('remaining_residual_pages')}",
        f"Remaining residual subsection blockers: {summary.get('remaining_residual_subsection_count')}",
        f"Empty production subsections: {summary.get('empty_production_subsections')}",
        f"Deterministic fix count: {summary.get('deterministic_fix_count')}",
        f"Page legacy fields synced: {summary.get('page_legacy_fields_synced')}",
        f"Section legacy fields synced: {summary.get('section_legacy_fields_synced')}",
        "",
        "Remaining reason counts:",
    ]
    for reason, count in sorted((summary.get("remaining_reason_counts") or {}).items()):
        lines.append(f"  - {reason}: {count}")
    nums = summary.get("remaining_residual_page_numbers") or []
    if nums:
        lines.extend(["", "Remaining page numbers:", "  " + ", ".join(map(str, nums))])
    sub_nums = summary.get("remaining_residual_subsection_start_pages") or []
    if sub_nums:
        lines.extend(["", "Remaining subsection start pages:", "  " + ", ".join(map(str, sub_nums))])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final hard production validator on Grade10_Maths production fields.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--remaining-csv", type=Path, default=DEFAULT_REMAINING_CSV)
    parser.add_argument("--pages", default=None, help="Optional PDF pages to audit only, e.g. 13,49,71")
    parser.add_argument("--strict-complete", action="store_true")
    args = parser.parse_args()

    selected = parse_pages_arg(args.pages)
    data = load_json(args.input)
    summary, page_rows, subsection_rows = run_residual_audit(data, selected_pages=selected)
    all_rows = page_rows + subsection_rows
    write_json(args.output, data)
    write_csv(args.remaining_csv, all_rows)
    write_report(args.report, input_path=args.input, output_path=args.output, remaining_csv=args.remaining_csv, summary=summary)

    print(f"Residual production audit remaining pages: {summary.get('remaining_residual_pages')}")
    print(f"Residual production audit remaining subsections: {summary.get('remaining_residual_subsection_count')}")
    print(f"Hard production blocker count: {summary.get('hard_production_blocker_count')}")
    print(f"Production status: {summary.get('gate_status')}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")
    print(f"Wrote: {args.remaining_csv}")

    if args.strict_complete and summary.get("gate_status") != "production_complete_ready":
        raise RuntimeError(
            "Strict complete failed after final hard production gate: "
            f"status={summary.get('gate_status')}, "
            f"hard_production_blocker_count={summary.get('hard_production_blocker_count')}, "
            f"remaining_residual_pages={summary.get('remaining_residual_pages')}, "
            f"remaining_residual_subsection_count={summary.get('remaining_residual_subsection_count')}, "
            f"excluded_lesson_body_pages={summary.get('excluded_lesson_body_pages')}, "
            f"empty_production_subsections={summary.get('empty_production_subsections')}. "
            f"See {args.report} and {args.remaining_csv}."
        )


if __name__ == "__main__":
    main()
