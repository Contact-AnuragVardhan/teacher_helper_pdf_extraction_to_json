#!/usr/bin/env python3
"""
Merge manual formula-review CSV edits into Grade10_Physics_formula_corrections.json.

Typical use from project root:

python app/physics_tenth/apply_review_csv_to_corrections.py ^
  --csv output/physics_tenth/formula_review_pack/review_items.csv ^
  --corrections app/physics_tenth/Grade10_Physics_formula_corrections.json ^
  --force

CSV rules:
- Put exact corrected textbook text in reviewed_text.
- Put yes in discard only for confirmed OCR garbage, duplicate fragments, captions/sidebar junk, or non-content.
- Leave both blank when not reviewed yet.
- If both reviewed_text and discard are filled, reviewed_text wins and the row is treated as corrected text.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRUTHY = {"1", "y", "yes", "true", "t", "discard", "x"}


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "2.0",
            "description": "Curated corrections for Grade 10 Physics formula/math review queue.",
            "review_policy": {
                "reviewed_blocks": "Exact text to insert into production text for the matching review_id.",
                "discard_review_ids": "Review IDs visually confirmed as non-content, duplicated formula fragments, or OCR garbage to omit.",
                "do_not_use_raw_corrupted_text": True,
            },
            "reviewed_blocks": {},
            "discard_review_ids": [],
            "global_replacements": {},
        }

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("schema_version", "2.0")
    data.setdefault("reviewed_blocks", {})
    data.setdefault("discard_review_ids", [])
    data.setdefault("global_replacements", {})
    data.setdefault(
        "review_policy",
        {
            "reviewed_blocks": "Exact text to insert into production text for the matching review_id.",
            "discard_review_ids": "Review IDs visually confirmed as non-content, duplicated formula fragments, or OCR garbage to omit.",
            "do_not_use_raw_corrupted_text": True,
        },
    )
    return data


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"review_id", "reviewed_text", "discard"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        return list(reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Edited review_items.csv")
    parser.add_argument(
        "--corrections",
        required=True,
        type=Path,
        help="Grade10_Physics_formula_corrections.json to update",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwriting --corrections.",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwrite of output file")
    args = parser.parse_args()

    csv_path = args.csv
    corrections_path = args.corrections
    output_path = args.output or corrections_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if output_path.exists() and output_path != corrections_path and not args.force:
        raise FileExistsError(f"Output exists. Use --force to overwrite: {output_path}")

    data = load_json(corrections_path)
    rows = read_csv_rows(csv_path)

    reviewed_blocks: dict[str, str] = dict(data.get("reviewed_blocks") or {})
    discard_ids = set(data.get("discard_review_ids") or [])

    csv_reviewed_count = 0
    csv_discard_count = 0
    untouched_count = 0
    both_reviewed_and_discard_count = 0

    for row in rows:
        review_id = clean_cell(row.get("review_id"))
        if not review_id:
            continue

        reviewed_text = clean_cell(row.get("reviewed_text"))
        discard_value = clean_cell(row.get("discard")).lower()
        should_discard = discard_value in TRUTHY

        if reviewed_text:
            csv_reviewed_count += 1
            if should_discard:
                both_reviewed_and_discard_count += 1
            reviewed_blocks[review_id] = reviewed_text
            discard_ids.discard(review_id)
        elif should_discard:
            csv_discard_count += 1
            reviewed_blocks.pop(review_id, None)
            discard_ids.add(review_id)
        else:
            untouched_count += 1

    data["reviewed_blocks"] = dict(sorted(reviewed_blocks.items()))
    data["discard_review_ids"] = sorted(discard_ids)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["last_csv_import"] = {
        "csv_path": str(csv_path),
        "csv_rows": len(rows),
        "csv_rows_with_reviewed_text": csv_reviewed_count,
        "csv_rows_marked_discard": csv_discard_count,
        "csv_rows_untouched": untouched_count,
        "csv_rows_with_both_reviewed_text_and_discard": both_reviewed_and_discard_count,
        "total_reviewed_blocks": len(data["reviewed_blocks"]),
        "total_discard_review_ids": len(data["discard_review_ids"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output exists. Use --force to overwrite: {output_path}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Updated corrections JSON: {output_path}")
    print(f"CSV rows: {len(rows)}")
    print(f"Added/updated reviewed_blocks from CSV: {csv_reviewed_count}")
    print(f"Added discard_review_ids from CSV: {csv_discard_count}")
    print(f"Untouched rows: {untouched_count}")
    if both_reviewed_and_discard_count:
        print(
            "Warning: rows had both reviewed_text and discard=yes; "
            f"reviewed_text was used for {both_reviewed_and_discard_count} rows."
        )
    print(f"Total reviewed_blocks now: {len(data['reviewed_blocks'])}")
    print(f"Total discard_review_ids now: {len(data['discard_review_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
