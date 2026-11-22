"""Prepare source-based review drafts without changing adjudication status."""

import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

from importData import clean_text, classify_year, normalize_address


FLOOR = re.compile(r"\bfloors?\b|地庫|地下|樓上|[0-9]+樓", re.I)
STREET = re.compile(
    r"\b(?:road|street|avenue|drive|boulevard|lane|terrace|crescent|way)\b|[街道路巷徑]",
    re.I,
)
BLOCK = re.compile(r"\b(?:block|phase|tower)\b|第?[0-9一二三四五六七八九十甲乙丙丁A-Z]+[座期]", re.I)
COMPONENTS = ("door_number", "street", "block_or_phase", "floor")


def transform(source: Path, destination: Path, prepare) -> None:
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(row["review_status"].strip().casefold() != "pending" for row in rows):
        raise ValueError("Use the original pending review template")
    prepared = [prepare(row) for row in rows]
    with destination.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prepared[0]))
        writer.writeheader()
        writer.writerows(prepared)


def holdout_suggestion(row: dict) -> dict:
    raw = json.loads(row["raw"])

    def join_parts(prefix, delimiter=" "):
        return delimiter.join(filter(None, (clean_text(raw.get(f"{prefix}{i}")) for i in (1, 2, 3))))

    english = normalize_address(join_parts("EnglishAddress", ", "))
    chinese = join_parts("ChineseAddress")
    notes = ["Provisional transcription based on source fields. Check exact spelling and address format."]
    if any(FLOOR.search(val) for val in (english, chinese)):
        notes.append("Check the floor-to-address association and approve the expected separator format.")
        english = chinese = "__SKIP__"
    row.update(
        review_status="draft",
        expected_english_name=join_parts("EnglishBuildingName"),
        expected_chinese_name=join_parts("ChineseBuildingName"),
        expected_english_address=english,
        expected_chinese_address=chinese,
        expected_year_category=classify_year(raw.get("YearBuild", ""))[0],
        review_note=" ".join(notes),
    )
    return row


def blind_suggestion(row: dict) -> dict:
    parts = [clean_text(value) for value in json.loads(row["raw_parts"])]
    english = row["field"] == "english_address"
    source = (", " if english else " ").join(filter(None, parts))
    candidate = row["address_to_review"]

    def folded(text):
        text = unicodedata.normalize("NFKC", text)
        value = normalize_address(text) if english else text
        return re.sub(r"[\s,，]", "", value.casefold())

    answers = dict.fromkeys(COMPONENTS, "__SKIP__")
    note = "Source or candidate is empty, or significant text differs. Review manually."
    if source and candidate and folded(source) == folded(candidate):
        as_written = normalize_address(source) if english else source
        # The lexical check verifies preservation in the input, including numbers.
        # It cannot establish which number belongs to which floor or verify the address.
        for label, pattern in (("door_number", r"\d"), ("street", STREET),
                               ("block_or_phase", BLOCK), ("floor", FLOOR)):
            if re.search(pattern, as_written):
                answers[label] = "yes"
        note = "Draft preservation check. Verify street, door and floor roles against the source."
    row.update(**answers, review_status="draft", review_note=note)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["holdout", "addresses"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    transform(args.input, args.out, holdout_suggestion if args.kind == "holdout"
              else blind_suggestion)


if __name__ == "__main__":
    main()
