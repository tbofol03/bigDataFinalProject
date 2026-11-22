"""Prepare a blinded paired review of four address components and score it."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from experiment import load_runs, write_json


COMPONENTS = ("door_number", "street", "block_or_phase", "floor")
LANGUAGES = {"english_address": "EnglishAddress", "chinese_address": "ChineseAddress"}


def variant_id(index: int, field: str, value: str) -> str:
    data = json.dumps([index, field, value], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:24]


def paired_rows(indices: Path, baseline: Path, hybrid: Path):
    ids = json.loads(indices.read_text(encoding="utf-8"))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid or duplicate entries in the review manifest")
    rules, mixed = load_runs(baseline), load_runs(hybrid)
    for idx in ids:
        if idx not in rules or idx not in mixed:
            raise ValueError(f"Missing record {idx} in paired runs")
        left, right = rules[idx], mixed[idx]
        if left["raw"] != right["raw"]:
            raise ValueError(f"Mismatched raw fields at {idx}")
        yield idx, left, right


def prepare(indices: Path, baseline: Path, hybrid: Path, output: Path, seed: int) -> None:
    variants = {}
    for index, left, right in paired_rows(indices, baseline, hybrid):
        raw = left["raw"]
        for field, prefix in LANGUAGES.items():
            parts = [raw.get(f"{prefix}{n}", "") for n in (1, 2, 3)]
            for item in (left, right):
                text = item[field]
                code = variant_id(index, field, text)
                variants[code] = {"variant_id": code, "source_index": index, "field": field,
                                  "raw_parts": json.dumps(parts, ensure_ascii=False),
                                  "address_to_review": text, "review_status": "pending"}
    rows = list(variants.values())
    random.Random(seed).shuffle(rows)
    columns = [*rows[0], *COMPONENTS, "review_note"]
    with output.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} unique address variants for blinded review")


def score(indices: Path, baseline: Path, hybrid: Path, labels: Path, output: Path,
          protocol: str = "human") -> None:
    reviews = {}
    with labels.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = row["variant_id"]
            if code in reviews or row["review_status"].strip().casefold() != "adjudicated":
                raise ValueError(f"Duplicate or unreviewed variant: {code}")
            if code != variant_id(int(row["source_index"]), row["field"], row["address_to_review"]):
                raise ValueError(f"Address text or identifier changed in review file: {code}")
            answers = {key: row[key].strip().casefold() for key in COMPONENTS}
            if any(value not in {"yes", "no", "__skip__"} for value in answers.values()):
                raise ValueError(f"Use yes, no or __SKIP__ for each component of {code}")
            reviews[code] = answers
    counts = {lang: {part: Counter() for part in COMPONENTS} for lang in LANGUAGES}
    for index, left, right in paired_rows(indices, baseline, hybrid):
        for lang, tallies in counts.items():
            keys = [variant_id(index, lang, run[lang]) for run in (left, right)]
            if not all(v in reviews for v in keys):
                raise ValueError(f"Missing component review for record {index}, {lang}")
            for part, tally in tallies.items():
                before, after = (reviews[v][part] for v in keys)
                if "__skip__" in (before, after):
                    tally["skipped_pairs"] += 1
                    continue
                tally["scored_pairs"] += 1
                tally["rules_correct"] += before == "yes"
                tally["hybrid_correct"] += after == "yes"
                tally["repairs"] += before == "no" and after == "yes"
                tally["regressions"] += before == "yes" and after == "no"
    total = sum(tally["scored_pairs"] for lang in counts.values() for tally in lang.values())
    if total == 0:
        raise ValueError("No components were scored")
    result = {}
    for lang, tallies in counts.items():
        result[lang] = {}
        for part, tally in tallies.items():
            n = tally["scored_pairs"]
            result[lang][part] = {
                **tally, "rules_accuracy": tally["rules_correct"] / n if n else None,
                "hybrid_accuracy": tally["hybrid_correct"] / n if n else None,
                "delta_percentage_points": 100 * (tally["hybrid_correct"] -
                                                   tally["rules_correct"]) / n if n else None,
            }
    method = ("blinded paired XML source-fidelity audit" if protocol == "source-fidelity"
              else "blinded paired human review")
    write_json(output, {"method": method, "components": result})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    prep, evaluation = (actions.add_parser(cmd) for cmd in ("prepare", "score"))
    for action in (prep, evaluation):
        for flag in ("indices", "baseline", "hybrid", "out"):
            action.add_argument(f"--{flag}", required=True, type=Path)
    prep.add_argument("--seed", default=42, type=int)
    evaluation.add_argument("--labels", required=True, type=Path)
    evaluation.add_argument("--protocol", choices=["human", "source-fidelity"], default="human")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.indices, args.baseline, args.hybrid, args.out, args.seed)
    else:
        score(args.indices, args.baseline, args.hybrid, args.labels, args.out, args.protocol)


if __name__ == "__main__":
    main()
