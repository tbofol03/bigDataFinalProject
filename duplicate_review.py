"""Build and score a review set of possible duplicate building-record pairs."""

import argparse
import csv
import random
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from experiment import write_json
from importData import clean_record, stream_records


def key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def exact_pair_candidate(left, right) -> bool:
    names = [(key(left.english_name), key(right.english_name)),
             (key(left.chinese_name), key(right.chinese_name))]
    addresses = [(key(left.english_address), key(right.english_address)),
                 (key(left.chinese_address), key(right.chinese_address))]
    for pairs in (names, addresses):
        comparable = [(a, b) for a, b in pairs if a and b]
        if not comparable or any(a != b for a, b in comparable):
            return False
    return True


def shared_primary_pair(left, right) -> bool:
    return any(
        key(left.raw.get(f"{language}BuildingName1", "")) ==
        key(right.raw.get(f"{language}BuildingName1", ""))
        and bool(key(left.raw.get(f"{language}BuildingName1", "")))
        for language in ("English", "Chinese")
    ) and any(
        key(getattr(left, field)) == key(getattr(right, field))
        and bool(key(getattr(left, field)))
        for field in ("english_address", "chinese_address")
    )


def make_review(xml: Path, output: Path, per_group: int, seed: int) -> None:
    records = [clean_record(raw, i) for i, raw in enumerate(stream_records(str(xml)), 1)]
    latter = {rec.source_index: rec for rec in records}
    index = defaultdict(list)
    for rec in records:
        for language in ("English", "Chinese"):
            name = key(rec.raw.get(f"{language}BuildingName1", ""))
            if name:
                index[("name", language, name)].append(rec.source_index)
        for field in ("english_address", "chinese_address"):
            address = key(getattr(rec, field))
            if address:
                index[("address", field, address)].append(rec.source_index)

    related = defaultdict(set)
    for (kind, _, _), ids in index.items():
        for pair in combinations(ids, 2):
            related[pair].add(kind)
    groups = defaultdict(list)
    for pair, basis in sorted(related.items()):
        left, right = (latter[i] for i in pair)
        if exact_pair_candidate(left, right):
            group = "exact_name_and_address"
        elif basis == {"name", "address"}:
            group = "same_primary_name_and_shared_address"
        elif "name" in basis:
            group = "same_primary_name"
        else:
            group = "shared_address"
        groups[group].append(pair)

    rng = random.Random(seed)
    selected = [(group, pair) for group, pool in sorted(groups.items())
                for pair in rng.sample(pool, min(per_group, len(pool)))]
    if not selected:
        raise ValueError("No related record pairs found")
    rng.shuffle(selected)
    columns = ["left_id", "right_id", "left_english_name", "right_english_name",
               "left_chinese_name", "right_chinese_name", "left_english_address",
               "right_english_address", "left_chinese_address", "right_chinese_address",
               "review_status", "same_entity", "review_note"]
    keys = ["english_name", "chinese_name", "english_address", "chinese_address"]
    with output.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for _, (a, b) in selected:
            row = {"left_id": a, "right_id": b, "review_status": "pending"}
            for side, rec_id in (("left", a), ("right", b)):
                row.update({f"{side}_{field}": getattr(latter[rec_id], field) for field in keys})
            writer.writerow(row)
    print("Pairs queued by sampling stratum:", dict(Counter(group for group, _ in selected)))
    print("Available pairs by stratum:", {group: len(pool) for group, pool in groups.items()})


def score_review(xml: Path, labels: Path, output: Path) -> None:
    records = {i: clean_record(raw, i) for i, raw in enumerate(stream_records(str(xml)), 1)}
    detectors = {"exact_full_name_and_address": exact_pair_candidate,
                 "shared_primary_name_and_address": shared_primary_pair}
    counts = {name: Counter() for name in detectors}
    seen = set()
    skipped = 0
    with labels.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["review_status"].strip().casefold() != "adjudicated":
                raise ValueError("Resolve every pair before scoring")
            pair = tuple(sorted((int(row["left_id"]), int(row["right_id"]))))
            if pair in seen or pair[0] == pair[1] or not all(i in records for i in pair):
                raise ValueError(f"Invalid or repeated pair: {pair}")
            for side in ("left", "right"):
                rec = records[int(row[f"{side}_id"])]
                for field in ("english_name", "chinese_name", "english_address", "chinese_address"):
                    display = row.get(f"{side}_{field}")
                    if display is not None and display != getattr(rec, field):
                        raise ValueError(f"Displayed source differs for pair {pair}, {side}, {field}")
            seen.add(pair)
            label = row["same_entity"].strip().casefold()
            if label == "__skip__":
                skipped += 1
                continue
            if label not in {"yes", "no"}:
                raise ValueError(f"Expected yes, no or __SKIP__ for {pair}")
            actual = label == "yes"
            for name, detector in detectors.items():
                pred = detector(*(records[i] for i in pair))
                counts[name]["tp" if pred and actual else "fp" if pred else
                             "fn" if actual else "tn"] += 1
    if len(seen) == skipped:
        raise ValueError("No adjudicated duplicate decisions to score")

    def rates(tally):
        tp, fp, fn, tn = (tally[k] for k in ("tp", "fp", "fn", "tn"))
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "positive_labels": tp + fn, "negative_labels": fp + tn}

    write_json(output, {
        "scope": "reviewed related pairs only",
        "pair_count": len(seen), "skipped": skipped,
        "methods": {name: rates(tally) for name, tally in counts.items()},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    prepare, score = (actions.add_parser(cmd) for cmd in ("prepare", "score"))
    for action in (prepare, score):
        action.add_argument("--xml", required=True, type=Path)
        action.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--per-group", type=int, default=24)
    prepare.add_argument("--seed", type=int, default=42)
    score.add_argument("--labels", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.per_group < 1:
            parser.error("--per-group must be positive")
        make_review(args.xml, args.out, args.per_group, args.seed)
    else:
        score_review(args.xml, args.labels, args.out)


if __name__ == "__main__":
    main()
