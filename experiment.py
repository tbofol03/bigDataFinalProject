"""Create fixed sample manifests and score paired runs against reviewed labels."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

from importData import clean_record, stream_records
from llm_cleaning import FIELDS


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def strata_for(rec) -> str:
    if rec.needs_llm:
        return "complex_address"
    if any(flag.startswith("missing_") or flag.startswith("no_") for flag in rec.issues):
        return "missing_fields"
    if rec.year_category != "numeric":
        return "special_year"
    return "ordinary"


def interleave(items, key, rng):
    queues = defaultdict(list)
    for item in items:
        queues[key(item)].append(item)
    ordered = []
    for label in sorted(queues):
        rng.shuffle(queues[label])
        ordered.append(deque(queues[label]))
    while ordered:
        ordered = [queue for queue in ordered if queue]
        for queue in ordered:
            yield queue.popleft()


def sample(xml: Path, output: Path, seed: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve existing manifests and labels")
    records = [clean_record(raw, i) for i, raw in enumerate(stream_records(str(xml)), 1)]
    if len(records) < 5000:
        raise ValueError("A snapshot of at least 5,000 records is required")
    with xml.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    rng = random.Random(seed)
    reviewed = []
    seen = set()
    # Use one representative per primary-name group for the independent quality split.
    for rec in interleave(records, strata_for, rng):
        group = (rec.raw.get("EnglishBuildingName1", "").strip().casefold(),
                 rec.raw.get("ChineseBuildingName1", "").strip())
        if any(group):
            if group in seen:
                continue
            seen.add(group)
        reviewed.append(rec)
        if len(reviewed) == 400:
            break
    if len(reviewed) < 400:
        raise ValueError("Fewer than 400 distinct primary-name groups")
    # Allocate a quarter of each stratum to development, then fill any rounding gap.
    groups = defaultdict(list)
    for rec in reviewed:
        groups[strata_for(rec)].append(rec)
    development = []
    for label in sorted(groups):
        members = groups[label]
        development.extend(members[:len(members) // 4])
    selected_ids = {rec.source_index for rec in development}
    development.extend([rec for rec in reviewed if rec.source_index not in selected_ids]
                       [:100 - len(development)])
    development_ids = {rec.source_index for rec in development}
    holdout = [rec for rec in reviewed if rec.source_index not in development_ids]
    pilot = list(interleave(development, strata_for, rng))[:40]
    holdout_ids = {rec.source_index for rec in holdout}
    remaining = [rec for rec in records if rec.source_index not in holdout_ids]
    rng.shuffle(remaining)
    output.mkdir(parents=True, exist_ok=True)
    for size in (500, 5000):
        # Uniform shuffle preserves the remaining population mix in expectation.
        sample_ids = sorted(holdout_ids | {r.source_index for r in remaining[:size - 300]})
        write_json(output / f"indices_{size}.json", sample_ids)
    write_json(output / "indices_full.json", [rec.source_index for rec in records])
    write_json(output / "indices_pilot.json", sorted(rec.source_index for rec in pilot))
    write_json(output / "metadata.json", {
        "xml_sha256": source_hash, "records": len(records), "seed": seed,
        "strata": dict(Counter(strata_for(r) for r in reviewed)),
        "development": len(development), "holdout": len(holdout),
    })
    for name, subset in (("development", development), ("holdout", holdout)):
        with (output / f"labels_{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            columns = ["source_index", "stratum", "raw", "review_status",
                       *[f"expected_{key}" for key in FIELDS], "expected_year_category"]
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for rec in sorted(subset, key=lambda item: item.source_index):
                writer.writerow({"source_index": rec.source_index, "stratum": strata_for(rec),
                                 "raw": json.dumps(rec.raw, ensure_ascii=False),
                                 "review_status": "pending"})


def load_runs(path: Path) -> dict:
    result = {}
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            item = json.loads(line)
            index = item["source_index"]
            if index in result:
                raise ValueError(f"Duplicate index {index} in {path}")
            result[index] = item
    return result


def score(labels_path: Path, baseline_path: Path, hybrid_path: Path, output: Path) -> None:
    baseline, hybrid = load_runs(baseline_path), load_runs(hybrid_path)
    if baseline.keys() != hybrid.keys():
        raise ValueError("Baseline and hybrid runs must contain the same record IDs")
    counts = {field: Counter() for field in FIELDS}
    statuses, year = Counter(), Counter()
    seen = set()
    with labels_path.open(encoding="utf-8-sig", newline="") as f:
        for label in csv.DictReader(f):
            if label["review_status"].strip().casefold() != "adjudicated":
                raise ValueError("All label rows must be marked adjudicated before scoring")
            index = int(label["source_index"])
            if index in seen or index not in baseline:
                raise ValueError(f"Duplicate or unmatched label index {index}")
            seen.add(index)
            if json.loads(label["raw"]) != baseline[index]["raw"] or (
                baseline[index]["raw"] != hybrid[index]["raw"]
            ):
                raise ValueError(f"Raw data differ for labelled record {index}")
            statuses[hybrid[index]["llm_status"]] += 1
            expected_year = label["expected_year_category"].strip()
            if expected_year:
                if expected_year not in {"numeric", "blank", "dash", "pre_1945", "other"}:
                    raise ValueError(f"Invalid expected year category for {index}")
                actual = hybrid[index]["year_category"]
                year["n"] += 1
                year["correct"] += actual == expected_year
                truth_anomaly = expected_year in {"blank", "dash", "other"}
                found_anomaly = actual in {"blank", "dash", "other"}
                year["tp"] += found_anomaly and truth_anomaly
                year["fp"] += found_anomaly and not truth_anomaly
                year["fn"] += not found_anomaly and truth_anomaly
            for field in FIELDS:
                expected = label[f"expected_{field}"].strip()
                if expected == "__SKIP__":
                    continue
                before, after = baseline[index][field], hybrid[index][field]
                tally = counts[field]
                tally["n"] += 1
                tally["baseline_correct"] += before == expected
                tally["hybrid_correct"] += after == expected
                tally["correct_repairs"] += before != expected and after == expected
                tally["regressions"] += before == expected and after != expected
                tally["changes"] += before != after
    if not seen:
        raise ValueError("No labels were scored")

    def ratio(num, denom):
        return num / denom if denom else None

    write_json(output, {
        "records": len(seen), "hybrid_statuses": dict(statuses),
        "fields": {
            field: {**tally, "baseline_accuracy": ratio(tally["baseline_correct"], tally["n"]),
                    "hybrid_accuracy": ratio(tally["hybrid_correct"], tally["n"]),
                    "delta_percentage_points": ratio(
                        100 * (tally["hybrid_correct"] - tally["baseline_correct"]), tally["n"]
                    )}
            for field, tally in counts.items()
        },
        "year_anomalies": {**year, "precision": ratio(year["tp"], year["tp"] + year["fp"]),
                           "recall": ratio(year["tp"], year["tp"] + year["fn"]),
                           "f1": ratio(2 * year["tp"], 2 * year["tp"] + year["fp"] + year["fn"])},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    samples = commands.add_parser("sample")
    samples.add_argument("--xml", required=True, type=Path)
    samples.add_argument("--out", required=True, type=Path)
    samples.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("score")
    evaluation.add_argument("--labels", required=True, type=Path)
    evaluation.add_argument("--baseline", required=True, type=Path)
    evaluation.add_argument("--hybrid", required=True, type=Path)
    evaluation.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        sample(args.xml, args.out, args.seed)
    else:
        score(args.labels, args.baseline, args.hybrid, args.out)


if __name__ == "__main__":
    main()
