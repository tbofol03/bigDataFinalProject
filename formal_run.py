"""Run matched workloads using the frozen XML snapshot and manifests."""

import argparse
import csv
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys

from experiment import write_json


SOURCE = Path(__file__).resolve().parent
SIZES = {"500": 500, "5000": 5000, "full": 12421}


def run(script: str, *args) -> None:
    subprocess.run([sys.executable, str(SOURCE / script), *map(str, args)], check=True)


def verify_inputs(xml: Path, folder: Path):
    snapshot = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    with xml.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != snapshot["xml_sha256"]:
        raise ValueError("The XML does not match the frozen experiment snapshot")
    manifests = {}
    for size, n in SIZES.items():
        path = folder / f"indices_{size}.json"
        ids = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(ids, list) or len(ids) != n or len(set(ids)) != n
                or any(type(i) is not int or not 1 <= i <= SIZES["full"] for i in ids)):
            raise ValueError(f"Unexpected record IDs for {path.name}")
        manifests[size] = (path, set(ids))
    if not manifests["500"][1] < manifests["5000"][1] < manifests["full"][1]:
        raise ValueError("The workload manifests do not nest")
    with (folder / "labels_holdout.csv").open(encoding="utf-8-sig", newline="") as source:
        ids = [int(row["source_index"]) for row in csv.DictReader(source)]
    if len(ids) != 300 or len(set(ids)) != 300 or not set(ids) <= manifests["500"][1]:
        raise ValueError("Expected the same 300 holdout records at all three workloads")
    return digest, manifests, sorted(ids)


def run_rules(xml: Path, out: Path, manifests: dict, holdout: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    request_plan = {}
    for size, (path, ids) in manifests.items():
        run("importData.py", "--file", xml, "--indices", path, "--format", "jsonl",
            "-o", out / f"rules-{size}.jsonl", "--metrics-json", out / f"rules-{size}.json",
            "--no-plot")
        report = json.loads((out / f"rules-{size}.json").read_text(encoding="utf-8"))
        request_plan[size] = {"records": len(ids),
                              "route_candidates": round(report["routed_record_ratio"] * len(ids))}
    write_json(out / "indices_holdout.json", holdout)
    write_json(out / "live_request_plan.json", request_plan)


def run_live(args, digest: str, manifests: dict) -> None:
    path, ids = manifests[args.size]
    out = args.out
    recorded = json.loads((out / f"rules-{args.size}.json").read_text(encoding="utf-8"))
    expected_manifest = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    if recorded["indices_sha256"] != expected_manifest or recorded["xml_sha256"] != digest:
        raise ValueError("The existing baseline is from a different workload")
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("Configure DEEPSEEK_API_KEY in this terminal first")
    candidates = round(recorded["routed_record_ratio"] * len(ids))
    if not candidates:
        raise ValueError("There are no DeepSeek candidates for this workload")
    if args.max_requests < candidates:
        raise ValueError(f"This workload can request {candidates} calls. "
                         "Provide a sufficient --max-requests limit.")
    prefix = "live" if not args.resume else "resumed"
    if args.concurrency != 1:
        prefix += f"-c{args.concurrency}"
    if args.requests_per_minute is not None:
        prefix += f"-rpm{args.requests_per_minute:g}"
    metrics = out / f"{prefix}-{args.size}.json"
    result = out / f"{prefix}-{args.size}.jsonl"
    cache = out / "cache" / args.size
    if metrics.exists() or result.exists():
        raise ValueError(f"An output already exists for {prefix}-{args.size}")
    if not args.resume and cache.exists() and any(cache.iterdir()):
        raise ValueError("Use a fresh cache for live timing or add --resume")
    options = (["--requests-per-minute", args.requests_per_minute]
               if args.requests_per_minute is not None else [])
    run("importData.py", "--file", args.xml, "--indices", path, "--use-llm",
        "--cache-dir", cache, "--format", "jsonl", "--model", args.model,
        "--max-llm-requests", args.max_requests, "--concurrency", args.concurrency,
        "--input-price-per-million", args.input_price_per_million,
        "--output-price-per-million", args.output_price_per_million,
        "-o", result, "--metrics-json", metrics, "--no-plot", *options)
    report = json.loads(metrics.read_text(encoding="utf-8"))
    print(f"Saved {report['llm_calls']} live calls and {report['llm_cache_hits']} cache hits")
    failures = report["llm_status_counts"].get("request_error", 0)
    if failures == candidates:
        raise ValueError("All requests failed. Check the credential, account and model before retrying.")
    if failures:
        print(f"ATTENTION: {failures} candidate requests failed. "
              "Their rule values were retained; inspect failures before treating this run as complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    baseline = commands.add_parser("rules")
    live = commands.add_parser("live")
    for command in (baseline, live):
        command.add_argument("--xml", type=Path, required=True)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    live.add_argument("--size", choices=SIZES, required=True)
    live.add_argument("--model", required=True)
    live.add_argument("--max-requests", type=int, required=True)
    live.add_argument("--input-price-per-million", type=float, required=True)
    live.add_argument("--output-price-per-million", type=float, required=True)
    live.add_argument("--resume", action="store_true")
    live.add_argument("--concurrency", type=int, default=1)
    live.add_argument("--requests-per-minute", type=float)
    args = parser.parse_args()
    try:
        digest, manifests, holdout = verify_inputs(args.xml, args.inputs)
        if args.phase == "rules":
            run_rules(args.xml, args.out, manifests, holdout)
        else:
            prices = (args.input_price_per_million, args.output_price_per_million)
            if args.max_requests < 1 or args.concurrency < 1 or any(
                not isfinite(p) or p < 0 for p in prices
            ):
                raise ValueError("Use positive request and concurrency limits and non-negative prices")
            if args.requests_per_minute is not None and (
                not isfinite(args.requests_per_minute) or args.requests_per_minute <= 0
            ):
                raise ValueError("Request pacing must be finite and positive")
            run_live(args, digest, manifests)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
