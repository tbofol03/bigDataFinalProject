#!/usr/bin/env python3
"""
Hybrid Rule + Selective DeepSeek Cleaning with Plots
Only low-confidence records consume tokens.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, asdict
from collections import Counter
from math import ceil, isfinite
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.request import urlopen
from xml.etree.ElementTree import iterparse

from floor_separators import FLOOR_RULE_REVISION, format_floor_sections
from llm_cleaning import PROMPT_VERSION, decide_changes, get_model_result, is_semantic_candidate
from request_pacing import PacedClient

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)

NULL_MARKERS = {"", "-", "N/A", "n/a", "NA", "None", "null", "NULL"}
ABBREV_MAP = {
    r"(?<!\w)rd\b\.?": "Road", r"(?<!\w)st\b\.?": "Street",
    r"(?<!\w)ave\b\.?": "Avenue", r"(?<!\w)blvd\b\.?": "Boulevard",
    r"(?<!\w)dr\b\.?": "Drive", r"\blane\b": "Lane",
    r"(?<!\w)blk\b\.?": "Block", r"(?<!\w)bl\b\.?": "Block", r"\bg/f\b": "G/F",
}
YEAR_PRE_1945_PATTERN = re.compile(r"pre\s*1945", re.IGNORECASE)
YEAR_NUMERIC_PATTERN = re.compile(r"^\d{4}$")

_converter = None
_convert_mode = None

def init_converter(mode: str) -> bool:
    global _converter, _convert_mode
    _convert_mode = mode
    if mode == "none":
        return True
    try:
        from opencc import OpenCC
        config = {"hk2s": "hk2s", "t2s": "t2s", "s2t": "s2t"}.get(mode, "t2s")
        _converter = OpenCC(config)
        return True
    except ImportError:
        try:
            import zhconv
            _converter = zhconv
            return True
        except ImportError:
            return False

def convert_chinese(text: str) -> str:
    if not text or _converter is None or _convert_mode == "none":
        return text
    try:
        if hasattr(_converter, "convert"):
            return _converter.convert(text)
        else:
            target = "zh-cn" if _convert_mode in ("t2s", "hk2s") else "zh-tw"
            return _converter.convert(text, target)
    except Exception:
        return text

@dataclass
class BuildingRecord:
    raw: Dict[str, str] = field(default_factory=dict)
    english_name: str = ""
    chinese_name: str = ""
    chinese_name_simplified: str = ""
    english_address: str = ""
    chinese_address: str = ""
    chinese_address_simplified: str = ""
    year_build_raw: str = ""
    year_category: str = ""
    year_numeric: Optional[int] = None
    owners_corporation: str = ""
    public_housing_type_en: str = ""
    public_housing_type_zh: str = ""
    public_housing_type_zh_simplified: str = ""
    issues: List[str] = field(default_factory=list)
    confidence: float = 1.0
    needs_llm: bool = False
    source_index: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_applied: bool = False
    llm_rationale: str = ""
    llm_confidence: str = ""
    unresolved: bool = False
    rule_values: Dict[str, str] = field(default_factory=dict)
    llm_proposal: Dict[str, Any] = field(default_factory=dict)
    llm_status: str = "not_requested"
    llm_origin: str = "none"
    llm_error: str = ""
    llm_requested: bool = False
    request_latency_seconds: Optional[float] = None

def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = value.strip()
    if text in NULL_MARKERS:
        return ""
    return re.sub(r"\s+", " ", text)

def normalize_address(text: str) -> str:
    if not text:
        return text
    result = text
    for pattern, replacement in ABBREV_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result.strip()

def concatenate_parts(parts: List[str], separator: str = " ") -> str:
    return separator.join(p for p in parts if p)

def classify_year(raw: str) -> tuple[str, Optional[int]]:
    if raw.strip() == "-":
        return "dash", None
    text = clean_text(raw)
    if not text:
        return "blank", None
    if YEAR_PRE_1945_PATTERN.search(text):
        return "pre_1945", None
    if YEAR_NUMERIC_PATTERN.match(text):
        return "numeric", int(text)
    return "other", None

def stream_records(source: str) -> Iterator[Dict[str, str]]:
    if source.startswith(("http://", "https://")):
        with urlopen(source, timeout=120) as response:
            context = iterparse(response, events=("end",))
            yield from _extract_records(context)
    else:
        with open(source, "rb") as f:
            context = iterparse(f, events=("end",))
            yield from _extract_records(context)

def _extract_records(context) -> Iterator[Dict[str, str]]:
    for event, elem in context:
        if elem.tag == "Record":
            record = {child.tag: (child.text or "") for child in elem}
            yield record
            elem.clear()
        if elem.tag == "data-set":
            elem.clear()

def detect_issues(rec: BuildingRecord) -> List[str]:
    issues = []
    if rec.english_name and not rec.chinese_name:
        issues.append("missing_chinese_name")
    if rec.chinese_name and not rec.english_name:
        issues.append("missing_english_name")
    if rec.english_address and not rec.chinese_address:
        issues.append("missing_chinese_address")
    if rec.chinese_address and not rec.english_address:
        issues.append("missing_english_address")
    if not rec.english_name and not rec.chinese_name:
        issues.append("no_building_name")
    if not rec.english_address and not rec.chinese_address:
        issues.append("no_address")
    if rec.year_category in {"blank", "dash", "other"}:
        issues.append(f"year_{rec.year_category}")
    if rec.english_address and re.search(r"\d+\s*-\s*\d+", rec.english_address):
        issues.append("address_range")
    if rec.english_address and "&" in rec.english_address:
        issues.append("address_ampersand")
    return issues

def compute_confidence(rec: BuildingRecord) -> float:
    score = 1.0
    if not rec.english_name and not rec.chinese_name:
        score -= 0.4
    if not rec.english_address and not rec.chinese_address:
        score -= 0.3
    if rec.year_category in {"blank", "other"}:
        score -= 0.15
    if any(i.startswith("missing_") for i in rec.issues):
        score -= 0.1
    return max(0.0, min(1.0, score))

def clean_record(raw: Dict[str, str], index: int, *, floor_separators: bool = False) -> BuildingRecord:
    rec = BuildingRecord(raw=raw.copy(), source_index=index)
    en_names = [clean_text(raw.get(f"EnglishBuildingName{i}")) for i in (1, 2, 3)]
    zh_names = [clean_text(raw.get(f"ChineseBuildingName{i}")) for i in (1, 2, 3)]
    en_addrs = [clean_text(raw.get(f"EnglishAddress{i}")) for i in (1, 2, 3)]
    zh_addrs = [clean_text(raw.get(f"ChineseAddress{i}")) for i in (1, 2, 3)]

    rec.english_name = concatenate_parts(en_names)
    rec.chinese_name = concatenate_parts(zh_names)
    rec.english_address = normalize_address(concatenate_parts(en_addrs, ", "))
    rec.chinese_address = concatenate_parts(zh_addrs)
    if floor_separators:
        rec.english_address = format_floor_sections(rec.english_address, "english")
        rec.chinese_address = format_floor_sections(rec.chinese_address, "chinese")
    rec.chinese_name_simplified = convert_chinese(rec.chinese_name)
    rec.chinese_address_simplified = convert_chinese(rec.chinese_address)
    rec.year_build_raw = raw.get("YearBuild", "")
    rec.year_category, rec.year_numeric = classify_year(rec.year_build_raw)
    rec.owners_corporation = clean_text(raw.get("OwnersCorporation"))
    rec.public_housing_type_en = clean_text(raw.get("EnglishPublicHousingType"))
    rec.public_housing_type_zh = clean_text(raw.get("ChinesePublicHousingType"))
    rec.public_housing_type_zh_simplified = convert_chinese(rec.public_housing_type_zh)
    rec.issues = detect_issues(rec)
    rec.confidence = compute_confidence(rec)
    rec.rule_values = {key: getattr(rec, key) for key in (
        "english_name", "chinese_name", "english_address", "chinese_address",
    )}
    rec.needs_llm = is_semantic_candidate(rec.rule_values)
    return rec

def call_deepseek(rec: BuildingRecord, client, model: str = "deepseek-chat",
                  cache_dir: Optional[Path] = None) -> None:
    result = get_model_result(rec, client, model, cache_dir)
    rec.llm_requested = result.requested
    rec.request_latency_seconds = result.latency
    rec.input_tokens = result.input_tokens
    rec.output_tokens = result.output_tokens
    rec.llm_error = result.error
    rec.llm_origin = result.status
    if result.status not in {"live", "cache_hit"}:
        rec.llm_status = result.status
        rec.unresolved = True
        return
    try:
        data = json.loads(result.content)
    except (ValueError, TypeError):
        rec.llm_status = "invalid_json"
        rec.unresolved = True
        return
    if isinstance(data, dict):
        rec.llm_proposal = data
    status, changes = decide_changes(data, rec.raw, rec.rule_values, normalize_address)
    rec.llm_status = status
    rec.llm_applied = status == "applied"
    rec.unresolved = status not in {"applied", "unchanged"}
    if isinstance(data, dict):
        rec.llm_rationale = str(data.get("rationale", ""))
        rec.llm_confidence = str(data.get("confidence", ""))
    for key, value in changes.items():
        setattr(rec, key, value)
    if "chinese_name" in changes:
        rec.chinese_name_simplified = convert_chinese(rec.chinese_name)
    if "chinese_address" in changes:
        rec.chinese_address_simplified = convert_chinese(rec.chinese_address)


def process_llm_records(records: Iterator[BuildingRecord], client, model: str,
                        cache_dir: Optional[Path], *, concurrency: int = 1,
                        max_requests: Optional[int] = None, replay_only: bool = False,
                        route: str = "selective") -> List[BuildingRecord]:
    """Bound in-flight work and reserve the live request budget before submission."""
    if concurrency < 1 or (max_requests is not None and max_requests < 0):
        raise ValueError("Invalid concurrency or request limit")
    cleaned = []
    live_calls = completed = 0
    pending = {}

    def mark_done(rec):
        nonlocal live_calls, completed
        live_calls += int(rec.llm_requested)
        completed += 1
        if completed % 10 == 0:
            print(f"  evaluated {completed:,} candidates, {live_calls:,} live requests",
                  file=sys.stderr, flush=True)

    def reap():
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            rec = pending.pop(future)
            future.result()
            mark_done(rec)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for rec in records:
            cleaned.append(rec)
            if not (rec.needs_llm or route == "all"):
                continue
            # Each unfinished job can consume one live request. Wait for its
            # actual usage before reusing a reservation released by a cache hit.
            while pending and (
                len(pending) >= concurrency
                or (max_requests is not None and live_calls + len(pending) >= max_requests)
            ):
                reap()
            capped = max_requests is not None and live_calls >= max_requests
            if capped:
                call_deepseek(rec, None, model, cache_dir)
                if rec.llm_status == "cache_miss" and not replay_only:
                    rec.llm_status = "budget_skipped"
                mark_done(rec)
                continue
            future = pool.submit(call_deepseek, rec, client, model, cache_dir)
            pending[future] = rec
        while pending:
            reap()
    return cleaned


def evaluate(records: List[BuildingRecord], archived_count: int, elapsed: float) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    bilingual_name_ok = sum(1 for r in records if (r.english_name and r.chinese_name) or (not r.english_name and not r.chinese_name))
    bilingual_addr_ok = sum(1 for r in records if (r.english_address and r.chinese_address) or (not r.english_address and not r.chinese_address))
    year_anomaly = sum(1 for r in records if r.year_category in {"blank", "dash", "other"})
    records_per_min = (n / elapsed * 60) if elapsed > 0 else 0
    routed_ratio = sum(1 for r in records if r.needs_llm) / n
    total_input_tokens = sum(r.input_tokens for r in records)
    total_output_tokens = sum(r.output_tokens for r in records)
    live = [r for r in records if r.llm_requested]
    latencies = sorted(r.request_latency_seconds for r in live
                       if r.request_latency_seconds is not None)

    def percentile(p: float) -> Optional[float]:
        return latencies[ceil(p * len(latencies)) - 1] if latencies else None

    return {
        "name_presence_agreement": bilingual_name_ok / n,
        "address_presence_agreement": bilingual_addr_ok / n,
        "names_both_blank": sum(not r.english_name and not r.chinese_name for r in records),
        "addresses_both_blank": sum(not r.english_address and not r.chinese_address
                                    for r in records),
        "year_anomaly_count": year_anomaly,
        "records_per_minute": records_per_min,
        "routed_record_ratio": routed_ratio,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "llm_calls": len(live),
        "llm_applied_count": sum(r.llm_applied for r in records),
        "llm_status_counts": dict(Counter(r.llm_status for r in records)),
        "llm_cache_hits": sum(r.llm_origin == "cache_hit" for r in records),
        "live_input_tokens": sum(r.input_tokens for r in live),
        "live_output_tokens": sum(r.output_tokens for r in live),
        "request_p50_seconds": percentile(0.50),
        "request_p95_seconds": percentile(0.95),
        "total_records": n,
        "elapsed_seconds": elapsed,
        "parsed_to_archived_ratio": n / archived_count if archived_count else None,
        "year_numeric": sum(1 for r in records if r.year_category == "numeric"),
        "year_blank": sum(1 for r in records if r.year_category == "blank"),
        "year_dash": sum(1 for r in records if r.year_category == "dash"),
        "year_pre_1945": sum(1 for r in records if r.year_category == "pre_1945"),
        "year_other": sum(1 for r in records if r.year_category == "other"),
        "english_name1_missing_pct": sum(1 for r in records if not clean_text(r.raw.get("EnglishBuildingName1"))) / n * 100,
        "chinese_name1_missing_pct": sum(1 for r in records if not clean_text(r.raw.get("ChineseBuildingName1"))) / n * 100,
        "address1_missing_pct": sum(1 for r in records if not clean_text(r.raw.get("EnglishAddress1")) and not clean_text(r.raw.get("ChineseAddress1"))) / n * 100,
        "owners_blank_pct": sum(1 for r in records if not r.owners_corporation) / n * 100,
        "public_housing_blank_pct": sum(1 for r in records if not r.public_housing_type_en and not r.public_housing_type_zh) / n * 100,
    }

def print_evaluation(metrics: Dict[str, Any]) -> None:
    print("\n" + "="*70, file=sys.stderr)
    print("CLEANING RUN SUMMARY", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(f"Total records                 : {metrics.get('total_records', 0):,}", file=sys.stderr)
    print(f"Name presence agreement       : {metrics.get('name_presence_agreement', 0):.4f}", file=sys.stderr)
    print(f"Address presence agreement    : {metrics.get('address_presence_agreement', 0):.4f}", file=sys.stderr)
    print(f"Year anomalies                : {metrics.get('year_anomaly_count', 0):,}", file=sys.stderr)
    print(f"Routed record ratio           : {metrics.get('routed_record_ratio', 0):.4f}", file=sys.stderr)
    print(f"Actual LLM calls              : {metrics.get('llm_calls', 0):,}", file=sys.stderr)
    print(f"Total input tokens            : {metrics.get('total_input_tokens', 0):,}", file=sys.stderr)
    print(f"Total output tokens           : {metrics.get('total_output_tokens', 0):,}", file=sys.stderr)
    print(f"Runtime                       : {metrics.get('elapsed_seconds', 0):.2f} s", file=sys.stderr)
    print(f"Records per minute            : {metrics.get('records_per_minute', 0):,.1f}", file=sys.stderr)
    print("="*70 + "\n", file=sys.stderr)

def plot_summary(records: List[BuildingRecord], metrics: Dict[str, Any], output_prefix: str) -> None:
    if not HAS_PLOT or not records:
        return
    n = len(records)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Building Record Cleaning Summary", fontsize=14, fontweight="bold")

    year_labels = ["numeric", "blank", "dash", "pre_1945", "other"]
    year_counts = [
        metrics.get("year_numeric", 0), metrics.get("year_blank", 0),
        metrics.get("year_dash", 0), metrics.get("year_pre_1945", 0), metrics.get("year_other", 0)
    ]
    axes[0, 0].bar(year_labels, year_counts, color=["#4CAF50", "#FF9800", "#9E9E9E", "#2196F3", "#F44336"])
    axes[0, 0].set_title("YearBuild Category Distribution")
    axes[0, 0].set_ylabel("Count")

    missing_labels = ["EN Name1", "ZH Name1", "Address1", "Owners", "Pub.Housing"]
    missing_values = [
        metrics.get("english_name1_missing_pct", 0),
        metrics.get("chinese_name1_missing_pct", 0),
        metrics.get("address1_missing_pct", 0),
        metrics.get("owners_blank_pct", 0),
        metrics.get("public_housing_blank_pct", 0),
    ]
    axes[0, 1].barh(missing_labels, missing_values, color="#5C6BC0")
    axes[0, 1].set_title("Missing / Blank Rates (%)")
    axes[0, 1].set_xlim(0, 100)

    confidences = [r.confidence for r in records]
    axes[1, 0].hist(confidences, bins=20, color="#26A69A", edgecolor="white")
    axes[1, 0].set_title("Confidence Score Distribution")
    axes[1, 0].set_xlabel("Confidence")

    llm_done = sum(1 for r in records if r.llm_applied)
    flagged_not_applied = sum(r.needs_llm and not r.llm_applied for r in records)
    axes[1, 1].pie(
        [n - flagged_not_applied - llm_done, flagged_not_applied, llm_done],
        labels=[f"Rules retained\n({n-flagged_not_applied-llm_done:,})",
                f"Flagged, rules retained\n({flagged_not_applied:,})", f"LLM applied\n({llm_done:,})"],
        autopct="%1.1f%%",
        colors=["#66BB6A", "#FFA726", "#EF5350"],
        startangle=90,
    )
    axes[1, 1].set_title("Routing & LLM Application")

    plt.tight_layout()
    plot_path = f"{output_prefix}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plots to: {plot_path}", file=sys.stderr)
    plt.close()

def write_csv(records: List[BuildingRecord], path: Path) -> None:
    if not records:
        return
    fieldnames = [
        "source_index", "english_name", "chinese_name", "chinese_name_simplified",
        "english_address", "chinese_address", "chinese_address_simplified",
        "year_build_raw", "year_category", "year_numeric",
        "owners_corporation", "public_housing_type_en", "public_housing_type_zh",
        "public_housing_type_zh_simplified", "confidence", "needs_llm", "issues",
        "input_tokens", "output_tokens", "llm_applied", "llm_confidence", "unresolved", "llm_rationale",
        "llm_status", "llm_origin", "llm_requested", "request_latency_seconds", "llm_error",
        "raw", "rule_values", "llm_proposal",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: getattr(rec, k, "") for k in fieldnames}
            row["issues"] = "|".join(rec.issues)
            row["year_numeric"] = rec.year_numeric or ""
            row["confidence"] = round(rec.confidence, 3)
            for name in ("raw", "rule_values", "llm_proposal"):
                row[name] = json.dumps(row[name], ensure_ascii=False)
            writer.writerow(row)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid rule + selective DeepSeek with plots")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url")
    source_group.add_argument("--file", type=Path)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--convert", choices=["none", "t2s", "hk2s", "s2t"], default="none")
    parser.add_argument("--archived-count", type=int, default=12421)
    parser.add_argument("--use-llm", action="store_true", help="Enable selective DeepSeek calls")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--indices", type=Path, help="JSON array of one-based XML record indices")
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--replay-only", action="store_true", help="Never call the API")
    parser.add_argument("--route", choices=["selective", "all"], default="selective")
    parser.add_argument("--max-llm-requests", type=int, help="Maximum live requests in this run")
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Maximum in-flight requests (default: 1)")
    parser.add_argument("--requests-per-minute", type=float,
                        help="Pace live requests across all workers")
    parser.add_argument("--floor-separators", action="store_true",
                        help="Use the development-derived rule for explicit floor sections")
    args = parser.parse_args()

    if args.limit < 0 or args.archived_count < 1:
        parser.error("Use a non-negative limit and a positive archived count")
    if args.max_llm_requests is not None and args.max_llm_requests < 0:
        parser.error("--max-llm-requests cannot be negative")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.requests_per_minute is not None and (
        not isfinite(args.requests_per_minute) or args.requests_per_minute <= 0
    ):
        parser.error("--requests-per-minute must be finite and positive")
    if args.replay_only and not (args.use_llm and args.cache_dir):
        parser.error("--replay-only requires --use-llm and --cache-dir")
    prices = (args.input_price_per_million, args.output_price_per_million)
    if (prices[0] is None) != (prices[1] is None) or any(
        p is not None and (p < 0 or not isfinite(p)) for p in prices
    ):
        parser.error("Provide both token prices as finite, non-negative numbers")
    indices = None
    if args.indices:
        if args.limit:
            parser.error("--indices and --limit cannot be combined")
        try:
            selected = json.loads(args.indices.read_text(encoding="utf-8"))
            if (not isinstance(selected, list) or not selected
                    or any(type(i) is not int or i <= 0 for i in selected)
                    or len(set(selected)) != len(selected)):
                raise ValueError("expected a nonempty array of unique positive integers")
            indices = set(selected)
        except (OSError, ValueError) as error:
            parser.error(f"Invalid --indices: {error}")
    if args.cache_dir and not args.replay_only:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not init_converter(args.convert):
        parser.error("--convert requires OpenCC or zhconv")

    client = None
    if args.use_llm and not args.replay_only and args.max_llm_requests != 0:
        if not HAS_OPENAI:
            print("Error: pip install openai", file=sys.stderr)
            sys.exit(1)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("Error: set DEEPSEEK_API_KEY environment variable", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com",
                        timeout=60.0, max_retries=0)
        if args.requests_per_minute is not None:
            client = PacedClient(client, args.requests_per_minute)

    source = args.url or str(args.file)
    source_hash = None
    if args.file:
        with args.file.open("rb") as xml_input:
            source_hash = hashlib.file_digest(xml_input, "sha256").hexdigest()
    print(f"Streaming records from: {source}", file=sys.stderr)
    start = time.perf_counter()

    last_requested_index = max(indices) if indices else None

    def selected_records():
        selected_count = 0
        for idx, raw in enumerate(stream_records(source), start=1):
            if indices is not None and idx not in indices:
                continue
            yield clean_record(raw, idx, floor_separators=args.floor_separators)
            selected_count += 1
            if selected_count % 200 == 0:
                print(f"  processed {selected_count:,} records", file=sys.stderr, flush=True)
            if (args.limit and idx >= args.limit) or idx == last_requested_index:
                break

    if args.use_llm:
        cleaned = process_llm_records(
            selected_records(), client, args.model, args.cache_dir,
            concurrency=args.concurrency, max_requests=args.max_llm_requests,
            replay_only=args.replay_only, route=args.route,
        )
    else:
        cleaned = list(selected_records())

    elapsed = time.perf_counter() - start
    print(f"Finished: {len(cleaned):,} records", file=sys.stderr)

    if not cleaned or (indices is not None and len(cleaned) != len(indices)):
        parser.error("XML did not contain all requested records")
    expected = len(indices) if indices else args.limit or args.archived_count
    metrics = evaluate(cleaned, expected, elapsed)
    metrics.update({
        "expected_records": expected,
        "xml_sha256": source_hash,
        "indices_sha256": hashlib.sha256(json.dumps(sorted(indices)).encode()).hexdigest()
                          if indices is not None else None,
        "mode": "replay" if args.replay_only else "hybrid" if args.use_llm else "rules",
        "floor_separators": args.floor_separators,
        "floor_rule_revision": FLOOR_RULE_REVISION if args.floor_separators else None,
        "route": args.route,
        "concurrency": args.concurrency if args.use_llm else 1,
        "requests_per_minute": args.requests_per_minute if args.use_llm else None,
        "model": args.model if args.use_llm else None,
        "prompt_version": PROMPT_VERSION if args.use_llm else None,
        "estimated_live_cost": (
            metrics["live_input_tokens"] * prices[0] +
            metrics["live_output_tokens"] * prices[1]
        ) / 1_000_000 if prices[0] is not None else None,
    })
    print_evaluation(metrics)

    output_path = Path(args.output)
    if args.format == "csv":
        write_csv(cleaned, output_path)
    else:
        with output_path.open("w", encoding="utf-8") as f:
            for rec in cleaned:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    print(f"Wrote {len(cleaned):,} records to {output_path}", file=sys.stderr)
    if args.metrics_json:
        args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        plot_summary(cleaned, metrics, str(output_path.with_suffix("")))

if __name__ == "__main__":
    main()
