#!/usr/bin/env python3
"""
Hybrid Rule + Selective DeepSeek Cleaning with Plots
Only low-confidence records consume tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.request import urlopen
from xml.etree.ElementTree import iterparse

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
    r"\brd\.?\b": "Road", r"\bst\.?\b": "Street", r"\bave\.?\b": "Avenue",
    r"\bblvd\.?\b": "Boulevard", r"\bdr\.?\b": "Drive", r"\blane\b": "Lane",
    r"\bblk\.?\b": "Block", r"\bbl\.?\b": "Block", r"\bg/f\b": "G/F",
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
    text = clean_text(raw)
    if not text:
        return "blank", None
    if text == "-":
        return "dash", None
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

def clean_record(raw: Dict[str, str], index: int) -> BuildingRecord:
    rec = BuildingRecord(raw=raw.copy(), source_index=index)
    en_names = [clean_text(raw.get(f"EnglishBuildingName{i}")) for i in (1, 2, 3)]
    zh_names = [clean_text(raw.get(f"ChineseBuildingName{i}")) for i in (1, 2, 3)]
    en_addrs = [clean_text(raw.get(f"EnglishAddress{i}")) for i in (1, 2, 3)]
    zh_addrs = [clean_text(raw.get(f"ChineseAddress{i}")) for i in (1, 2, 3)]

    rec.english_name = concatenate_parts(en_names)
    rec.chinese_name = concatenate_parts(zh_names)
    rec.english_address = normalize_address(concatenate_parts(en_addrs, ", "))
    rec.chinese_address = concatenate_parts(zh_addrs)
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
    rec.needs_llm = rec.confidence < 0.7 or bool(rec.issues)
    return rec

def call_deepseek(rec: BuildingRecord, client: OpenAI, model: str = "deepseek-chat") -> None:
    prompt = f"""You are cleaning a bilingual Hong Kong building record.
Raw fields:
{json.dumps(rec.raw, ensure_ascii=False, indent=2)}

Current rule-based result:
English name: {rec.english_name}
Chinese name: {rec.chinese_name}
English address: {rec.english_address}
Chinese address: {rec.chinese_address}
Year category: {rec.year_category}
Issues: {rec.issues}

Return STRICT JSON only with these keys:
{{
  "english_name": "string or null",
  "chinese_name": "string or null",
  "english_address": "string or null",
  "chinese_address": "string or null",
  "rationale": "short explanation",
  "confidence": "high|medium|low",
  "unresolved": true/false
}}

Rules:
- Do NOT invent a name or address if the raw record has no supporting evidence.
- Prefer "unresolved": true when evidence is insufficient.
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        rec.input_tokens = usage.prompt_tokens if usage else 0
        rec.output_tokens = usage.completion_tokens if usage else 0
        content = response.choices[0].message.content
        data = json.loads(content)
        if data.get("english_name"):
            rec.english_name = data["english_name"]
        if data.get("chinese_name"):
            rec.chinese_name = data["chinese_name"]
        if data.get("english_address"):
            rec.english_address = data["english_address"]
        if data.get("chinese_address"):
            rec.chinese_address = data["chinese_address"]
        rec.llm_rationale = data.get("rationale", "")
        rec.llm_confidence = data.get("confidence", "")
        rec.unresolved = bool(data.get("unresolved", False))
        rec.llm_applied = True
    except Exception as e:
        rec.llm_rationale = f"LLM error: {e}"
        rec.unresolved = True

def evaluate(records: List[BuildingRecord], archived_count: int, elapsed: float) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    bilingual_name_ok = sum(1 for r in records if (r.english_name and r.chinese_name) or (not r.english_name and not r.chinese_name))
    bilingual_addr_ok = sum(1 for r in records if (r.english_address and r.chinese_address) or (not r.english_address and not r.chinese_address))
    year_anomaly = sum(1 for r in records if r.year_category in {"blank", "dash", "other", "pre_1945"})
    records_per_min = (n / elapsed * 60) if elapsed > 0 else 0
    routed_ratio = sum(1 for r in records if r.needs_llm) / n
    total_input_tokens = sum(r.input_tokens for r in records)
    total_output_tokens = sum(r.output_tokens for r in records)
    llm_calls = sum(1 for r in records if r.llm_applied)

    return {
        "name_bilingual_consistency": bilingual_name_ok / n,
        "address_bilingual_consistency": bilingual_addr_ok / n,
        "year_anomaly_count": year_anomaly,
        "records_per_minute": records_per_min,
        "routed_record_ratio": routed_ratio,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "llm_calls": llm_calls,
        "total_records": n,
        "elapsed_seconds": elapsed,
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
    print("EVALUATION REPORT (Hybrid Selective Routing)", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(f"Total records                 : {metrics.get('total_records', 0):,}", file=sys.stderr)
    print(f"Name bilingual consistency    : {metrics.get('name_bilingual_consistency', 0):.4f}", file=sys.stderr)
    print(f"Address bilingual consistency : {metrics.get('address_bilingual_consistency', 0):.4f}", file=sys.stderr)
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
    fig.suptitle("Hybrid Cleaning Summary (Selective LLM)", fontsize=14, fontweight="bold")

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

    needs = sum(1 for r in records if r.needs_llm)
    llm_done = sum(1 for r in records if r.llm_applied)
    axes[1, 1].pie(
        [n - needs, needs - llm_done, llm_done],
        labels=[f"Rule-only\n({n-needs:,})", f"Flagged\n({needs-llm_done:,})", f"LLM applied\n({llm_done:,})"],
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
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: getattr(rec, k, "") for k in fieldnames}
            row["issues"] = "|".join(rec.issues)
            row["year_numeric"] = rec.year_numeric or ""
            row["confidence"] = round(rec.confidence, 3)
            writer.writerow(row)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid rule + selective DeepSeek with plots")
    parser.add_argument("--url")
    parser.add_argument("--file")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--convert", choices=["none", "t2s", "hk2s", "s2t"], default="none")
    parser.add_argument("--archived-count", type=int, default=12421)
    parser.add_argument("--use-llm", action="store_true", help="Enable selective DeepSeek calls")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error("Either --url or --file must be provided")

    init_converter(args.convert if args.convert != "none" else "none")

    client = None
    if args.use_llm:
        if not HAS_OPENAI:
            print("Error: pip install openai", file=sys.stderr)
            sys.exit(1)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("Error: set DEEPSEEK_API_KEY environment variable", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    source = args.url or args.file
    print(f"Streaming records from: {source}", file=sys.stderr)
    start = time.perf_counter()

    cleaned: List[BuildingRecord] = []
    for idx, raw in enumerate(stream_records(source), start=1):
        rec = clean_record(raw, idx)
        if args.use_llm and rec.needs_llm and client is not None:
            call_deepseek(rec, client, args.model)
        cleaned.append(rec)
        if args.limit and idx >= args.limit:
            break
        if idx % 200 == 0:
            print(f"  processed {idx:,} records...", file=sys.stderr)

    elapsed = time.perf_counter() - start
    print(f"Finished: {len(cleaned):,} records", file=sys.stderr)

    metrics = evaluate(cleaned, args.archived_count, elapsed)
    print_evaluation(metrics)

    output_path = Path(args.output)
    if args.format == "csv":
        write_csv(cleaned, output_path)
    else:
        with output_path.open("w", encoding="utf-8") as f:
            for rec in cleaned:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    print(f"Wrote {len(cleaned):,} records to {output_path}", file=sys.stderr)
    plot_summary(cleaned, metrics, output_path.stem)

if __name__ == "__main__":
    main()
