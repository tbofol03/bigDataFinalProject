"""Audit the measured runs and assemble reproducible, source-based results.

This command never calls a model API. Request failures remain visible in every
summary; a completed cache-resume run is used when one exists.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import address_review
import experiment
from formal_run import SIZES, verify_inputs
import source_review


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measured_chain(folder, size, digest, expected_ids):
    initial = sorted(folder.glob(f"live*-{size}.json"))
    resumed = sorted(folder.glob(f"resumed*-{size}.json"), key=lambda p: p.stat().st_mtime_ns)
    if len(initial) != 1:
        raise ValueError(f"Expected one original measured run for {size}")
    chain = initial + resumed
    reports = []
    previous = None
    expected_hash = hashlib.sha256(json.dumps(sorted(expected_ids)).encode()).hexdigest()
    for position, path in enumerate(chain):
        report = json.loads(path.read_text(encoding="utf-8"))
        records = experiment.load_runs(path.with_suffix(".jsonl"))
        if set(records) != expected_ids or report["total_records"] != len(records):
            raise ValueError(f"Record IDs do not match the frozen workload: {path}")
        if report["xml_sha256"] != digest or report["indices_sha256"] != expected_hash:
            raise ValueError(f"The run uses different input data: {path}")
        if dict(Counter(r["llm_status"] for r in records.values())) != report["llm_status_counts"]:
            raise ValueError(f"Run status counts do not match the output: {path}")
        for token in ("input_tokens", "output_tokens"):
            if sum(r[token] for r in records.values()) != report["total_" + token]:
                raise ValueError(f"Token counts do not match the output: {path}")
        if position == 0 and report["llm_cache_hits"]:
            raise ValueError("The original measured run must have an empty cache")
        if previous:
            if report["llm_calls"] != previous["llm_status_counts"].get("request_error", 0):
                raise ValueError("A recovery did not request exactly the previously failed records")
        previous = report
        reports.append(report)
    return chain, reports, records


def component_reviews(indices, baseline, hybrid, references, raw, target):
    # Keep output variants anonymous and reuse only labels for identical text.
    known = {r["variant_id"]: r for r in source_review.load_rows(references)}
    rows = {}
    for index, left, right in address_review.paired_rows(indices, baseline, hybrid):
        for field, prefix in address_review.LANGUAGES.items():
            for item in (left, right):
                value = item[field]
                code = address_review.variant_id(index, field, value)
                row = {"variant_id": code, "source_index": str(index), "field": field,
                       "raw_parts": json.dumps(source_review.parts(raw[index], prefix), ensure_ascii=False),
                       "address_to_review": value, "review_status": "pending"}
                if code in known:
                    old = known[code]
                    if any(row[k] != old[k] for k in ("source_index", "field", "raw_parts", "address_to_review")):
                        raise ValueError(f"Reference text changed for address variant {code}")
                    row = old
                else:
                    row = source_review.blind_row(row, raw[index])
                    if row["review_note"].startswith("Source and displayed output differ"):
                        raise ValueError(f"Additional source-component inspection required for {code}")
                rows[code] = row
    write_csv(target, [rows[code] for code in sorted(rows)])


def finalize(workspace, *, inputs=None, runs=None, output=None, xml=None,
             labels=None, component_labels=None, supporting=None, bundle=None):
    source = Path(__file__).resolve().parent
    inputs = inputs or workspace / "inputs"
    runs = runs or workspace / "runs"
    output = output or workspace / "results"
    xml = xml or workspace / "data/bnb-u.xml"
    labels = labels or inputs / "labels_holdout_reviewed.csv"
    component_labels = component_labels or inputs / "component_reference.csv"
    supporting = supporting or workspace / "supporting"
    bundle = bundle or workspace / "experiment_results.zip"
    output.mkdir(parents=True, exist_ok=True)
    digest, manifests, holdout = verify_inputs(xml, inputs)
    holdout_path = inputs / "indices_holdout.json"
    if not holdout_path.exists():
        holdout_path = runs / "indices_holdout.json"
    if sorted(json.loads(holdout_path.read_text(encoding="utf-8"))) != holdout:
        raise ValueError("The review manifest must match the frozen holdout")
    raw = source_review.records(xml)
    references = source_review.load_rows(labels)
    if len(references) != 300 or {int(r["source_index"]) for r in references} != set(holdout):
        raise ValueError("The same 300 reference records must be used at every scale")
    summaries, artifacts, failures = [], [labels, holdout_path], []
    nonempty = sum(bool(r["expected_" + f]) for r in references for f in address_review.LANGUAGES)
    for size in SIZES:
        chain, reports, records = measured_chain(runs, size, digest, manifests[size][1])
        baseline = runs / f"rules-{size}.jsonl"
        rules = experiment.load_runs(baseline)
        if rules.keys() != records.keys() or any(
            rules[i]["raw"] != records[i]["raw"] or records[i]["raw"] != raw[i] for i in records
        ):
            raise ValueError("The source fields differ between runs or from the XML")
        current = chain[-1].with_suffix(".jsonl")
        fields_path = output / f"scores-{size}.json"
        components_path = output / f"components-{size}.json"
        variants_path = output / f"address-review-{size}.csv"
        experiment.score(labels, baseline, current, fields_path)
        component_reviews(holdout_path, baseline, current, component_labels,
                          raw, variants_path)
        address_review.score(holdout_path, baseline, current, variants_path,
                             components_path, "source-fidelity")
        scores = json.loads(fields_path.read_text(encoding="utf-8"))
        exact = sum(scores["fields"][f]["hybrid_correct"] for f in address_review.LANGUAGES)
        nonempty_exact = sum(records[int(r["source_index"])][f] == r["expected_" + f]
                             for r in references for f in address_review.LANGUAGES if r["expected_" + f])
        failed = [r for r in records.values() if r["llm_status"] == "request_error"]
        for r in failed:
            failures.append({"size": size, "source_index": r["source_index"],
                             "error": r.get("llm_error", ""), "result_file": current.name})
        summary = {
            "scale": size, "records": len(records),
            "status": "complete" if not failed else "needs_request_recovery",
            "routed_records": sum(r["needs_llm"] for r in records.values()),
            "initial_requests": reports[0]["llm_calls"],
            "requests_across_runs": sum(r["llm_calls"] for r in reports),
            "provider_responses": sum(r["needs_llm"] for r in records.values()) - len(failed),
            "remaining_request_errors": len(failed),
            "applied": reports[-1]["llm_applied_count"],
            "rejected_proposals": sum(reports[-1]["llm_status_counts"].get(s, 0)
                                      for s in ("invalid_schema", "unsupported_change")),
            "initial_seconds": reports[0]["elapsed_seconds"],
            "processing_seconds_across_runs": sum(r["elapsed_seconds"] for r in reports),
            "estimated_cost_cny": sum(r["estimated_live_cost"] for r in reports),
            "live_input_tokens_across_runs": sum(r["live_input_tokens"] for r in reports),
            "live_output_tokens_across_runs": sum(r["live_output_tokens"] for r in reports),
            "english_address_exact": scores["fields"]["english_address"]["hybrid_correct"],
            "chinese_address_exact": scores["fields"]["chinese_address"]["hybrid_correct"],
            "address_exact_including_empty": exact, "address_fields_including_empty": 600,
            "nonempty_address_exact": nonempty_exact, "nonempty_address_fields": nonempty,
            "nonempty_exact_percent": 100 * nonempty_exact / nonempty,
            "holdout_request_errors": scores["hybrid_statuses"].get("request_error", 0),
            "result_file": current.name, "measured_reports": [p.name for p in chain],
        }
        summaries.append(summary)
        artifacts.extend([baseline, baseline.with_suffix(".json"), *chain,
                          *(p.with_suffix(".jsonl") for p in chain)])
    dump(output / "summary.json", {
        "all_requests_completed": not failures, "xml_sha256": digest,
        "scoring_scope": "Formatting agreement with 300 XML-grounded references; no external registry validation",
        "cost_basis": "Sum of recorded estimated_live_cost; no duplicate charging of cached tokens",
        "scales": summaries, "remaining_failures": failures,
    })
    write_csv(output / "scaling_results.csv", summaries)
    dump(output / "remaining_failures.json", failures)
    table = ["| Records | Routed | Applied | Request errors | Processing seconds (sum) | Estimated CNY | Exact addresses | Nonempty exact |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        table.append(f"| {s['records']:,} | {s['routed_records']:,} | {s['applied']:,} | "
                     f"{s['remaining_request_errors']} | {s['processing_seconds_across_runs']:.2f} | "
                     f"{s['estimated_cost_cny']:.6f} | {s['address_exact_including_empty']}/600 | "
                     f"{s['nonempty_address_exact']}/{nonempty} ({s['nonempty_exact_percent']:.2f}%) |")
    complete = not failures
    lines = ["# Measured three-scale results", "",
             "All routed requests have provider responses." if complete else
             f"Request recovery is still required for {len(failures)} records. Scores below include their rule fallbacks.",
             "", *table, "",
             "Every scale is scored against the same 300 XML-grounded reference records. These are a stratified, "
             "difficulty-enriched sample, not an estimate of population-wide accuracy. Exact agreement includes "
             "punctuation and whitespace and counts verified empty fields; the nonempty denominator is 506.", "",
             "The original rules match 560/600 address fields, or 466/506 nonempty fields (92.09%). "
             "This study measures source-based formatting agreement, not physical address correctness. "
             "The component audit normalizes whitespace, commas and documented English abbreviations. "
             "Its labels are based on XML comparison; they are not independent human or registry validation. "
             "No loss was found in the scored door-number, street and floor components. There are no "
             "scorable block/phase examples in this sample.", "",
             "The 500-record run used one worker; the 5,000-record run used 32 workers followed by an "
             "8-worker, 60-request-per-minute recovery; the full run used 8 workers and 60 requests per minute. "
             "Thinking settings were preserved. The summed processing times include failed attempts but "
             "exclude time between runs and output writing. They are not matched-concurrency scaling benchmarks.", "",
             "Costs sum the recorded live-request estimates across the initial and recovery runs. Reused cached "
             "tokens are counted once. The recorded price basis is CNY 1 per million input tokens and CNY 4 "
             "per million output tokens. These estimates cover reported usage, not a provider invoice.", "",
             "The development-derived floor-separator rule baseline reaches 598/600 address fields "
             "(504/506 nonempty) without API calls. The v2 guard was inspected after holdout use and must "
             "be described as a diagnostic. The LLM improves the original baseline but does not establish "
             "superiority over the enhanced rules. Differences between repeated model responses at different "
             "scales show residual formatting variability.", "",
             "The duplicate-pair audit contains 48 supported negative pairs and 24 unresolved pairs; there "
             "are no confirmed positives, so duplicate recall and F1 cannot be established.", "",
             "## Reproduction", "",
             "From the repository root: `python finalize_results.py` (no API calls).", "",
             "From the ZIP, reproduce field scoring with:", "",
             "```powershell",
             f"python src/experiment.py score --labels inputs/labels_holdout_reviewed.csv --baseline runs/rules-full.jsonl --hybrid runs/{summaries[-1]['result_file']} --out scores-rerun.json",
             "```", "",
             "The ZIP includes measured outputs, reference labels, component audits and the scorer source. "
             "It excludes API keys and response caches. Earlier measured files remain unchanged.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    # Assemble an evidence package rather than reporting from summaries alone.
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.glob("*.py")):
            archive.write(path, "src/" + path.name)
        for path in sorted(source.glob("tests/*.py")):
            archive.write(path, "src/tests/" + path.name)
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, "results/" + path.name)
        for path in dict.fromkeys(artifacts):
            destination = "inputs/" + path.name if path in (labels, holdout_path) else "runs/" + path.name
            archive.write(path, destination)
        for relative in ("scores-original-vs-enhanced-v1.json",
                         "scores-original-vs-enhanced-v2.json", "duplicate_scores.json"):
            path = supporting / relative
            if path.exists():
                archive.write(path, "supporting/" + path.name)
    print(json.dumps({"all_requests_completed": complete,
                      "remaining_request_errors": len(failures),
                      "full_address_exact": summaries[-1]["address_exact_including_empty"],
                      "summary": str(output / "summary.json"), "bundle": str(bundle)}, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    for flag in ("inputs", "runs", "out", "xml", "labels", "component-labels", "supporting", "bundle"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    options = {k: v.resolve() for k, v in vars(args).items() if k != "workspace" and v is not None}
    if "out" in options:
        options["output"] = options.pop("out")
    finalize(args.workspace.resolve(), **options)


if __name__ == "__main__":
    main()
