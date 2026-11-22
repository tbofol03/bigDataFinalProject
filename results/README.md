# Measured three-scale results

All routed requests have provider responses.

| Records | Routed | Applied | Request errors | Processing seconds (sum) | Estimated CNY | Exact addresses | Nonempty exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 149 | 37 | 0 | 936.95 | 0.833774 | 599/600 | 505/506 (99.80%) |
| 5,000 | 1,654 | 370 | 0 | 868.68 | 8.719451 | 599/600 | 505/506 (99.80%) |
| 12,421 | 4,217 | 897 | 0 | 4837.14 | 22.147108 | 594/600 | 500/506 (98.81%) |

Every scale is scored against the same 300 XML-grounded reference records. These are a stratified, difficulty-enriched sample, not an estimate of population-wide accuracy. Exact agreement includes punctuation and whitespace and counts verified empty fields; the nonempty denominator is 506.

The original rules match 560/600 address fields, or 466/506 nonempty fields (92.09%). This study measures source-based formatting agreement, not physical address correctness. The component audit normalizes whitespace, commas and documented English abbreviations. Its labels are based on XML comparison; they are not independent human or registry validation. No loss was found in the scored door-number, street and floor components. There are no scorable block/phase examples in this sample.

The 500-record run used one worker; the 5,000-record run used 32 workers followed by an 8-worker, 60-request-per-minute recovery; the full run used 8 workers and 60 requests per minute. Thinking settings were preserved. The summed processing times include failed attempts but exclude time between runs and output writing. They are not matched-concurrency scaling benchmarks.

Costs sum the recorded live-request estimates across the initial and recovery runs. Reused cached tokens are counted once. The recorded price basis is CNY 1 per million input tokens and CNY 4 per million output tokens. These estimates cover reported usage, not a provider invoice.

The development-derived floor-separator rule baseline reaches 598/600 address fields (504/506 nonempty) without API calls. The v2 guard was inspected after holdout use and must be described as a diagnostic. The LLM improves the original baseline but does not establish superiority over the enhanced rules. Differences between repeated model responses at different scales show residual formatting variability.

The duplicate-pair audit contains 48 supported negative pairs and 24 unresolved pairs; there are no confirmed positives, so duplicate recall and F1 cannot be established.

## Reproduction

From the repository root: `python finalize_results.py` (no API calls).

From the ZIP, reproduce field scoring with:

```powershell
python src/experiment.py score --labels inputs/labels_holdout_reviewed.csv --baseline runs/rules-full.jsonl --hybrid runs/resumed-c8-rpm60-full.jsonl --out scores-rerun.json
```

The ZIP includes measured outputs, reference labels, component audits and the scorer source. It excludes API keys and response caches. Earlier measured files remain unchanged.
