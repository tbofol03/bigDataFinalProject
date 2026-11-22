# Bilingual Building Data Cleaning

Rule-based cleaning with selective DeepSeek assistance for complex bilingual
building addresses. The pipeline preserves source records, rule outputs, model
proposals and final values. Changes must pass source-evidence validation before
they are accepted. Failed requests and rejected proposals retain the rule values.

## Repository layout

```text
*.py           Cleaning, sampling, evaluation and result assembly
tests/         Request limits, concurrency, cache and rate-limit checks
data/          Archived XML snapshot
inputs/        Frozen workload manifests and source-based reference labels
runs/          Original measured outputs and recovery outputs
results/       Scores, component audits and the three-scale summary
supporting/    Enhanced-rule and duplicate-pair comparisons
```

Python 3.11 or newer is required. Offline cleaning and scoring use the standard
library. `requirements.txt` records the locally used versions of the API SDK and
optional plotting library. API calls require `DEEPSEEK_API_KEY` in the environment.

## Completed results

All three workloads have completed, with zero remaining request errors.
Each scale is scored against the same 300 XML-grounded reference records.

| Records | Model-routed records | Applied proposals | Nonempty address exact agreement |
|---:|---:|---:|---:|
| 500 | 149 | 37 | 505/506 (99.80%) |
| 5,000 | 1,654 | 370 | 505/506 (99.80%) |
| 12,421 | 4,217 | 897 | 500/506 (98.81%) |

The original rule baseline matches 466/506 nonempty address fields (92.09%).
Agreement here measures source-based formatting, including spaces and commas.
The difficulty-enriched reference sample does not estimate population accuracy.
The component audit checks preservation of explicit source components; it is not
an external address-registry check. Details and timing/cost limitations are in
[results/README.md](results/README.md).

The enhanced floor-separator rule baseline reaches 504/506 nonempty fields.
The v2 guard was inspected after holdout use and is a diagnostic. The experiment
does not establish that the LLM outperforms enhanced rules. Duplicate-pair data
contains no confirmed positives, so duplicate recall and F1 are not established.

## Reproduce results without API calls

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python finalize_results.py
```

The second command verifies XML and workload hashes, checks source-record
alignment and request/recovery accounting, recomputes the three-scale scores,
and produces `experiment_results.zip`. It does not call the model API.

Reproduce just the full-workload field score:

```bash
python experiment.py score --labels inputs/labels_holdout_reviewed.csv --baseline runs/rules-full.jsonl --hybrid runs/resumed-c8-rpm60-full.jsonl --out scores-rerun.json
```

Reproduce the address-component audit:

```bash
python address_review.py score --indices inputs/indices_holdout.json --baseline runs/rules-full.jsonl --hybrid runs/resumed-c8-rpm60-full.jsonl --labels results/address-review-full.csv --protocol source-fidelity --out components-rerun.json
```

Reproduce the duplicate-pair score:

```bash
python duplicate_review.py score --xml data/bnb-u.xml --labels supporting/duplicate_pairs_reviewed.csv --out duplicate-rerun.json
```

An empty expected field is a verified empty value; `__SKIP__` excludes a field
from scoring. Source-based labels are identified as such. Independently collected
labels should record their own review protocol and adjudication status.

## Run new experiments

Create new manifests in a separate directory when intentionally changing the
sample. Keep the supplied frozen manifests for comparisons with measured runs.

```bash
python experiment.py sample --xml data/bnb-u.xml --out new-inputs --seed 42
python importData.py --file data/bnb-u.xml --format jsonl -o new-rules.jsonl --metrics-json new-rules.json --no-plot
```

Install the optional live dependencies and set the API key in your own terminal.
The example below starts a new workload and incurs API usage:

```bash
python -m pip install -r requirements.txt
python importData.py --file data/bnb-u.xml --indices inputs/indices_500.json --format jsonl -o new-hybrid.jsonl --metrics-json new-hybrid.json --use-llm --model deepseek-flash --cache-dir runs/new-cache-500 --max-llm-requests 149 --concurrency 8 --requests-per-minute 60 --no-plot
```

Concurrent requests preserve the original API arguments and default thinking
settings. Rate-limit responses trigger a shared cooldown. Successful responses
are cached; failed calls remain visible and are not silently retried. Use
`--replay-only` with an existing matching cache for offline response replay.
Caches are not included in this repository package.

`formal_run.py` verifies frozen inputs and orchestrates matched workloads; its
help lists the required paths and explicit request/pricing limits. Live timing
requires a fresh cache. Recovery timing includes cached work and must be reported
separately. The original 32-worker run and its paced recovery are both retained.

The dataset source is the [Hong Kong Rating and Valuation Department XML feed](https://www.rvd.gov.hk/datagovhk/bnb-u.xml).
The archived snapshot hash is recorded in `inputs/metadata.json`. Keep original
measured outputs intact when changing code or comparing future experiments.
