# RailPulse Dortmund 🚆

**Data Engineering & Analytics of railway timetable updates and cancellations at Dortmund stations.**

Python · DuckDB · Parquet · SQL · Streamlit

> **Project status:** September 2026 source data has been processed through GitHub Actions. Read the [first verified descriptive findings](docs/RESULTS_2026_09.md). Updated times may reflect planned-time fallbacks, so these are **not verified actual arrival delays**. Deutsche Bahn rail records only, not DSW21 buses or Stadtbahnen.

## Quick start

Requires Python 3.10+ and disk space for one monthly Germany-wide Parquet file.

```bash
python -m venv .venv
# Activate virtualenv: source .venv/bin/activate (Linux/macOS)
# Windows: .venv\Scripts\Activate.ps1
pip install -e '.[dashboard,dev]'
python -m railpulse download --month 2026-09
python -m railpulse extract --month 2026-09
python -m railpulse analyze --month 2026-09
streamlit run dashboard/app.py
python -m pytest -q
```

Or `python -m railpulse run --month 2026-09` to run ingestion and reports.

The project **does not include train datasets**. Downloaded files are kept locally and ignored by Git. If a release is unavailable, choose an available `--month YYYY-MM`.

## Explicit update validation (second milestone)

**Verified September 2026 milestone:** 129,500 / 132,450 eligible Dortmund scheduled arrivals have at least one explicitly recorded arrival-time update in the parsed snapshots (**97.77% coverage**). These are timetable updates, **not confirmed actual train arrivals**. See [data-backed validation results](docs/EXPLICIT_UPDATES_RESULTS_2026_09.md) and the [successful GitHub Actions run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43643412457).



The processed monthly archive fills absent arrival changes with planned times. To identify **explicitly recorded arrival updates**, RailPulse now cross-checks stop IDs and EVA station codes against the original parsed change snapshots. This distinguishes rows with an observed updated timestamp from rows that cannot be verified. **Even explicitly recorded updates are not independently confirmed physical arrivals.**

```bash
# Once the processed September subset has been downloaded and extracted:
python -m railpulse validate-updates --month 2026-09
streamlit run dashboard/app.py
```

Run the [validation workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/validate-updates.yml) on GitHub to produce **aggregate-only** evidence without committing raw train records. See the [validation method](docs/EXPLICIT_UPDATES_VALIDATION.md). The dashboard displays these metrics in a separate section to avoid mistaking predicted timetable changes for verified punctuality.

## Experimental machine-learning baseline — Milestone 3

**First real test result:** September's leakage-aware logistic baseline achieved **ROC AUC 0.816**, **average precision 0.535** and **Brier 0.102** versus constant-prevalence ROC AUC 0.500 / average precision 0.160 / Brier 0.135. **Recall is only 32.5% at a 0.5 threshold**. [Read the evaluation and its limitations](docs/ML_RESULTS_2026_09.md); these labels are **retrospective timetable updates, not independently verified physical delays**.



A chronological logistic-regression baseline predicts **whether the latest explicitly recorded arrival-time update is 6+ minutes after schedule** using *only* station, train category, scheduled hour, and scheduled weekday. The constant-prevalence baseline provides a no-skill comparison. The first 20 days of a month train the model and days 21–end evaluate it; no future timetable updates are features.

**This is a timetable-update prediction, not independently verified actual train punctuality.** The sample includes only stops with explicit updates, so selection effects remain. See [full ML methodology](docs/ML_BASELINE.md) and the [reproducible ML workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-baseline.yml).

```bash
pip install -e '.[ml,dev]'
python -m railpulse ml-baseline --month 2026-09
```

The ML run writes only `reports/ml_baseline_metrics.json` and `reports/ml_baseline_comparison.csv`, never row-level records or model artifacts, to downloadable aggregate-only CI artifacts.

## Selected model for the project — recall-focused champion

**Current portfolio choice:** validation-selected **logistic regression with alert threshold 0.225** for identifying **explicit historical timetable arrival updates at least 6 minutes later than scheduled**. It is the best *evaluated* option **for our chosen goal of catching more potential late-update events while enforcing at least 45% precision on the validation split**. This is an experimental classifier, **not a deployed live arrival-delay model**.

**Held-out September 21–30 evaluation:** recall **57.9%**, precision **45.9%**, F1 **0.512**, ROC AUC **0.817**, and accuracy **82.3%**. The older threshold-0.50 logistic baseline had recall **32.5%**, precision **70%**, and accuracy **87.0%**. A lower threshold catches more potentially late updates but creates more false alerts. The 45% precision constraint was set on **validation**, not guaranteed for every later dataset.

We **retain both configurations** for reproducibility: the preferred model is generated by `python -m railpulse ml-improve --month 2026-09`; the earlier higher-precision comparison stays available through `python -m railpulse ml-baseline --month 2026-09`. The model and threshold were selected without looking at held-out test results. [Full comparison and limitations](docs/ML_IMPROVED_RESULTS_2026_09.md).

**Important:** the outcome is an **archived explicit API timetable update**, which may be a prediction, not a verified physical arrival. We will revalidate on later months before calling this a generalizable model.

## Out-of-month test — September-trained model → October 2026

**October 2026 external-month result:** The fixed September-trained model scored **ROC AUC 0.769, recall 53.9%, precision 36.5%, accuracy 79.3%** at the locked 0.225 threshold, compared with September **ROC AUC 0.817, recall 58.0%, precision 45.7%, accuracy 82.2%**. This is a **performance decline**, particularly in false alerts. [See the real out-of-month results](docs/OUT_OF_MONTH_RESULTS_2026_10.md). We retain this as a research benchmark, **not a production passenger-warning system**.

Our current alert policy is **locked**: logistic regression trained on **September 1–16**, chosen using the **September 17–20** validation period, and a fixed probability alert threshold of **0.225**. We now test it on **October 2026** without learning from October records. September 21–30 remains a reference holdout.

Reproduce with `python -m railpulse ml-cross-month --month 2026-09 --test-month 2026-10`, or run the [cross-month validation workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-cross-month.yml). All downloaded railway records are excluded from GitHub; the workflow uploads aggregated metrics only. See [methodology and limitations](docs/OUT_OF_MONTH_VALIDATION.md).

**Model outcome:** a retrospectively recorded explicit arrival-time update of **6 minutes or more**, **not** an independently confirmed physical arrival delay.

## Pipeline

`Hugging Face monthly_processed_data/*.parquet` → select Dortmund station records with DuckDB → validate missing values, cancellations and plan coverage → calculate planned-vs-changed arrival time differences → CSV summaries → local Streamlit dashboard.

**Caution:** `arrival_change_time` may reflect updated predictions rather than independently verified actual arrival times. Missing changes are *unknown*, not on time. Cancellation statistics are reported separately. Source times are local German wall times and need DST care.

## Data provenance and licences

Underlying rail timetable-update data from Deutsche Bahn, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Historical collection and processing by Piet Brömmel, [Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data) / [Hugging Face dataset](https://huggingface.co/datasets/piebro/deutsche-bahn-data). Filtered and aggregated for Dortmund. Independent educational project; no affiliation or endorsement.

The repository's **original code uses MIT**; the source dataset has its own CC BY 4.0 licence. See [DATA_SOURCES](docs/DATA_SOURCES.md), [METHODS_AND_LIMITATIONS](docs/METHODS_AND_LIMITATIONS.md) and [PRIVACY](docs/PRIVACY.md).

## Planned next work

1. Validate the September 2026 Dortmund subset and its coverage.
2. Compare stations, hours, train categories and cancellations.
3. Extend across months after quality checks.
4. Evaluate forecast baselines with strict time-based train/test splits.
5. Test operational scenarios without making unsupported causal claims.
