# RailPulse Dortmund 🚆

**Data Engineering & Analytics of railway timetable updates and cancellations at Dortmund stations.**

Python · DuckDB · Parquet · SQL · Streamlit

> **Project status:** Starter pipeline. No real delay results or models are claimed until a real monthly dataset has been downloaded and validated. Deutsche Bahn rail records, not DSW21 bus or Stadtbahn records.

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
