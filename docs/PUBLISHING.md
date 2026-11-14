# Safe GitHub publication

This repository contains source code, documentation, tests and an empty local-data pipeline only.

## Local use

```bash
pip install -e '.[dashboard,dev]'
python -m railpulse download --month 2026-09
python -m railpulse extract --month 2026-09
python -m railpulse analyze --month 2026-09
streamlit run dashboard/app.py
```

## Before sharing

Check `git status`, `git ls-files` and commit diffs. Never commit:
- `data/raw/`, `data/processed/`, `reports/`
- Parquet files, DuckDB databases, credentials, `.env` files
- row-level licensed records copied into notebooks or test fixtures

Synthetic data in tests is clearly artificial. Public results should contain source attribution to Deutsche Bahn / Piet Brömmel, method notes, source month and date retrieved. The MIT code licence does not relicense source data.

## Roadmap

Validate September 2026 coverage, add more months, investigate service reliability, build prediction baselines with time-based validation and benchmark proposed interventions. Never claim causality based only on timetable updates.
