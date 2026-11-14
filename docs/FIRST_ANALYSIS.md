# First real Dortmund railway analysis

The first **September 2026** analysis was configured as a [GitHub Actions workflow](../.github/workflows/analyze.yml). The input source is only the documented Deutsche Bahn archive by Piet Brömmel (CC BY 4.0).

## How to run or rerun

1. Open [Actions → Analyze historical Dortmund train data](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/analyze.yml).
2. Click **Run workflow**, choose a processed release month (default: `2026-09`), and start the run.
3. If successful, open the completed run and download `dortmund-aggregates-YYYY-MM` under **Artifacts**.
4. Extract `quality.csv`, `station_quality.csv`, `by_station.csv`, `by_hour.csv`, `by_train_type.csv`, `by_weekday.csv`, `by_day.csv` and `provenance.json` to a local `reports/` directory.
5. Start `streamlit run dashboard/app.py` locally (after `pip install -e '.[dashboard,dev]'`).

For a local end-to-end run, use `python -m railpulse run --month 2026-09`.

## What is and is not published

The GitHub run downloads one national Parquet archive into **ephemeral runner storage** and filters the rows to Dortmund stations. **The original file and extracted stop-level data are never uploaded as artifacts or committed.** The artifact has **aggregated results only** plus provenance, with **7-day retention**. Public GitHub Actions artifacts can be accessible to other users.

These are **planned-vs-changed arrival time differences**; they are not independently validated actual train arrival times. Only rows with both timestamps and non-cancelled, non-replacement scheduled arrival records enter delay-summary comparisons. Cancellations, missing updated times, and incomplete plan records are shown separately in quality metrics.

The analysis includes a minimum station sample-size filter; changing that threshold can change the ranking. No inference of real delay causes from this dataset alone is justified.

## Interpreting the results

- `quality.csv`: coverage, cancellations and eligibility. Inspect this before using mean/median changes.
- `station_quality.csv`: completeness by station. Exclude low-coverage stations from comparisons.
- `by_station.csv`: station mean, median, p90 and share with updates 6 minutes or more after schedule, among eligible rows only.
- `by_train_type.csv`: categories with the same inclusion rule.
- `by_day.csv` / `by_hour.csv`: temporal patterns.
- `provenance.json`: source, licence, date, month and caveats.

If the job fails due to Hugging Face access, space, or download limits, no data-backed result is claimed. Rerun locally; never invent Dortmund statistics.

## Attribution

Rail timetable-update data: Deutsche Bahn, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Collected/processed by Piet Brömmel, [Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data); filtered and aggregated by this independent portfolio project.
