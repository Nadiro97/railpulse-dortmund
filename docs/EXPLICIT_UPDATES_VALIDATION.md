# Validation milestone: distinguish explicit timetable updates from fallback values

## Why this is necessary

The archive's monthly processed release fills absent changes with planned times using `COALESCE(f.arrival_change_time,p.arrival_planned_time,f.arrival_planned_time)`. Therefore a zero-minute difference in the processed release **does not prove a train arrived on time**.

The independent [parsed change snapshots](https://huggingface.co/datasets/piebro/deutsche-bahn-data) include one row per stop and API snapshot. We now validate whether a **non-null arrival change timestamp** was explicitly recorded in that feed.

## Reproduce the audit

In [GitHub Actions](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/validate-updates.yml), run **Validate explicit Deutsche Bahn arrival updates** with month `2026-09`, or run locally:

```bash
pip install -e '.[dashboard,dev]'
python -m railpulse download --month 2026-09
python -m railpulse extract --month 2026-09
python -m railpulse validate-updates --month 2026-09
```

The audit downloads the same month's parsed `monthly_processed_data_change/data-YYYY-MM.parquet`. Data remain in `data/raw/` and `data/processed/`, both excluded from Git.

## Matching and caveats

- Join parsed-change snapshots to processed Dortmund records on **(stop ID, EVA station code)**; no names-only joins.
- For each stop, retain the **latest snapshot with a non-null explicit arrival change**; do not deduce actual physical arrival.
- Only scheduled, non-cancelled, non-additional, non-replacement September arrivals with a plan record enter the coverage denominator.
- A record with an explicit update equal to the planned time is **observed**, but not evidence of a physically on-time train.
- A record without explicit update is **unknown**, not a punctual train.
- Distinguish records with no change snapshot at all from records with change snapshots but no explicit arrival timestamp.
- Calculate station, train-category and hour summaries only for the explicit update group, and separately report its **coverage fraction**. Never generalize this subset to all trains without identifying selection bias.
- Compare processed arrival times against latest explicitly observed arrival-time updates and report disagreement.
- Observation time is the API request snapshot, **not train arrival time**. The source cannot independently certify actual arrival.

## Outputs

`reports/validated_quality.csv`, `validated_by_station.csv`, `validated_by_train_type.csv`, `validated_by_hour.csv`, `validated_provenance.json`. These are aggregates only. Action artifacts retain them 7 days. Source records and stop-level tables must **never** be uploaded.

## Licensing

Deutsche Bahn Timetables data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Historical archive curated by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). The repository's original code uses MIT. Independent learning/portfolio analysis, without operator endorsement.
