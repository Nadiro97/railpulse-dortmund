# Methods and limitations

## Observational unit
One archive-processed railway stop, not one passenger. The SQL filter `station_name ILIKE 'Dortmund%'` identifies named Dortmund station stops; verify station coverage after extraction.

## Time-change terminology
The archived `arrival_change_time` may contain updated forecast or reported changes and does not independently prove physical arrival time. Define:

`changed_arrival_delta_min = (arrival_change_time - arrival_planned_time) / 60 seconds`.

Do not equate that automatically with measured actual delay. The `delay_in_min` field may be based on departure where arrival is unavailable; analytics use the explicit arrival-time difference instead.

## Inclusion/exclusion
Only analyze records with planned and changed arrival timestamps, non-cancelled arrivals, known plan record (where available), and not additional/replacement stops. Restrict summary values to [-60, 240] minutes as an explicit quality threshold. Missing change time is unknown, never zero delay. Cancellations are counted separately, not included as late arrivals.

## Temporal coverage
All source timestamps are Europe/Berlin naive local time. Daylight saving transitions require care. The archive documents missing collection blocks in earlier months. Compare months only after reporting coverage and schema changes.

## Causality
This dataset does not identify why delays occurred. Weather, construction, traffic and staffing effects cannot be asserted without independently licensed covariates and appropriate study design. Recommendations are hypothetical until tested.

## Important upstream fallback discovered during validation (2026-11-14)

The monthly processed release creates `arrival_change_time` with
`COALESCE(f.arrival_change_time, p.arrival_planned_time, f.arrival_planned_time)`.
This is documented in [the archive's processing code](https://github.com/piebro/deutsche-bahn-data/blob/main/scripts/create_monthly_data_release.py).
Consequently, a zero difference **may mean no observed change update**, rather than a confirmed on-time arrival. Timestamp presence (often 100%) is not observation completeness. Results are descriptive processed timetable-update differences only. For separately observable change snapshots, inspect the parsed `monthly_processed_data_change` dataset in future work.
