# First observed Dortmund summary — September 2026

**Status: real source data processed successfully by the [GitHub Actions analysis run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43539495458).** This page uses aggregated outputs only; no raw or per-train data is committed. Public analysis is **descriptive** and cannot be interpreted as confirmed physical arrival punctuality.

## Extracted data quality

| Metric | September 2026 |
|---|---:|
| Railway-station-stop records selected for Dortmund | 155,800 |
| Distinct station names | 44 |
| Source rows with planned arrivals | 141,266 |
| Arrival-cancellation flagged records | 7,396 |
| Rows with no matching plan record | 2,108 |
| Rows flagged as replacement trains | 1,671 |
| Eligible rows for planned-vs-processed-time-difference summaries | 132,448 |
| Duplicate stop IDs found by the extraction | 0 |

**Critical interpretation:** The source [monthly-release processing code](https://github.com/piebro/deutsche-bahn-data/blob/main/scripts/create_monthly_data_release.py) computes

`COALESCE(f.arrival_change_time, p.arrival_planned_time, f.arrival_planned_time)`

as the published `arrival_change_time`. Therefore the resulting 100% scheduled-arrival timestamp presence in the processed release is a **processing fallback**, **not** proof that live updates were observed or that those trains were on time. We must not describe it as update observation coverage. To measure actual update availability, examine the separately published parsed plan/change snapshots.

## Planned-versus-processed updated arrival differences

For 132,448 eligible records, the count-weighted approximate mean time difference is **3.43 minutes**, and approximately **15.58%** have a processed arrival time at least 6 minutes after the planned arrival (calculated from the rounded category aggregate results, so approximate).

These are **not official Deutsche Bahn punctuality statistics** and should not be compared as if they were verified train arrivals. A zero difference can simply reflect fallback to planned time.

### Dortmund stations (selected examples)

Only stations with at least 1,000 eligible records are displayed.

| Station | Eligible rows | Mean changed-vs-planned (min) | Median (min) | Share 6+ min |
|---|---:|---:|---:|---:|
| Dortmund-Kurl | 2,167 | 10.81 | 5 | 47.49% |
| Dortmund-Scharnhorst | 1,708 | 9.30 | 4 | 41.39% |
| Dortmund Hbf | 17,890 | 9.25 | 4 | 40.11% |
| Dortmund-Mengede | 5,089 | 6.34 | 3 | 34.70% |
| Dortmund-Derne | 2,614 | 5.13 | 3 | 30.99% |
| Dortmund Universität | 3,465 | 1.98 | 1 | 7.33% |

Station comparisons can be affected by train category mix, routes, missing API observations, substitutions, and other operational factors. The results establish no causes of delay.

### Selected train categories

| Train type | Eligible records | Mean difference (min) | Share 6+ min |
|---|---:|---:|---:|
| S (S-Bahn) | 85,089 | 1.83 | 8.35% |
| RB | 20,778 | 3.18 | 16.55% |
| ERB | 13,749 | 3.61 | 19.25% |
| NX | 6,857 | 14.34 | 63.26% |
| RE | 2,037 | 5.84 | 33.04% |
| ICE | 2,270 | 23.33 | 79.78% |

Train-type groups reflect operators/services in the processed data and should not be interpreted as a controlled or like-for-like performance ranking. Lower-n categories have higher sampling uncertainty.

## Time and coverage caveats

The daily summary contains **8 eligible records with planned arrival dated 2026-08-31** because the archive release is partitioned by a processed `time` field; arrivals can fall across month boundaries. Do not present the 8 August records as a typical September day. German local timestamps are timezone naive, so further analysis must handle DST carefully.

This one month is not sufficient for seasonality, causal attribution, event impacts, or predictions. Before proposing delay interventions, validate the upstream change history using `monthly_processed_data_change` and `monthly_processed_data_plan`.

## Source and licence

- Underlying rail timetable-update records: Deutsche Bahn, **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.
- Historical archive and preprocessing: Piet Brömmel, [Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data) and [Hugging Face archive](https://huggingface.co/datasets/piebro/deutsche-bahn-data).
- Transformations performed: filter to Dortmund stations; select and qualify fields; compute arrival differences; group by station, hour, day, and train category.
- Executed: **2026-11-14 UTC**. Month: **2026-09**.
- Aggregates available in the [workflow artifact](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43539495458) (7-day retention).

Not affiliated with, sponsored by, or endorsed by Deutsche Bahn.
