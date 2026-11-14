# September 2026: Explicit arrival-update validation

**Validation status: succeeded** — [GitHub Actions run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43643412457), executed 2026-11-14 UTC. All figures below were computed from real Deutsche Bahn timetable snapshots published in Piet Brömmel's archive. No raw or per-stop observations were committed to this repository.

## Research question

**Which published arrival timestamps reflect an explicit record in the parsed change snapshots, rather than being a fallback to the originally scheduled arrival time?**

The upstream monthly processed data use `COALESCE(f.arrival_change_time, p.arrival_planned_time, f.arrival_planned_time)`. A zero-minute difference there can mean there was no explicit observed change. We independently checked the parsed `monthly_processed_data_change/data-2026-09.parquet` source against the processed Dortmund subset using stop `id` **and** `eva`, selecting the latest non-null explicit arrival update by `snapshot_timestamp`.

## Sample and completeness

| Metric | September 2026 |
|---|---:|
| Processed Dortmund station-stop records | 155,800 |
| Stations | 44 |
| Eligible scheduled arrival rows | **132,450** |
| Eligible records with at least one parsed change snapshot | 129,538 |
| Eligible records with explicitly recorded arrival-time update | **129,500** |
| Eligible records without an explicit arrival-time update | **2,950** |
| Of these, without any matching change snapshot | 2,912 |
| Explicit updates passing sanity range [-60, 240] minutes | 129,490 |
| Explicit updates that disagreed with the processed arrival time | 0 |
| Arrival-cancellation-flagged records (all Dortmund stop rows) | 7,396 |
| **Explicit update coverage, among eligible rows** | **97.77%** |
| No explicit update, among eligible rows | 2.23% |

The remaining 38 records without explicit arrival times have some matching change snapshot information. These distinctions are important: **no update is unknown, not evidence of zero delay**.

The two analyses use slightly different eligible samples due to explicit September planned-arrival bounds and upstream release boundaries; do not mix denominators.

## Ranking by observed planned-vs-updated arrival difference

These are descriptive **timetable-update differences**, not verified physical arrival delays or officially published punctuality figures. The station examples below have at least 1,000 explicit eligible updates.

| Station | Explicit updates | Update coverage | Mean difference | Median difference | 6+ min share within explicit updates |
|---|---:|---:|---:|---:|---:|
| Dortmund-Kurl | 2,130 | 98.34% | 10.96 min | 5 min | 48.26% |
| Dortmund Hbf | 17,136 | 95.79% | 9.65 min | 4 min | 41.86% |
| Dortmund-Scharnhorst | 1,671 | 97.89% | 9.44 min | 4 min | 42.25% |
| Dortmund-Mengede | 4,969 | 97.66% | 6.49 min | 3 min | 35.52% |
| Dortmund-Derne | 2,516 | 96.25% | 5.33 min | 3 min | 32.19% |
| Dortmund-Hörde | 4,416 | 97.76% | 3.09 min | 1 min | 14.86% |

### Train-category comparisons

| Train type | Explicit updates | Update coverage | Mean difference | 6+ min share |
|---|---:|---:|---:|---:|
| ICE | 2,269 | 100.00% | 23.32 min | 79.77% |
| NX | 6,853 | 99.99% | 14.31 min | 63.26% |
| RE | 2,028 | 99.56% | 5.87 min | 33.19% |
| ERB | 13,481 | 98.00% | 3.68 min | 19.62% |
| RB | 20,427 | 98.31% | 3.23 min | 16.83% |
| S-Bahn (S) | 83,599 | 98.25% | 1.86 min | 8.50% |

These train categories serve different routes and market segments, so **this is not a controlled operator performance ranking**.

## What was established, and what wasn't

**Established:** The processed release's zero-time fallback was real; the archive contains explicit arrival updates for 97.77% of our eligible September records; the 129,500 explicit updates agree with the processed arrival timestamps; differences can be computed for observed updates and reported with station- and service-specific coverage.

**Not established:** That a train actually arrived at the updated time, that a missing update means on-time arrival, why any particular update was made, or that an operational intervention would reduce delays.

The parsed feed is sampled periodically; snapshot timestamps represent the API request time. It can miss intermediate changes, and explicit updates can remain predictions. This must be addressed before using this work as a true train punctuality benchmark or as machine-learning labels.

## Reproduce without sharing source records

Use the [explicit-update validation workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/validate-updates.yml) or run `python -m railpulse validate-updates --month 2026-09` after the first extraction. The [successful workflow run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43643412457) contains a 7-day artifact named `dortmund-explicit-updates-2026-09` with aggregate CSVs and provenance only.

### Licence and attribution

Underlying timetable updates: **Deutsche Bahn**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); historical dataset collected and processed by **Piet Brömmel**, [Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data) / [Hugging Face archive](https://huggingface.co/datasets/piebro/deutsche-bahn-data). We filtered Dortmund station stops, matched parsed change events by `(id,eva)`, selected latest explicit arrival updates, and aggregated the results. Independent educational project without Deutsche Bahn endorsement.
