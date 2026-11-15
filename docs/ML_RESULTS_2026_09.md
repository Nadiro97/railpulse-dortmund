# Milestone 3 — September 2026 temporal ML baseline results

**Source:** September 2026 historical Deutsche Bahn timetable updates in [Piet Brömmel's archive](https://github.com/piebro/deutsche-bahn-data). **Reproducible CI run:** [GitHub Actions #43776022230](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43776022230). The model ran on actual archived records, not fabricated or mock observations. Only aggregate metrics were retained.

## Research question

Using only station name, train type, scheduled local arrival hour, and weekday, can we identify planned train stops whose **latest explicitly recorded arrival-time update** is **6 minutes or more later** than the planned arrival? This is a **retrospective timetable-update prediction task** and **NOT an independently confirmed physical-arrival delay forecast**.

## Temporal evaluation design

| Partition | Scheduled arrival dates | Records | ≥6-minute update fraction |
|---|---|---:|---:|
| Training | 2026-09-01 to 2026-09-20 | 85,855 | 15.8814% |
| Held-out test | 2026-09-21 to 2026-09-30 | 43,635 | 16.0238% |

All rows use the original explicitly recorded arrival-time change from the parsed change snapshots, matched by train stop ID and EVA station ID. Cancellations, replacement trains and additional stops are removed; only explicit timestamps with a difference in [-60,240] minutes are eligible. Stop-level data are not published. The time-sorted split prevents future calendar days from appearing in the training set, although it does **not** independently certify the physical-arrival labels.

## Test results

| Metric | Constant train-prevalence baseline | Logistic regression |
|---|---:|---:|
| ROC AUC ↑ | **0.500** | **0.816** |
| Average precision / PR AUC ↑ | 0.160 | **0.535** |
| Brier probability score ↓ | 0.13456 | **0.10172** |
| Accuracy ↑ | 83.98% | **86.96%** |
| Precision, threshold = 0.5 ↑ | 0% | **70.0%** |
| Recall, threshold = 0.5 ↑ | 0% | **32.54%** |
| F1, threshold = 0.5 ↑ | 0 | **0.444** |

### Test-set confusion matrix

| | Predicted under 6 min | Predicted 6+ min |
|---|---:|---:|
| Observed explicit update under 6 min | 35,668 (TN) | 975 (FP) |
| Observed explicit update 6+ min | 4,717 (FN) | 2,275 (TP) |

**Interpretation:** The simple model improves ranking and probabilistic metrics over a constant prediction. But with the fixed 0.5 decision threshold, **it misses around 67% of positive timetable-update events**. Accuracy alone is misleading because roughly 84% of test records do not have a 6+ minute explicit update.

### Scope and limitations

- The target is **an archived updated timetable arrival**, not an independently verified physical arrival time. API updates may be predictions.
- Only records with explicit updates are modeled. The prior September validation found 97.77% explicit-update coverage in eligible scheduled arrivals; the uncovered subset cannot be assumed on-time, and selection bias may remain.
- Model inputs are limited to attributes known ahead of arrival: station, train type, scheduled hour and scheduled weekday. No event-time changed timestamp, future snapshot information, or target is used as a feature.
- Only **one month** is studied, and the train and test sets share station and service categories. Results should not be presented as performance on different months or unseen stations.
- No hyperparameter tuning was performed using the test set. No deployed live forecasts or intervention outcomes are claimed.
- Service categories and station routes are confounded. The model does not establish *why* an update occurred or that any action reduces delays.
- Local German timestamps have DST caveats; September is outside the fall clock-change transition.

## Reproduce

Run the [ML workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-baseline.yml) or follow [the setup instructions](ML_BASELINE.md). The [successful run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43776022230) provides a **seven-day aggregate-only artifact** with `ml_baseline_metrics.json` and `ml_baseline_comparison.csv`.

### Attribution and licence

Underlying timetable data: **Deutsche Bahn** under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), historical acquisition and processing by **Piet Brömmel**, [Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data) / [Hugging Face](https://huggingface.co/datasets/piebro/deutsche-bahn-data). Filtered to Dortmund and processed into an independent educational model. No Deutsche Bahn sponsorship or endorsement.
