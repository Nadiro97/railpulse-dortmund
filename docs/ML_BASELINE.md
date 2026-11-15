# Milestone 3 — Chronological machine-learning baseline

## Goal

Predict whether the **latest explicitly recorded timetable arrival update** for a Dortmund train stop is **six minutes or more later than the planned arrival time**. This is an **API timetable-update prediction** task. We have not observed independently confirmed physical arrivals, so this is **not** a reliable claim of train punctuality prediction.

## Reproducibility

```bash
pip install -e '.[ml,dev]'
python -m railpulse download --month 2026-09
python -m railpulse extract --month 2026-09
python -m railpulse ml-baseline --month 2026-09
```

Or use [GitHub Actions](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-baseline.yml). Download its summary-only artifact and inspect `ml_baseline_metrics.json` and `ml_baseline_comparison.csv`. Raw national Parquet, Dortmund rows, and trained model objects are **not uploaded or committed**.

## Design

- Labels: explicit arrival-update difference >= 6 min, with sanity limits [-60,240] min.
- Rows: joined on `(id,eva)` to the latest **non-null explicit change snapshot**. Non-cancelled scheduled arrivals with matching plan, excluding additional and replacement stops.
- Model inputs restricted to attributes expected before the scheduled arrival: **station name, train type, scheduled hour, scheduled weekday**. No changed arrival time, recorded delay, snapshot time, target, or cancellation flag as predictive input.
- Historical split: **1st–20th of the selected month to train; 21st–end to test**. No random row split.
- Baselines: train prevalence (constant probability for all stops) and logistic regression with one-hot station/type features.
- Metrics: ROC AUC, average precision (PR AUC), Brier score, accuracy, precision, recall, F1 and confusion matrix at probability threshold 0.5.
- No hyperparameter optimization on test data, station leakage across time splits, or cherry-picked test outcome.

## Known limitations

1. The target is an **updated timetable value**, not confirmed physical arrival.
2. Availability of explicitly recorded updates is **selection-biased**. Prior September audit found 97.77% of eligible rows with one or more explicit arrival changes; those without are excluded, not assigned on-time.
3. A latest retrospective update may differ from what could have been known *before* the event. We do **not** include it as a feature, but the target remains retrospective and may still be a prediction.
4. A one-month split is illustrative, not rigorous seasonality/generalization evidence. Extend across multiple months before using this for operational recommendations.
5. Different services and stations have differing coverage and route structure. Performance is not causality and does not demonstrate reduced delay.
6. Timestamps are Europe/Berlin naive local time in the underlying archive. Calendar features use scheduled local time.
7. No personal data is needed, no raw licensed records are pushed, and model outputs include licence/source attribution.

Underlying timetable data: Deutsche Bahn under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), archived by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). Independent educational project.
