# Milestone 4: Improve the timetable-update classifier without test leakage

## Target and caveat
The model labels a scheduled arrival **positive** if the latest explicitly recorded Deutsche Bahn arrival timetable update is at least 6 minutes later than planned. It is **not** a verified physical-train-arrival delay target.

## Strict time split and evaluation

- Train: 1–16 September 2026.
- Validate: 17–20 September 2026.
- Untouched test: 21–30 September 2026.
- Compare predeclared logistic regression and histogram gradient boosting using validation average precision (Brier tie-break).
- Pick the classification threshold **on validation** by maximizing F2 (recall emphasized), constrained to at least 45% validation precision, with an explicit fallback if no threshold satisfies it.
- Once both model and threshold are frozen, assess test performance, including ROC AUC, average precision, Brier, precision, recall, F1, F2, and confusion matrices. No test-set tuning.
- Both models use only station name, train type, scheduled hour and scheduled weekday (attributes known ahead of the scheduled arrival).
- A constant-prevalence baseline is retained for comparison. Realized change times, API snapshots, cancellations and retrospective delays are never model inputs.

## Run

```bash
pip install -e '.[ml,dev]'
python -m railpulse download --month 2026-09
python -m railpulse extract --month 2026-09
python -m railpulse ml-improve --month 2026-09
```

Alternatively run [the automated GitHub Actions job](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-improved.yml).

Only two **aggregate** files are made public as an artifact: `ml_improved_metrics.json` and `ml_improved_comparison.csv`. No underlying Parquet, train-stop table or trained model is published.

The next serious research milestone is validating performance on **later months**, since a split within one month only demonstrates limited temporal generalization. Improving recall increases false-alarm risk; report precision jointly.

Source attribution: Deutsche Bahn timetable-update data, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); historical archive by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). Independent educational analysis.
