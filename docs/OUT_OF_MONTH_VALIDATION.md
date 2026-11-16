# Milestone 5 — Test our locked model on a genuinely unseen month

**Decision frozen before this test:** Logistic regression, trained on **September 1–16, 2026**, using station name, train category, scheduled arrival hour and weekday; warn for probabilities >= **0.225**. Chosen on September 17–20 validation. We evaluate September 21–30 for reference and **October 1–31, 2026** as the unseen month.

This is a test of how consistently the model predicts **historically recorded updated timetable times** (at least 6 minutes later than scheduled). It is **not** independently verified physical-train arrival prediction.

## Reproduce without distributing row-level data

```bash
pip install -e '.[ml,dev]'
python -m railpulse ml-cross-month --month 2026-09 --test-month 2026-10
```

This automatically downloads the September/October processed sources, extracts Dortmund locally, and fetches both months' parsed change snapshots. It trains only on the early-September records using the locked logistic-regression configuration, then evaluates September holdout and all October data. **It does not retrain using October or choose the threshold using October labels.**

See the [cross-month GitHub Actions workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-cross-month.yml) and its aggregate-only artifacts.

## Results and interpretation

The CSV summarizes both months, with sample size, positive prevalence, accuracy, precision, recall, F1, F2, ROC AUC, average precision, Brier score and confusion matrix at the fixed 0.225 threshold. A constant-prevalence comparison is also included. If the test fails or October data are not yet published, no generalization result is claimed.

Important limitations: (1) updates can be forecasts, not actual arrivals; (2) no-update records are excluded, potentially introducing selection bias; (3) October includes the German DST clock transition and naive local source times; (4) changes to collection coverage or train mix may influence observed metrics; (5) this experiment tests retrospective API updates and does not establish intervention benefits.

Data licence: Deutsche Bahn [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); historical acquisition and processing by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). No direct affiliation. Raw Parquet and per-stop records never leave the ephemeral runner in the GitHub workflow.
