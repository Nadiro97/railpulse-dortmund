# Milestone 4 — Improved September 2026 timetable-update classifier

**Result: increased detection of 6+ minute *archived timetable-update deviations*, with a meaningful precision / false-alert trade-off.**

This was evaluated on actual Deutsche Bahn historical timetable records (CC BY 4.0), independently archived by Piet Brömmel. It is **not** a validated physical train arrival delay prediction model.

[Successful reproducible GitHub Actions run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43779340881) · [Model evaluation source](../railpulse/ml_improved.py) · [Workflow](../.github/workflows/ml-improved.yml)

## Evaluation design

| Group | Dates | Records | Positive rate |
|---|---|---:|---:|
| Train | Sep 1–16 | 69,345 | 16.0848% |
| Validation (choose model & alert threshold) | Sep 17–20 | 16,510 | 15.0273% |
| Untouched test | Sep 21–30 | 43,635 | 16.0238% |

Target: explicitly recorded arrival update at least 6 minutes after the timetable. Only station name, train category, scheduled hour and weekday are features. We excluded time-change outcomes, snapshot timestamps, cancellation flags and downstream information. The fixed candidate models were logistic regression and histogram gradient boosting. Logistic regression was chosen on **validation average precision** (0.58515 vs 0.58118). Threshold was chosen on **validation F2**, requiring precision >=45%; selected threshold **0.225** for logistic regression. The held-out test was not used for selection.

## Held-out test comparison

| Metric | Previous logistic model, 0.50 threshold | Selected logistic model, 0.225 threshold | Boosting candidate, 0.25 threshold |
|---|---:|---:|---:|
| Recall ↑ | 32.5% | **57.895%** | 54.448% |
| Precision ↑ | **70.0%** | 45.948% | 48.814% |
| F1 ↑ | 0.444 | 0.51234 | 0.51477 |
| F2 ↑ | — | **0.55033** | 0.53219 |
| ROC AUC ↑ | 0.816 | 0.81656 | 0.81110 |
| Average precision ↑ | 0.535 | 0.53471 | 0.54037 |
| Brier ↓ | 0.10172 | 0.10198 | 0.10126 |

**The chosen model is logistic regression** based on validation (not on which candidate happened to perform best on test). Although boosting slightly outperformed on test average precision, Brier and F1, using those held-out scores to change the model now would be test-set overfitting.

The reported performance improvement mainly comes from a **lower decision threshold**, not from a new algorithm with substantially better ROC AUC. Relative to the old version, threshold tuning improves positive-event recall by approximately **25.4 percentage points**, while precision decreases by approximately **24.1 percentage points**.

### Test confusion matrix at threshold 0.225

| | Predicted below threshold | Predicted alert |
|---|---:|---:|
| No 6+ minute archived update | 31,881 TN | 4,762 FP |
| 6+ minute archived update | 2,944 FN | 4,048 TP |

At the old 0.50 threshold, there were around 975 false positives and 2,275 true positives. The new threshold catches more real positives **but also increases false alerts significantly**. Select a threshold based on the intended cost of missed updates versus unnecessary warnings, never accuracy alone.

### Interpretation limitations

- The target is the **latest explicitly observed historical API timetable update**, which can be a prediction, **not** a confirmed real-world arrival time.
- Records without explicit updates are omitted rather than classified as on time; the sample can be selection biased.
- Only **September 2026** was tested; this does **not** demonstrate robustness across other months or unusual events.
- Both models use the same limited pre-scheduled-time features. They do not explain *why* services are late or prove any delay reduction.
- Both models and the threshold policy were selected using the validation period. The test is evaluated only after model and threshold selection; future changes should be evaluated on fresh, later months rather than continually adjusting to September test data.
- The archived train categories may reflect route/operator mix; results should not be treated as a causal operator ranking.

## Reproduce

```bash
pip install -e '.[ml,dev]'
python -m railpulse download --month 2026-09
python -m railpulse extract --month 2026-09
python -m railpulse ml-improve --month 2026-09
```

GitHub Actions provides a downloadable, **aggregate-only** artifact containing `ml_improved_metrics.json` and `ml_improved_comparison.csv`. No row-level historical data or train-specific model artifacts are committed or uploaded.

**Licence:** Deutsche Bahn Timetables data, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); historical collection and processing by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). Independently transformed and analyzed for a portfolio, without Deutsche Bahn sponsorship.
