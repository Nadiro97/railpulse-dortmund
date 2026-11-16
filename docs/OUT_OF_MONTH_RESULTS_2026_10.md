# Milestone 5 — First out-of-month evaluation (October 2026)

**Completed with real source records on 16 November 2026 UTC.** [Successful GitHub Actions run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43961941340). Source: Deutsche Bahn railway timetable-update data under CC BY 4.0, historically archived by Piet Brömmel.

## Experiment was locked before October was evaluated

- **Model:** logistic regression, same fixed hyperparameters and pre-schedule features as our September validation experiment.
- **Training:** September 1–16, 2026, **69,345** eligible explicit arrival-update records.
- **Model choice:** logistic regression chosen using September 17–20 validation, without using October labels.
- **Threshold:** **0.225**, also chosen on September validation. Not changed for October.
- **Reference holdout:** September 21–30, 2026, **43,635** records.
- **Genuinely unseen month:** October 1–31, 2026, **122,882** eligible records.
- **Target:** a **retrospective explicitly recorded arrival-time update** >= 6 minutes later than planned. **Not confirmed physical train arrival delay.**

## Real-world month-to-month performance

| Metric | September holdout | Unseen October 2026 | Interpretation |
|---|---:|---:|---|
| Sample | 43,635 | 122,882 | October has a much larger sample |
| Positive rate | 16.02% | 14.80% | The monthly class balance shifted |
| ROC AUC ↑ | 0.817 | **0.769** | Ranking quality decreased |
| Average precision ↑ | 0.534 | **0.423** | Distinguishing positives became harder |
| Brier ↓ | 0.102 | **0.107** | Probability accuracy worsened somewhat |
| Accuracy at threshold .225 ↑ | 82.25% | **79.29%** | Not the primary goal |
| Precision at threshold .225 ↑ | 45.75% | **36.49%** | More false warnings per true warning |
| Recall at threshold .225 ↑ | 57.98% | **53.87%** | Still detects more than half of labelled events |
| F1 ↑ | 0.511 | **0.435** | Precision/recall balance deteriorated |

**Important:** The locked model did **not** generalize as well to October. Its October precision was **below the 45% constraint used when selecting the alert threshold on September validation**. That constraint was a September validation criterion, not a guarantee of future-month performance.

### October confusion matrix at locked threshold 0.225

| | Predicted no alert | Predicted alert |
|---|---:|---:|
| No 6+ minute explicit update | 87,641 TN | 17,054 FP |
| 6+ minute explicit update | 8,390 FN | 9,797 TP |

The positive count is **18,187**, and the model misses **8,390** of them. There are **17,054 false alerts**, meaning far more false positives than true positives. This can be too noisy for passenger-facing use.

### Reference benchmark

A constant-probability predictor trained on the early-September positive rate scored **ROC AUC 0.500** and **average precision 0.148** in October. It achieved **85.20% accuracy** by never flagging a late update at threshold 0.225—illustrating why accuracy alone is not an adequate metric for an early-warning system.

## Decision

- **Keep the September-selected logistic regression model as a reproducible research baseline**, not a production-ready passenger-warning system.
- **Do not tune the threshold using October test labels** after viewing the results. That would spoil this out-of-month holdout for future claims.
- Gather multiple future monthly holdouts, inspect update/coverage changes, and pre-register an operational cost of false alarms before considering a production alert threshold.
- If developing a revised model, tune using earlier months and evaluate on **another untouched later month**, not on the October set used here.

## Provenance and caveats

The model uses only `station_name`, `train_type`, `scheduled_hour`, and `day_of_week`. It does not use retrospective changes or snapshot timestamps as features. We selected only noncancelled, planned, nonreplacement/nonadditional Dortmund station stops with explicitly recorded arrival changes in the parsed archive and a plausible updated-minus-planned difference.

The archived update may still be a **forecast**, not a physical train arrival. Missing updates are unknown and excluded. The October comparison includes the autumn daylight-saving clock change; source times are Europe/Berlin naive. Route and train-type mix, collection gaps, and service changes can affect generalization.

No raw national data, stop-level rows, individual predictions or trained model files were committed to GitHub. The [workflow artifact](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43961941340) contains only `ml_cross_month_metrics.json` and `ml_cross_month_comparison.csv`, with seven-day retention.

Data attribution: underlying railway timetable updates © Deutsche Bahn under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), historical collection and processing by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). Filtered, matched and statistically summarized for this independent educational portfolio. No affiliation or endorsement.
