# Multi-month model diagnostics — October 2026 (exploratory)

**Status: successful real-data experiment**, [GitHub Actions run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43964271353). The training and benchmark figures below come from the aggregate-only `railpulse-multimonth-diagnostics` workflow artifact. Source: Deutsche Bahn historical timetable updates, CC BY 4.0, archived by Piet Brömmel.

## Chronology and selection
- **Training:** July and August 2026, **243,594** explicitly updated eligible stop records.
- **Model selection and threshold:** September 2026, **129,490** records. Predeclared selection rule: minimum 55% validation precision, then choose the best validation F1 alert threshold, and prefer the candidate with highest validation average precision.
- **Benchmark:** October 2026, **122,882** eligible records. **Important:** October had already been used in earlier work, so this is **not a fresh blind test** and cannot establish a definitive improvement.

The validation-picked candidate was **logistic regression** with probability threshold **0.325**. On September validation it achieved **55.04% precision** and **38.66% recall**. Histogram gradient boosting was also tested, but logistic regression had slightly higher validation average precision (0.48088 versus 0.47625).

## October exploratory comparison

| Metric | Existing model: logistic, 0.225 | Candidate: multi-month logistic, 0.325 | Boosted candidate, 0.375 |
|---|---:|---:|---:|
| Precision ↑ | 36.49% | **47.25%** | 50.34% |
| Recall ↑ | **53.87%** | 36.74% | 32.57% |
| F1 ↑ | **0.435** | 0.413 | 0.395 |
| ROC AUC ↑ | 0.76887 | 0.76960 | 0.76999 |
| Average precision ↑ | 0.42315 | 0.42118 | 0.42435 |
| Brier ↓ | 0.10724 | 0.10626 | 0.10617 |
| False alerts | 17,054 | **7,460** | 5,844 |
| False alerts / 1,000 eligible stops | 138.78 | **60.71** | 47.56 |

The logistic candidate reduces false alerts by about **56%** (9,594 fewer), but correctly flags fewer 6+ minute update records: **6,682 vs 9,797**. The stronger precision is largely associated with a more conservative decision threshold. The probability-ranking metrics barely change.

## Conclusion

**Do not replace the current experimental model yet.** The candidate may be useful when false warnings are costly, but it sacrifices recall. It needs an **untouched later month** to support a fair model-selection decision. October is no longer eligible as a fresh blind test, and November 2026 is not complete on the experiment date.

These models predict **explicit archived timetable updates** at least 6 minutes after schedule, **not actual independently verified train arrivals**. Missing change snapshots are excluded, and differences in route mix and source coverage can shift estimates.

## Reproducibility and privacy

Run `python -m railpulse ml-multimonth` after installing `pip install -e '.[ml,dev]'`, or use the [manual GitHub Actions workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-multimonth.yml). Raw Parquet files and row-level records stay on an ephemeral runner and are never uploaded to the repository; the artifact holds only aggregate CSV and JSON metrics.

**Attribution:** Underlying data: Deutsche Bahn, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Historical collection and processing: [Piet Brömmel / Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data). Independently filtered and aggregated for the RailPulse Dortmund portfolio.
