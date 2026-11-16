# Multi-month experiment — reduction of false alerts

## Question

Can a broader training history improve forecasts of 6+ minute **historical timetable updates** and reduce false warnings?

**Predeclared chronology:** train on July–August 2026, select logistic regression versus histogram gradient boosting and threshold using September 2026 validation, and compare the result with the previously published September-trained logistic baseline (threshold 0.225) on October 2026.

**October is NOT a new blind test.** Our previous model already used October as an evaluation month, and we have looked at those results. Therefore any improvement here is exploratory; **do not promote a new champion based solely on this comparison**. A later untouched complete month will be necessary for that decision.

The experiment requires minimum **55% precision on September validation**, compared with the earlier 45% constraint. Candidate alert thresholds span 0.10–0.90, in increments of 0.025. Select threshold by validation F1 subject to the minimum precision requirement. Select among feasible models by highest September validation average precision, using Brier score to break ties. If neither meets 55% validation precision, **no candidate is selected** rather than silently changing the requirement.

The October benchmark, after the choice has been frozen, reports: positive rate, sample size, ROC AUC, average precision, Brier score, thresholded precision/recall/F1, and false alerts per 1,000 stops.

## Run the experiment

```bash
pip install -e '.[ml,dev]'
python -m railpulse ml-multimonth
```

Or use the [GitHub Actions workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-multimonth.yml). It downloads four publicly licensed processed monthly archives and their parsed change snapshots into **ephemeral local storage**. It uploads only aggregate metrics, never raw national data, stop-level observations or trained model objects.

## Interpretation limits

An `arrival_change_time` update in the historical Deutsche Bahn archive is **not** proof that the train physically arrived at that timestamp: it can be a prediction. Records without an explicit update are excluded, and collection coverage may vary. The data have Europe/Berlin local timestamps and DST ambiguity. The model predicts an archival timetable-update property; it is not a passenger-facing operational predictor or a proven delay-reduction intervention.

Licence: underlying timetable data from Deutsche Bahn, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); historical collection and preprocessing by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). This is an independent educational analysis.
