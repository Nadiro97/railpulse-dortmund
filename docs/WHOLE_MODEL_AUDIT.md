# Whole-model reliability review — all train categories and Dortmund stations

**Status:** Diagnostic audit. The existing logistic regression is an experimental **archival timetable-update classifier**, *not* a reliable model of independently measured real train arrival delays. Good software tests or an attractive AUC score do not imply real-world punctuality accuracy.

## What the audit covers

1. **Target integrity:** The label is the most recent explicitly recorded updated arrival timestamp at least six minutes after the planned timestamp. It is not proof a train actually arrived at that time. Zero-minute processed records can be imputed; missing explicit updates are unknown.
2. **Population and exclusions:** Non-cancelled planned Dortmund arrivals with explicit update timestamps and plausible differences are eligible. Cancellations, replacement or additional stops, missing records, and unobserved updates are outside the current classifier, so a single score cannot describe all railway travel.
3. **Leakage and timeline:** Training remains limited to September 1–16, 2026, using station, train category, scheduled hour, and scheduled weekday only. We evaluate on September 21–30 and October 1–31 without changing model or threshold. October is already inspected; its results are exploratory.
4. **All cohorts:** Audit **every station and every train category** with at least 100 eligible records, including the non-S-Bahn services. Report predicted-versus-observed archival event rate, ranking, Brier score, precision, recall, and false alerts per 1,000 eligible stops. Smaller cohorts are omitted and counted separately.
5. **Calibration:** Compare scores in 10% bands with the actual *archived label* frequency, looking for systematic over- and underestimation. Even good archive-label calibration would not imply calibration to physical arrival times.
6. **Observation coverage:** Report eligible scheduled arrivals, how often snapshots and explicit updates are missing, cancellation counts, and update disagreement with processed timestamps by station and train type.
7. **Time-bound prediction readiness:** An important unresolved issue is whether the latest snapshot is available *before a real passenger decision*. The existing label uses the latest retrospectively recorded value, so we cannot call this a timely operational forecast without a defined prediction-time cutoff and further data verification.

## Reproduce

From the repo in a local Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,ml,dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m railpulse audit-model
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The audit reuses local September and October sources if already downloaded, otherwise fetches only the vetted licensed archive files. Processing may need several GB of storage. The [manual GitHub workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/model-audit.yml) can produce the same aggregate-only outputs using temporary runner storage.

Files produced under `reports/`: `model_audit_summary.json`, `model_audit_cohorts.csv`, `model_audit_calibration.csv`, and `model_audit_coverage.csv`. Per-train rows and learned coefficients are never uploaded to GitHub.

## Before we rebuild a real-world delay model

We need an outcome that matches the passenger experience and an honest definition of what is being predicted (for instance, actual arrival at the platform at least 3 or 6 minutes after the scheduled arrival, and cancellations separately). That requires **independently trustworthy actual-arrival observations**, a strictly timestamped as-of prediction cutoff, sufficient coverage across all relevant services, time-ordered multi-month evaluation, model calibration and error audits, and a completely fresh final evaluation period.

Until those conditions are established, we will **not retune thresholds to make numbers look better** or promote the present archival classifier as an actual-delay prediction system.

Historical archive: Deutsche Bahn timetable updates (CC BY 4.0), collected and processed by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). Independent educational work, not affiliated with Deutsche Bahn.
