# Rebuild — evidence-first Dortmund minutes estimates (research, NOT proven real arrivals)

## Decision and most important source-level finding

We audited the actual upstream processing source in [piebro/deutsche-bahn-data/scripts/create_monthly_data_release.py](https://github.com/piebro/deutsche-bahn-data/blob/main/scripts/create_monthly_data_release.py). Its processed `arrival_change_time` and `departure_change_time` use `COALESCE(f.change_time, p.planned_time, f.planned_time)` and its `delay_in_min` is a planned-vs-updated timestamp difference, not a station-verified physical measurement. A source XML `ct` is a *changed timetable timestamp* and `pt` is planned ([parser](https://github.com/piebro/deutsche-bahn-data/blob/main/scripts/parse_monthly_xml.py)). Consequently, claiming the existing project predicts actual real-world Verspätung would be unjustified.

Our new target is **nonnegative minutes of explicit post-scheduled-event timetable change at Dortmund**. This is still a *proxy*, not a verified physical arrival/departure; even after the planned event, an API change can be an unconfirmed forecast.

## What we changed in the data model

- The time reference is always a **Dortmund railway stop**, and arrival and departure are **distinct tasks**. When the user selects Dortmund → Düsseldorf, model **departure from Dortmund**, not arrival at Düsseldorf. It is NOT a station-to-station travel-time forecast.
- Use the published `monthly_processed_data_plan` and `monthly_processed_data_change` **snapshot Parquet** records joined by stop id and EVA, together with the locally extracted Dortmund scheduled stops. Never train on the processed `delay_in_min` field.
- At the prediction cutoff, use only the latest **planned snapshot at least 60 minutes before scheduled Dortmund time**. This bounds availability of the planned route and train information. Do not pull route features from later changed snapshots.
- For historical *outcome proxies*, require a non-null explicit `<arrival|departure>_change_time` in a snapshot from **15 minutes to 9 hours after the scheduled Dortmund event**. This does not verify that the train arrived physically. Missing records remain UNKNOWN and are excluded from model fitting; monthly coverage is reported.
- Exclude cancellations, replacement/additional stops, missing plan snapshots, and out-of-plausibility-range deltas. Track counts and selection attrition, rather than treating exclusions as punctual.
- Reject differences outside -20 to +180 minutes and cap large positive *proxy labels* at 120 minutes; count capped records. Exclude ambiguous local times in the 02:00 autumn DST transition hour. This means the model cannot claim to predict disruptions beyond the training cap.
- Keep the Dortmund stop's **route position** when the upstream `train_line_station_num` exists. The **first listed upstream station** in the arrival plan path is a partial-path indicator, NOT necessarily the true train origin; the final listed downstream station is a path endpoint, NOT evidence that it serves arbitrary intermediate cities. Record unknowns explicitly.
- `Bus` in Deutsche Bahn timetable labels does **not** imply broad coverage of local city buses. Do not present forecasts for unsupported vehicles.

## Feature engineering (inputs from planned snapshots only)

Categoricals: **Dortmund station, train type, line number, train number, first listed previous station, last listed following station**.

Numerics: **planned hour, weekday, ordered route-stop index, upstream and downstream path lengths**, plus **weekend flag, rush-hour flag, cyclic sine/cosine hour, total route hops, progress fraction, and origin/destination availability indicators**. Unknown fields remain distinct; unknown routes are not invented.

Do not use a late update itself, a retrospective delay calculation, cancellation flag, future snapshot, or observed label as an input.

## Chronological evaluation, no artificial performance claims

| Period | Purpose | Allowed for model selection? |
|---|---|---|
| July–August 2026 | Fit model and feature encoding | Training only |
| September 2026 | Choose winning model based on predeclared metric | Yes |
| October 2026 | Previously inspected benchmark diagnostics | **No** |
| Later complete and untouched month | Future honest out-of-month verification | Only after candidate frozen |

Candidates: trivial all-zero, training mean and median, simpler station/category ridge, route-aware ridge, and feature-engineered target-encoded histogram gradient boosting. Target encoding uses internal cross-fitting on training data to reduce leakage; September and October labels are not available during fitting. No imputed punctual stops.

**Primary metric: weighted mean absolute error (minutes)**, with recorded proxy delays weighted 1× for under 6 minutes, 3× for 6–14 minutes, and 5× for 15+ minutes. Ordinary MAE, RMSE, MAE on 6+/15+, share of forecasts within 5 minutes, false-**Pünktlich** rate for 15+ events, and precision of punctual claims are reported. This guards against the model winning by always predicting zero.

**Hard quality gates** before *any* user-facing result:
- Algorithm must beat the always-zero weighted-MAE baseline by at least 3% on September.
- On September, no more than half of the 15+ minute archival update events may be called punctual and at least 80% of punctual claims must correspond to under 3-minute archived updates.
- At least 50% of eligible stops in every evaluated month (training, validation and already-inspected October diagnostics) must have a usable post-event proxy label and early planned snapshot.
- The inspected October model weighted MAE must also beat the trivial zero-prediction baseline, and its 15+ minute missed-as-punctual rate must be at most 60%. **This is a conservative veto, not a way to choose a model on October.**
- At least 20 earlier records must match the chosen station, category, line, upstream route indicator, and destination. Otherwise: **Keine belastbare Schätzung**.
- These are **provisional research safeguards**, not evidence of physical arrival reliability. Passing them does not establish real-world accuracy.

The UI no longer shows a decorative percent as the primary outcome. It says **Pünktlich**, **Voraussichtliche Verspätung: X Minuten**, or **Keine belastbare Schätzung** and shows held-out errors. A score cannot turn a timetable-change proxy into a proven real-world delay.

## Reproducible experiment

Run locally on Windows after cloning the project and setting up Python:

```powershell
cd C:\Users\znadi\railpulse-dortmund
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,ml,dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m railpulse train-evidence --mode both
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

**Important:** Four months of Germany-wide plan + change + processed source files are large. The command will download source archives to ignored local folders; allow sufficient disk space. Trained `.joblib` files stay in ignored `models/`, not in Git or uploaded workflow artifacts. Only load **your own locally trained** joblib file, never an untrusted download (joblib deserialization can execute code). Aggregate metrics are in `reports/evidence_minutes_{arrival,departure}_evaluation.json`. The [manual CI workflow](../.github/workflows/evidence-minutes.yml) uploads only aggregates; it may be unsuitable for standard GitHub-hosted runners if the source archives exhaust their disk.

## What must happen before calling this a real delay predictor

1. Obtain independent, trustworthy **actual arrival and actual departure observations**, preferably with a recorded verification state and timestamp. Source update timestamps alone are not enough.
2. Measure coverage across all services (especially cancellations, no updates, train category, rare routes, cross-day services and DST). Document major missingness and possible bias.
3. Define the prediction as-of time relative to Dortmund (e.g. one hour before) and ensure all features are available by then.
4. Run genuine future time holdouts not used in design choices; calibrate intervals and monitor changes by station, line and direction.
5. Only then consider passenger-facing claims. Never silently change the ground-truth definition to make MAE smaller.

Underlying source data: Deutsche Bahn timetable records CC BY 4.0, archived and processed by [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). Independent research project, no endorsement.
