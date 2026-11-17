# RailPulse Dortmund 🚆

**Research project: evidence-first minutes estimates for Dortmund train timetable updates, with explicit uncertainty and missing-data checks.**

**Critical limitation:** Deutsche Bahn's historical changed timetable timestamps are not independently verified actual arrival/departure times. The upstream processed data fills some missing updated timestamps with planned values. **The application must not claim proven real-world punctuality**, and will decline to estimate when validation or coverage is weak.

## All months excluded? Fixed label extraction, not a lower quality standard

Earlier `history-rail` releases required the **latest explicit update to be in a snapshot recorded 15 minutes to nine hours AFTER the scheduled Dortmund stop**. This is not how many archived timetable updates are captured. The strict timestamp condition excluded legitimate archival timetable-change observations before the event. We now label **the latest explicitly recorded timetable changed timestamp, regardless of snapshot capture time**, and independently report the share with actual post-event snapshot observations. The label is a *historical update proxy*, **not independently verified physical punctuality**; no-update records remain unknown.

The new `label_schema_version` marker automatically invalidates old stop-level feature caches that were produced with the post-event-only rule. Already downloaded national source files are reused. Run `python -m railpulse history-rail --mode both` again after `git pull`, then use `python -m railpulse model-status`. If a month still fails, `reports/history_rail_<arrival|departure>_quality.json` now lists its counts and rates to distinguish weak planned-snapshot coverage, absent updates, and unusable labels. The original **50% overall label-coverage threshold and ten-month minimum remain unchanged** pending evidence.

## Multi-year research estimate versus old quality gate (fixed)

If you previously saw **"Keine belastbare Schätzung" for every route**, the former top-of-page form was loading `models/evidence_minutes_<arrival|departure>.joblib` from the *old four-month experiment* instead of the new model. Run `python -m railpulse model-status` to check separately for newer `models/history_rail_<arrival|departure>_research.joblib` files, completed evaluation reports and failed month-quality reports. The dashboard now uses the multi-year research model **if it exists**, and otherwise explains why the new model is missing. It never silently falls back to the older predictor.

With a successfully trained multi-year local model, the dashboard displays a **numeric experimental archived timetable-update estimate in minutes**, with its model name, historical support, validation/test MAE and warnings. The result is **not verified physical delay**. In particular, a zero-minute estimate must never be presented as a confirmed "Pünktlich" train. We do not bypass poor model accuracy by changing a global quality flag; the numerical research estimate is accompanied by explicit accuracy diagnostics and limitations.

If your training completed with too few usable months, inspect `reports/history_rail_arrival_quality.json` and `reports/history_rail_departure_quality.json`. You can also run a data-only inventory with `python -m railpulse history-rail --inventory-only`; that command **does not train**. Existing local training-month caches can regenerate route selection menus without needing to rerun the entire multi-year training.

## Multi-year Dortmund rail inventory and retrospective 80/20 evaluation

The **new rail-only historical workflow** audits every complete published month from **July 2024 through the last completed calendar month**. It restricts records to reviewed railway-service codes (S, RB, RE, ICE, IC, EC and other rail codes), excluding buses, and evaluates Dortmund **arrivals and departures separately**.

Run the lightweight source inventory first (no nationwide parquet downloads):

```powershell
.\.venv\Scripts\python.exe -m railpulse history-rail --inventory-only
```

If you have ample disk space, memory and network bandwidth, run full historical extraction and research training:

```powershell
.\.venv\Scripts\python.exe -m railpulse history-rail --mode both
```

The workflow inventories all three published source formats per month, then extracts only Dortmund rail stop records with pre-event plan information and explicit after-event timetable updates. Monthly quality checks exclude missing or unreliable periods. From **usable whole months**, about 80% are assigned to training/validation (a separate recent validation period) and the newest 20% to retrospective testing. Models are compared using **weighted MAE in minutes** and monthly, category, line and station diagnostics. This is **not yet a validated real-world delay predictor**: historical timetable changes are not confirmed physical arrival observations, and the newest historical months have been examined before.

Results appear in the dashboard's **Mehrjährige Datenprüfung** section once generated. See [step-by-step method and data-volume warnings](docs/HISTORICAL_RAIL_TRAINING.md), or run the [metadata-only inventory workflow](.github/workflows/history-month-inventory.yml).

## Rebuilt evidence-first pipeline (experimental)

The new implementation uses the raw **plan snapshots** and **explicit change snapshots** rather than treating the processed `delay_in_min` as physical arrival ground truth. It only uses scheduled features from planned snapshots at least **60 minutes before the Dortmund event**, then finds an explicit updated time in snapshots **15 minutes to 9 hours after the scheduled event**. Those late snapshots are still only a *proxy* for the outcome, not verified physical arrival evidence. Missing snapshots are **unknown, never silently punctual**.

Feature engineering adds Dortmund stop, train type, line number, train number, first listed upstream station, final listed downstream station, route stop index, number of listed stops, weekday/weekend, rush hour, cyclic hour representation, route progress and missing-route flags. For a train going Dortmund → Düsseldorf, use **departure at Dortmund**; for a train coming from Aachen to Dortmund, use **arrival at Dortmund**. Neither distant city delay nor full journey origin is inferred from a partial route.

Train **July–August 2026**, choose among a no-delay dummy, median/mean, simple ridge, route ridge and route-aware gradient boosting using **September** validation only, and report **October as an already inspected diagnostic**. Our primary criterion is **weighted MAE in minutes** (5× weight for archived 15+ minute updates, 3× for 6–14 minute updates, 1× otherwise), alongside ordinary MAE, severe-delay misses, punctual-claim reliability, and fraction of cases within 5 minutes. The dashboard shows **Pünktlich**, **Voraussichtliche Verspätung: X Minuten**, or **Keine belastbare Schätzung**. It does not display a fake percentage in place of minute estimates. It refuses to show a claim unless validation improves on the dummy and quality/coverage thresholds are met.

After updating your local checkout in PowerShell:

```powershell
cd C:\Users\znadi\railpulse-dortmund
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,ml,dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m railpulse train-evidence --mode both
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

**Data storage warning:** This experiment downloads four months of Germany-wide monthly archives, including original plan and change snapshots. Ensure sufficient storage before starting; downloads can be large. Learned joblib models stay in your local Git-ignored `models/` folder. **Never load joblib models from unknown sources**: it uses Python pickle internally. Aggregated validation reports are written under `reports/` and are never mistaken for real train arrival verification.

See [full methodology and blockers](docs/EVIDENCE_FIRST_REBUILD.md) and the [manual aggregate-only GitHub workflow](.github/workflows/evidence-minutes.yml). **The new model's real-data evaluation has not been completed merely by passing unit tests.** Earlier archived-record experiments remain in this repo for historical reference, not as claims of real-life punctuality.

---

## Legacy work (reproducibility only)

## Explicit update validation (second milestone)

**Verified September 2026 milestone:** 129,500 / 132,450 eligible Dortmund scheduled arrivals have at least one explicitly recorded arrival-time update in the parsed snapshots (**97.77% coverage**). These are timetable updates, **not confirmed actual train arrivals**. See [data-backed validation results](docs/EXPLICIT_UPDATES_RESULTS_2026_09.md) and the [successful GitHub Actions run](https://github.com/Nadiro97/railpulse-dortmund/actions/runs/43643412457).



The processed monthly archive fills absent arrival changes with planned times. To identify **explicitly recorded arrival updates**, RailPulse now cross-checks stop IDs and EVA station codes against the original parsed change snapshots. This distinguishes rows with an observed updated timestamp from rows that cannot be verified. **Even explicitly recorded updates are not independently confirmed physical arrivals.**

```bash
# Once the processed September subset has been downloaded and extracted:
python -m railpulse validate-updates --month 2026-09
streamlit run dashboard/app.py
```

Run the [validation workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/validate-updates.yml) on GitHub to produce **aggregate-only** evidence without committing raw train records. See the [validation method](docs/EXPLICIT_UPDATES_VALIDATION.md). The dashboard displays these metrics in a separate section to avoid mistaking predicted timetable changes for verified punctuality.

## Experimental machine-learning baseline — Milestone 3

**First real test result:** September's leakage-aware logistic baseline achieved **ROC AUC 0.816**, **average precision 0.535** and **Brier 0.102** versus constant-prevalence ROC AUC 0.500 / average precision 0.160 / Brier 0.135. **Recall is only 32.5% at a 0.5 threshold**. [Read the evaluation and its limitations](docs/ML_RESULTS_2026_09.md); these labels are **retrospective timetable updates, not independently verified physical delays**.



A chronological logistic-regression baseline predicts **whether the latest explicitly recorded arrival-time update is 6+ minutes after schedule** using *only* station, train category, scheduled hour, and scheduled weekday. The constant-prevalence baseline provides a no-skill comparison. The first 20 days of a month train the model and days 21–end evaluate it; no future timetable updates are features.

**This is a timetable-update prediction, not independently verified actual train punctuality.** The sample includes only stops with explicit updates, so selection effects remain. See [full ML methodology](docs/ML_BASELINE.md) and the [reproducible ML workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-baseline.yml).

```bash
pip install -e '.[ml,dev]'
python -m railpulse ml-baseline --month 2026-09
```

The ML run writes only `reports/ml_baseline_metrics.json` and `reports/ml_baseline_comparison.csv`, never row-level records or model artifacts, to downloadable aggregate-only CI artifacts.

## Selected model for the project — recall-focused champion

**Current portfolio choice:** validation-selected **logistic regression with alert threshold 0.225** for identifying **explicit historical timetable arrival updates at least 6 minutes later than scheduled**. It is the best *evaluated* option **for our chosen goal of catching more potential late-update events while enforcing at least 45% precision on the validation split**. This is an experimental classifier, **not a deployed live arrival-delay model**.

**Held-out September 21–30 evaluation:** recall **57.9%**, precision **45.9%**, F1 **0.512**, ROC AUC **0.817**, and accuracy **82.3%**. The older threshold-0.50 logistic baseline had recall **32.5%**, precision **70%**, and accuracy **87.0%**. A lower threshold catches more potentially late updates but creates more false alerts. The 45% precision constraint was set on **validation**, not guaranteed for every later dataset.

We **retain both configurations** for reproducibility: the preferred model is generated by `python -m railpulse ml-improve --month 2026-09`; the earlier higher-precision comparison stays available through `python -m railpulse ml-baseline --month 2026-09`. The model and threshold were selected without looking at held-out test results. [Full comparison and limitations](docs/ML_IMPROVED_RESULTS_2026_09.md).

**Important:** the outcome is an **archived explicit API timetable update**, which may be a prediction, not a verified physical arrival. We will revalidate on later months before calling this a generalizable model.

## Out-of-month test — September-trained model → October 2026

**October 2026 external-month result:** The fixed September-trained model scored **ROC AUC 0.769, recall 53.9%, precision 36.5%, accuracy 79.3%** at the locked 0.225 threshold, compared with September **ROC AUC 0.817, recall 58.0%, precision 45.7%, accuracy 82.2%**. This is a **performance decline**, particularly in false alerts. [See the real out-of-month results](docs/OUT_OF_MONTH_RESULTS_2026_10.md). We retain this as a research benchmark, **not a production passenger-warning system**.

Our current alert policy is **locked**: logistic regression trained on **September 1–16**, chosen using the **September 17–20** validation period, and a fixed probability alert threshold of **0.225**. We now test it on **October 2026** without learning from October records. September 21–30 remains a reference holdout.

Reproduce with `python -m railpulse ml-cross-month --month 2026-09 --test-month 2026-10`, or run the [cross-month validation workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-cross-month.yml). All downloaded railway records are excluded from GitHub; the workflow uploads aggregated metrics only. See [methodology and limitations](docs/OUT_OF_MONTH_VALIDATION.md).

**Model outcome:** a retrospectively recorded explicit arrival-time update of **6 minutes or more**, **not** an independently confirmed physical arrival delay.

## Multi-month experiment — reducing false alerts

The new research comparison trains on **July and August 2026**, uses **September 2026** for model/threshold choice, and reports October as an **already inspected historical benchmark**, not a fresh blind test. We prioritize alert precision with a predeclared **55% validation precision** requirement, then measure how many 6+ minute timetable updates are caught and how many false alerts arise per 1,000 eligible stops. If the target cannot be met, no new model is promoted.

Run `python -m railpulse ml-multimonth` after `pip install -e '.[ml,dev]'`, or use the [multi-month workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/ml-multimonth.yml). Read [methodology](docs/MULTIMONTH_DIAGNOSTICS.md). The published artifacts contain **aggregates only**. The current September-selected logistic model remains the research reference until a new, genuinely untouched month supports a promotion decision.

## Four-month diagnostics and reliability guardrails

**Exploratory October 2026 comparison:** The July–August-trained logistic candidate with a **September-selected 0.325 alert threshold** produced **7,460 false alerts** in October, against **17,054** for our existing September-trained 0.225-threshold reference. This represents **56% fewer false alerts**, but recall dropped from **53.9% to 36.7%**. Its ROC AUC stayed about 0.77. Because October results were previously inspected, **we have not promoted this candidate**. See [the multi-month results](docs/MULTIMONTH_RESULTS_2026_10.md).

**CI reliability:** Every push automatically runs dependency validation, syntax compilation, unit/integration tests, and **CLI–workflow contract tests**. The contract tests parse each GitHub Actions YAML workflow and verify that every `python -m railpulse <command>` actually exists and accepts `--help`. This catches missing CLI registrations before attempting large archive downloads. Heavy data workflows run manually rather than on each push.

## Interactive prediction demo — local Streamlit

The dashboard offers a form to choose a **Dortmund railway station**, **train category**, **scheduled arrival hour** and **weekday**. It displays the experimental logistic-regression score for whether a recorded timetable arrival update will be **6+ minutes later than the scheduled time**. It is NOT a live train prediction or independently verified physical arrival forecast. The score has not been calibrated for operational use; the October test showed false alerts.

Run the following locally after cloning the repository:

~~~bash
python -m pip install -e ".[dashboard,ml,dev]"
python -m railpulse train-demo
streamlit run dashboard/app.py
~~~

Training uses **only September 1–16, 2026** explicitly updated eligible historical stops to reproduce the chosen 0.225-threshold classifier. The downloaded archive, training rows and local JSON coefficient file in **models/** are **Git-ignored**. The public repository contains only training/inference code and synthetic tests; no trained model, per-train records or snapshots are included.

The form works even if the analytic CSV reports have not been generated. It requires a locally trained model. See [the demo methodology](docs/PREDICTION_DEMO.md).

## Whole-model reliability audit (all stations and train categories)

Our existing demo is **not a validated real-world train-delay predictor**. The archival target is a **recorded timetable change of 6+ minutes**, not an independent measurement of physical arrival. Records without explicit arrival updates, cancellations, and replacements are omitted; differences between train types, routes, and months can distort a prediction. A model score of 10–20% cannot be interpreted as a guaranteed real-world on-time rate.

Run `python -m railpulse audit-model` after installing `.[dashboard,ml,dev]`. The audit reuses September and October archived sources where available and writes only aggregate files: `reports/model_audit_summary.json`, `model_audit_cohorts.csv`, `model_audit_calibration.csv`, and `model_audit_coverage.csv`. These check all station/type groups with sufficient samples, calibration on archival labels, and missing-change coverage. Or use the [manual GitHub workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/model-audit.yml). See [audit methodology](docs/WHOLE_MODEL_AUDIT.md).

**Policy:** Do not promote or claim a real-world passenger-delay predictor until reliable independently verified arrival-time ground truth, a time-bounded prediction cutoff, comprehensive service coverage, and fresh untouched temporal evaluations are available. The dashboard now explicitly labels its score as exploratory archival-update analysis.

## Route-aware minutes model — Dortmund arrival OR departure

**Experimental new model:** We now estimate **nonnegative archived timetable-update minutes at Dortmund**, not just a 6+ minute yes/no category. Two separate models use **arrival at Dortmund** and **departure from Dortmund**, respectively. Inputs are **Dortmund stop, train category, line number, train's final destination, scheduled hour, and weekday**. Line and destination are included only when published; no route origin, intermediate stop, or passenger destination is fabricated. Long-distance trains often have no line number. Bus-labelled records are railway-timetable services, **not comprehensive city-bus coverage**.

The dashboard only offers service/direction combinations that appeared repeatedly in training. It compares the route-aware minutes model against simpler station/type and constant-mean models using **September 17–20 validation**, then reports **September 21–30 holdout metrics**. There is also an uncalibrated experimental 6+ minute archival-update score. A route-aware model must beat the simpler baseline on validation before we call it an improvement.

On Windows PowerShell, after pulling updates:

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,ml,dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m railpulse train-route-delay
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Only September 2026 data are needed for this local experiment. Downloaded files, per-stop observations and learned models remain local and Git-ignored. The [model methodology](docs/ROUTE_AWARE_MINUTES.md) and [manual training/evaluation workflow](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/route-minutes.yml) explain the controls.

**Not verified physical delay:** The targets are the *latest explicitly recorded API timetable updates*, measured against the planned Dortmund time. Predictions are **not** physical arrival/departure times, not live forecasts, and not reliable for making real travel decisions. A train's final destination does not prove a particular intermediate stop is on its route.

## Pipeline

`Hugging Face monthly_processed_data/*.parquet` → select Dortmund station records with DuckDB → validate missing values, cancellations and plan coverage → calculate planned-vs-changed arrival time differences → CSV summaries → local Streamlit dashboard.

**Caution:** `arrival_change_time` may reflect updated predictions rather than independently verified actual arrival times. Missing changes are *unknown*, not on time. Cancellation statistics are reported separately. Source times are local German wall times and need DST care.

## Data provenance and licences

Underlying rail timetable-update data from Deutsche Bahn, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Historical collection and processing by Piet Brömmel, [Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data) / [Hugging Face dataset](https://huggingface.co/datasets/piebro/deutsche-bahn-data). Filtered and aggregated for Dortmund. Independent educational project; no affiliation or endorsement.

The repository's **original code uses MIT**; the source dataset has its own CC BY 4.0 licence. See [DATA_SOURCES](docs/DATA_SOURCES.md), [METHODS_AND_LIMITATIONS](docs/METHODS_AND_LIMITATIONS.md) and [PRIVACY](docs/PRIVACY.md).

## Planned next work

1. Validate the September 2026 Dortmund subset and its coverage.
2. Compare stations, hours, train categories and cancellations.
3. Extend across months after quality checks.
4. Evaluate forecast baselines with strict time-based train/test splits.
5. Test operational scenarios without making unsupported causal claims.
