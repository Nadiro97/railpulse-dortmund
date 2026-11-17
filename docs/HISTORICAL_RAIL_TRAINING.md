# Multi-year Dortmund rail data inventory and chronological 80/20 backtest

## Scope

This workflow implements the approved first step toward using **all available complete published source months from July 2024 to the last fully completed calendar month**, restricted to scheduled **rail** station stops in Dortmund. It DOES NOT claim actual physical punctuality. Source records are historical Deutsche Bahn timetable **planned** and **explicit changed** timestamps from the Piet Brömmel archive (CC BY 4.0).

The default classification covers `S`, `RE`, `RB`, `R`, `ICE`, `IC`, `EC`, `ECE`, `IRE`, `MEX`, `ERB`, `NX`, `RRX`, `REX` and additional reviewed rail category codes. `Bus` and city transport types are excluded; **unknown codes are not automatically treated as trains**. Review the labels in individual archive releases for drift or newly introduced rail operators.

Arrival at Dortmund and departure from Dortmund are evaluated separately. A destination extracted from a planned route path is not proof of a passenger's complete journey. Pre-event planned route and train details only can be used as features; post-event timetable snapshots are for retrospective proxy labels only.

## Phase 1: metadata-only archive inventory

```powershell
cd C:\Users\znadi\railpulse-dortmund
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,ml,dev]"
.\.venv\Scripts\python.exe -m railpulse history-rail --inventory-only
```

This uses `HfApi.list_repo_tree` on **three explicit monthly source folders**, without downloading huge national Parquet data. For every complete calendar month it records presence/absence of the **processed**, **plan** and **change** files. It never assumes that a missing source or incomplete month exists. Outputs: local Git-ignored `reports/history_month_inventory.csv` and `reports/history_month_inventory.json`.

Only a month with **all three published source files** can enter the first proposed split. If fewer than ten months are usable, the program refuses to invent an 80/20 partition. The current month is always excluded even if partially published.

**This metadata inventory is not yet a row-level audit:** it cannot know whether all Dortmund stations, routes, services, hours and labels were collected. It makes no accuracy claims. Review source coverage changes around late 2025 and missing archive collection hours.

## Phase 2: quality-audited multi-month study

```powershell
.\.venv\Scripts\python.exe -m railpulse history-rail --mode both
```

**Disk/memory warning:** Downloading three Germany-wide monthly archives per month (2024 onwards) may require **very large** disk capacity, time, memory, and network bandwidth. Each source file is cached locally. Restarting without `--refresh` reuses the private Dortmund monthly feature cache and coverage report; it does not need to re-extract those months. An omitted or failed month is reported explicitly. Only run the full pipeline on a machine with sufficient resources and a stable connection.

For each published source triple, the pipeline extracts non-cancelled, non-replacement, non-additional **rail-only Dortmund scheduled stops**. It requires **planned snapshot information observed at least 60 minutes before** the station event and **explicit changed timestamps recorded between 15 minutes and nine hours afterward**. These later updates can still be API forecasts, **not verified actual arrival/departure**. No explicit update is treated as **unknown**, not punctual. Label deltas are quality checked. A month requires at least **100 labelled examples** and **50% explicit update coverage** to qualify for the training/backtest partition. Counts, missingness, excluded months and their reasons are reported.

For all remaining eligible months, order chronologically. Use the **oldest ~80% of months for development** and **newest ~20% as retrospective test**, with the last ~10% of development months (at least two months) reserved for model selection and the rest for fitting. Keep **entire months** together—never random rows across time. The fraction is by **month count**, so records will not necessarily be exactly 80/20.

Models compared: always-zero baseline, training mean, simple Ridge, route-aware Ridge, and target-encoded histogram gradient boosting. The primary score is **weighted MAE in minutes**, alongside ordinary MAE, RMSE, performance on seriously delayed timetable updates, erroneous `Pünktlich` decisions, and **per-month, per-station, per-category and per-line** error summaries. Training sees only earlier months; validation chooses the winner; recent test months do not choose the algorithm. No promising percentage or reduction in minutes is claimed before executing these runs.

The newest months have **already been examined** in earlier research, so this is a transparent **retrospective backtest, not a previously unseen blind test**. A future complete untouched period will still be required before a defensible operational deployment claim.

Outputs:
- `reports/history_month_inventory.csv/json`: published sources and preliminary chronological split
- `data/processed/history_rail/YYYY-MM_arrival.parquet` and `YYYY-MM_departure.parquet`: **private stop-level monthly cache** (never upload)
- `data/processed/history_rail/YYYY-MM_<event>_coverage.json`: per-month selection/label quality
- `reports/history_rail_arrival_evaluation.json` and `history_rail_departure_evaluation.json`: aggregated retrospective error summaries
- `models/history_rail_<event>_research.joblib`: **local and untrusted-source-unsafe research models** (never upload or load from a download)

The existing Streamlit dashboard can display the source inventory and the monthly **backtest MAE chart**, if local report files exist. The historical research artifacts are deliberately **not promoted to live passenger-delay predictions**. The existing UI's `Keine belastbare Schätzung` behavior is unchanged until we have evidence for accurate arrival/departure outcomes.

## Follow-on data engineering steps

- Quantify station/line population drift before vs. after the archive's station-coverage expansion, and report the month-by-month missing-hour coverage. An available file is not necessarily a complete month.
- Verify that train type codes reflect actual rail services, and expand the reviewed allowlist with evidence rather than silently including buses.
- Independently verify physical arrival/departure ground truth and cancellation handling before advertising `Pünktlich` as a real-world statement.
- Optimize archive extraction and learning for local storage/memory; the raw national snapshots can be too large for free CI runners. Train on your own device or appropriately sized compute.
- Calibrate and evaluate new prediction vintages on genuinely untouched future dates.

**Provenance:** Deutsche Bahn timetable data, CC BY 4.0; historical archive: [Piet Brömmel / Deutsche Bahn Data](https://huggingface.co/datasets/piebro/deutsche-bahn-data). This independent project is not affiliated with DB.

## Interface wiring and diagnosis

The former experimental on-page form used the **older** `evidence_minutes_<event>.joblib` model, even if the multi-year historical model had been trained. The new UI explicitly loads `models/history_rail_<event>_research.joblib` and, for compatibility with pre-existing local models, builds a supported service catalogue **only from the older training-month cached rail stops**. It does not read test or validation months for these dropdowns. There is no global `deployment_allowed` gate in the historical research artifact: exploratory numerical minutes are reported **as archival timetable-change estimates only**, with visible model/test errors. The user is not told physical punctuality has been verified. If a file, catalogue, or report is missing, the UI indicates which component is missing rather than repeatedly showing the older vague warning.

Diagnose with `python -m railpulse model-status` after `git pull`. If it reports no newer trained model, the multi-year `history-rail --mode both` run must successfully complete on the local machine first. If quality reports list insufficient monthly coverage, inspect their excluded-month reasons; do not relabel missing updates as punctual or loosen gates without evidence.
