# Route-aware regression: approximate archive-update minutes at Dortmund

## What exactly is predicted?

The experimental target is **positive minutes by which the latest explicitly recorded timetable update differs from the scheduled event at Dortmund**, truncated to 0–240 minutes. It does **not** measure an independently verified train or bus arrival/departure. A negative difference becomes 0 positive-delay minutes; absent update records are **unknown**, never silently set to zero.

There are two separate tasks:

- **Arriving at Dortmund:** compare the scheduled and most recently recorded updated arrival timestamps **at the Dortmund stop**. Useful for trains coming *into* Dortmund.
- **Departing from Dortmund:** compare scheduled and most recently recorded updated departure timestamps **at the Dortmund stop**. Useful for services travelling *out of* Dortmund.

This is *not* a prediction of the delay at Düsseldorf or Aachen. The `final_destination_station` field means the service's final destination, **not** its full sequence of stops or the passenger's desired destination. A service terminating in Aachen cannot automatically be claimed to stop at Düsseldorf. The archive currently extracted by RailPulse does **not** reconstruct a definitive service origin; unknown origins remain unknown. Data labelled `Bus` are not a complete local-bus dataset.

## Features

Planned stop properties only: `station_name`, `train_type`, `line_number`, `final_destination_station`, scheduled local hour, weekday. Missing lines/destinations are explicitly recorded as `(unknown)`. **Train number is retained in the local processed archive but not yet used as a feature** because many individual numbers would have too few observations; it needs a coverage and generalization study first.

Changed timestamps, computed delay minutes and snapshot timestamps are excluded from model features to avoid directly looking up the historical answer. We do not claim real-time information or a pre-departure snapshot cutoff.

## Reproducible experiment

From Windows PowerShell in your repository:

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,ml,dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m railpulse train-route-delay --mode both
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The command fetches September 2026 archived source records, if not already available locally, and joins **explicit** arrival/departure changes at Dortmund. It fits the two local models separately. Raw rows and learned coefficients are kept in Git-ignored local directories. The manual [GitHub Actions experiment](https://github.com/Nadiro97/railpulse-dortmund/actions/workflows/route-minutes.yml) uploads aggregate-only evaluation metrics and no fitted models.

Training dates: **September 1–16**; validation dates: **September 17–20**; untouched within-month test dates: **September 21–30**, 2026. Models are selected using **validation MAE in minutes only**. The report shows holdout MAE, RMSE, error for 6+ minute archival events, percentage predictions within 5 minutes, plus an experimental route-aware logistic score for whether the *archived change* is 6+ minutes. Baselines: overall training mean and a station+category+time-only ridge model. If route features don't help, the dashboard explicitly notes they are *not an improvement*.

Only station/type/line/destination combinations with at least **5 training examples** appear in the form. Combinations are not invented or stitched into unverified origin–destination journeys. Small sample selections trigger a warning. The linear regression score is clipped to 0–240 minutes. This is a point estimate, **not** a reliable real-world journey forecast or a prediction interval.

The test dates are from the September 2026 archival data. We already examined October for prior models, so a genuine promotion would need **later unseen complete-month evaluation** of both arrival and departure targets and external physical-arrival verification.

## Source, licensing and caution

Underlying timetable updates: Deutsche Bahn, CC BY 4.0. Historical collection and processing: [Piet Brömmel](https://github.com/piebro/deutsche-bahn-data). Independent portfolio research, not endorsed by Deutsche Bahn.

**Never rely on these forecasts to decide whether to leave for a train or make a connection.** Planned vs. updated API timestamps may be forecasts or change snapshots, not confirmed arrival/departure observations. Cancelling trains and missing updates require separate treatment.
