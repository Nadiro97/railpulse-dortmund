# Interactive timetable-update risk demo

## What users can do

Run the Streamlit app, choose a **Dortmund railway station**, **train type**, **scheduled arrival hour (local Europe/Berlin time)** and **weekday**, and receive a score for the risk that the **latest explicitly recorded historical timetable arrival update** is at least six minutes after the original planned time. A score of 22.5% or above triggers the **experimental flag**.

**This is not a prediction of actual physical train arrival, not a live feed, and not a validated probability of being delayed on a future journey.** A past explicitly recorded update can itself be a forecast. Data without observed changes are excluded, not considered on-time.

## Reproduce locally

Clone the repository and use Python 3.10+:

~~~bash
python -m pip install -e ".[dashboard,ml,dev]"
python -m pytest -q
python -m railpulse train-demo
streamlit run dashboard/app.py
~~~

The training command downloads the September 2026 processed monthly source and its original parsed-change snapshots if necessary, extracts Dortmund stops, and trains the existing logistic-regression specification **using only planned arrivals from September 1–16, 2026**. It stores a small validated **JSON coefficient artifact** in \`models/railpulse_demo_2026-09.json\`, not a pickle or joblib object. No row-level railway observations, fitted model coefficients, snapshots, or full datasets are committed to GitHub. Raw data and local reports remain ignored.

The form works without CSV analytical reports; those reports enable optional plots elsewhere in the dashboard. The model artifact itself is local-only, so a GitHub clone without the trained file displays training instructions instead of fabricated predictions.

## Technical contract

- Pipeline and model configuration are copied from the reviewed September experiment in \`railpulse/ml_improved.py\` (logistic regression with one-hot station/train-type features plus hour and weekday).
- The alert threshold **0.225** was chosen on September 17–20 validation, **not** recalibrated with October data.
- The JSON includes ordered category names, finite coefficients, model dates, train count, a reviewed schema identifier, source attribution, and explicit \`is_live_prediction=false\` and \`is_verified_actual_delay=false\` flags. Malformed or unexpected artifacts are rejected.
- Only known station and train categories and valid scheduled hours 0–23 and weekdays 1–7 can be scored. Unknown categories are rejected rather than silently imputed.
- Synthetic automated tests compare JSON-only inference against scikit-learn logistic probabilities and test artifact validation and the CLI–workflow contract.
- The previous October 2026 out-of-month experiment found a decline in performance (precision around **36.5%**, recall around **53.9%**, at threshold 0.225), so the demo must not suggest operational reliability. This is a portfolio demonstration only.

## How the score should be interpreted

For a selected station, train type, weekday and hour, a score of 0.65 means that the trained experimental model assigns **65% to the historical binary timetable-update label**. It does **not** mean the passenger has a calibrated 65% real-world probability of arriving six minutes late; the label and calibration have important limitations.

## Licensing

The underlying published timetable-update data originate from Deutsche Bahn, CC BY 4.0, with historical archival processing by Piet Brömmel ([project source](https://github.com/piebro/deutsche-bahn-data)). This independent project filters and transforms Dortmund stop records. It is not endorsed by Deutsche Bahn.
