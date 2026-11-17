"""RailPulse Dortmund – interactive local dashboard of aggregated update metrics."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RailPulse Dortmund", page_icon="🚆", layout="wide")
st.title("RailPulse Dortmund 🚆")
st.caption("Independent, educational railway timetable-update analytics · Deutsche Bahn / Piet Brömmel")
st.warning(
    "The archive's processed **arrival_change_time is filled with the scheduled time when no change is recorded**. "
    "A zero-minute difference therefore may mean no observed update, **not** a verified on-time arrival. "
    "Positive differences are timetable-update deviations, not independently confirmed physical arrival delays."
)


# Route/direction-aware experimental estimates; origin and intermediate stops are not inferred.
st.subheader("Route-aware delay minutes at Dortmund (experimental)")
st.caption(
    "Choose an observed line and its train's final destination. Select arrival or departure "
    "AT DORTMUND, not delay at the distant destination. Data are archival timetable updates, "
    "NOT verified physical train arrival/departure times."
)
from railpulse.route_delay import (
    CAT_ROUTE, MISSING, local_model_path, load_route_model, estimate_at_dortmund
)
event_label = st.radio(
    "Where should Dortmund time be measured?",
    ("Departure from Dortmund", "Arrival at Dortmund"),
    horizontal=True,
)
event_mode = "departure" if event_label.startswith("Departure") else "arrival"
if local_model_path(event_mode).is_file():
    try:
        route_artifact = load_route_model(event_mode)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Route model cannot be loaded: {exc}. Retrain with python -m railpulse train-route-delay.")
    else:
        route_list = route_artifact["routes"]
        st.caption("Only routes with repeated examples in our September training data appear below.")
        route_stations = sorted({r["station_name"] for r in route_list})
        route_station = st.selectbox("Dortmund stop", route_stations, key="r_station")
        candidates = [r for r in route_list if r["station_name"] == route_station]
        route_types = sorted({r["train_type"] for r in candidates})
        route_type = st.selectbox("Train / replacement-service category", route_types, key="r_type")
        candidates = [r for r in candidates if r["train_type"] == route_type]
        route_lines = sorted({r["line_number"] for r in candidates})
        route_line = st.selectbox(
            "Line number (may be unavailable for long-distance trains)",
            route_lines, key="r_line"
        )
        candidates = [r for r in candidates if r["line_number"] == route_line]
        route_destinations = sorted({r["final_destination_station"] for r in candidates})
        route_destination = st.selectbox(
            "Train's final destination (not necessarily your stop)",
            route_destinations, key="r_destination"
        )
        chosen_route = next(r for r in candidates
                            if r["final_destination_station"] == route_destination)
        r1, r2 = st.columns(2)
        with r1:
            route_hour = st.selectbox(
                "Scheduled hour at Dortmund (Europe/Berlin)", list(range(24)),
                index=8, key="r_hour"
            )
        with r2:
            route_day = st.selectbox(
                "Weekday at Dortmund",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                key="r_weekday"
            )
        st.caption(
            f"Historical support for this exact station/type/line/destination combination: "
            f"{chosen_route['training_examples']} training records."
        )
        if st.button("Estimate minutes at Dortmund", key="route_estimate"):
            result = estimate_at_dortmund(
                route_artifact, chosen_route, int(route_hour),
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(route_day) + 1,
            )
            a, b = st.columns(2)
            a.metric(
                "Estimated positive timetable-update minutes at Dortmund",
                f"{result['estimated_archived_delay_minutes']:.1f} min"
            )
            b.metric(
                "Score for 6+ min archived update",
                f"{result['estimated_archived_6plus_probability']:.1%}"
            )
            if chosen_route["training_examples"] < 30:
                st.warning(
                    "Only a few matching historical examples. This service estimate is especially uncertain."
                )
            st.error(
                "Not a live or verified physical delay prediction. Do not rely on these numbers "
                "to plan a connection or decide when to leave for the station."
            )
        st.caption(
            "Training: September 1–16, 2026. Route-aware September validation MAE: "
            f"{route_artifact['validation_route_mae']:.1f} min; "
            f"simple model MAE: {route_artifact['validation_baseline_mae']:.1f} min. "
            "A smaller error is better. Both refer only to retrospective archived updates."
        )
        if route_artifact["selected_for_minutes"] != "station_line_destination":
            st.warning(
                "The route-aware minutes model was NOT the lowest-error model on validation. "
                "Its score is shown for research only, not promoted as an improvement."
            )
        st.caption(
            "The original journey start and intermediate stops are not inferred here. "
            "A service shown as 'toward Aachen' cannot automatically be assumed to stop at Düsseldorf."
        )
else:
    st.info(
        f"No local {event_mode} route model yet. In PowerShell, run "
        r'.\.venv\Scripts\python.exe -m railpulse train-route-delay '
        "after git pull. This uses the September archive locally and "
        "does not upload learned coefficients."
    )
st.divider()

# Local-only interactive model score. The form works without report CSV files.
st.subheader("Explore the archival model (not a real-world delay predictor)")
st.caption(
    "The model estimates an archived timetable-update label in a selected sample, "
    "NOT how likely your train will really be late. Its cross-month reliability is limited."
)
from railpulse.demo_model import MODEL_PATH, load_artifact, predict_probability
if MODEL_PATH.is_file():
    try:
        demo = load_artifact(MODEL_PATH)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        st.error(f"Local model artifact is invalid: {exc}. Retrain with 'python -m railpulse train-demo'.")
    else:
        with st.form("timetable_update_demo"):
            c1, c2 = st.columns(2)
            with c1:
                station = st.selectbox("Dortmund rail station", demo["categories"]["station_name"])
                hour = st.selectbox("Scheduled arrival hour (Europe/Berlin)", list(range(24)), index=8)
            with c2:
                category = st.selectbox("Train category", demo["categories"]["train_type"])
                day = st.selectbox(
                    "Scheduled weekday",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                )
            calculate = st.form_submit_button("Estimate timetable-update risk")
        if calculate:
            chance = predict_probability(
                demo, station, category, int(hour),
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day) + 1,
            )
            st.metric("Experimental score for archived 6+ min timetable update", f"{chance:.1%}")
            st.progress(chance)
            if chance >= demo["threshold"]:
                st.warning("Experimental archive-update flag (not a verified late-arrival warning).")
            else:
                st.info("Below the archive-update flag threshold; this does NOT mean the train is likely on time.")
            st.warning("Do not use this score to decide whether to catch a train. Real arrival delays are not verified by this dataset.")
            st.caption(
                "This is an experimental retrospective-score demonstration. "
                "The percentage is NOT a calibrated personal travel probability or a live train prediction. "
                "The model was trained on September 1–16, 2026 archive data and "
                "showed reduced precision on October holdout data."
            )
else:
    st.info(
        "No local trained model found. Run 'pip install -e \".[dashboard,ml,dev]\"' "
        "and then 'python -m railpulse train-demo'. "
        "Training downloads September historical source data to local ignored folders; "
        "the resulting local JSON coefficients stay in models/."
    )
st.divider()

reports = Path("reports")
required = [
    "quality", "station_quality", "by_station", "by_hour",
    "by_train_type", "by_day", "by_weekday"
]
if not all((reports / f"{name}.csv").is_file() for name in required):
    st.info(
        "No complete aggregate reports available locally. "
        "Run `python -m railpulse run --month 2026-09` first, "
        "or download the reports-only artifact from GitHub Actions into `reports/`."
    )
    st.stop()

frames = {name: pd.read_csv(reports / f"{name}.csv") for name in required}
q = frames["quality"].iloc[0]
total = int(q["total_station_stop_rows"])
eligible = int(q["eligible_changed_arrivals"])
coverage = q.get("processed_arrival_timestamp_presence_pct", q.get("scheduled_arrival_change_coverage_pct"))
cancelled = int(q["cancelled_arrival_records"])

st.subheader("Coverage & quality")
a, b, c, d = st.columns(4)
a.metric("Dortmund station stops", f"{total:,}")
b.metric("Eligible processed arrival rows", f"{eligible:,}")
c.metric("Processed arrival timestamp presence", "N/A" if pd.isna(coverage) else f"{coverage:.1f}%")
d.metric("Cancelled arrival records", f"{cancelled:,}")
with st.expander("Full quality diagnostics / methodology"):
    st.dataframe(frames["quality"], use_container_width=True, hide_index=True)
    st.dataframe(frames["station_quality"], use_container_width=True, hide_index=True)
    st.caption(
        "The processed archive fills absent change timestamps with planned timestamps, so 100% presence is expected "
        "and does not measure API observation coverage or punctuality. "
        "Cancelled arrivals are record counts, not passenger journeys."
    )

by_station = frames["by_station"]
by_type = frames["by_train_type"]
by_hour = frames["by_hour"]
by_day = frames["by_day"]

tab_stations, tab_time, tab_types = st.tabs(["Stations", "Time patterns", "Train categories"])

with tab_stations:
    minimum = st.number_input("Minimum eligible records for station comparison", min_value=1, value=100, step=50)
    filtered = by_station[by_station["eligible_rows"] >= minimum].copy()
    sort_by = st.selectbox(
        "Rank stations by",
        ["Median arrival change (min)", "Mean arrival change (min)", "6+ min share (%)"]
    )
    metric = {
        "Median arrival change (min)": "median_changed_arrival_delta_min",
        "Mean arrival change (min)": "mean_changed_arrival_delta_min",
        "6+ min share (%)": "share_6min_plus_pct",
    }[sort_by]
    if filtered.empty:
        st.info("No stations meet the minimum sample size. Reduce the threshold.")
    else:
        filtered = filtered.sort_values(metric, ascending=False)
        st.plotly_chart(
            px.bar(
                filtered, x=metric, y="station_name", orientation="h",
                hover_data=["eligible_rows", "p90_changed_arrival_delta_min"],
                labels={"station_name": "Station", metric: sort_by},
                title="Station comparison (only eligible records)",
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            use_container_width=True,
        )
        st.dataframe(filtered, hide_index=True, use_container_width=True)

with tab_time:
    if not by_day.empty:
        st.plotly_chart(
            px.line(
                by_day, x="scheduled_date", y="mean_changed_arrival_delta_min", markers=True,
                hover_data=["eligible_rows", "share_6min_plus_pct"],
                labels={"scheduled_date": "Date", "mean_changed_arrival_delta_min": "Mean difference (min)"},
                title="Daily mean planned-vs-changed arrival time difference"
            ),
            use_container_width=True,
        )
    st.plotly_chart(
        px.line(
            by_hour, x="planned_hour", y="mean_changed_arrival_delta_min", markers=True,
            hover_data=["eligible_rows", "share_6min_plus_pct"],
            labels={"planned_hour": "Scheduled hour (Europe/Berlin)",
                    "mean_changed_arrival_delta_min": "Mean difference (min)"},
            title="Typical pattern by planned arrival hour"
        ), use_container_width=True
    )
    st.dataframe(frames["by_weekday"], hide_index=True, use_container_width=True)

with tab_types:
    by_type = by_type.sort_values("eligible_rows", ascending=False)
    st.plotly_chart(
        px.bar(
            by_type, x="train_type", y="median_changed_arrival_delta_min",
            hover_data=["eligible_rows", "share_6min_plus_pct"],
            labels={"train_type": "Train type", "median_changed_arrival_delta_min": "Median difference (min)"},
            title="Train categories (eligible records only)"
        ), use_container_width=True
    )
    st.dataframe(by_type, hide_index=True, use_container_width=True)

# Separate observed-update audit — never conflate this sample with all services.
validated_files = [
    "validated_quality", "validated_by_station",
    "validated_by_train_type", "validated_by_hour"
]
st.divider()
st.subheader("Explicit timetable update validation")
st.caption(
    "This independent audit uses non-null arrival updates recorded in the upstream "
    "parsed change snapshots; it does not establish actual vehicle arrival times."
)
if all((reports / f"{name}.csv").is_file() for name in validated_files):
    verified = {name: pd.read_csv(reports / f"{name}.csv") for name in validated_files}
    vq = verified["validated_quality"].iloc[0]
    v1, v2, v3 = st.columns(3)
    v1.metric("Eligible planned arrivals", f"{int(vq['eligible_scheduled_arrival_rows']):,}")
    v2.metric("Explicitly updated arrivals", f"{int(vq['eligible_with_explicit_arrival_update']):,}")
    cover = vq["explicit_update_coverage_pct"]
    v3.metric("Explicit update coverage", "N/A" if pd.isna(cover) else f"{cover:.1f}%")
    st.warning(
        "These are **observed API timetable updates**, not verified physical train arrivals. "
        "Stations and services with different update coverage must not be directly "
        "ranked as if this were a complete population sample."
    )
    min_valid = st.number_input(
        "Minimum explicit updates for the validated station chart",
        min_value=1, value=100, step=50, key="validated_minimum"
    )
    vs = verified["validated_by_station"].copy()
    vs = vs[vs["explicit_update_rows"] >= min_valid]
    if not vs.empty:
        st.plotly_chart(
            px.bar(
                vs.sort_values("median_explicit_update_delta_min", ascending=False),
                x="median_explicit_update_delta_min", y="station_name", orientation="h",
                hover_data=["explicit_update_rows", "explicit_update_coverage_pct"],
                title="Explicitly observed updates — median planned-vs-updated time difference",
                labels={"station_name": "Station",
                        "median_explicit_update_delta_min": "Median update difference (min)"}
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            use_container_width=True,
        )
    st.dataframe(verified["validated_by_station"], use_container_width=True, hide_index=True)
    with st.expander("Explicit-update validation diagnostics and by-type coverage"):
        st.dataframe(verified["validated_quality"], use_container_width=True, hide_index=True)
        st.dataframe(verified["validated_by_train_type"], use_container_width=True, hide_index=True)
        st.dataframe(verified["validated_by_hour"], use_container_width=True, hide_index=True)
else:
    st.info(
        "Explicit-update validation not loaded yet. Run "
        "`python -m railpulse validate-updates --month 2026-09`, "
        "or download its aggregated GitHub Actions artifact into `reports/`."
    )

# Optional historical baseline metrics; only aggregates, never model artifacts or raw rows.
baseline_metrics = reports / "ml_baseline_metrics.json"
st.divider()
st.subheader("Experimental ML baseline")
st.caption(
    "Predicts whether a retrospectively recorded timetable arrival-time update is "
    "6+ minutes later than scheduled; not actual train arrival delay prediction. "
    "Training uses earlier calendar days, testing later days."
)
if baseline_metrics.is_file():
    import json
    ml = json.loads(baseline_metrics.read_text(encoding="utf-8"))
    st.warning(
        "Model evaluation includes only stops with explicit recorded arrival updates; "
        "this subset may be biased. A model beating the baseline does not establish "
        "causes or real-world delay-reduction benefits."
    )
    model = ml["logistic_regression"]
    constant = ml["prevalence_baseline"]
    ma, mb, mc = st.columns(3)
    ma.metric("Train records", f"{ml['train_n']:,}")
    mb.metric("Later test records", f"{ml['test_n']:,}")
    mc.metric("Test ROC AUC", "N/A" if model["roc_auc"] is None else f"{model['roc_auc']:.3f}")
    comparison = pd.DataFrame([
        {"model": "Constant prevalence", "ROC AUC": constant["roc_auc"],
         "Average precision": constant["average_precision"], "Brier": constant["brier"]},
        {"model": "Logistic regression", "ROC AUC": model["roc_auc"],
         "Average precision": model["average_precision"], "Brier": model["brier"]},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption(
        f"Train: {ml['train_window'][0]}–{ml['train_window'][1]}; "
        f"Test: {ml['test_window'][0]}–{ml['test_window'][1]}. "
        "Lower Brier is better, higher ROC AUC and average precision are better."
    )
else:
    st.info(
        "No baseline metrics yet. Run `python -m railpulse ml-baseline --month 2026-09`, "
        "or download the ML GitHub Actions metrics artifact into `reports/`."
    )

# Model improvements selected on validation-only September dates.
improved_metrics = reports / "ml_improved_metrics.json"
st.divider()
st.subheader("Validation-tuned model comparison")
if improved_metrics.is_file():
    import json
    tuned = json.loads(improved_metrics.read_text(encoding="utf-8"))
    selected = tuned["selected_model"]
    rows = []
    for model_name, values in tuned["test"].items():
        row = values["validation_selected_threshold"]
        ranking = values["ranking"]
        rows.append({
            "Model": model_name,
            "Chosen on validation": model_name == selected,
            "Validation-picked threshold": row["threshold"],
            "Recall (test)": row["recall"],
            "Precision (test)": row["precision"],
            "F2 (test)": row["f2"],
            "ROC AUC (test)": ranking["roc_auc"],
            "Average precision (test)": ranking["average_precision"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Model: {selected}; threshold: {tuned['selected_threshold']}. "
        f"Training {tuned['train_window'][0]}–{tuned['train_window'][1]}, "
        f"validation {tuned['validation_window'][0]}–{tuned['validation_window'][1]}, "
        f"held-out test {tuned['test_window'][0]}–{tuned['test_window'][1]}."
    )
    st.warning(
        "Improvements on this historical test may not generalize to later months. "
        "The target describes retrospectively recorded timetable updates, not verified train arrivals."
    )
else:
    st.info("Run `python -m railpulse ml-improve --month 2026-09` or download the aggregate-only workflow artifact to `reports/`.")

# An out-of-month validation should be viewed separately from September tuning.
cross_month = reports / "ml_cross_month_metrics.json"
st.divider()
st.subheader("Out-of-month model check")
if cross_month.is_file():
    import json
    external = json.loads(cross_month.read_text(encoding="utf-8"))
    periods = [
        ("September reference holdout", external["reference_september_holdout"]),
        ("October unseen month", external["unseen_month_holdout"]),
    ]
    rows = []
    for title, values in periods:
        rank = values["selected_model"]["ranking"]
        alert = values["selected_model"]["locked_alert_threshold"]
        rows.append({
            "Period": title, "Records": values["sample_n"],
            "Positive fraction": values["positive_fraction"],
            "ROC AUC": rank["roc_auc"], "Average precision": rank["average_precision"],
            "Accuracy": alert["accuracy"], "Recall": alert["recall"],
            "Precision": alert["precision"], "F1": alert["f1"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.warning(
        "October is a truly unseen month; the chosen September model and alert threshold "
        "were held fixed. Lower precision means more false alerts. "
        "This is experimental retrospective timetable-update classification, "
        "NOT verified train-arrival delay prediction."
    )
    st.caption("Locked model: logistic regression; alert threshold: 0.225. See docs/OUT_OF_MONTH_RESULTS_2026_10.md.")
else:
    st.info(
        "For external-month performance, run `python -m railpulse ml-cross-month "
        "--month 2026-09 --test-month 2026-10` or place its aggregate artifact in `reports/`."
    )

# Exploratory 4-month development comparison (October was already inspected).
multimonth = reports / "ml_multimonth_comparison.csv"
st.divider()
st.subheader("Multi-month model diagnostics")
if multimonth.is_file():
    data = pd.read_csv(multimonth)
    st.warning(
        "The October dataset was previously inspected. This comparison is **exploratory**, "
        "not an independent blind test. No candidate is promoted automatically."
    )
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.caption(
        "Models trained on July–August; September used for validation; "
        "October used only for previously inspected diagnostics. "
        "Precision, recall and false-alert counts must be considered together."
    )
else:
    st.info(
        "Run `python -m railpulse ml-multimonth` or download the aggregate-only "
        "multi-month GitHub Actions artifact into `reports/`."
    )


st.divider()
st.subheader("Whole-model reliability audit")
st.caption(
    "Research check across all train categories and Dortmund stations. "
    "A model passing software tests is not evidence of correct real-world train-delay probabilities."
)
audit_summary_path = reports / "model_audit_summary.json"
audit_cohorts_path = reports / "model_audit_cohorts.csv"
audit_calibration_path = reports / "model_audit_calibration.csv"
audit_coverage_path = reports / "model_audit_coverage.csv"
if all(path.is_file() for path in (
    audit_summary_path, audit_cohorts_path, audit_calibration_path, audit_coverage_path
)):
    import json
    audit_report = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    st.error(
        "The current model is NOT validated for passenger arrival delays. "
        "The comparison below concerns recorded timetable updates in the archive only."
    )
    overview = []
    for name, details in audit_report["periods"].items():
        overview.append({
            "Period": name,
            "Archived records": details["sample_n"],
            "Observed event share": details["observed_archived_event_fraction"],
            "Average score": details["average_model_score"],
            "Score–outcome gap": details["score_event_gap"],
            "Brier score": details["brier"],
            "ROC AUC": details["roc_auc"],
            "False alerts per 1,000 records": details["false_alerts_per_1000"],
        })
    st.dataframe(pd.DataFrame(overview), hide_index=True, use_container_width=True)
    with st.expander("Station and train-category audit"):
        cohort_table = pd.read_csv(audit_cohorts_path)
        period = st.selectbox("Audit period", ["october_holdout", "september_holdout"])
        st.dataframe(cohort_table[cohort_table["period"] == period],
                     hide_index=True, use_container_width=True)
    with st.expander("Calibration and missing-update coverage"):
        st.caption(
            "A model score of 20% is only meaningful if comparable archived cases "
            "show that label about 20% of the time. This does not verify actual arrivals."
        )
        st.dataframe(pd.read_csv(audit_calibration_path), hide_index=True, use_container_width=True)
        st.dataframe(pd.read_csv(audit_coverage_path), hide_index=True, use_container_width=True)
else:
    st.info(
        "Whole-model audit not generated locally. To check station, train-category, "
        "score calibration and missing update coverage, run "
        "python -m railpulse audit-model (or use its manual GitHub workflow)."
    )

st.divider()
st.caption(
    "Source: Deutsche Bahn timetable updates, CC BY 4.0; archive by Piet Brömmel. "
    "Filtered and aggregated for Dortmund. Not affiliated with Deutsche Bahn. "
    "See docs/DATA_SOURCES.md and docs/METHODS_AND_LIMITATIONS.md."
)
