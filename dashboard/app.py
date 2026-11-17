"""RailPulse Dortmund — evidence-first archival-update minutes research UI.

Never imply that an API timetable update proves physical punctuality.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from railpulse.evidence_model import (
    UNKNOWN, model_path, report_path, load_local_model, estimate_minutes,
)

st.set_page_config(page_title="RailPulse Dortmund", page_icon="🚆", layout="wide")
st.title("RailPulse Dortmund 🚆")
st.caption("Dortmund as the measurement point · DB timetable archive · experimental research")
st.error(
    "WICHTIG: Our data records timetable *updates*, not independently confirmed "
    "physical train arrivals. No reliable real-world Verspätung prediction is "
    "established yet. Never rely on this demo to catch a train."
)
st.subheader("Voraussichtliche Verspätung am Dortmunder Halt")
st.caption(
    "First choose whether to measure arrival AT Dortmund or departure FROM Dortmund. "
    "A final destination is NOT proof that the service stops at every intermediate city. "
    "Bus-labelled entries are DB timetable records, not a full city-bus feed."
)
event = st.radio(
    "Dortmund event", ("Abfahrt aus Dortmund", "Ankunft in Dortmund"),
    horizontal=True,
)
mode = "departure" if event.startswith("Abfahrt") else "arrival"
if model_path(mode).is_file():
    try:
        local = load_local_model(mode)
    except (KeyError, ValueError, OSError, TypeError) as exc:
        st.error(f"Invalid local model: {exc}")
        local = None
    if local:
        services = local["service_rows"]
        station = st.selectbox("Dortmunder Bahnhof", sorted({x["station_name"] for x in services}))
        choices = [x for x in services if x["station_name"] == station]
        category = st.selectbox("Zug-/Linienart", sorted({x["train_type"] for x in choices}))
        choices = [x for x in choices if x["train_type"] == category]
        line = st.selectbox("Linie (falls bekannt)", sorted({x["line_number"] for x in choices}))
        choices = [x for x in choices if x["line_number"] == line]
        origin = st.selectbox(
            "Erster aufgeführter vorheriger Halt (nicht garantiert Zugstart)",
            sorted({x["route_origin"] for x in choices}),
        )
        choices = [x for x in choices if x["route_origin"] == origin]
        dest = st.selectbox(
            "Letzter aufgeführter Zielhalt (nicht garantiert Fahrgastziel)",
            sorted({x["route_destination"] for x in choices}),
        )
        choice = next(x for x in choices if x["route_destination"] == dest)
        left, right = st.columns(2)
        with left:
            hour = st.selectbox("Planmäßige Stunde in Dortmund", list(range(24)), index=8)
        with right:
            weekday = st.selectbox(
                "Wochentag", ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                               "Freitag", "Samstag", "Sonntag"]
            )
        st.caption(
            f"Historische Beispiele für diese Haltestelle/Linie/Richtung: {choice['support']}. "
            "Das ist keine Verifikation der tatsächlichen Ankunft."
        )
        if st.button("Verspätung schätzen", type="primary"):
            output = estimate_minutes(
                local, choice, int(hour),
                ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                 "Freitag", "Samstag", "Sonntag"].index(weekday) + 1,
            )
            if output["minutes"] is None:
                st.warning(
                    "Keine belastbare Schätzung: Datenqualität oder Testgüte reicht "
                    "für eine seriöse Minutenangabe nicht aus."
                )
            elif output["minutes"] == 0:
                st.success("Pünktlich")
            else:
                st.warning(output["status"])
            st.caption(
                "Das Ergebnis beschreibt ausschließlich historische Zeitplanänderungen "
                "an diesem Dortmunder Halt. Keine Echtzeitdaten oder verifizierten "
                "tatsächlichen Ankunfts-/Abfahrtszeiten."
            )
        if report_path(mode).is_file():
            import json
            report = json.loads(report_path(mode).read_text(encoding="utf-8"))
            st.subheader("Wie gut ist dieses Modell wirklich?")
            s = report["validation"].get(report["selected"], {})
            a,b,c = st.columns(3)
            a.metric("Validierung: Fehler in Minuten (MAE)", f"{s.get('mae_minutes', float('nan')):.1f}")
            b.metric("Gewichteter Fehler (MAE)", f"{s.get('weighted_mae_minutes', float('nan')):.1f}")
            c.metric("Training auf historischen Stops", f"{report['training_count']:,}")
            if not report["eligible_for_user_facing_estimates"]:
                st.warning("Model quality gates failed: we refuse to show invented certainty.")
            st.caption(
                "Train: July–August 2026 · Selection: September 2026 · October diagnostics "
                "were already inspected. See the methodology before interpreting these scores."
            )
else:
    st.info(
        "No rebuilt local model yet. In the project directory run "
        r'.\.venv\Scripts\python.exe -m railpulse train-evidence --mode both '
        "after git pull. This downloads large public monthly archives locally."
    )
st.divider()
st.subheader("Historical archived timetable analytics (not ground-truth delay statistics)")
st.caption("These legacy research charts are kept for reproducibility, not live forecasts.")

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
