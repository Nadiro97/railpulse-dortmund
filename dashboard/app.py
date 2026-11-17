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
st.subheader("Verspätung am Dortmunder Halt — Forschungsmodell")
st.caption(
    "Das neue Mehrmonatsmodell ist separat vom alten Juli–Oktober-Modell. "
    "Diese Schätzung ist eine historische DB-Zeitplanänderung in Minuten, "
    "keine unabhängig bestätigte tatsächliche Zugverspätung."
)
from railpulse.history_inference import (
    history_model_path, load_history_model, estimate_archived_minutes, local_diagnostics,
)

event = st.radio(
    "Am Dortmunder Halt", ("Abfahrt aus Dortmund", "Ankunft in Dortmund"),
    horizontal=True, key="history_event",
)
mode = "departure" if event.startswith("Abfahrt") else "arrival"
status = local_diagnostics(mode)

if status["has_multi_year_model"]:
    try:
        local = load_history_model(mode)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        st.error(
            "Das Mehrmonatsmodell kann nicht geladen werden: "
            f"{exc}. Prüfe die lokalen Trainingsdateien bzw. den Bericht."
        )
        local = None
    if local is not None:
        services = local["service_rows"]
        if not services:
            st.warning(
                "Kein Zug/Linien-Verlauf hat mindestens 20 passende "
                "Trainingsbeobachtungen. Keine Schätzung möglich."
            )
        else:
            station = st.selectbox(
                "Dortmunder Bahnhof",
                sorted({r["station_name"] for r in services}),
                key="history_station",
            )
            candidates = [r for r in services if r["station_name"] == station]
            category = st.selectbox(
                "Zugart", sorted({r["train_type"] for r in candidates}),
                key="history_train_type",
            )
            candidates = [r for r in candidates if r["train_type"] == category]
            line = st.selectbox(
                "Linie", sorted({r["line_number"] for r in candidates}),
                key="history_line",
            )
            candidates = [r for r in candidates if r["line_number"] == line]
            train_number = st.selectbox(
                "Zugnummer (aus echten Trainingsverbindungen)",
                sorted({r["train_number"] for r in candidates}),
                key="history_train_number",
            )
            candidates = [r for r in candidates if r["train_number"] == train_number]
            origin = st.selectbox(
                "Erster aufgeführter vorheriger Halt (nicht zwingend Start)",
                sorted({r["route_origin"] for r in candidates}),
                key="history_origin",
            )
            candidates = [r for r in candidates if r["route_origin"] == origin]
            destination = st.selectbox(
                "Letzter aufgeführter Zielhalt (nicht zwingend Fahrgastziel)",
                sorted({r["route_destination"] for r in candidates}),
                key="history_destination",
            )
            chosen = next(r for r in candidates
                          if r["route_destination"] == destination)
            left, right = st.columns(2)
            with left:
                hour = st.selectbox(
                    "Planmäßige Stunde in Dortmund", list(range(24)),
                    index=8, key="history_hour",
                )
            with right:
                weekday = st.selectbox(
                    "Wochentag", [
                        "Montag", "Dienstag", "Mittwoch", "Donnerstag",
                        "Freitag", "Samstag", "Sonntag",
                    ], key="history_weekday",
                )
            st.caption(
                f"Trainingsbeispiele für diese konkrete Linie/Richtung/Zugnummer: "
                f"{chosen['support']}. Alle Eingaben stammen aus früheren Fahrplandaten."
            )
            if st.button("Archivierte Minuten schätzen", type="primary",
                         key="history_estimate"):
                estimate = estimate_archived_minutes(
                    local, chosen, int(hour),
                    ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag",
                     "Samstag", "Sonntag"].index(weekday) + 1,
                )
                if estimate["minutes"] is None:
                    st.warning(estimate["reason"])
                else:
                    st.metric(
                        "Experimentell geschätzte Zeitplanabweichung in Dortmund",
                        f"{estimate['minutes']:.1f} Minuten",
                    )
                    if estimate["is_baseline"]:
                        st.warning(
                            "Auf der Validierung gewann nur ein einfaches "
                            "Vergleichsmodell. Diese Zahl ist keine belastbare "
                            "individuelle Verspätungsprognose."
                        )
                    elif estimate["minutes"] <= 1.5:
                        st.info(
                            "Der archivbasierte Modellwert liegt nahe null. "
                            "Das beweist nicht 'Pünktlich' im tatsächlichen Betrieb."
                        )
                    st.warning(
                        "WICHTIG: Nur ein experimenteller Zahlenwert für alte "
                        "API-Fahrplanänderungen. Keine Live-Daten und keine "
                        "bestätigte tatsächliche Ankunfts-/Abfahrtsverspätung."
                    )
        st.caption(
            "Trainiert auf historischen Monaten: "
            + ", ".join(local["train_months"])
            + " | Validierungsmonate: "
            + ", ".join(local["validation_months"])
            + " | Modell: "
            + str(local["model_name"])
        )
elif status["has_old_2026_model"]:
    st.warning(
        "Gefunden: nur das alte Juli–Oktober-Modell. Dessen globaler "
        "Qualitätsfilter blockierte alle Linien. Das neue Mehrmonatsmodell "
        "wurde hier nicht erfolgreich erzeugt oder ist noch nicht vorhanden."
    )
    st.code(
        r'.\.venv\Scripts\python.exe -m railpulse history-rail --mode both',
        language="powershell",
    )
else:
    st.info(
        "Noch kein abgeschlossenes Mehrmonats-Forschungsmodell vorhanden. "
        "Eine Quelleninventur allein erzeugt keine Modellschätzung."
    )
    st.code(
        r'.\.venv\Scripts\python.exe -m railpulse history-rail --mode both',
        language="powershell",
    )

if status.get("training_blocker"):
    st.error("Grund für fehlendes Training: " + status["training_blocker"])
    if status.get("excluded_months"):
        with st.expander("Warum wurden Monate ausgeschlossen?"):
            st.dataframe(pd.DataFrame([
                {"Monat": month, "Grund": reason}
                for month, reason in status["excluded_months"].items()
            ]), hide_index=True, use_container_width=True)

if status["has_multi_year_evaluation"]:
    import json
    multi_report = json.loads(
        Path(f"reports/history_rail_{mode}_evaluation.json")
        .read_text(encoding="utf-8")
    )
    with st.expander("Wie genau war die Mehrmonats-Schätzung?"):
        val = multi_report["validation_models"][multi_report["selected_model"]]
        test = multi_report["test_overall"]
        left, right = st.columns(2)
        left.metric("Validierung — MAE", f"{val['mae_minutes']:.2f} min")
        right.metric("Rückblickender Test — MAE", f"{test['mae_minutes']:.2f} min")
        st.caption(
            f"Validierung — gewichteter MAE: {val['weighted_mae_minutes']:.2f} min. "
            f"Rückblickender Test — gewichteter MAE: "
            f"{test['weighted_mae_minutes']:.2f} min. "
            "Auf den historischen API-Zeitplanupdates, NICHT auf verifizierten "
            "echten Ankünften. Die Testmonate wurden bereits erforscht."
        )
        st.caption(
            "Der ausgewählte Algorithmus überschreitet die "
            "3%-Verbesserungshürde gegen das Null-Minuten-Basismodell: "
            + ("Ja" if multi_report.get("selected_beats_zero_baseline_by_at_least_3pct")
               else "Nein")
        )
else:
    st.caption(
        "Kein lokaler Mehrmonats-Evaluationsbericht vorhanden. "
        "Prüfe reports/history_rail_<arrival|departure>_quality.json."
    )

st.divider()
st.subheader("Mehrjährige Datenprüfung · Schienenverkehr")
st.caption(
    "Monthly inventory and chronological holdouts for S/RE/RB/ICE/IC/EC "
    "and other reviewed railway-service codes. Local bus services are excluded. "
    "Older months may have much narrower station coverage."
)
inventory_file = Path("reports/history_month_inventory.json")
if inventory_file.is_file():
    import json
    inventory_report = json.loads(inventory_file.read_text(encoding="utf-8"))
    monthly = pd.DataFrame(inventory_report["rows"])
    st.metric("Monate mit allen drei Archiven",
              f"{inventory_report['months_with_all_sources']} / {inventory_report['months_catalogued']}")
    st.dataframe(monthly, hide_index=True, use_container_width=True)
    if inventory_report.get("split"):
        split = inventory_report["split"]
        st.caption(
            f"Quellen-Metadaten-Split: {len(split['train'])} Monate Training, "
            f"{len(split['validation'])} Validierung, "
            f"{len(split['test'])} retrospektive Tests. "
            "Qualitätsprüfungen können Monate vor dem endgültigen Split ausschließen."
        )
else:
    st.info(
        "Keine historische Quellen-Inventur vorhanden. Zuerst "
        "python -m railpulse history-rail --inventory-only ausführen — "
        "keine großen Parquet-Downloads nötig."
    )

for event in ("arrival", "departure"):
    path = Path(f"reports/history_rail_{event}_evaluation.json")
    if not path.is_file():
        continue
    import json
    evaluated = json.loads(path.read_text(encoding="utf-8"))
    st.subheader(f"Mehrmonats-Backtest: {'Ankunft' if event == 'arrival' else 'Abfahrt'} Dortmund")
    st.caption(
        "Retrospektive, bereits erforschte Testmonate; historische API-Zeitplanupdates, "
        "nicht verifizierte tatsächliche Zugverspätungen."
    )
    stats = evaluated["test_overall"]
    aa, bb, cc = st.columns(3)
    aa.metric("MAE (Minuten)", f"{stats['mae_minutes']:.1f}")
    bb.metric("Gewichteter MAE (Minuten)", f"{stats['weighted_mae_minutes']:.1f}")
    cc.metric("Getestete Stops", f"{stats['n']:,}")
    st.caption(f"Gewähltes Modell auf Validierung: {evaluated['selected_model']}")
    monthly_metrics = pd.DataFrame([
        {"Monat": m, "MAE (Minuten)": r["models"][evaluated["selected_model"]]["mae_minutes"],
         "Gewichteter MAE": r["models"][evaluated["selected_model"]]["weighted_mae_minutes"],
         "Stops": r["n"]}
        for m, r in evaluated["test_monthly"].items()
    ])
    if not monthly_metrics.empty:
        st.plotly_chart(
            px.line(monthly_metrics, x="Monat", y="MAE (Minuten)", markers=True,
                    title=f"Archivierte Zeitplanänderung: Testfehler je Monat ({event})"),
            use_container_width=True,
        )
        st.dataframe(monthly_metrics, hide_index=True, use_container_width=True)

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
