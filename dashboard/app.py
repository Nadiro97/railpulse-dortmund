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

st.divider()
st.caption(
    "Source: Deutsche Bahn timetable updates, CC BY 4.0; archive by Piet Brömmel. "
    "Filtered and aggregated for Dortmund. Not affiliated with Deutsche Bahn. "
    "See docs/DATA_SOURCES.md and docs/METHODS_AND_LIMITATIONS.md."
)
