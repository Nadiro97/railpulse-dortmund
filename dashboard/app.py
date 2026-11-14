"""Local-only Streamlit dashboard for Dartmouth? No: Dortmund railway data."""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="RailPulse Dortmund", layout="wide")
st.title("RailPulse Dortmund 🚆")
st.caption("Independent educational analysis of Deutsche Bahn timetable updates")
st.warning("Changed arrival times can be predictions, not confirmed actual arrivals. Missing changes are not interpreted as punctual.")
reports = Path("reports")
needed = [reports / f"{name}.csv" for name in ("quality", "by_station", "by_hour", "by_train_type")]
if not all(p.exists() for p in needed):
    st.info("No local data processed yet. Run: python -m railpulse run --month 2026-09")
    st.stop()

quality = pd.read_csv(reports / "quality.csv")
st.subheader("Coverage and data quality")
st.dataframe(quality, use_container_width=True, hide_index=True)
by_station = pd.read_csv(reports / "by_station.csv")
by_hour = pd.read_csv(reports / "by_hour.csv")
by_type = pd.read_csv(reports / "by_train_type.csv")
st.subheader("Changed arrival-time difference by station")
st.plotly_chart(px.bar(
    by_station, x="station_name", y="mean_changed_arrival_delta_min",
    hover_data=["eligible_rows", "share_6min_plus_pct"],
    labels={"station_name": "Dortmund station",
            "mean_changed_arrival_delta_min": "Mean change relative to schedule (min)"}
), use_container_width=True)
st.subheader("Arrival-time difference by scheduled hour")
st.plotly_chart(px.line(
    by_hour, x="planned_hour", y="mean_changed_arrival_delta_min", markers=True,
    labels={"planned_hour": "Scheduled hour (Europe/Berlin)",
            "mean_changed_arrival_delta_min": "Mean change (minutes)"}
), use_container_width=True)
st.subheader("By train type")
st.dataframe(by_type, use_container_width=True, hide_index=True)
st.caption("Source: Deutsche Bahn, CC BY 4.0; archive by Piet Brömmel. See docs/DATA_SOURCES.md.")
