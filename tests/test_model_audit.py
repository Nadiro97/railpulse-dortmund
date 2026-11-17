"""Whole-model archive audit: every train category, every station, calibration and privacy."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from railpulse.model_audit import (
    _group_row, calibration_bins, evaluate_fixed_archival_model,
    grouped_audit, score_summary, MIN_GROUP_SIZE,
)


def synth_month(year, month, days, rows_per_day=20):
    base = date(year, month, 1)
    data = []
    for day in range(days):
        dt = base + timedelta(days=day)
        for i in range(rows_per_day):
            data.append({
                "station_name": ["Dortmund Hbf", "Dortmund-Kurl", "Dortmund-Hörde"][i % 3],
                "train_type": ["S", "ICE", "RE", "RB", "Bus"][i % 5],
                "scheduled_hour": (day + i + 5) % 24,
                "day_of_week": dt.isoweekday(),
                "planned_date": dt,
                "target_6plus": int(i % 4 == 0 or (day + i) % 7 == 0),
            })
    return pd.DataFrame(data)


def test_month_wide_audit_outputs_only_aggregates(tmp_path):
    sep = synth_month(2026, 9, 30)
    octo = synth_month(2026, 10, 31)
    report = evaluate_fixed_archival_model(sep, octo, reports_root=tmp_path)
    assert report["fit_n"] == 16 * 20
    assert report["target_is_verified_physical_arrival"] is False
    assert report["ready_for_passenger_delay_predictions"] is False
    assert report["october_has_been_inspected_before"] is True
    assert report["needs_verified_arrival_ground_truth"] is True
    assert report["periods"]["october_holdout"]["sample_n"] == len(octo)
    assert report["periods"]["october_holdout"]["reported_groups"] == 8
    summaries = list(tmp_path.glob("*"))
    assert {f.name for f in summaries} == {
        "model_audit_summary.json",
        "model_audit_cohorts.csv",
        "model_audit_calibration.csv",
    }
    text = (tmp_path / "model_audit_summary.json").read_text()
    assert '"planned_date"' not in text and "Dortmund Hbf" not in text
    assert "Dortmund Hbf" in (tmp_path / "model_audit_cohorts.csv").read_text()


def test_cohort_suppression_and_all_category_groups():
    frame = synth_month(2026, 10, 31)
    probabilities = np.linspace(0.05, 0.8, len(frame))
    rows, omitted = grouped_audit(frame, probabilities, minimum=MIN_GROUP_SIZE)
    assert omitted["station_name"] == 0
    assert omitted["train_type"] == 0
    assert {r["value"] for r in rows if r["cohort"] == "station_name"} == {
        "Dortmund Hbf", "Dortmund-Kurl", "Dortmund-Hörde"
    }
    assert {r["value"] for r in rows if r["cohort"] == "train_type"} == {
        "S", "ICE", "RE", "RB", "Bus"
    }
    assert all(r["n"] >= MIN_GROUP_SIZE for r in rows)
    assert _group_row("train_type", "RARE", [1, 0], [0.8, 0.2]) is None


def test_calibration_bins_and_gap():
    frame = synth_month(2026, 9, 30)
    p = np.full(len(frame), 0.2)
    buckets = calibration_bins(frame, p)
    assert len(buckets) == 1
    assert buckets[0]["n"] == len(frame)
    assert buckets[0]["mean_model_score"] == pytest.approx(0.2)
    assert buckets[0]["observed_archived_event_fraction"] == pytest.approx(frame["target_6plus"].mean(), abs=1e-6)
    scored = score_summary(frame["target_6plus"], p)
    assert scored["average_model_score"] == pytest.approx(0.2)
    assert scored["sample_n"] == len(frame)


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        score_summary([0, 1], [0.2, 1.5])
    with pytest.raises(ValueError):
        score_summary([0, 1], [float("nan"), 0.8])
    frame = synth_month(2026, 9, 30)
    with pytest.raises(ValueError):
        grouped_audit(frame, np.ones(len(frame) - 1))


def test_audit_does_not_treat_any_class_as_real_arrival(tmp_path):
    result = evaluate_fixed_archival_model(
        synth_month(2026, 9, 30), synth_month(2026, 10, 31), tmp_path,
    )
    assert "actual" in " ".join(result["notes"]).lower()
    for period in result["periods"].values():
        assert 0 < period["observed_archived_event_fraction"] < 1
