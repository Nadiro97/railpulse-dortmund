"""Synthetic regression tests for route-aware estimates at Dortmund only."""
from datetime import date, datetime, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from railpulse.route_delay import (
    CAT_BASE, CAT_ROUTE, MISSING, NUMERIC, SCHEMA, _model, _probability_model,
    extract_route_rows, fit_route_minutes, estimate_at_dortmund, load_route_model,
)


def month_frame():
    base = date(2026, 9, 1)
    rows = []
    for day in range(30):
        d = base + timedelta(days=day)
        for i in range(25):
            kind = ("S", "RE", "ICE")[i % 3]
            line = ("S1", "RE1", MISSING)[i % 3]
            destination = ("Düsseldorf Hbf", "Aachen Hbf", "Köln Hbf")[i % 3]
            minutes = [0.0, 3.0, 14.0, 20.0, 7.0][(i + day) % 5]
            rows.append({
                "station_name": "Dortmund Hbf" if i % 2 else "Dortmund-Kurl",
                "train_type": kind,
                "line_number": line,
                "final_destination_station": destination,
                "scheduled_hour": 6 + (i % 15),
                "day_of_week": d.isoweekday(),
                "planned_date": d,
                "archived_delay_min": minutes,
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("mode", ["arrival", "departure"])
def test_train_route_model_and_json_inference_equals_sklearn(tmp_path, mode):
    frame = month_frame()
    report = fit_route_minutes(frame, mode, tmp_path / "reports", tmp_path / "models")
    assert set(report["validation"]) == {
        "mean_baseline", "station_type_hour", "station_line_destination"
    }
    artifact = load_route_model(mode, tmp_path / "models")
    assert artifact["schema"] == SCHEMA
    assert artifact["n_train"] == 16 * 25
    assert not artifact["is_verified_real_arrival"]
    assert not artifact["route_origin_known"]
    assert artifact["mode"] == mode
    route = artifact["routes"][0]
    result = estimate_at_dortmund(artifact, route, 11, 1)
    assert 0 <= result["estimated_archived_delay_minutes"] <= 240
    assert 0 <= result["estimated_archived_6plus_probability"] <= 1
    assert result["is_verified_actual_delay"] is False

    original_train = frame[frame["planned_date"] <= date(2026, 9, 16)]
    reference = _model(CAT_ROUTE)
    reference.fit(original_train[CAT_ROUTE + NUMERIC], original_train["archived_delay_min"])
    probability_reference = _probability_model()
    probability_reference.fit(original_train[CAT_ROUTE + NUMERIC],
                              original_train["archived_delay_min"] >= 6)
    example = pd.DataFrame([{**{k: route[k] for k in CAT_ROUTE},
                              "scheduled_hour": 11, "day_of_week": 1}])
    expected = float(np.clip(reference.predict(example)[0], 0, 240))
    expected_prob = float(probability_reference.predict_proba(example)[0, 1])
    assert result["estimated_archived_delay_minutes"] == pytest.approx(round(expected, 1))
    assert result["estimated_archived_6plus_probability"] == pytest.approx(round(expected_prob, 4))
    assert set((tmp_path / "models").glob("*.json"))
    assert "Dortmund Hbf" not in (tmp_path / "reports" /
                                f"route_minutes_{mode}_metrics.json").read_text()


def test_test_labels_cannot_affect_fitted_model(tmp_path):
    frame = month_frame()
    fit_route_minutes(frame, "arrival", tmp_path / "a", tmp_path / "a-model")
    original = load_route_model("arrival", tmp_path / "a-model")
    mutated = frame.copy()
    mutated.loc[mutated["planned_date"] >= date(2026, 9, 21), "archived_delay_min"] = 99
    fit_route_minutes(mutated, "arrival", tmp_path / "b", tmp_path / "b-model")
    future_changed = load_route_model("arrival", tmp_path / "b-model")
    assert original["minutes_model"] == future_changed["minutes_model"]
    assert original["probability_model"] == future_changed["probability_model"]
    assert original["validation_route_mae"] == future_changed["validation_route_mae"]


def test_unknown_route_and_bad_inputs_rejected(tmp_path):
    fit_route_minutes(month_frame(), "arrival", tmp_path, tmp_path / "models")
    artifact = load_route_model("arrival", tmp_path / "models")
    known = artifact["routes"][0]
    with pytest.raises(ValueError, match="not observed"):
        estimate_at_dortmund(artifact, {**known, "line_number": "invented"}, 9, 1)
    for h in (-1, 24, True, 5.3):
        with pytest.raises(ValueError, match="hour"):
            estimate_at_dortmund(artifact, known, h, 1)
    for wd in (0, 8, True, 1.3):
        with pytest.raises(ValueError, match="Weekday"):
            estimate_at_dortmund(artifact, known, 9, wd)
    wrong = {**artifact, "route_origin_known": True}
    with pytest.raises(ValueError, match="origin"):
        estimate_at_dortmund(wrong, known, 9, 1)
    missing_weights = {**artifact, "minutes_model": {
        **artifact["minutes_model"],
        "weights": artifact["minutes_model"]["weights"][:-1],
    }}
    with pytest.raises(ValueError, match="shape"):
        estimate_at_dortmund(missing_weights, known, 9, 1)


def test_reject_missing_labels():
    frame = month_frame()
    frame["archived_delay_min"] = np.nan
    with pytest.raises(ValueError, match="between 0 and 240"):
        fit_route_minutes(frame, "arrival")


def test_extract_separate_departure_and_arrival_from_explicit_snapshots(tmp_path):
    pytest.importorskip("duckdb")
    processed = pd.DataFrame([
        {
            "id": "stop1", "eva": "8000080",
            "station_name": "Dortmund Hbf",
            "train_type": "RE", "line_number": "RE1",
            "final_destination_station": "Aachen Hbf",
            "arrival_planned_time": pd.Timestamp("2026-09-07 08:00:00"),
            "departure_planned_time": pd.Timestamp("2026-09-07 08:05:00"),
            "arrival_is_canceled": False, "departure_is_canceled": False,
            "has_plan_record": True,
            "is_additional_stop": False, "is_replacement_train": False,
        },
        {
            "id": "cancelled", "eva": "8000080",
            "station_name": "Dortmund Hbf",
            "train_type": "S", "line_number": "S1",
            "final_destination_station": "Düsseldorf Hbf",
            "arrival_planned_time": pd.Timestamp("2026-09-07 09:00:00"),
            "departure_planned_time": pd.Timestamp("2026-09-07 09:05:00"),
            "arrival_is_canceled": True, "departure_is_canceled": True,
            "has_plan_record": True,
            "is_additional_stop": False, "is_replacement_train": False,
        },
    ])
    parsed = pd.DataFrame([
        {"id": "stop1", "eva": "8000080",
         "snapshot_timestamp": pd.Timestamp("2026-09-07 07:00:00"),
         "arrival_change_time": pd.Timestamp("2026-09-07 08:07:00"),
         "departure_change_time": pd.Timestamp("2026-09-07 08:07:00")},
        {"id": "stop1", "eva": "8000080",
         "snapshot_timestamp": pd.Timestamp("2026-09-07 08:10:00"),
         "arrival_change_time": pd.Timestamp("2026-09-07 08:11:00"),
         "departure_change_time": pd.Timestamp("2026-09-07 08:18:00")},
        {"id": "cancelled", "eva": "8000080",
         "snapshot_timestamp": pd.Timestamp("2026-09-07 08:30:00"),
         "arrival_change_time": pd.Timestamp("2026-09-07 09:30:00"),
         "departure_change_time": pd.Timestamp("2026-09-07 09:40:00")},
    ])
    pp, cc = tmp_path / "processed.parquet", tmp_path / "changes.parquet"
    processed.to_parquet(pp)
    parsed.to_parquet(cc)
    arrival = extract_route_rows("2026-09", pp, cc, "arrival")
    departure = extract_route_rows("2026-09", pp, cc, "departure")
    assert len(arrival) == len(departure) == 1
    assert arrival.iloc[0]["archived_delay_min"] == pytest.approx(11)
    assert departure.iloc[0]["archived_delay_min"] == pytest.approx(13)
    assert arrival.iloc[0]["line_number"] == "RE1"
    assert departure.iloc[0]["final_destination_station"] == "Aachen Hbf"
    assert departure.iloc[0]["station_name"] == "Dortmund Hbf"
    # A parsed snapshot with no departure update cannot establish punctuality.
    parsed["departure_change_time"] = pd.NaT
    parsed.to_parquet(cc)
    assert extract_route_rows("2026-09", pp, cc, "departure").empty
