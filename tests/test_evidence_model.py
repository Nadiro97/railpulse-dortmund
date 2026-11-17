"""Contract and synthetic tests for evidence-first Dortmund punctuality research."""
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from railpulse.event_data import extract_labels, PLAN_LEAD_MINUTES
from railpulse.evidence_model import (
    FEATURES, UNKNOWN, build_features, metrics, train_mode, estimate_minutes,
    load_local_model, MIN_SERVICE_SUPPORT,
)


def fake_month(y, m, days):
    start = date(y, m, 1)
    data = []
    for d in range(days):
        day = start + timedelta(days=d)
        for k in range(16):
            kind = ["S", "RE", "ICE", "RB"][k % 4]
            late = 18.0 if (k + d) % 7 == 0 else (
                7.0 if (k + d) % 4 == 0 else 0.0
            )
            data.append({
                "station_name": "Dortmund Hbf",
                "train_type": kind,
                "line_number": ["S1", "RE1", UNKNOWN, "RB59"][k % 4],
                "train_number": str(1000 + k % 4),
                "route_origin": ["Bochum Hbf", "Hamm Hbf", "Köln Hbf", "Unna"][k % 4],
                "route_destination": ["Düsseldorf Hbf", "Aachen Hbf",
                                      "Berlin Hbf", "Soest"][k % 4],
                "planned_date": day, "scheduled_hour": 8 + k % 10,
                "weekday": day.isoweekday(),
                "route_stop_index": float(k % 8),
                "hops_before": float(k % 8),
                "hops_after": float(10 - k % 8),
                "observed_update_minutes": late,
            })
    return pd.DataFrame(data)


def test_feature_engineering_contains_only_scheduled_properties():
    frame = build_features(fake_month(2026, 9, 8))
    assert set(FEATURES) <= set(frame.columns)
    assert "observed_update_minutes" not in FEATURES
    assert "latest_snapshot" not in FEATURES
    assert "raw_change_minutes" not in FEATURES
    assert frame["route_progress"].between(0, 1).all()
    assert frame["hour_sin"].between(-1, 1).all()
    assert frame["hour_cos"].between(-1, 1).all()


def test_punctual_naive_model_is_penalized_for_severe_delays():
    y = [0, 0, 0, 0, 20]
    pred = [0, 0, 0, 0, 0]
    summary = metrics(y, pred)
    assert summary["weighted_mae_minutes"] > summary["mae_minutes"]
    assert summary["severe_missed_as_punctual_fraction"] == 1.0
    assert summary["punctual_claim_precision"] == 0.8
    assert summary["punctual_claim_count"] == 5


def test_train_only_earlier_months_then_select_without_october(tmp_path):
    pytest.importorskip("sklearn")
    frames = {
        "2026-07": fake_month(2026, 7, 31),
        "2026-08": fake_month(2026, 8, 31),
        "2026-09": fake_month(2026, 9, 30),
        "2026-10": fake_month(2026, 10, 31),
    }
    coverage = {k: {"label_coverage_fraction": 0.85} for k in frames}
    report = train_mode(frames, coverage, "arrival", tmp_path / "models", tmp_path / "reports")
    assert report["train_months"] == ["2026-07", "2026-08"]
    assert report["selection_month"] == "2026-09"
    assert report["previously_inspected_diagnostic_month"] == "2026-10"
    assert report["is_independent_physical_delay"] is False
    assert report["model_was_promoted_to_real_delay"] is False
    assert report["validation_count"] == 30 * 16
    assert "always_punctual" in report["validation"]
    assert "route_boost" in report["validation"]
    assert "simple_ridge" in report["validation"]
    assert report["supported_service_combinations"] >= 4
    artifact = load_local_model("arrival", tmp_path / "models")
    assert artifact["observed_update_proxy_only"] is True
    service = artifact["service_rows"][0]
    result = estimate_minutes(artifact, service, 8, 1)
    assert result["status"].startswith(("Pünktlich", "Voraussichtliche Verspätung",
                                        "Keine belastbare Schätzung"))
    if not report["eligible_for_user_facing_estimates"]:
        assert result["minutes"] is None
    assert json.loads(
        (tmp_path / "reports" / "evidence_minutes_arrival_evaluation.json").read_text()
    )["is_independent_physical_delay"] is False


def test_unsupported_service_cannot_claim_punctuality():
    from railpulse.evidence_model import estimate_minutes
    artifact = {
        "service_rows": [{
            "station_name": "Dortmund Hbf", "train_type": "S",
            "line_number": "S1", "route_origin": "Bochum",
            "route_destination": "Düsseldorf", "support": MIN_SERVICE_SUPPORT,
        }],
        "deployment_allowed": False,
    }
    service = {**artifact["service_rows"][0], "line_number": "S99"}
    output = estimate_minutes(artifact, service, 8, 1)
    assert output["minutes"] is None
    assert "belastbare" in output["status"]
    for bad_hour in [-1, 24, True, 8.3]:
        with pytest.raises(ValueError, match="Hour"):
            estimate_minutes(artifact, service, bad_hour, 1)


@pytest.mark.parametrize("mode", ["arrival", "departure"])
def test_snapshot_label_requires_event_time_after_planned_and_early_plan(tmp_path, mode):
    pytest.importorskip("duckdb")
    start = pd.Timestamp("2026-09-07 08:00:00")
    rows = []
    for name in ["valid", "missing"]:
        rows.append({
            "id": name, "eva": "8000080",
            "station_name": "Dortmund Hbf",
            "arrival_planned_time": start,
            "departure_planned_time": start + pd.Timedelta(minutes=5),
            "arrival_is_canceled": False, "departure_is_canceled": False,
            "has_plan_record": True, "is_additional_stop": False,
            "is_replacement_train": False,
            "train_line_ride_id": name, "train_line_station_num": 4,
        })
    processed = pd.DataFrame(rows)
    plan_rows = []
    for name in ("valid", "missing"):
        for snap, dest in [
            (pd.Timestamp("2026-09-07 05:00"), "Düsseldorf Hbf"),
            (pd.Timestamp("2026-09-07 07:45"), "changed-too-late"),
        ]:
            plan_rows.append({
                "id": name, "eva": "8000080",
                "snapshot_timestamp": snap,
                "train_type": "S", "train_number": "999",
                "arrival_line_number": "S1", "departure_line_number": "S1",
                "arrival_planned_path": ["Witten Hbf", "Bochum Hbf"],
                "departure_planned_path": ["Dortmund-Kurl", dest],
            })
    plan = pd.DataFrame(plan_rows)
    changes = pd.DataFrame([
        {
            "id": "valid", "eva": "8000080",
            "snapshot_timestamp": pd.Timestamp("2026-09-07 07:50"),
            "arrival_change_time": start + pd.Timedelta(minutes=2),
            "departure_change_time": start + pd.Timedelta(minutes=9),
        },
        {
            "id": "valid", "eva": "8000080",
            "snapshot_timestamp": pd.Timestamp("2026-09-07 08:30"),
            "arrival_change_time": start + pd.Timedelta(minutes=12),
            "departure_change_time": start + pd.Timedelta(minutes=17),
        },
        {
            "id": "missing", "eva": "8000080",
            "snapshot_timestamp": pd.Timestamp("2026-09-07 07:50"),
            "arrival_change_time": start,
            "departure_change_time": start + pd.Timedelta(minutes=5),
        },
    ])
    files = {}
    for key, frame in (("dortmund", processed), ("plan", plan), ("change", changes)):
        path = tmp_path / f"{key}.parquet"
        frame.to_parquet(path)
        files[key] = path
    out, quality = extract_labels("2026-09", mode, files)
    assert len(out) == 1
    assert quality["scheduled_eligible"] == 2
    assert quality["labelled_eligible"] == 1
    assert quality["known_post_event_explicit_change"] == 1
    assert quality["missing_updates_are_assumed_on_time"] is False
    assert quality["pre_event_plan_lead_minutes"] == PLAN_LEAD_MINUTES
    assert out.iloc[0]["route_destination"] == "Düsseldorf Hbf"
    assert out.iloc[0]["route_origin"] == "Witten Hbf"
    assert out.iloc[0]["observed_update_minutes"] == pytest.approx(
        12 if mode == "arrival" else 12
    )
    assert out.iloc[0]["train_number"] == "999"
    assert out.iloc[0]["line_number"] == "S1"
