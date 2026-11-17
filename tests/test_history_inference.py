"""The multi-month predictor must not use the older global quality flag.

Tests exercise legacy research artifact loading (without re-training), catalog
construction using TRAINING months only, numeric archive proxy inference, and
safety labels. The results are synthetic and are not actual DB arrival times.
"""
from pathlib import Path

import joblib
import pandas as pd
import pytest

from railpulse.history_inference import (
    build_service_catalog, estimate_archived_minutes, history_model_path,
    load_history_model, local_diagnostics,
)


def rows(*, target=40, line="RE1", train_number="1234"):
    return pd.DataFrame({
        "station_name": ["Dortmund Hbf"] * target,
        "train_type": ["RE"] * target,
        "line_number": [line] * target,
        "train_number": [train_number] * target,
        "route_origin": ["Bochum Hbf"] * target,
        "route_destination": ["Aachen Hbf"] * target,
        "route_stop_index": [7.] * target,
        "hops_before": [4.] * target,
        "hops_after": [10.] * target,
        "observed_update_minutes": [5.] * target,
    })


def model_record():
    return {
        "schema": "railpulse-rail-history-backtest-v1",
        "mode": "departure",
        "model_name": "training_mean",
        "estimator": 5.25,
        "train_months": ["2024-07"],
        "validation_months": ["2024-08"],
        "not_verified_physical_delay": True,
        "not_approved_for_live_forecast": True,
    }


def test_old_multiyear_artifact_reuses_training_caches_without_retraining(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    data_root = tmp_path / "data"
    path = data_root / "processed/history_rail/2024-07_departure.parquet"
    path.parent.mkdir(parents=True)
    rows().to_parquet(path, index=False)
    # The validation-only route must NEVER enter the training catalogue.
    val_path = path.with_name("2024-08_departure.parquet")
    rows(target=100, line="RE999").to_parquet(val_path, index=False)
    joblib.dump(model_record(), history_model_path("departure", model_root))

    artifact = load_history_model("departure", model_dir=model_root, data_root=data_root)
    assert len(artifact["service_rows"]) == 1
    assert artifact["service_rows"][0]["line_number"] == "RE1"
    assert artifact["service_rows"][0]["support"] == 40
    assert artifact["service_rows"][0]["route_stop_index_median"] == 7.
    assert (model_root / "history_rail_departure_catalog.json").is_file()

    service = artifact["service_rows"][0]
    estimate = estimate_archived_minutes(artifact, service, 15, 2)
    assert estimate["minutes"] == pytest.approx(5.2, abs=0.15)
    assert estimate["is_baseline"] is True
    assert estimate["is_verified_actual_delay"] is False
    assert estimate["can_claim_punctual"] is False
    assert estimate["status"] == "experimental_archive_update"
    # A later load should work even if raw caches are no longer accessible.
    path.unlink()
    second = load_history_model("departure", model_dir=model_root, data_root=data_root)
    assert len(second["service_rows"]) == 1


def test_only_supported_routes_get_numeric_estimates(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    data_root = tmp_path / "data"
    path = data_root / "processed/history_rail/2024-07_departure.parquet"
    path.parent.mkdir(parents=True)
    rows().to_parquet(path, index=False)
    joblib.dump(model_record(), history_model_path("departure", model_root))
    artifact = load_history_model("departure", model_dir=model_root, data_root=data_root)
    choice = artifact["service_rows"][0]
    unknown = {**choice, "line_number": "RE998"}
    estimate = estimate_archived_minutes(artifact, unknown, 12, 2)
    assert estimate["minutes"] is None
    assert estimate["status"] == "not_enough_training_support"
    for hour in (-1, 24, True):
        with pytest.raises(ValueError):
            estimate_archived_minutes(artifact, choice, hour, 1)


def test_status_distinguishes_old_model_from_absent_new_model(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "evidence_minutes_arrival.joblib").write_bytes(b"old-not-loaded")
    result = local_diagnostics("arrival", reports_dir=tmp_path / "reports",
                               model_dir=model_root)
    assert result["has_old_2026_model"] is True
    assert result["has_multi_year_model"] is False
    assert "history-rail" in result["guidance"]
    with pytest.raises(ValueError):
        history_model_path("bus")


def test_training_catalog_uses_only_pre_event_route_dimensions(tmp_path):
    data_root = tmp_path / "data"
    path = data_root / "processed/history_rail/2024-07_departure.parquet"
    path.parent.mkdir(parents=True)
    rows().to_parquet(path, index=False)
    selected = build_service_catalog(["2024-07"], "departure", data_root)
    assert len(selected) == 1
    assert set(selected[0]) == {
        "station_name", "train_type", "line_number", "train_number",
        "route_origin", "route_destination", "support",
        "route_stop_index_median", "hops_before_median",
        "hops_after_median",
    }
    with pytest.raises(FileNotFoundError):
        build_service_catalog(["2024-07", "2024-08"], "departure", data_root)
