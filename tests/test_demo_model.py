"""Synthetic model and safe inference checks; no licensed records in tests."""
from datetime import date, timedelta
import json
import math
import pandas as pd
import pytest

from railpulse.demo_model import (
    MODEL_SCHEMA, ALERT_THRESHOLD, fit_demo, load_artifact,
    predict_probability, validate_artifact,
)


def synthetic_frame():
    rows = []
    start = date(2026, 9, 1)
    for day in range(25):
        for i in range(18):
            rows.append({
                "planned_date": start + timedelta(days=day),
                "station_name": ["Dortmund Hbf", "Dortmund-Kurl"][i % 2],
                "train_type": ["RE", "S", "ICE"][i % 3],
                "scheduled_hour": (5 + i) % 24,
                "day_of_week": (start + timedelta(days=day)).isoweekday(),
                "target_6plus": int(i % 4 == 0 or (day + i) % 9 == 0),
            })
    return pd.DataFrame(rows)


def test_train_only_early_september_and_save_json(tmp_path):
    pytest.importorskip("sklearn")
    artifact_path = tmp_path / "demo.json"
    artifact = fit_demo(synthetic_frame(), artifact_path)
    assert artifact["schema"] == MODEL_SCHEMA
    assert artifact["n_train"] == 16 * 18
    assert artifact["fit_dates"] == ["2026-09-01", "2026-09-16"]
    assert artifact["threshold"] == ALERT_THRESHOLD
    loaded = load_artifact(artifact_path)
    assert not loaded["is_verified_actual_delay"]
    assert not loaded["is_live_prediction"]
    p = predict_probability(loaded, "Dortmund Hbf", "ICE", 8, 1)
    assert 0 < p < 1
    assert math.isfinite(p)
    assert '"planned_date"' not in artifact_path.read_text()
    assert '"target_6plus"' not in artifact_path.read_text()


def test_scores_equal_sklearn_implementation(tmp_path):
    pytest.importorskip("sklearn")
    from railpulse.ml_improved import make_models
    from railpulse.ml_baseline import CATEGORICAL, NUMERIC
    frame = synthetic_frame()
    train = frame.loc[frame["planned_date"] <= date(2026, 9, 16)]
    model = make_models()["logistic_regression"]
    model.fit(train[CATEGORICAL + NUMERIC], train["target_6plus"].astype(int))
    artifact = fit_demo(frame, tmp_path / "demo.json")
    for station in artifact["categories"]["station_name"]:
        for category in artifact["categories"]["train_type"]:
            row = pd.DataFrame([{
                "station_name": station, "train_type": category,
                "scheduled_hour": 8, "day_of_week": 1,
            }])
            expected = float(model.predict_proba(row)[0, 1])
            assert predict_probability(artifact, station, category, 8, 1) == pytest.approx(
                expected, rel=0, abs=1e-10
            )


def test_reject_unknown_or_incorrect_inputs(tmp_path):
    artifact = fit_demo(synthetic_frame(), tmp_path / "demo.json")
    with pytest.raises(ValueError, match="Unknown station"):
        predict_probability(artifact, "Essen Hbf", "RE", 8, 1)
    with pytest.raises(ValueError, match="Unknown train"):
        predict_probability(artifact, "Dortmund Hbf", "RB", 8, 1)
    for bad_hour in (-1, 24, True, 8.5):
        with pytest.raises(ValueError, match="hour"):
            predict_probability(artifact, "Dortmund Hbf", "RE", bad_hour, 1)
    for bad_weekday in (0, 8, True, 3.3):
        with pytest.raises(ValueError, match="Weekday"):
            predict_probability(artifact, "Dortmund Hbf", "RE", 8, bad_weekday)


def test_reject_modified_model_metadata_and_coefficients(tmp_path):
    artifact = fit_demo(synthetic_frame(), tmp_path / "demo.json")
    cases = [
        {"schema": "pickle-v1"},
        {"month": "2026-10"},
        {"threshold": 0.5},
        {"is_verified_actual_delay": True},
        {"is_live_prediction": True},
        {"coefficient_vector": artifact["coefficient_vector"][:-1]},
        {"coefficient_vector": [float("nan")] * len(artifact["coefficient_vector"])},
    ]
    for modification in cases:
        tampered = {**artifact, **modification}
        with pytest.raises(ValueError):
            validate_artifact(tampered)
    huge = tmp_path / "huge.json"
    huge.write_text("x" * 512001)
    with pytest.raises(ValueError, match="large"):
        load_artifact(huge)


def test_training_cannot_switch_month(tmp_path):
    with pytest.raises(ValueError, match="September"):
        fit_demo(synthetic_frame(), tmp_path / "model.json", month="2026-10")


def test_models_directory_not_tracked():
    from pathlib import Path
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "/models/" in ignore
    assert "*.pkl" in ignore
    assert "*.joblib" in ignore
