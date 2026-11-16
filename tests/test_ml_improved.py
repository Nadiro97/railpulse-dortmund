"""Synthetic temporal evaluation tests (no railway records)."""
from datetime import date, timedelta
import pandas as pd
import pytest

from railpulse.ml_improved import (
    validation_month_splits, select_threshold, threshold_metrics, fit_improved,
)

def fake_data():
    rows = []
    start = date(2026,9,1)
    for i in range(30):
        for j in range(18):
            d = start + timedelta(days=i)
            rows.append({
                "station_name": ["Dortmund Hbf","Dortmund-Kurl","Dortmund-Hörde"][j%3],
                "train_type": ["RE","S","ICE"][j%3],
                "scheduled_hour": j % 24,
                "day_of_week": d.isoweekday(),
                "planned_date": d,
                "target_6plus": int(j%4==0 or (i+j)%9==0),
            })
    return pd.DataFrame(rows)

def test_disjoint_chronological_split():
    frame = fake_data()
    expanded = pd.concat([frame, frame], ignore_index=True)
    train, val, test = validation_month_splits(expanded, "2026-09")
    assert (len(train), len(val), len(test)) == (576, 144, 360)
    assert train.planned_date.max() < val.planned_date.min()
    assert val.planned_date.max() < test.planned_date.min()

def test_threshold_uses_precision_constraint():
    y = [1,1,0,0,0,0]
    scores = [0.85,0.7,0.6,0.1,0.1,0.1]
    best, okay = select_threshold(y,scores,min_precision=0.6)
    assert okay
    assert best["precision"] >= 0.6
    assert 0.05 <= best["threshold"] <= 0.8
    assert threshold_metrics(y,scores,0.5)["tp"] == 2

def test_realistic_synthetic_pipeline(tmp_path):
    pytest.importorskip("sklearn")
    frame = fake_data()
    # Training/validation/test require >=100 rows each; expand to 30/day.
    extra = frame.copy()
    extra["scheduled_hour"] = (extra["scheduled_hour"]+1)%24
    frame = pd.concat([frame,extra],ignore_index=True)
    result=fit_improved(frame,"2026-09",tmp_path)
    assert result["test_never_used_for_tuning"]
    assert result["selected_model"] in ["logistic_regression","histogram_gradient_boosting"]
    assert result["test_n"] == 360
    assert result["validation_n"] == 144
    assert result["train_n"] == 576
    assert result["test"][result["selected_model"]]["validation_selected_threshold"]["threshold"] == result["selected_threshold"]
    assert (tmp_path/"ml_improved_metrics.json").exists()
    assert (tmp_path/"ml_improved_comparison.csv").exists()
