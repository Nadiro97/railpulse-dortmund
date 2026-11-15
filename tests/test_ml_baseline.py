"""ML tests use synthetic dates and fake station-stop patterns only."""
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from railpulse.ml_baseline import month_bounds, fit_and_evaluate, CATEGORICAL, NUMERIC

def sample_frame():
    rows = []
    base = date(2026, 9, 1)
    for day in range(30):
        for j in range(14):
            timestamp = base + timedelta(days=day)
            rows.append({
                "station_name": "Dortmund Hbf" if j % 2 else "Dortmund-Kurl",
                "train_type": "RE" if j % 3 else "S",
                "scheduled_hour": 6 + j,
                "day_of_week": timestamp.isoweekday(),
                "planned_date": timestamp,
                "target_6plus": int(j % 3 == 0 or (day + j) % 7 == 0),
            })
    return pd.DataFrame(rows)

def test_month_split():
    assert month_bounds("2026-09") == (date(2026, 9, 1), date(2026, 9, 21), date(2026, 10, 1))
    assert month_bounds("2026-12") == (date(2026, 12, 1), date(2026, 12, 21), date(2027, 1, 1))

def test_no_leaky_features():
    assert CATEGORICAL == ["station_name", "train_type"]
    assert NUMERIC == ["scheduled_hour", "day_of_week"]
    assert "delay_in_min" not in CATEGORICAL + NUMERIC
    assert "arrival_change_time" not in CATEGORICAL + NUMERIC

def test_temporal_metrics_and_no_raw_records(tmp_path):
    pytest.importorskip("sklearn")
    report = fit_and_evaluate(sample_frame(), "2026-09", tmp_path / "reports")
    assert report["train_n"] == 280
    assert report["test_n"] == 140
    assert report["train_window"] == ["2026-09-01", "2026-09-20"]
    assert report["test_window"] == ["2026-09-21", "2026-09-30"]
    assert 0 <= report["logistic_regression"]["roc_auc"] <= 1
    assert 0 <= report["prevalence_baseline"]["brier"] <= 1
    assert not report["label_is_confirmed_physical_arrival"]
    assert not any(x in report["features_used"] for x in report["forbidden_post_event_features"])
    assert (tmp_path / "reports" / "ml_baseline_comparison.csv").exists()
    contents = (tmp_path / "reports" / "ml_baseline_metrics.json").read_text()
    assert '"station_name": "Dortmund Hbf"' not in contents

def test_reject_small_samples(tmp_path):
    with pytest.raises(ValueError, match="100"):
        fit_and_evaluate(sample_frame().head(30), "2026-09", tmp_path)
