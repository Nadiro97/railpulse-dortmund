"""Synthetic unseen-month holdout checks; no train records or labels from October to training."""
from datetime import date, timedelta

import pandas as pd
import pytest

from railpulse.ml_cross_month import ALERT_THRESHOLD, MODEL, fixed_partitions, evaluate_locked_months


def synthetic_month(year, month, days, rows_per_day=12):
    base = date(year, month, 1)
    rows = []
    for i in range(days):
        d = base + timedelta(days=i)
        for j in range(rows_per_day):
            rows.append({
                "planned_date": d,
                "station_name": ["Dortmund Hbf","Dortmund-Kurl","Dortmund-Hörde"][j % 3],
                "train_type": ["S", "RE", "ICE"][j % 3],
                "scheduled_hour": (j + i) % 24,
                "day_of_week": d.isoweekday(),
                "target_6plus": int(j % 4 == 0 or (i+j) % 9 == 0),
            })
    return pd.DataFrame(rows)


def test_fixed_model_and_locked_threshold():
    assert MODEL == "logistic_regression"
    assert ALERT_THRESHOLD == 0.225


def test_only_previous_month_used_to_train():
    sep = synthetic_month(2026, 9, 30)
    octo = synthetic_month(2026, 10, 31)
    train, september_holdout, october = fixed_partitions(sep, octo, "2026-09", "2026-10")
    assert train.planned_date.max() == date(2026, 9, 16)
    assert september_holdout.planned_date.min() == date(2026, 9, 21)
    assert september_holdout.planned_date.max() == date(2026, 9, 30)
    assert october.planned_date.min() == date(2026, 10, 1)
    assert len(october) == 372


def test_future_or_same_month_not_accepted():
    frame = synthetic_month(2026, 9, 30)
    with pytest.raises(ValueError, match="after"):
        fixed_partitions(frame, frame, "2026-09", "2026-09")


def test_comparison_outputs_only_aggregates(tmp_path):
    pytest.importorskip("sklearn")
    sep = synthetic_month(2026, 9, 30)
    octo = synthetic_month(2026, 10, 31)
    result = evaluate_locked_months(sep, octo, "2026-09", "2026-10", tmp_path)
    assert result["training_dates"] == ["2026-09-01", "2026-09-16"]
    assert result["unseen_test_dates"] == ["2026-10-01", "2026-10-31"]
    assert result["model_configuration"] == "logistic_regression"
    assert result["locked_probability_threshold"] == 0.225
    assert result["untouched_test_labels_never_used_to_tune"]
    assert result["unseen_month_holdout"]["sample_n"] == 372
    assert (tmp_path / "ml_cross_month_comparison.csv").exists()
    metrics = (tmp_path / "ml_cross_month_metrics.json").read_text()
    assert '"station_name": "Dortmund' not in metrics
