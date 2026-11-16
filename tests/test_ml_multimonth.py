"""Synthetic-only tests: month chronology, no benchmark-based model tuning."""
from datetime import date, timedelta
import pandas as pd
import pytest

from railpulse.ml_multimonth import (
    TRAIN_MONTHS, VALIDATION_MONTH, BENCHMARK_MONTH, TARGET_VALIDATION_PRECISION,
    _check_months, select_reduced_alert_threshold, fit_multi_month,
)


def synthetic_month(year, month, days, per_day=12):
    records = []
    base = date(year, month, 1)
    for day in range(days):
        d = base + timedelta(days=day)
        for i in range(per_day):
            records.append({
                "station_name": ["Dortmund Hbf", "Dortmund-Kurl", "Dortmund-Hörde"][i % 3],
                "train_type": ["ICE", "RE", "S"][i % 3],
                "scheduled_hour": (i + day) % 24,
                "day_of_week": d.isoweekday(),
                "planned_date": d,
                "target_6plus": int(i % 4 == 0 or (day + i) % 9 == 0),
            })
    return pd.DataFrame(records)


def synthetic_frames():
    return {
        "2026-07": synthetic_month(2026, 7, 31),
        "2026-08": synthetic_month(2026, 8, 31),
        "2026-09": synthetic_month(2026, 9, 30),
        "2026-10": synthetic_month(2026, 10, 31),
    }


def test_fixed_months():
    assert TRAIN_MONTHS == ("2026-07", "2026-08")
    assert VALIDATION_MONTH == "2026-09"
    assert BENCHMARK_MONTH == "2026-10"
    _check_months(TRAIN_MONTHS, VALIDATION_MONTH, BENCHMARK_MONTH)
    with pytest.raises(ValueError, match="strictly increasing"):
        _check_months(("2026-08", "2026-07"), "2026-09", "2026-10")
    with pytest.raises(ValueError, match="distinct"):
        _check_months(("2026-08", "2026-09"), "2026-09", "2026-10")


def test_precision_constraint_not_silently_relaxed():
    assert TARGET_VALIDATION_PRECISION == 0.55
    y = [1, 1, 0, 0, 0, 0]
    scores = [0.7, 0.6, 0.5, 0.2, 0.1, 0.1]
    good = select_reduced_alert_threshold(y, scores, min_precision=0.55)
    assert good is not None and good["precision"] >= 0.55
    assert select_reduced_alert_threshold(y, scores, min_precision=1.01) is None


def test_historical_benchmark_is_not_called_blind(tmp_path):
    pytest.importorskip("sklearn")
    result = fit_multi_month(synthetic_frames(), reports_root=tmp_path)
    assert "NOT an independent blind test" in result["warning"]
    assert result["candidate_promoted_to_production"] is False
    assert result["reference_threshold"] == 0.225
    assert result["monthly_samples"]["2026-10"]["n"] == 372
    assert result["monthly_samples"]["2026-09"]["n"] == 360
    assert (tmp_path / "ml_multimonth_metrics.json").is_file()
    assert (tmp_path / "ml_multimonth_comparison.csv").is_file()
    assert '"station_name": "Dortmund Hbf"' not in (
        tmp_path / "ml_multimonth_metrics.json"
    ).read_text()


def test_out_of_order_month_rejected(tmp_path):
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError):
        fit_multi_month(
            synthetic_frames(), reports_root=tmp_path,
            train_months=("2026-08", "2026-07"),
        )
