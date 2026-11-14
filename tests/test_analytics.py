from pathlib import Path
from railpulse.analytics import ELIGIBLE, report_sql

def test_aggregate_queries():
    queries = report_sql("read_parquet('synthetic.parquet')")
    assert set(queries) == {
        "quality", "station_quality", "by_station", "by_hour",
        "by_train_type", "by_weekday", "by_day"
    }
    assert "median_changed_arrival_delta_min" in queries["by_station"]
    assert "p90_changed_arrival_delta_min" in queries["by_station"]
    assert "scheduled_arrival_change_coverage_pct" in queries["quality"]
    assert "COUNT(*)" in queries["quality"]

def test_missing_change_is_not_on_time():
    assert "arrival_change_time IS NOT NULL" in ELIGIBLE
    assert "arrival_is_canceled IS FALSE" in ELIGIBLE
    assert "changed_arrival_delta_min BETWEEN -60 AND 240" in ELIGIBLE

def test_data_directories_ignored():
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "/data/raw/" in ignore
    assert "/data/processed/" in ignore
    assert "/reports/" in ignore
