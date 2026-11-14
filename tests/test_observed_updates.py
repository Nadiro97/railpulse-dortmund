"""Synthetic fixtures only; never commit third-party train-stop observations."""
import csv
import pytest
from railpulse.observed_updates import (
    change_filename, check_schema, report_queries, REQUIRED_CHANGES, validate_month
)

def test_allowlisted_parsed_change_path():
    assert change_filename("2026-09") == "monthly_processed_data_change/data-2026-09.parquet"
    with pytest.raises(ValueError):
        change_filename("../../secret")

def test_schema_check():
    check_schema(REQUIRED_CHANGES, REQUIRED_CHANGES, "change")
    with pytest.raises(ValueError, match="arrival_change_time"):
        check_schema(REQUIRED_CHANGES - {"arrival_change_time"}, REQUIRED_CHANGES, "change")

def test_only_explicit_updates_enter_metrics():
    sql = report_queries("2026-09")
    assert set(sql) == {
        "validated_quality", "validated_by_station",
        "validated_by_train_type", "validated_by_hour"
    }
    assert "explicit_arrival_update_count > 0" in sql["validated_by_station"]
    assert "2026-09-01" in sql["validated_by_station"]
    assert "2026-10-01" in sql["validated_by_station"]

def test_synthetic_full_validation(tmp_path):
    db_module = pytest.importorskip("duckdb")
    processed = tmp_path / "processed.parquet"
    changes = tmp_path / "changes.parquet"
    db = db_module.connect()
    try:
        db.execute("""
            CREATE TABLE p (
              id VARCHAR, eva VARCHAR, station_name VARCHAR, train_type VARCHAR,
              arrival_planned_time TIMESTAMP, arrival_change_time TIMESTAMP,
              arrival_is_canceled BOOLEAN, has_plan_record BOOLEAN,
              is_additional_stop BOOLEAN, is_replacement_train BOOLEAN
            )
        """)
        db.execute("""
            INSERT INTO p VALUES
              ('A','101','Dortmund Hbf','S','2026-09-01 08:00','2026-09-01 08:00',false,true,false,false),
              ('B','101','Dortmund Hbf','RE','2026-09-01 09:00','2026-09-01 09:07',false,true,false,false),
              ('C','102','Dortmund-Aplerbeck','RB','2026-09-01 10:00','2026-09-01 10:05',true,true,false,false),
              ('D','101','Dortmund Hbf','S','2026-09-01 11:00','2026-09-01 11:00',false,true,false,false),
              ('E','101','Dortmund Hbf','RE','2026-09-01 12:00','2026-09-01 12:01',false,true,false,false),
              ('F','201','Essen Hbf','RE','2026-09-01 13:00','2026-09-01 13:10',false,true,false,false),
              ('G','101','Dortmund Hbf','S','2026-08-31 23:59','2026-08-31 23:59',false,true,false,false)
        """)
        db.execute("""
            CREATE TABLE ch (
              id VARCHAR, eva VARCHAR, station_name VARCHAR,
              snapshot_timestamp TIMESTAMP, arrival_change_time TIMESTAMP
            )
        """)
        db.execute("""
            INSERT INTO ch VALUES
              ('B','101','Dortmund Hbf','2026-09-01 08:30','2026-09-01 09:10'),
              ('B','101','Dortmund Hbf','2026-09-01 08:45','2026-09-01 09:07'),
              ('D','101','Dortmund Hbf','2026-09-01 10:10','2026-09-01 11:00'),
              ('E','101','Dortmund Hbf','2026-09-01 11:10',NULL),
              ('F','201','Essen Hbf','2026-09-01 12:00','2026-09-01 13:10'),
              ('D','999','Wrong EVA','2026-09-01 10:30','2026-09-01 12:00')
        """)
        db.execute(f"COPY p TO '{processed}' (FORMAT PARQUET)")
        db.execute(f"COPY ch TO '{changes}' (FORMAT PARQUET)")
    finally:
        db.close()
    outputs, meta = validate_month("2026-09", processed, changes, tmp_path / "reports")
    assert len(outputs) == 5
    assert meta["processed_dortmund_rows"] == 6
    with (tmp_path / "reports" / "validated_quality.csv").open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert int(row["eligible_scheduled_arrival_rows"]) == 4
    assert int(row["eligible_with_any_change_snapshot"]) == 3
    assert int(row["eligible_with_explicit_arrival_update"]) == 2
    assert int(row["eligible_without_explicit_arrival_update"]) == 2
    assert int(row["eligible_without_change_snapshot"]) == 1
    assert int(row["explicit_updates_in_valid_range"]) == 2
    assert float(row["explicit_update_coverage_pct"]) == 50.0
    with (tmp_path / "reports" / "validated_by_station.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    hbf = next(r for r in rows if r["station_name"] == "Dortmund Hbf")
    assert float(hbf["avg_explicit_update_delta_min"]) == 3.5
    assert float(hbf["explicit_updates_6plus_pct"]) == 50.0
