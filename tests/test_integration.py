"""Local test with synthetic records only; no third-party dataset is committed."""
import csv
import pytest
from railpulse.pipeline import ingest_dortmund, quoted
from railpulse.analytics import build_reports

def test_end_to_end_synthetic_parquet(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    source = data_root / "raw" / "monthly_processed_data" / "data-2026-09.parquet"
    source.parent.mkdir(parents=True)
    db = duckdb.connect()
    try:
        db.execute("""
          CREATE TABLE toy(
            id VARCHAR, station_name VARCHAR, eva VARCHAR,
            arrival_planned_time TIMESTAMP, arrival_change_time TIMESTAMP,
            arrival_is_canceled BOOLEAN, has_plan_record BOOLEAN,
            is_additional_stop BOOLEAN, is_replacement_train BOOLEAN,
            train_type VARCHAR)
        """)
        db.execute("""
          INSERT INTO toy VALUES
           ('a','Dortmund Hbf','8000080','2026-09-01 08:00','2026-09-01 08:08',FALSE,TRUE,FALSE,FALSE,'RE'),
           ('b','Dortmund Hbf','8000080','2026-09-01 09:00','2026-09-01 09:00',FALSE,TRUE,FALSE,FALSE,'RE'),
           ('c','Dortmund Hbf','8000080','2026-09-01 10:00',NULL,TRUE,TRUE,FALSE,FALSE,'ICE'),
           ('d','Dortmund-Dorstfeld','8001556','2026-09-01 11:00','2026-09-01 11:01',FALSE,TRUE,FALSE,TRUE,'RE'),
           ('e','Essen Hbf','8000098','2026-09-01 09:00','2026-09-01 09:15',FALSE,TRUE,FALSE,FALSE,'RE')
        """)
        db.execute(f"COPY toy TO {quoted(source)} (FORMAT PARQUET)")
    finally:
        db.close()
    output, meta = ingest_dortmund(source, data_root)
    assert output.is_file()
    assert meta["dortmund_rows"] == 4
    report_files = build_reports(output, tmp_path / "reports")
    assert len(report_files) == 5
    with (tmp_path / "reports" / "quality.csv").open(newline="") as file:
        quality = next(csv.DictReader(file))
    assert int(quality["cancelled_arrival_records"]) == 1
    assert int(quality["eligible_changed_arrivals"]) == 2
    with (tmp_path / "reports" / "by_station.csv").open(newline="") as file:
        first_station = next(csv.DictReader(file))
    assert first_station["station_name"] == "Dortmund Hbf"
    assert abs(float(first_station["mean_changed_arrival_delta_min"]) - 4.0) < 0.01
