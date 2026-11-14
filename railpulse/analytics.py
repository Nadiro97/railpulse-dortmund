"""Produce aggregate analytical reports from local processed Parquet."""
from pathlib import Path

# Missing change times are UNKNOWN, never assumed punctual.
ELIGIBLE = """
    arrival_planned_time IS NOT NULL
    AND arrival_change_time IS NOT NULL
    AND arrival_is_canceled IS FALSE
    AND has_plan_record IS DISTINCT FROM FALSE
    AND is_additional_stop IS DISTINCT FROM TRUE
    AND is_replacement_train IS DISTINCT FROM TRUE
    AND changed_arrival_delta_min BETWEEN -60 AND 240
"""

def report_sql(source_sql: str) -> dict[str, str]:
    basic = f"FROM {source_sql}"
    eligible = f"{basic} WHERE {ELIGIBLE}"
    return {
        "quality": f"""
            SELECT COUNT(*) AS total_station_stop_rows,
                COUNT(DISTINCT station_name) AS station_count,
                COUNT(*) FILTER (WHERE arrival_planned_time IS NOT NULL) AS scheduled_arrivals,
                COUNT(*) FILTER (WHERE arrival_change_time IS NOT NULL) AS changed_arrival_records,
                COUNT(*) FILTER (WHERE arrival_is_canceled IS TRUE) AS cancelled_arrival_records,
                COUNT(*) FILTER (WHERE is_additional_stop IS TRUE) AS additional_stop_rows,
                COUNT(*) FILTER (WHERE is_replacement_train IS TRUE) AS replacement_train_rows,
                COUNT(*) FILTER (WHERE has_plan_record IS FALSE) AS missing_plan_rows,
                COUNT(*) FILTER (WHERE {ELIGIBLE}) AS eligible_changed_arrivals
            {basic}
        """,
        "by_station": f"""
            SELECT station_name, COUNT(*) AS eligible_rows,
                ROUND(AVG(changed_arrival_delta_min), 2) AS mean_changed_arrival_delta_min,
                ROUND(quantile_cont(changed_arrival_delta_min, 0.5), 2) AS median_changed_arrival_delta_min,
                ROUND(100.0 * COUNT(*) FILTER (WHERE changed_arrival_delta_min >= 6) / COUNT(*), 2) AS share_6min_plus_pct
            {eligible}
            GROUP BY 1 ORDER BY eligible_rows DESC, station_name
        """,
        "by_hour": f"""
            SELECT EXTRACT(hour FROM arrival_planned_time)::INTEGER AS planned_hour,
                COUNT(*) AS eligible_rows,
                ROUND(AVG(changed_arrival_delta_min), 2) AS mean_changed_arrival_delta_min,
                ROUND(100.0 * COUNT(*) FILTER (WHERE changed_arrival_delta_min >= 6) / COUNT(*), 2) AS share_6min_plus_pct
            {eligible}
            GROUP BY 1 ORDER BY 1
        """,
        "by_train_type": f"""
            SELECT COALESCE(train_type, '(unknown)') AS train_type, COUNT(*) AS eligible_rows,
                ROUND(AVG(changed_arrival_delta_min), 2) AS mean_changed_arrival_delta_min
            {eligible}
            GROUP BY 1 ORDER BY eligible_rows DESC
        """,
        "by_weekday": f"""
            SELECT dayname(arrival_planned_time) AS weekday,
                EXTRACT(isodow FROM arrival_planned_time)::INTEGER AS weekday_number,
                COUNT(*) AS eligible_rows,
                ROUND(AVG(changed_arrival_delta_min), 2) AS mean_changed_arrival_delta_min
            {eligible}
            GROUP BY 1, 2 ORDER BY 2
        """,
    }

def build_reports(processed_file: Path, reports_root: Path = Path("reports")) -> list[Path]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install -e .") from exc
    from .pipeline import quoted
    processed_file = Path(processed_file)
    if not processed_file.is_file():
        raise FileNotFoundError(processed_file)
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    source_sql = f"read_parquet({quoted(processed_file.resolve())})"
    output_paths = []
    conn = duckdb.connect()
    try:
        for name, sql in report_sql(source_sql).items():
            destination = reports_root / f"{name}.csv"
            conn.execute(f"COPY ({sql}) TO {quoted(destination.resolve())} (HEADER, DELIMITER ',')")
            output_paths.append(destination)
    finally:
        conn.close()
    return output_paths
