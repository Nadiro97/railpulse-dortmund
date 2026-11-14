"""Cross-check processed railway arrivals against explicit parsed-change snapshots.

An explicit update is NOT proof of a physical train arrival. Do not upload rows.
"""
from pathlib import Path
from datetime import datetime, timezone
import json

from .config import DATASET_REPO, checked_month
from .pipeline import quoted

CHANGE_PREFIX = "monthly_processed_data_change"
REQUIRED_PROCESSED = {
    "id", "eva", "station_name", "train_type", "arrival_planned_time",
    "arrival_change_time", "arrival_is_canceled", "has_plan_record",
    "is_additional_stop", "is_replacement_train",
}
REQUIRED_CHANGES = {"id", "eva", "snapshot_timestamp", "arrival_change_time"}


def change_filename(month: str) -> str:
    return f"{CHANGE_PREFIX}/data-{checked_month(month)}.parquet"


def download_change_month(month: str, data_root: Path = Path("data")) -> Path:
    """Fetch only the allowlisted published snapshot release."""
    from huggingface_hub import hf_hub_download
    raw = Path(data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset",
        filename=change_filename(month), local_dir=str(raw)
    ))


def check_schema(columns: set[str], expected: set[str], source: str) -> None:
    missing = expected - columns
    if missing:
        raise ValueError(f"{source} missing fields: {', '.join(sorted(missing))}")


def report_queries(month: str) -> dict[str, str]:
    year, mn = (int(x) for x in checked_month(month).split("-"))
    after_year, after_month = (year + 1, 1) if mn == 12 else (year, mn + 1)
    start, end = f"{year:04d}-{mn:02d}-01", f"{after_year:04d}-{after_month:02d}-01"
    eligible = f"""
        arrival_planned_time >= TIMESTAMP '{start}'
        AND arrival_planned_time < TIMESTAMP '{end}'
        AND arrival_is_canceled IS FALSE
        AND has_plan_record IS TRUE
        AND is_additional_stop IS FALSE
        AND is_replacement_train IS FALSE
    """
    observed = f"({eligible}) AND explicit_arrival_update_count > 0 AND explicit_delta_min BETWEEN -60 AND 240"
    def percentage(condition, denominator):
        return (f"ROUND(100.0 * COUNT(*) FILTER (WHERE {condition}) / "
                f"NULLIF(COUNT(*) FILTER (WHERE {denominator}), 0),2)")

    return {
        "validated_quality": f"""
            SELECT COUNT(*) AS all_processed_dortmund_rows,
                COUNT(DISTINCT station_name) AS stations,
                COUNT(*) FILTER (WHERE {eligible}) AS eligible_scheduled_arrival_rows,
                COUNT(*) FILTER (WHERE ({eligible}) AND matching_snapshot_count > 0)
                    AS eligible_with_any_change_snapshot,
                COUNT(*) FILTER (WHERE ({eligible}) AND explicit_arrival_update_count > 0)
                    AS eligible_with_explicit_arrival_update,
                COUNT(*) FILTER (WHERE ({eligible}) AND explicit_arrival_update_count = 0)
                    AS eligible_without_explicit_arrival_update,
                COUNT(*) FILTER (WHERE ({eligible}) AND matching_snapshot_count = 0)
                    AS eligible_without_change_snapshot,
                COUNT(*) FILTER (WHERE {observed}) AS explicit_updates_in_valid_range,
                COUNT(*) FILTER (WHERE ({eligible}) AND explicit_arrival_update_count > 0
                    AND processed_updated_time IS DISTINCT FROM latest_explicit_arrival_time)
                    AS explicit_updates_disagreeing_with_processed_time,
                COUNT(*) FILTER (WHERE arrival_is_canceled IS TRUE) AS cancelled_arrival_records,
                {percentage('('+eligible+') AND explicit_arrival_update_count > 0', eligible)}
                    AS explicit_update_coverage_pct,
                {percentage('('+eligible+') AND explicit_arrival_update_count = 0', eligible)}
                    AS no_explicit_update_share_pct
            FROM matched
        """,
        "validated_by_station": f"""
            SELECT station_name,
                COUNT(*) FILTER (WHERE {eligible}) AS eligible_scheduled_rows,
                COUNT(*) FILTER (WHERE {observed}) AS explicit_update_rows,
                {percentage(observed, eligible)} AS explicit_update_coverage_pct,
                ROUND(AVG(explicit_delta_min) FILTER (WHERE {observed}),2)
                    AS avg_explicit_update_delta_min,
                ROUND(MEDIAN(explicit_delta_min) FILTER (WHERE {observed}),2)
                    AS median_explicit_update_delta_min,
                ROUND(100.0 * COUNT(*) FILTER (WHERE ({observed}) AND explicit_delta_min >= 6)
                    / NULLIF(COUNT(*) FILTER (WHERE {observed}),0),2)
                    AS explicit_updates_6plus_pct
            FROM matched GROUP BY station_name ORDER BY eligible_scheduled_rows DESC, station_name
        """,
        "validated_by_train_type": f"""
            SELECT COALESCE(train_type,'(unknown)') AS train_type,
                COUNT(*) FILTER (WHERE {eligible}) AS eligible_scheduled_rows,
                COUNT(*) FILTER (WHERE {observed}) AS explicit_update_rows,
                {percentage(observed, eligible)} AS explicit_update_coverage_pct,
                ROUND(AVG(explicit_delta_min) FILTER (WHERE {observed}),2)
                    AS avg_explicit_update_delta_min,
                ROUND(MEDIAN(explicit_delta_min) FILTER (WHERE {observed}),2)
                    AS median_explicit_update_delta_min,
                ROUND(100.0 * COUNT(*) FILTER (WHERE ({observed}) AND explicit_delta_min >= 6)
                    / NULLIF(COUNT(*) FILTER (WHERE {observed}),0),2)
                    AS explicit_updates_6plus_pct
            FROM matched GROUP BY train_type ORDER BY eligible_scheduled_rows DESC
        """,
        "validated_by_hour": f"""
            SELECT EXTRACT(hour FROM arrival_planned_time)::INTEGER AS scheduled_hour,
                COUNT(*) FILTER (WHERE {eligible}) AS eligible_scheduled_rows,
                COUNT(*) FILTER (WHERE {observed}) AS explicit_update_rows,
                {percentage(observed, eligible)} AS explicit_update_coverage_pct,
                ROUND(AVG(explicit_delta_min) FILTER (WHERE {observed}),2)
                    AS avg_explicit_update_delta_min
            FROM matched WHERE arrival_planned_time IS NOT NULL
            GROUP BY scheduled_hour ORDER BY scheduled_hour
        """,
    }


def validate_month(
    month: str, processed_file: Path, change_file: Path,
    reports_root: Path = Path("reports")
) -> tuple[list[Path], dict]:
    """Validate via station IDs and stop IDs; return only aggregate report paths."""
    import duckdb
    month = checked_month(month)
    processed_file, change_file = Path(processed_file), Path(change_file)
    for source in (processed_file, change_file):
        if not source.is_file():
            raise FileNotFoundError(source)
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect()
    try:
        conn.execute("SET memory_limit='4GB'")
        conn.execute(f"SET temp_directory={quoted((reports_root / 'duckdb_spill').resolve())}")
        p_src = f"read_parquet({quoted(processed_file.resolve())})"
        c_src = f"read_parquet({quoted(change_file.resolve())})"
        for sql, label, required in (
            (p_src, "processed", REQUIRED_PROCESSED),
            (c_src, "change", REQUIRED_CHANGES),
        ):
            columns = {r[0] for r in conn.execute(f"DESCRIBE SELECT * FROM {sql}").fetchall()}
            check_schema(columns, required, label)
        conn.execute(f"""
            CREATE TEMP TABLE base AS
            SELECT id, eva, station_name, train_type, arrival_planned_time,
                arrival_change_time AS processed_updated_time,
                arrival_is_canceled, has_plan_record,
                is_additional_stop, is_replacement_train
            FROM {p_src}
            WHERE station_name ILIKE 'Dortmund%'
        """)
        conn.execute(f"""
            CREATE TEMP TABLE snapshots AS
            SELECT c.id, c.eva,
                COUNT(*) AS matching_snapshot_count,
                COUNT(*) FILTER (WHERE c.arrival_change_time IS NOT NULL)
                    AS explicit_arrival_update_count,
                arg_max(c.arrival_change_time, c.snapshot_timestamp)
                    FILTER (WHERE c.arrival_change_time IS NOT NULL)
                    AS latest_explicit_arrival_time
            FROM {c_src} c
            WHERE c.eva IN (SELECT DISTINCT eva FROM base)
            GROUP BY c.id, c.eva
        """)
        conn.execute("""
            CREATE TEMP TABLE matched AS
            SELECT b.*,
                COALESCE(s.matching_snapshot_count,0) AS matching_snapshot_count,
                COALESCE(s.explicit_arrival_update_count,0) AS explicit_arrival_update_count,
                s.latest_explicit_arrival_time,
                CASE WHEN s.latest_explicit_arrival_time IS NOT NULL
                        AND b.arrival_planned_time IS NOT NULL
                    THEN date_diff('second', b.arrival_planned_time,
                        s.latest_explicit_arrival_time)/60.0
                    ELSE NULL END AS explicit_delta_min
            FROM base b
            LEFT JOIN snapshots s USING (id, eva)
        """)
        paths = []
        for name, sql in report_queries(month).items():
            out = reports_root / f"{name}.csv"
            conn.execute(f"COPY ({sql}) TO {quoted(out.resolve())} (FORMAT CSV, HEADER TRUE)")
            paths.append(out)
        n = conn.execute("SELECT COUNT(*) FROM matched").fetchone()[0]
    finally:
        conn.close()
    meta = {
        "month": month,
        "source": "https://huggingface.co/datasets/piebro/deutsche-bahn-data",
        "processed_file": processed_file.name,
        "parsed_change_file": change_file.name,
        "source_license": "CC BY 4.0",
        "attribution": "Deutsche Bahn; archive by Piet Brömmel",
        "processed_dortmund_rows": n,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Latest non-null arrival change from parsed snapshots matched by (id, eva)",
        "limitation": "Explicit API update is NOT confirmed physical arrival; no-update rows unknown",
        "privacy": "Aggregate-only outputs; no source or stop-level rows uploaded",
    }
    meta_path = reports_root / "validated_provenance.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return paths + [meta_path], meta
