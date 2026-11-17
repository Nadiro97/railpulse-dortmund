"""Extract Dortmund station records from a licensed monthly Parquet file."""
from pathlib import Path
import json
from datetime import datetime, timezone

REQUIRED = {"station_name", "eva", "id", "arrival_planned_time", "arrival_change_time", "arrival_is_canceled"}
OPTIONAL_TYPES = {
    "train_type": "VARCHAR",
    "train_number": "VARCHAR",
    "line_number": "VARCHAR",
    "train_line_ride_id": "VARCHAR",
    "train_line_station_num": "DOUBLE",
    "final_destination_station": "VARCHAR",
    "time": "TIMESTAMP",
    "delay_in_min": "DOUBLE",
    "departure_planned_time": "TIMESTAMP",
    "departure_change_time": "TIMESTAMP",
    "departure_is_canceled": "BOOLEAN",
    "has_plan_record": "BOOLEAN",
    "is_additional_stop": "BOOLEAN",
    "is_replacement_train": "BOOLEAN",
}

def quoted(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"

def validate_columns(columns: set[str]) -> None:
    missing = REQUIRED - set(columns)
    if missing:
        raise ValueError(f"Source schema missing required columns: {sorted(missing)}")

def selected_expressions(columns: set[str]) -> list[str]:
    validate_columns(columns)
    expressions = [f'"{name}"' for name in sorted(REQUIRED)]
    for name, typ in OPTIONAL_TYPES.items():
        expressions.append(f'"{name}"' if name in columns else f'CAST(NULL AS {typ}) AS "{name}"')
    expressions.append(
        "CASE WHEN arrival_planned_time IS NOT NULL AND arrival_change_time IS NOT NULL "
        "THEN date_diff('second', arrival_planned_time, arrival_change_time) / 60.0 "
        "ELSE NULL END AS changed_arrival_delta_min"
    )
    return expressions

def ingest_dortmund(source_file: Path, data_root: Path = Path("data")) -> tuple[Path, dict]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install -e .") from exc
    source_file = Path(source_file)
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing input {source_file}; run 'railpulse download' first.")
    outdir = Path(data_root) / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"dortmund_{source_file.stem.removeprefix('data-')}.parquet"
    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet({quoted(source_file.resolve())})"
        columns = {row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()}
        expressions = selected_expressions(columns)
        query = f"SELECT {', '.join(expressions)} FROM {source_sql} WHERE station_name ILIKE 'Dortmund%'"
        count = conn.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
        if not count:
            raise ValueError("No Dortmund rows found; validate station names and release month.")
        conn.execute(f"COPY ({query}) TO {quoted(output.resolve())} (FORMAT PARQUET, COMPRESSION ZSTD)")
        stations = conn.execute(
            f"SELECT COUNT(DISTINCT station_name) FROM read_parquet({quoted(output.resolve())})"
        ).fetchone()[0]
    finally:
        conn.close()
    metadata = {
        "source": "https://huggingface.co/datasets/piebro/deutsche-bahn-data",
        "upstream_file": source_file.name,
        "local_output": str(output),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "dortmund_rows": count,
        "dortmund_stations": stations,
        "station_filter": "station_name ILIKE 'Dortmund%'",
        "licence": "CC BY 4.0",
        "time_zone": "Europe/Berlin (timezone-naive source local time)",
        "arrival_change_time_semantics": "changed/updated time, not verified actual arrival",
    }
    (outdir / "extraction_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output, metadata
