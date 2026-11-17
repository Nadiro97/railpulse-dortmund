"""Event-time aware, route-rich Dortmund timetable labels.

The upstream monthly stop table fills missing change times with planned times.
NEVER use that table's delay_in_min/time as a physical-arrival ground truth.
Join original plan and change snapshots instead; keep exact observation
coverage and the snapshot's temporal relationship to the Dortmund stop.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .config import checked_month, source_filename, DATASET_REPO
from .observed_updates import change_filename
from .pipeline import quoted, ingest_dortmund

PLAN_PREFIX = "monthly_processed_data_plan"
UNKNOWN = "(unknown)"
MODES = ("arrival", "departure")
MINUTES_CAP = 120.0
POST_EVENT_MINUTES = 15
POST_EVENT_HOURS = 9
PLAN_LEAD_MINUTES = 60


def plan_filename(month: str) -> str:
    return f"{PLAN_PREFIX}/data-{checked_month(month)}.parquet"


def get_month_sources(month: str, data_root: Path = Path("data")):
    """Download each source into private local cache if absent; refresh old schema."""
    from huggingface_hub import hf_hub_download
    import duckdb

    root = Path(data_root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, filename in (
        ("processed", source_filename(month)),
        ("plan", plan_filename(month)),
        ("change", change_filename(month)),
    ):
        path = raw / filename
        if not path.is_file():
            path = Path(hf_hub_download(
                repo_id=DATASET_REPO, repo_type="dataset",
                filename=filename, local_dir=str(raw),
            ))
        files[kind] = path
    local = root / "processed" / f"dortmund_{month}.parquet"
    if local.is_file():
        con = duckdb.connect()
        try:
            columns = {r[0] for r in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({quoted(local.resolve())})"
            ).fetchall()}
        finally:
            con.close()
        if not {"train_line_station_num", "train_line_ride_id"} <= columns:
            local.unlink()
    if not local.is_file():
        ingest_dortmund(files["processed"], root)
    files["dortmund"] = local
    return files


def extract_labels(month: str, mode: str, files: dict):
    """Return eligible labelled observations in memory plus aggregate coverage.

    Features: last PLAN snapshot >=60 minutes before the stop.
    Label: last explicit CHANGED timestamp recorded from 15 minutes after
    the stop to 9 hours after it. This reduces but DOES NOT eliminate
    forecast-vs-actual ambiguity; no actual-arrival claim is made.
    """
    import duckdb

    if mode not in MODES:
        raise ValueError("Mode must be arrival or departure")
    start = date.fromisoformat(checked_month(month) + "-01")
    end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    required = ("dortmund", "plan", "change")
    for name in required:
        if name not in files or not Path(files[name]).is_file():
            raise FileNotFoundError(f"Missing private {name} file for {month}")

    planned = f"{mode}_planned_time"
    updated = f"{mode}_change_time"
    canceled = f"{mode}_is_canceled"
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='4GB'")
        local = f"read_parquet({quoted(Path(files['dortmund']).resolve())})"
        plan = f"read_parquet({quoted(Path(files['plan']).resolve())})"
        change = f"read_parquet({quoted(Path(files['change']).resolve())})"
        expected = [
            (local, {"id", "eva", "station_name", planned, canceled,
                     "has_plan_record", "train_line_station_num", "train_line_ride_id",
                     "is_additional_stop", "is_replacement_train"}, "Dortmund stops"),
            (plan, {"id", "eva", "snapshot_timestamp", "train_type", "train_number",
                    "arrival_planned_path", "departure_planned_path",
                    "arrival_line_number", "departure_line_number"}, "plan snapshots"),
            (change, {"id", "eva", "snapshot_timestamp", updated}, "change snapshots"),
        ]
        for source, fields, name in expected:
            columns = {r[0] for r in con.execute(
                f"DESCRIBE SELECT * FROM {source}"
            ).fetchall()}
            missing = fields - columns
            if missing:
                raise ValueError(f"{name} missing columns: {sorted(missing)}")

        con.execute(f"""
            CREATE TEMP TABLE base AS
            SELECT * EXCLUDE (rn) FROM (
                SELECT id, eva, station_name, CAST({planned} AS TIMESTAMP) AS scheduled,
                       CAST(train_line_station_num AS DOUBLE) AS route_stop_index,
                       train_line_ride_id AS ride_id,
                       ROW_NUMBER() OVER (PARTITION BY id, eva
                           ORDER BY {planned}, station_name) AS rn
                FROM {local}
                WHERE {planned} >= TIMESTAMP '{start}'
                  AND {planned} < TIMESTAMP '{end}'
                  AND {canceled} IS FALSE
                  AND has_plan_record IS TRUE
                  AND is_additional_stop IS FALSE
                  AND is_replacement_train IS FALSE
            ) WHERE rn = 1
        """)

        # The feature snapshot cannot be collected after the prediction cutoff.
        # Take the latest *planned* snapshot available at least an hour before.
        con.execute(f"""
            CREATE TEMP TABLE plan_before AS
            SELECT p.id, p.eva, p.snapshot_timestamp AS plan_snapshot_timestamp,
                   p.train_type AS train_type, p.train_number AS train_number,
                   COALESCE(p.{mode}_line_number,
                       p.arrival_line_number, p.departure_line_number) AS line_number,
                   list_extract(p.arrival_planned_path, 1) AS route_origin,
                   CASE
                     WHEN p.departure_planned_path IS NOT NULL
                       THEN list_extract(p.departure_planned_path, -1)
                     ELSE NULL
                   END AS route_destination,
                   length(p.arrival_planned_path) AS hops_before,
                   length(p.departure_planned_path) AS hops_after
            FROM {plan} p
            JOIN base b ON b.id = p.id AND b.eva = p.eva
            WHERE p.snapshot_timestamp <=
                b.scheduled - INTERVAL '{PLAN_LEAD_MINUTES} minutes'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY p.id, p.eva
                       ORDER BY p.snapshot_timestamp DESC) = 1
        """)

        # Label update must exist in an EXPLICIT parsed change snapshot AFTER
        # the scheduled event; no post-event status is used as a feature.
        con.execute(f"""
            CREATE TEMP TABLE recorded_after AS
            SELECT c.id, c.eva,
                   arg_max(c.{updated}, c.snapshot_timestamp) AS changed_at,
                   max(c.snapshot_timestamp) AS latest_snapshot,
                   count(*) AS explicit_post_event_snapshots
            FROM {change} c
            JOIN base b ON c.id = b.id AND c.eva = b.eva
            WHERE c.{updated} IS NOT NULL
              AND c.snapshot_timestamp >=
                  b.scheduled + INTERVAL '{POST_EVENT_MINUTES} minutes'
              AND c.snapshot_timestamp <=
                  b.scheduled + INTERVAL '{POST_EVENT_HOURS} hours'
            GROUP BY c.id, c.eva
        """)

        con.execute("""
            CREATE TEMP TABLE matched AS
            SELECT b.*, p.plan_snapshot_timestamp, p.train_type, p.train_number,
                   p.line_number, p.route_origin, p.route_destination,
                   p.hops_before, p.hops_after,
                   u.changed_at, u.latest_snapshot,
                   date_diff('second', b.scheduled, u.changed_at) / 60.0
                     AS raw_change_minutes,
                   u.explicit_post_event_snapshots
            FROM base b
            LEFT JOIN plan_before p USING (id, eva)
            LEFT JOIN recorded_after u USING (id, eva)
        """)

        row = con.execute(f"""
            SELECT count(*) AS scheduled_eligible,
                   count(*) FILTER (WHERE plan_snapshot_timestamp IS NOT NULL)
                       AS known_pre_event_plan,
                   count(*) FILTER (WHERE latest_snapshot IS NOT NULL)
                       AS known_post_event_explicit_change,
                   count(*) FILTER (
                       WHERE plan_snapshot_timestamp IS NOT NULL
                         AND raw_change_minutes BETWEEN -20 AND 180
                   ) AS labelled_eligible,
                   count(*) FILTER (
                       WHERE plan_snapshot_timestamp IS NOT NULL
                         AND raw_change_minutes > 120 AND raw_change_minutes <= 180
                   ) AS labels_capped_above_120,
                   count(*) FILTER (
                       WHERE plan_snapshot_timestamp IS NOT NULL
                         AND raw_change_minutes < -20
                   ) AS implausible_negative,
                   count(*) FILTER (
                       WHERE plan_snapshot_timestamp IS NOT NULL
                         AND raw_change_minutes > 180
                   ) AS implausible_positive,
                   count(*) FILTER (
                       WHERE plan_snapshot_timestamp IS NOT NULL
                         AND raw_change_minutes BETWEEN -20 AND 180
                         AND strftime(scheduled, '%m-%d') = '10-25'
                         AND extract(hour FROM scheduled) = 2
                   ) AS ambiguous_dst_records,
                   count(*) FILTER (WHERE route_origin IS NULL) AS missing_origin,
                   count(*) FILTER (WHERE route_destination IS NULL)
                       AS missing_destination,
                   count(*) FILTER (WHERE line_number IS NULL)
                       AS missing_line,
                   count(*) FILTER (WHERE train_number IS NULL)
                       AS missing_train_number
            FROM matched
        """).fetchone()
        columns = [
            "scheduled_eligible", "known_pre_event_plan",
            "known_post_event_explicit_change", "labelled_eligible",
            "labels_capped_above_120", "implausible_negative",
            "implausible_positive", "ambiguous_dst_records",
            "missing_origin", "missing_destination",
            "missing_line", "missing_train_number",
        ]
        quality = dict(zip(columns, [int(x) for x in row]))
        denominator = quality["scheduled_eligible"]
        quality["label_coverage_fraction"] = round(
            quality["labelled_eligible"] / denominator, 6
        ) if denominator else 0.0
        quality.update({
            "month": month,
            "event": mode,
            "pre_event_plan_lead_minutes": PLAN_LEAD_MINUTES,
            "minimum_post_event_minutes": POST_EVENT_MINUTES,
            "post_event_window_hours": POST_EVENT_HOURS,
            "outcome_source": "explicit parsed changes after scheduled Dortmund event",
            "is_independent_actual_time": False,
            "missing_updates_are_assumed_on_time": False,
        })

        # Distinct stop labels, never numeric features based on post-event data.
        data = con.execute(f"""
            SELECT station_name,
                   COALESCE(NULLIF(TRIM(train_type), ''), '{UNKNOWN}')
                       AS train_type,
                   COALESCE(NULLIF(TRIM(train_number), ''), '{UNKNOWN}')
                       AS train_number,
                   COALESCE(NULLIF(TRIM(line_number), ''), '{UNKNOWN}')
                       AS line_number,
                   COALESCE(NULLIF(TRIM(route_origin), ''), '{UNKNOWN}')
                       AS route_origin,
                   COALESCE(NULLIF(TRIM(route_destination), ''), '{UNKNOWN}')
                       AS route_destination,
                   CAST(scheduled AS DATE) AS planned_date,
                   EXTRACT(hour FROM scheduled)::INTEGER AS scheduled_hour,
                   EXTRACT(isodow FROM scheduled)::INTEGER AS weekday,
                   route_stop_index,
                   COALESCE(CAST(hops_before AS DOUBLE), -1.0) AS hops_before,
                   COALESCE(CAST(hops_after AS DOUBLE), -1.0) AS hops_after,
                   LEAST({MINUTES_CAP},
                       GREATEST(0.0, raw_change_minutes)) AS observed_update_minutes
            FROM matched
            WHERE plan_snapshot_timestamp IS NOT NULL
              AND raw_change_minutes BETWEEN -20 AND 180
              AND NOT (
                  strftime(scheduled, '%m-%d') = '10-25'
                  AND extract(hour FROM scheduled) = 2
              )
            ORDER BY planned_date, station_name
        """).df()
        return data, quality
    finally:
        con.close()
