"""Research-only inference from the multi-year Dortmund rail backtest.

IMPORTANT: The stored estimator predicts *archived timetable-update minutes*,
not independently verified physical arrival or departure times. A zero here
MUST NEVER be presented as verified 'Pünktlich'.

The first multi-year artifact format lacked its own train-service catalogue.
For existing local models, reconstruct that catalogue from *training-month
caches only*, without retraining or looking at validation/test observations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .evidence_model import MODEL_ROOT, UNKNOWN, build_features, _pred
from .history_train import MIN_ROUTE_COUNT, cache_path, history_report_path
from .pipeline import quoted

MODEL_SCHEMA = "railpulse-rail-history-backtest-v1"
SERVICE_COLS = (
    "station_name", "train_type", "line_number", "train_number",
    "route_origin", "route_destination",
)
NUMERIC_MEDIANS = ("route_stop_index", "hops_before", "hops_after")
CATALOG_VERSION = 1
CATALOG_MAX_ROUTES = 5000


def history_model_path(mode, root=MODEL_ROOT):
    if mode not in ("arrival", "departure"):
        raise ValueError("Invalid Dortmund event")
    return Path(root) / f"history_rail_{mode}_research.joblib"


def catalog_cache_path(mode, root=MODEL_ROOT):
    if mode not in ("arrival", "departure"):
        raise ValueError("Invalid Dortmund event")
    return Path(root) / f"history_rail_{mode}_catalog.json"


def _model_metadata(artifact, mode):
    if not isinstance(artifact, dict) or artifact.get("schema") != MODEL_SCHEMA:
        raise ValueError("Not a recognised multi-year rail research model.")
    if artifact.get("mode") != mode or artifact.get("not_verified_physical_delay") is not True:
        raise ValueError("Model metadata inconsistent with Dortmund event and outcome.")
    if artifact.get("not_approved_for_live_forecast") is not True:
        raise ValueError("Untrusted or unsupported research model metadata.")
    if not isinstance(artifact.get("train_months"), list) or not artifact["train_months"]:
        raise ValueError("Training months missing from model.")
    if not isinstance(artifact.get("validation_months"), list):
        raise ValueError("Validation months missing from model.")
    return artifact


def _valid_catalog(rows):
    if not isinstance(rows, list):
        raise ValueError("Missing training service catalogue.")
    for entry in rows:
        if not isinstance(entry, dict):
            raise ValueError("Invalid service catalogue entry.")
        if any(not isinstance(entry.get(c), str) for c in SERVICE_COLS):
            raise ValueError("Invalid service identity.")
        if type(entry.get("support")) is not int or entry["support"] < MIN_ROUTE_COUNT:
            raise ValueError("Insufficient training support in service catalogue.")
        if any(not math.isfinite(float(entry.get(c + "_median", float("nan"))))
               for c in NUMERIC_MEDIANS):
            raise ValueError("Incomplete route-position feature medians.")
    return rows


def build_service_catalog(train_months, mode, data_root=Path("data")):
    """Aggregate ONLY pre-test cached railway training rows, with DuckDB.

    Reads only the six categorical identifiers and three planned route
    attributes; never reads labels or change timestamps for inference.
    Does not access validation/test-month caches.
    """
    import duckdb

    if not train_months or train_months != sorted(set(train_months)):
        raise ValueError("Training month list must be ascending and unique.")
    paths = [cache_path(m, mode, data_root).resolve() for m in train_months]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing local training caches for the multi-year model: " +
            ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
        )
    parquet_array = "[" + ", ".join(quoted(path) for path in paths) + "]"
    identities = ", ".join('"' + col + '"' for col in SERVICE_COLS)
    con = duckdb.connect()
    try:
        rows = con.execute(f"""
            SELECT {identities}, CAST(COUNT(*) AS BIGINT) AS support,
                   COALESCE(MEDIAN(route_stop_index), -1.0)
                       AS route_stop_index_median,
                   COALESCE(MEDIAN(hops_before), -1.0) AS hops_before_median,
                   COALESCE(MEDIAN(hops_after), -1.0) AS hops_after_median
            FROM read_parquet({parquet_array})
            GROUP BY {identities}
            HAVING COUNT(*) >= {MIN_ROUTE_COUNT}
            ORDER BY support DESC, {identities}
            LIMIT {CATALOG_MAX_ROUTES}
        """).fetchall()
        names = list(SERVICE_COLS) + [
            "support", "route_stop_index_median", "hops_before_median",
            "hops_after_median",
        ]
        result = []
        for record in rows:
            entry = dict(zip(names, record))
            for col in SERVICE_COLS:
                entry[col] = str(entry[col]) if entry[col] is not None else UNKNOWN
            entry["support"] = int(entry["support"])
            for col in NUMERIC_MEDIANS:
                entry[col + "_median"] = float(entry[col + "_median"])
            result.append(entry)
        return _valid_catalog(result)
    finally:
        con.close()


def load_history_model(mode, *, model_dir=MODEL_ROOT, data_root=Path("data")):
    """Only load locally trained joblib; do not deserialize downloaded models."""
    import joblib

    model_file = history_model_path(mode, model_dir)
    if model_file.stat().st_size > 150_000_000:
        raise ValueError("Unexpectedly large research model artifact.")
    artifact = _model_metadata(joblib.load(model_file), mode)
    rows = artifact.get("service_rows")
    if rows is None:
        # Compatibility: prior multi-year training did not store route choices.
        # A catalogue can be rebuilt without re-fitting or re-downloading.
        cache_file = catalog_cache_path(mode, model_dir)
        if cache_file.is_file():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if (data.get("schema") == MODEL_SCHEMA
                and data.get("train_months") == artifact["train_months"]
                and data.get("version") == CATALOG_VERSION):
                rows = data.get("service_rows")
        if rows is None:
            rows = build_service_catalog(artifact["train_months"], mode, data_root)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({
                "schema": MODEL_SCHEMA, "version": CATALOG_VERSION,
                "train_months": artifact["train_months"],
                "service_rows": rows,
            }, indent=2) + "\n", encoding="utf-8")
    artifact["service_rows"] = _valid_catalog(rows)
    return artifact


def estimate_archived_minutes(artifact, service, hour, weekday):
    """Experimental numeric archive-update estimate; NOT a real delay claim.

    Reports baseline status and selection warning instead of the legacy
    global deployment_allowed flag, which does not apply to this artifact.
    """
    import pandas as pd

    mode = artifact.get("mode")
    _model_metadata(artifact, mode)
    if type(hour) is not int or not 0 <= hour <= 23:
        raise ValueError("Hour must be an integer in 0..23")
    if type(weekday) is not int or not 1 <= weekday <= 7:
        raise ValueError("Weekday must be an integer in 1..7")
    if not isinstance(service, dict) or any(not isinstance(service.get(x), str)
                                            for x in SERVICE_COLS):
        raise ValueError("A fully specified, observed service is required.")
    match = next(
        (record for record in artifact.get("service_rows", [])
         if all(record[col] == service[col] for col in SERVICE_COLS)),
        None,
    )
    if match is None or match["support"] < MIN_ROUTE_COUNT:
        return {
            "minutes": None,
            "reason": "Diese Kombination hat weniger als 20 passende Trainingsbeispiele.",
            "status": "not_enough_training_support",
        }
    data = {
        **{col: match[col] for col in SERVICE_COLS},
        "scheduled_hour": hour,
        "weekday": weekday,
        **{col: float(match[col + "_median"]) for col in NUMERIC_MEDIANS},
    }
    frame = build_features(pd.DataFrame([data]))
    estimate = float(_pred(artifact["estimator"], frame)[0])
    name = artifact.get("model_name", "unknown")
    return {
        "minutes": round(estimate, 1),
        "status": "experimental_archive_update",
        "support": match["support"],
        "model_name": name,
        "is_baseline": name in {"always_punctual", "training_mean"},
        "is_verified_actual_delay": False,
        "can_claim_punctual": False,
        "note": "Nur historisch archivierte Zeitplanänderung, nicht tatsächliche Verspätung.",
    }


def local_diagnostics(mode, reports_dir=Path("reports"), model_dir=MODEL_ROOT):
    """Read-only status, suitable for dashboard and troubleshooting commands."""
    if mode not in ("arrival", "departure"):
        raise ValueError("Invalid Dortmund event")
    modern = history_model_path(mode, model_dir)
    historical_report = history_report_path(mode, reports_dir)
    quality_report = Path(reports_dir) / f"history_rail_{mode}_quality.json"
    old_model = Path(model_dir) / f"evidence_minutes_{mode}.joblib"
    old_report = Path(reports_dir) / f"evidence_minutes_{mode}_evaluation.json"
    status = {
        "mode": mode,
        "has_multi_year_model": modern.is_file(),
        "has_multi_year_evaluation": historical_report.is_file(),
        "has_old_2026_model": old_model.is_file(),
        "has_old_2026_evaluation": old_report.is_file(),
    }
    if historical_report.is_file():
        record = json.loads(historical_report.read_text(encoding="utf-8"))
        status["selected_model"] = record.get("selected_model")
        status["test_overall"] = record.get("test_overall")
        status["train_months"] = record.get("split", {}).get("train", [])
        status["test_months"] = record.get("split", {}).get("test", [])
        status["tested_on_verified_actual_arrivals"] = record.get(
            "target_is_verified_physical_arrival", False)
    if quality_report.is_file():
        quality = json.loads(quality_report.read_text(encoding="utf-8"))
        status["training_blocker"] = quality.get("error")
        status["usable_months"] = quality.get("usable_months", [])
        status["excluded_months"] = quality.get("excluded_months", {})
    if not modern.is_file():
        status["guidance"] = (
            "Das Mehrmonats-Modell fehlt. Prüfe reports/history_rail_" + mode +
            "_quality.json; falls genug Monate vorhanden: "
            "python -m railpulse history-rail --mode " + mode
        )
    return status
