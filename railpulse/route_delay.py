"""Route-aware archived timetable-update minutes at Dortmund, not actual arrivals.

Arrival and departure are separate targets and separate local JSON models.
Station/type/line/final destination, weekday, and scheduled hour are known from
the planned stop. Updated times are used ONLY as historical labels.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

from .config import checked_month
from .ml_baseline import month_bounds
from .observed_updates import check_schema, change_filename, download_change_month
from .pipeline import quoted

MONTH = "2026-09"
CAT_BASE = ["station_name", "train_type"]
CAT_ROUTE = ["station_name", "train_type", "line_number", "final_destination_station"]
NUMERIC = ["scheduled_hour", "day_of_week"]
MISSING = "(unknown)"
SCHEMA = "railpulse-dortmund-route-minutes-v1"
MODEL_DIR = Path("models")
MODES = ("arrival", "departure")
MIN_ROUTE_ROWS = 5
MAX_ROUTES = 2000


def local_model_path(mode, root=MODEL_DIR):
    if mode not in MODES:
        raise ValueError("Select arrival or departure at Dortmund.")
    return Path(root) / f"route_minutes_{mode}_2026-09.json"


def extract_route_rows(month, processed_file, change_file, mode="arrival"):
    """Join explicit snapshots only; no planned-time fallback or post-event features.

    The processed data is restricted to Dortmund station stops. A full journey
    origin is NOT reconstructed; destination is the train's final endpoint.
    """
    import duckdb

    checked_month(month)
    if mode not in MODES:
        raise ValueError("mode must be arrival or departure")
    start, _, end = month_bounds(month)
    for path in (processed_file, change_file):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='4GB'")
        processed = f"read_parquet({quoted(Path(processed_file).resolve())})"
        changes = f"read_parquet({quoted(Path(change_file).resolve())})"
        proc_cols = {row[0] for row in connection.execute(
            f"DESCRIBE SELECT * FROM {processed}"
        ).fetchall()}
        changes_cols = {row[0] for row in connection.execute(
            f"DESCRIBE SELECT * FROM {changes}"
        ).fetchall()}
        plan = f"{mode}_planned_time"
        updated = f"{mode}_change_time"
        cancelled = f"{mode}_is_canceled"
        check_schema(proc_cols, {
            "id", "eva", "station_name", "train_type", "line_number",
            "final_destination_station", plan, cancelled, "has_plan_record",
            "is_additional_stop", "is_replacement_train",
        }, "processed stops")
        check_schema(changes_cols, {
            "id", "eva", "snapshot_timestamp", updated
        }, "parsed change snapshots")

        # An empty route field means 'unknown', never invent an origin or line.
        connection.execute(f"""
            CREATE TEMP TABLE base AS
            SELECT id, eva, station_name,
                   COALESCE(NULLIF(TRIM(train_type), ''), '{MISSING}') AS train_type,
                   COALESCE(NULLIF(TRIM(line_number), ''), '{MISSING}') AS line_number,
                   COALESCE(NULLIF(TRIM(final_destination_station), ''), '{MISSING}') AS final_destination_station,
                   {plan} AS planned
            FROM {processed}
            WHERE station_name ILIKE 'Dortmund%'
              AND {plan} >= TIMESTAMP '{start}'
              AND {plan} < TIMESTAMP '{end}'
              AND {cancelled} IS FALSE
              AND has_plan_record IS TRUE
              AND is_additional_stop IS FALSE
              AND is_replacement_train IS FALSE
        """)
        connection.execute(f"""
            CREATE TEMP TABLE updates AS
            SELECT id, eva,
                   arg_max({updated}, snapshot_timestamp) AS explicit_updated
            FROM {changes}
            WHERE eva IN (SELECT DISTINCT eva FROM base)
              AND {updated} IS NOT NULL
            GROUP BY id, eva
        """)
        # Earlier updated times count as zero positive delay, not negative delay.
        return connection.execute("""
            SELECT b.station_name, b.train_type, b.line_number,
                   b.final_destination_station,
                   EXTRACT(hour FROM b.planned)::INTEGER AS scheduled_hour,
                   EXTRACT(isodow FROM b.planned)::INTEGER AS day_of_week,
                   CAST(b.planned AS DATE) AS planned_date,
                   GREATEST(0.0, date_diff('second', b.planned, u.explicit_updated) / 60.0)
                       AS archived_delay_min
            FROM base b JOIN updates u USING (id, eva)
            WHERE date_diff('second', b.planned, u.explicit_updated) / 60.0
                  BETWEEN -60 AND 240
            ORDER BY planned_date
        """).df()
    finally:
        connection.close()


def _date_values(frame):
    return frame["planned_date"].map(
        lambda value: value.date() if hasattr(value, "date") else (
            date.fromisoformat(value) if isinstance(value, str) else value
        )
    )


def _normalise(frame):
    import pandas as pd

    required = set(CAT_ROUTE + NUMERIC + ["archived_delay_min", "planned_date"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Required route-delay columns missing: {sorted(missing)}")
    frame = frame.copy()
    for column in CAT_ROUTE:
        frame[column] = frame[column].fillna(MISSING).astype(str).str.strip()
        frame.loc[frame[column] == "", column] = MISSING
    y = pd.to_numeric(frame["archived_delay_min"], errors="coerce")
    if not y.notna().all() or not y.between(0, 240).all():
        raise ValueError("Archived delay labels must be finite and between 0 and 240 minutes.")
    return frame


def _splits(frame, month):
    if month != MONTH:
        raise ValueError("This experimental model is locked to the September 2026 release.")
    frame = _normalise(frame)
    dates = _date_values(frame)
    train = frame.loc[(dates >= date(2026, 9, 1)) &
                      (dates <= date(2026, 9, 16))].copy()
    validation = frame.loc[(dates >= date(2026, 9, 17)) &
                           (dates <= date(2026, 9, 20))].copy()
    test = frame.loc[(dates >= date(2026, 9, 21)) &
                     (dates <= date(2026, 9, 30))].copy()
    for label, group in (
        ("training", train), ("validation", validation), ("test", test)
    ):
        if len(group) < 100 or not (group["archived_delay_min"] >= 6).any():
            raise ValueError(f"{label} requires >=100 records and some 6+ minute updates.")
    return train, validation, test


def _model(categorical):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    return Pipeline([
        ("encode", ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("calendar", "passthrough", NUMERIC),
        ])),
        ("model", Ridge(alpha=20.0, solver="lsqr")),
    ])


def _probability_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    return Pipeline([
        ("encode", ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CAT_ROUTE),
            ("calendar", "passthrough", NUMERIC),
        ])),
        ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])


def _minutes(model, frame, cats):
    import numpy as np

    values = model.predict(frame[cats + NUMERIC])
    return np.clip(values, 0.0, 240.0)


def _metrics(y, estimated):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y, p = np.asarray(y, dtype=float), np.asarray(estimated, dtype=float)
    err = np.abs(y - p)
    late = y >= 6
    return {
        "n": int(len(y)),
        "archived_delay_mean": round(float(y.mean()), 4),
        "archived_delay_p50": round(float(np.median(y)), 4),
        "event_6plus_fraction": round(float(late.mean()), 6),
        "mae_minutes": round(float(mean_absolute_error(y, p)), 4),
        "rmse_minutes": round(float(math.sqrt(mean_squared_error(y, p))), 4),
        "mae_on_6plus_events": round(float(err[late].mean()), 4) if late.any() else None,
        "within_5min_fraction": round(float((err <= 5).mean()), 6),
        "avg_estimate_minutes": round(float(p.mean()), 4),
    }


def _coefs(pipeline):
    encoder = pipeline.named_steps["encode"].named_transformers_["categories"]
    learned = pipeline.named_steps["model"]
    categories = [[str(v) for v in values] for values in encoder.categories_]
    weights = [float(v) for v in learned.coef_.reshape(-1)]
    intercept = float(learned.intercept_[0]) if hasattr(learned.intercept_, "__len__") else float(learned.intercept_)
    if len(weights) != sum(map(len, categories)) + len(NUMERIC):
        raise ValueError("Model coefficient shape does not match encoded features.")
    if not all(math.isfinite(v) for v in weights + [intercept]):
        raise ValueError("Non-finite model coefficients.")
    return {"intercept": intercept, "weights": weights,
            "categories": dict(zip(CAT_ROUTE, categories))}


def fit_route_minutes(frame, mode="arrival", reports_dir=Path("reports"),
                      model_dir=MODEL_DIR):
    """Lock the selection to September validation; keep models/ ignored by Git."""
    import numpy as np
    from sklearn.metrics import brier_score_loss, roc_auc_score

    if mode not in MODES:
        raise ValueError("mode must be arrival or departure")
    train, valid, test = _splits(frame, MONTH)
    models = {
        "mean_baseline": None,
        "station_type_hour": _model(CAT_BASE),
        "station_line_destination": _model(CAT_ROUTE),
    }
    ytrain, yval, ytest = (
        group["archived_delay_min"].to_numpy(dtype=float)
        for group in (train, valid, test)
    )
    for key, m in models.items():
        if m is not None:
            m.fit(train[(CAT_ROUTE if key == "station_line_destination" else CAT_BASE) + NUMERIC], ytrain)
    def output(model_name, sample):
        m = models[model_name]
        if m is None:
            return np.full(len(sample), ytrain.mean())
        cats = CAT_ROUTE if model_name == "station_line_destination" else CAT_BASE
        return _minutes(m, sample, cats)

    validation = {key: _metrics(yval, output(key, valid)) for key in models}
    chosen = min(validation, key=lambda key: (validation[key]["mae_minutes"],
                                             list(models).index(key)))
    holdout = {key: _metrics(ytest, output(key, test)) for key in models}
    # Also fit a route-aware probability model. No threshold is selected here.
    classifier = _probability_model()
    classifier.fit(train[CAT_ROUTE + NUMERIC], (ytrain >= 6).astype(int))
    prob = classifier.predict_proba(test[CAT_ROUTE + NUMERIC])[:, 1]
    prob_metrics = {
        "roc_auc_archived_6plus": (
            round(float(roc_auc_score(ytest >= 6, prob)), 5)
            if len(set(ytest >= 6)) > 1 else None
        ),
        "brier_archived_6plus": round(float(brier_score_loss(ytest >= 6, prob)), 5),
        "sample_n": int(len(test)),
    }

    # List only combinations observed repeatedly in training, not synthetic routes.
    counts = (train.groupby(CAT_ROUTE, dropna=False).size()
              .reset_index(name="records").sort_values(
                  ["records", *CAT_ROUTE], ascending=[False] + [True] * len(CAT_ROUTE)
              ))
    route_rows = counts.loc[counts["records"] >= MIN_ROUTE_ROWS].head(MAX_ROUTES)
    if route_rows.empty:
        raise ValueError("Too few repeated line/destination combinations for the demo.")
    routes = []
    for _, r in route_rows.iterrows():
        routes.append({**{name: str(r[name]) for name in CAT_ROUTE},
                       "training_examples": int(r["records"])})
    artifact = {
        "schema": SCHEMA,
        "mode": mode,
        "location": "Dortmund station stop",
        "month": MONTH,
        "train_dates": ["2026-09-01", "2026-09-16"],
        "validation_dates": ["2026-09-17", "2026-09-20"],
        "test_dates": ["2026-09-21", "2026-09-30"],
        "n_train": int(len(train)),
        "num_features": NUMERIC,
        "features": CAT_ROUTE + NUMERIC,
        "categorical": CAT_ROUTE,
        "minutes_model": _coefs(models["station_line_destination"]),
        "probability_model": _coefs(classifier),
        "validation_route_mae": validation["station_line_destination"]["mae_minutes"],
        "validation_baseline_mae": validation["station_type_hour"]["mae_minutes"],
        "selected_for_minutes": chosen,
        "routes": routes,
        "is_verified_real_arrival": False,
        "is_live": False,
        "route_origin_known": False,
        "target": "nonnegative latest explicit timetable change minus scheduled time at Dortmund, clipped 0..240",
        "source": "Deutsche Bahn CC BY 4.0; archived by Piet Brömmel",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = local_model_path(mode, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    report = {
        "mode": mode,
        "location": "Dortmund stop only",
        "target_is_verified_real_arrival": False,
        "source": artifact["source"],
        "train_n": len(train), "validation_n": len(valid), "test_n": len(test),
        "validation": validation,
        "heldout_september": holdout,
        "heldout_route_probability": prob_metrics,
        "selected_on_validation_mae": chosen,
        "route_candidate_is_better_than_simple_on_validation":
            validation["station_line_destination"]["mae_minutes"] <
            validation["station_type_hour"]["mae_minutes"],
        "route_choices": len(routes),
        "feature_availability": {
            name + "_unknown_fraction": round(float((train[name] == MISSING).mean()), 6)
            for name in ("line_number", "final_destination_station")
        },
        "caveat": "Retrospective API changed times are not verified train arrivals/departures. "
                  "No start station or intermediate station is inferred from final destination.",
    }
    dest = Path(reports_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"route_minutes_{mode}_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def validate_artifact(artifact):
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA:
        raise ValueError("Unsupported route artifact schema.")
    if artifact.get("month") != MONTH or artifact.get("train_dates") != ["2026-09-01", "2026-09-16"]:
        raise ValueError("Unreviewed training window.")
    if artifact.get("mode") not in MODES:
        raise ValueError("Invalid Dortmund event mode.")
    if artifact.get("is_verified_real_arrival") is not False or artifact.get("is_live") is not False:
        raise ValueError("Invalid model limitations metadata.")
    if artifact.get("route_origin_known") is not False:
        raise ValueError("Cannot claim origin without reconstructed route data.")
    if artifact.get("categorical") != CAT_ROUTE or artifact.get("features") != CAT_ROUTE + NUMERIC:
        raise ValueError("Unexpected route input features.")
    for key in ("minutes_model", "probability_model"):
        model = artifact.get(key)
        if not isinstance(model, dict) or set(model.get("categories", {})) != set(CAT_ROUTE):
            raise ValueError("Malformed model category schema.")
        for col in CAT_ROUTE:
            values = model["categories"][col]
            if not isinstance(values, list) or not values or len(values) != len(set(values)) or not all(isinstance(x, str) and x for x in values):
                raise ValueError(f"Invalid categories for {col}.")
        count = sum(len(model["categories"][col]) for col in CAT_ROUTE) + len(NUMERIC)
        weights = model.get("weights")
        if not isinstance(weights, list) or len(weights) != count:
            raise ValueError("Wrong model coefficient shape.")
        if not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
                   for x in weights + [model.get("intercept")]):
            raise ValueError("Invalid model coefficients.")
    routes = artifact.get("routes")
    if not isinstance(routes, list) or not routes or len(routes) > MAX_ROUTES:
        raise ValueError("No learned route combinations available.")
    for route in routes:
        if not all(isinstance(route.get(x), str) for x in CAT_ROUTE):
            raise ValueError("Invalid route combination.")
        if type(route.get("training_examples")) is not int or route["training_examples"] < MIN_ROUTE_ROWS:
            raise ValueError("Insufficient historical route support.")
    return artifact


def load_route_model(mode="arrival", root=MODEL_DIR):
    path = local_model_path(mode, root)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Unexpectedly large route artifact.")
    return validate_artifact(json.loads(path.read_text(encoding="utf-8")))


def _score_linear(model, features):
    total = model["intercept"]
    idx = 0
    for col in CAT_ROUTE:
        categories = model["categories"][col]
        val = features[col]
        if val in categories:
            total += model["weights"][idx + categories.index(val)]
        idx += len(categories)
    return total + model["weights"][idx] * features["scheduled_hour"] + model["weights"][idx+1] * features["day_of_week"]


def estimate_at_dortmund(artifact, route, scheduled_hour, day_of_week):
    """Return experimental minutes and archived-update probability at Dortmund."""
    validate_artifact(artifact)
    if type(scheduled_hour) is not int or not 0 <= scheduled_hour <= 23:
        raise ValueError("Scheduled hour must be 0..23.")
    if type(day_of_week) is not int or not 1 <= day_of_week <= 7:
        raise ValueError("Weekday must be 1..7.")
    if not isinstance(route, dict) or not all(isinstance(route.get(f), str) for f in CAT_ROUTE):
        raise ValueError("Route fields are required.")
    if not any(all(route[col] == r[col] for col in CAT_ROUTE) for r in artifact["routes"]):
        raise ValueError("This route was not observed sufficiently often in training.")
    row = {**{col: route[col] for col in CAT_ROUTE},
           "scheduled_hour": scheduled_hour, "day_of_week": day_of_week}
    minutes = max(0.0, min(240.0, _score_linear(artifact["minutes_model"], row)))
    logit = _score_linear(artifact["probability_model"], row)
    if logit >= 0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        t = math.exp(logit)
        probability = t / (1.0 + t)
    return {"estimated_archived_delay_minutes": round(float(minutes), 1),
            "estimated_archived_6plus_probability": round(float(probability), 4),
            "location": row["station_name"], "mode": artifact["mode"],
            "is_verified_actual_delay": False}


def train_local_route_models(data_root=Path("data"), model_dir=MODEL_DIR,
                             reports_dir=Path("reports"), mode="both"):
    """Generate distinct arrival and departure models, using explicit snapshots."""
    from .config import source_filename
    from .download import download_month
    from .pipeline import ingest_dortmund

    if mode not in (*MODES, "both"):
        raise ValueError("Mode must be arrival, departure, or both.")
    root = Path(data_root)
    processed = root / "processed" / "dortmund_2026-09.parquet"
    national = root / "raw" / source_filename(MONTH)
    if not processed.is_file():
        if not national.is_file():
            national = download_month(MONTH, root)
        ingest_dortmund(national, root)
    changes = root / "raw" / change_filename(MONTH)
    if not changes.is_file():
        changes = download_change_month(MONTH, root)

    report = {}
    for event in (MODES if mode == "both" else (mode,)):
        frame = extract_route_rows(MONTH, processed, changes, event)
        report[event] = fit_route_minutes(frame, event, reports_dir, model_dir)
    return report
