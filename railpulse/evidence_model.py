"""Train/evaluate an evidence-first, feature-engineered minutes estimator.

An explicit post-event timetable update is a *proxy*, NOT a physical arrival
observation. Missing labels are unknown, not punctual. Never use a future
month for tuning. No row-level data or serialized models are committed.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .event_data import extract_labels, get_month_sources, MODES, UNKNOWN

TRAIN_MONTHS = ("2026-07", "2026-08")
VALIDATION_MONTH = "2026-09"
INSPECTED_MONTH = "2026-10"
CATEGORICAL = [
    "station_name", "train_type", "line_number", "train_number",
    "route_origin", "route_destination",
]
NUMERIC = [
    "scheduled_hour", "weekday", "route_stop_index", "hops_before",
    "hops_after", "is_weekend", "rush_hour", "hour_sin", "hour_cos",
    "route_total_hops", "route_progress", "origin_known",
    "destination_known",
]
SIMPLE_CATEGORICAL = ["station_name", "train_type"]
SIMPLE_NUMERIC = ["scheduled_hour", "weekday", "is_weekend", "rush_hour"]
FEATURES = CATEGORICAL + NUMERIC
SCHEMA = "railpulse-evidence-first-minutes-v1"
MIN_VALIDATION_RECORDS = 100
MIN_SERVICE_SUPPORT = 20
ON_TIME_CUTOFF = 1.5
MODEL_ROOT = Path("models")
REPORT_ROOT = Path("reports")


def model_path(mode, directory=MODEL_ROOT):
    if mode not in MODES:
        raise ValueError("Mode must be arrival or departure.")
    return Path(directory) / f"evidence_minutes_{mode}.joblib"


def report_path(mode, directory=REPORT_ROOT):
    if mode not in MODES:
        raise ValueError("Mode must be arrival or departure.")
    return Path(directory) / f"evidence_minutes_{mode}_evaluation.json"


def build_features(frame):
    import numpy as np
    import pandas as pd

    required = set(CATEGORICAL + [
        "scheduled_hour", "weekday", "route_stop_index", "hops_before",
        "hops_after",
    ])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing scheduled pre-event features: {sorted(missing)}")
    out = frame.copy()
    for col in CATEGORICAL:
        out[col] = out[col].fillna(UNKNOWN).astype(str).str.strip()
        out.loc[out[col] == "", col] = UNKNOWN
    for col in ("scheduled_hour", "weekday", "route_stop_index",
                "hops_before", "hops_after"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["scheduled_hour"].isna().any() or not out["scheduled_hour"].between(0, 23).all():
        raise ValueError("Scheduled hours must be between 0 and 23.")
    if out["weekday"].isna().any() or not out["weekday"].between(1, 7).all():
        raise ValueError("Weekdays must be between 1 and 7.")
    out["is_weekend"] = (out["weekday"] >= 6).astype(float)
    out["rush_hour"] = (out["scheduled_hour"].isin([6, 7, 8, 9, 16, 17, 18, 19])).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * out["scheduled_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["scheduled_hour"] / 24)
    before = out["hops_before"].clip(lower=0)
    after = out["hops_after"].clip(lower=0)
    total = before + after
    out["route_total_hops"] = total
    out["route_progress"] = (before / total.replace(0, np.nan)).fillna(-1.0)
    out["origin_known"] = (out["route_origin"] != UNKNOWN).astype(float)
    out["destination_known"] = (out["route_destination"] != UNKNOWN).astype(float)
    return out


def _pipeline(kind):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, TargetEncoder

    if kind == "route_boost":
        cats = CATEGORICAL
        transformer = TargetEncoder(target_type="continuous", cv=5, smooth=20.0,
                                    shuffle=True, random_state=17)
        estimator = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=140, max_leaf_nodes=25,
            max_depth=6, learning_rate=0.055, l2_regularization=5,
            min_samples_leaf=40, random_state=17,
        )
        nums = NUMERIC
    elif kind == "simple_ridge":
        cats, nums = SIMPLE_CATEGORICAL, SIMPLE_NUMERIC
        transformer = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=25,
        )
        estimator = Ridge(alpha=30.0, solver="lsqr")
    elif kind == "route_ridge":
        cats, nums = CATEGORICAL, NUMERIC
        transformer = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=50,
        )
        estimator = Ridge(alpha=40.0, solver="lsqr")
    else:
        raise ValueError(f"Unknown candidate: {kind}")
    preprocessor = ColumnTransformer([
        ("cat", transformer, cats),
        ("num", SimpleImputer(strategy="constant", fill_value=-1), nums),
    ])
    return Pipeline([("features", preprocessor), ("model", estimator)])


def _weights(actual):
    import numpy as np

    y = np.asarray(actual, dtype=float)
    return np.where(y >= 15, 5.0, np.where(y >= 6, 3.0, 1.0))


def metrics(actual, predicted):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Missing or invalid evaluation labels/scores.")
    p = np.clip(p, 0, 120)
    residual = np.abs(y-p)
    serious = y >= 6
    severe = y >= 15
    punctual = y < 3
    announced_punctual = p <= ON_TIME_CUTOFF
    weights = _weights(y)
    return {
        "n": len(y),
        "observed_archive_6plus_fraction": round(float(serious.mean()), 5),
        "mae_minutes": round(float(mean_absolute_error(y, p)), 4),
        "weighted_mae_minutes": round(float(np.average(residual, weights=weights)), 4),
        "rmse_minutes": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "mae_on_6plus": round(float(residual[serious].mean()), 4) if serious.any() else None,
        "mae_on_15plus": round(float(residual[severe].mean()), 4) if severe.any() else None,
        "within_5_minutes": round(float((residual <= 5).mean()), 5),
        "severe_missed_as_punctual_fraction": round(
            float((announced_punctual[severe]).mean()), 5
        ) if severe.any() else None,
        "punctual_claim_precision": round(
            float(punctual[announced_punctual].mean()), 5
        ) if announced_punctual.any() else None,
        "punctual_claim_count": int(announced_punctual.sum()),
        "mean_actual_archived_minutes": round(float(y.mean()), 3),
        "mean_prediction_minutes": round(float(p.mean()), 3),
    }


def _pred(model, frame):
    import numpy as np

    if model is None:
        return np.zeros(len(frame), dtype=float)
    if isinstance(model, (int, float)):
        return np.full(len(frame), float(model))
    return np.clip(model.predict(frame), 0, 120)


def _as_date(frame):
    from datetime import date

    return frame["planned_date"].map(
        lambda x: x.date() if hasattr(x, "date") else
        date.fromisoformat(x) if isinstance(x, str) else x
    )


def cohort_errors(frame, prediction, minimum=100):
    """Aggregate performance by all sufficiently represented services, no rows."""
    import numpy as np

    values = np.asarray(prediction, dtype=float)
    if len(values) != len(frame):
        raise ValueError("Cohort predictions must match the frame length.")
    summaries = []
    for column in ("station_name", "train_type", "line_number"):
        for name, positions in frame.groupby(column, dropna=False).indices.items():
            if len(positions) < minimum:
                continue
            actual = frame.iloc[positions]["observed_update_minutes"].to_numpy()
            result = metrics(actual, values[positions])
            summaries.append({
                "group": column, "value": str(name),
                "n": result["n"],
                "mae_minutes": result["mae_minutes"],
                "weighted_mae_minutes": result["weighted_mae_minutes"],
                "severe_missed_as_punctual_fraction":
                    result["severe_missed_as_punctual_fraction"],
            })
    return sorted(summaries, key=lambda x: (
        -x["weighted_mae_minutes"], -x["n"], x["group"], x["value"],
    ))


def _service_key(row):
    return tuple(str(row[col]) for col in (
        "station_name", "train_type", "line_number", "route_origin",
        "route_destination",
    ))


def train_mode(frames, coverage, mode, model_dir=MODEL_ROOT,
               report_dir=REPORT_ROOT):
    """Select on September only; October is a previously inspected diagnostic."""
    import numpy as np
    import pandas as pd
    import joblib

    if mode not in MODES:
        raise ValueError("Unknown Dortmund event type")
    for month in (*TRAIN_MONTHS, VALIDATION_MONTH, INSPECTED_MONTH):
        if month not in frames:
            raise ValueError(f"Missing month {month}")
    train = build_features(pd.concat(
        [frames[month] for month in TRAIN_MONTHS], ignore_index=True
    ))
    validation = build_features(frames[VALIDATION_MONTH])
    inspected = build_features(frames[INSPECTED_MONTH])
    for name, data in (("train", train), ("validation", validation), ("inspected", inspected)):
        if len(data) < MIN_VALIDATION_RECORDS:
            raise ValueError(f"Too few valid {name} labels.")
        if "observed_update_minutes" not in data.columns:
            raise ValueError(f"Missing actual archival-update label for {name}.")
        vals = data["observed_update_minutes"]
        if vals.isna().any() or not vals.between(0, 120).all():
            raise ValueError(f"Invalid {name} archival-minute labels.")
    training_y = train["observed_update_minutes"]
    median = float(training_y.median())
    mean = float(training_y.mean())
    candidates = {
        "always_punctual": None,
        "training_median": median,
        "training_mean": mean,
        "simple_ridge": _pipeline("simple_ridge"),
        "route_ridge": _pipeline("route_ridge"),
        "route_boost": _pipeline("route_boost"),
    }
    for model in candidates.values():
        if hasattr(model, "fit"):
            model.fit(train, training_y)
    validation_scores = {name: metrics(validation["observed_update_minutes"],
                                       _pred(model, validation))
                         for name, model in candidates.items()}
    # Predeclared operationally conservative selection: PRIMARY weighted MAE,
    # with ordinary MAE as tie break. No October labels can affect selection.
    selected = min(validation_scores, key=lambda n: (
        validation_scores[n]["weighted_mae_minutes"],
        validation_scores[n]["mae_minutes"],
        list(candidates).index(n),
    ))
    diagnostics = {name: metrics(inspected["observed_update_minutes"],
                                _pred(model, inspected))
                   for name, model in candidates.items()}
    validation_cohorts = cohort_errors(validation, _pred(candidates[selected], validation))
    october_cohorts = cohort_errors(inspected, _pred(candidates[selected], inspected))
    # Fitted models and support dictionary remain strictly local.
    key_columns = [
        "station_name", "train_type", "line_number",
        "route_origin", "route_destination",
    ]
    supports = (train.groupby(key_columns, dropna=False).size()
                .reset_index(name="support"))
    supports = supports.loc[supports["support"] >= MIN_SERVICE_SUPPORT]
    medians = (train.groupby(key_columns, dropna=False)[
        ["route_stop_index", "hops_before", "hops_after"]
    ].median().reset_index())
    supports = supports.merge(medians, on=key_columns, how="left")
    service_rows = []
    for _, row in supports.sort_values("support", ascending=False).iterrows():
        service_rows.append({
            **{col: str(row[col]) for col in key_columns},
            "support": int(row["support"]),
            "route_stop_index_median": float(row["route_stop_index"]) if not pd.isna(row["route_stop_index"]) else -1.,
            "hops_before_median": float(row["hops_before"]) if not pd.isna(row["hops_before"]) else -1.,
            "hops_after_median": float(row["hops_after"]) if not pd.isna(row["hops_after"]) else -1.,
        })
    # For the user-facing interface, abstain if there are no supported services.
    if not service_rows:
        raise ValueError("Insufficient repeating pre-event routes for a reliable demo.")
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline_loss = validation_scores["always_punctual"]["weighted_mae_minutes"]
    winner_loss = validation_scores[selected]["weighted_mae_minutes"]
    deployment_allowed = (
        selected not in {"always_punctual", "training_median", "training_mean"}
        and winner_loss < baseline_loss * 0.97
        and validation_scores[selected]["severe_missed_as_punctual_fraction"] is not None
        and validation_scores[selected]["severe_missed_as_punctual_fraction"] <= 0.50
        and validation_scores[selected]["punctual_claim_precision"] is not None
        and validation_scores[selected]["punctual_claim_precision"] >= 0.80
        and all(coverage.get(m, {}).get("label_coverage_fraction", 0) >= 0.50
                for m in (*TRAIN_MONTHS, VALIDATION_MONTH, INSPECTED_MONTH))
        and diagnostics[selected]["weighted_mae_minutes"] <
            diagnostics["always_punctual"]["weighted_mae_minutes"]
        and (diagnostics[selected]["severe_missed_as_punctual_fraction"] is not None)
        and diagnostics[selected]["severe_missed_as_punctual_fraction"] <= 0.60
    )
    artifact = {
        "schema": SCHEMA,
        "mode": mode,
        "source_model": selected,
        "model": candidates[selected],
        "service_rows": service_rows,
        "min_service_support": MIN_SERVICE_SUPPORT,
        "schema_features": FEATURES,
        "training_months": list(TRAIN_MONTHS),
        "validation_month": VALIDATION_MONTH,
        "inspected_month": INSPECTED_MONTH,
        "deployment_allowed": bool(deployment_allowed),
        "physically_verified": False,
        "actual_arrival_verified": False,
        "observed_update_proxy_only": True,
        "source": "Deutsche Bahn CC BY 4.0; Piet Broemmel archive",
    }
    joblib.dump(artifact, model_path(mode, model_dir))
    report = {
        "schema": SCHEMA,
        "mode": mode,
        "outcome": "Post-scheduled-event explicitly recorded timetable change vs planned Dortmund event, capped positive minutes",
        "is_independent_physical_delay": False,
        "train_months": list(TRAIN_MONTHS),
        "selection_month": VALIDATION_MONTH,
        "previously_inspected_diagnostic_month": INSPECTED_MONTH,
        "selection_primary": "weighted MAE: 1x if <6, 3x if >=6, 5x if >=15 archival-update minutes",
        "selection_secondary": "ordinary MAE",
        "on_time_display_threshold_minutes": ON_TIME_CUTOFF,
        "selected": selected,
        "eligible_for_user_facing_estimates": bool(deployment_allowed),
        "eligibility_reason": ("Passed provisional archive-label quality gates" if deployment_allowed
                               else "No reliable estimated punctual/delay display; failed archive-proxy quality gates"),
        "validation": validation_scores,
        "already_inspected_october": diagnostics,
        "validation_cohort_errors": validation_cohorts,
        "october_cohort_errors": october_cohorts,
        "month_quality": coverage,
        "training_count": int(len(train)),
        "validation_count": int(len(validation)),
        "october_count": int(len(inspected)),
        "supported_service_combinations": len(service_rows),
        "model_was_promoted_to_real_delay": False,
        "caveats": [
            "Latest change snapshot after scheduled time can still be a forecast, not a verified physical arrival/departure.",
            "No explicit post-event change => UNKNOWN, not punctual.",
            "Show no estimate if pre/post snapshot label coverage falls below 50% in training, validation, or inspected diagnostics.",
            "October is used only as a conservative block on unsafe generalization, NOT to choose the algorithm or tune its threshold.",
            "Route origin is first station listed in the planned arrival path; may not represent true journey origin.",
            "All inputs known at least 60 min before planned Dortmund event.",
            "October was previously inspected and is NOT an untouched future holdout.",
            "A smaller weighted MAE does not guarantee correct estimates for each journey.",
            "Canceled and replacement trains have separate operational outcomes and are excluded.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path(mode, report_dir).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_local_model(mode, directory=MODEL_ROOT):
    import joblib
    path = model_path(mode, directory)
    if path.stat().st_size > 30_000_000:
        raise ValueError("Unusually large model artifact.")
    # WARNING: joblib is pickle-based; use ONLY the file generated locally by
    # this training command, never a downloaded or untrusted model file.
    artifact = joblib.load(path)
    if (artifact.get("schema") != SCHEMA or artifact.get("mode") != mode or
        artifact.get("physically_verified") is not False or
        artifact.get("observed_update_proxy_only") is not True):
        raise ValueError("Invalid or outdated local model metadata.")
    return artifact


def estimate_minutes(artifact, service: dict, hour: int, weekday: int):
    """Returns only Pünktlich or Verspätung: X Min. Abstains if no support."""
    import pandas as pd

    if type(hour) is not int or not 0 <= hour < 24:
        raise ValueError("Hour must be an integer 0-23.")
    if type(weekday) is not int or not 1 <= weekday <= 7:
        raise ValueError("Weekday must be an integer 1-7.")
    required = ("station_name", "train_type", "line_number",
                "route_origin", "route_destination")
    if not isinstance(service, dict) or any(not isinstance(service.get(x), str)
                                            for x in required):
        raise ValueError("A known line and direction must be supplied.")
    match = next((s for s in artifact["service_rows"]
                  if all(s[col] == service[col] for col in required)), None)
    if match is None or match["support"] < MIN_SERVICE_SUPPORT:
        return {"status": "Keine belastbare Schätzung",
                "minutes": None, "reason": "Insufficient historic coverage."}
    if not artifact.get("deployment_allowed", False):
        return {"status": "Keine belastbare Schätzung", "minutes": None,
                "reason": "Failed validation quality criteria; do not pretend to know punctuality."}
    # No features after the prediction cutoff are used here. Path position
    # statistics must be stored per support group in a subsequent improvement;
    # never synthesize downstream live information.
    row = {**{c: service.get(c, UNKNOWN) for c in CATEGORICAL},
           "scheduled_hour": hour, "weekday": weekday,
           "route_stop_index": float(match.get("route_stop_index_median", -1)),
           "hops_before": float(match.get("hops_before_median", -1)),
           "hops_after": float(match.get("hops_after_median", -1))}
    frame = build_features(pd.DataFrame([row]))
    raw = float(_pred(artifact["model"], frame)[0])
    if raw <= ON_TIME_CUTOFF:
        return {"status": "Pünktlich", "minutes": 0,
                "is_verified_actual": False, "support": match["support"]}
    estimated = max(2, round(raw))
    return {"status": f"Voraussichtliche Verspätung: {estimated} Minuten",
            "minutes": estimated, "is_verified_actual": False,
            "support": match["support"]}


def run_training(data_root=Path("data"), model_dir=MODEL_ROOT,
                 report_dir=REPORT_ROOT, mode="both"):
    """Download public licensed archives locally; save aggregate reports only."""
    if mode not in (*MODES, "both"):
        raise ValueError("Select arrival, departure or both.")
    months = (*TRAIN_MONTHS, VALIDATION_MONTH, INSPECTED_MONTH)
    modes = MODES if mode == "both" else (mode,)
    results = {}
    data, quality = {}, {}
    for month in months:
        sources = get_month_sources(month, data_root)
        for event in modes:
            data[(month, event)], quality[(month, event)] = extract_labels(
                month, event, sources
            )
    for event in modes:
        coverage = {month: quality[(month, event)] for month in months}
        frames = {month: data[(month, event)] for month in months}
        results[event] = train_mode(
            frames, coverage, event, model_dir, report_dir
        )
    return results
