"""Evidence-first audit of the entire archival timetable-update model.

NOT a verified physical-arrival delay evaluation. This assesses the fixed,
September-trained logistic model on historical archive labels, with all train
types and Dortmund stations included in the population. Raw train rows stay
private; only large-group summaries are written to disk.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from .ml_baseline import CATEGORICAL, NUMERIC, evaluate_binary, extract_training_rows
from .ml_cross_month import ALERT_THRESHOLD
from .ml_improved import make_models, threshold_metrics

TRAIN_MONTH = "2026-09"
COMPARISON_MONTH = "2026-10"
FEATURES = CATEGORICAL + NUMERIC
MIN_GROUP_SIZE = 100


def _dates(frame):
    return frame["planned_date"].map(
        lambda v: v.date() if hasattr(v, "date") else (
            date.fromisoformat(v) if isinstance(v, str) else v
        )
    )


def _check_frame(frame, name):
    required = set(FEATURES + ["target_6plus", "planned_date"])
    if not required.issubset(frame.columns):
        raise ValueError(f"{name}: required pre-schedule columns and target missing")
    if len(frame) < 100 or frame["target_6plus"].nunique() != 2:
        raise ValueError(f"{name}: needs at least 100 records and both outcomes")


def score_summary(labels, probabilities, threshold=ALERT_THRESHOLD):
    """Meaningful only for the *archived explicit update* population."""
    import numpy as np

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not len(y) or len(y) != len(p):
        raise ValueError("Scores and labels must be nonempty and have equal length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid binary labels or model scores")
    decision = threshold_metrics(y, p, threshold)
    return {
        "sample_n": int(len(y)),
        "observed_archived_event_fraction": round(float(y.mean()), 6),
        "average_model_score": round(float(p.mean()), 6),
        "score_event_gap": round(float(p.mean() - y.mean()), 6),
        "roc_auc": evaluate_binary(y, p)["roc_auc"] if y.min() != y.max() else None,
        "average_precision": evaluate_binary(y, p)["average_precision"] if y.min() != y.max() else None,
        "brier": evaluate_binary(y, p)["brier"],
        "threshold": decision["threshold"],
        "precision": decision["precision"],
        "recall": decision["recall"],
        "false_alerts": decision["fp"],
        "true_alerts": decision["tp"],
        "missed_events": decision["fn"],
        "false_alerts_per_1000": round(float(1000 * decision["fp"] / len(y)), 2),
    }


def _group_row(group_type, value, labels, scores, minimum=MIN_GROUP_SIZE):
    """Exclude small groups to avoid implying meaningful subgroup estimates."""
    if len(labels) < minimum:
        return None
    result = score_summary(labels, scores)
    return {
        "cohort": group_type,
        "value": str(value),
        "n": result["sample_n"],
        "observed_6plus_update_fraction": result["observed_archived_event_fraction"],
        "average_model_score": result["average_model_score"],
        "score_minus_observed_fraction": result["score_event_gap"],
        "roc_auc": result["roc_auc"],
        "average_precision": result["average_precision"],
        "brier": result["brier"],
        "precision_at_0225": result["precision"],
        "recall_at_0225": result["recall"],
        "true_alerts": result["true_alerts"],
        "false_alerts": result["false_alerts"],
        "missed_events": result["missed_events"],
        "false_alerts_per_1000": result["false_alerts_per_1000"],
    }


def grouped_audit(frame, probabilities, minimum=MIN_GROUP_SIZE):
    """Check all station and train-type cohorts, not just a chosen service."""
    import numpy as np

    _check_frame(frame, "audit")
    scores = np.asarray(probabilities, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("Probability count does not match frame")
    rows, omitted = [], {}
    for name in CATEGORICAL:
        suppressed = 0
        for value, positions in frame.groupby(name, dropna=False, sort=True).indices.items():
            subset = frame.iloc[positions]
            result = _group_row(
                name, value, subset["target_6plus"].to_numpy(), scores[positions],
                minimum=minimum,
            )
            if result is None:
                suppressed += 1
            else:
                rows.append(result)
        omitted[name] = suppressed
    return rows, omitted


def calibration_bins(frame, probabilities, minimum=MIN_GROUP_SIZE):
    """Reliability diagram for archival labels only, not actual arrivals."""
    import numpy as np

    _check_frame(frame, "calibration")
    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(frame) or not np.isfinite(p).all():
        raise ValueError("Calibration scores invalid")
    y = frame["target_6plus"].to_numpy(dtype=int)
    bin_idx = np.minimum((p * 10).astype(int), 9)
    rows = []
    for i in range(10):
        selected = bin_idx == i
        n = int(selected.sum())
        if n < minimum:
            continue
        observed = float(y[selected].mean())
        predicted = float(p[selected].mean())
        rows.append({
            "bin_lower": round(i / 10, 1),
            "bin_upper": round((i + 1) / 10, 1),
            "n": n,
            "mean_model_score": round(predicted, 6),
            "observed_archived_event_fraction": round(observed, 6),
            "absolute_gap": round(abs(predicted - observed), 6),
        })
    return rows


def evaluate_fixed_archival_model(reference_frame, next_frame,
                                  reports_root=Path("reports")):
    """Train once on September 1–16; September 21–end/October are diagnostics."""
    import pandas as pd

    _check_frame(reference_frame, "September")
    _check_frame(next_frame, "October")
    train_dates = _dates(reference_frame)
    future_dates = _dates(next_frame)
    train = reference_frame.loc[
        (train_dates >= date(2026, 9, 1)) & (train_dates <= date(2026, 9, 16))
    ].copy()
    september_holdout = reference_frame.loc[
        (train_dates >= date(2026, 9, 21)) & (train_dates <= date(2026, 9, 30))
    ].copy()
    october_holdout = next_frame.loc[
        (future_dates >= date(2026, 10, 1)) & (future_dates <= date(2026, 10, 31))
    ].copy()
    for name, data in [
        ("training", train), ("September holdout", september_holdout),
        ("October holdout", october_holdout),
    ]:
        _check_frame(data, name)
    model = make_models()["logistic_regression"]
    model.fit(train[FEATURES], train["target_6plus"].astype(int))
    report = {
        "title": "Whole-model integrity and archived-timetable-label diagnostic",
        "model": "September 1–16 2026 logistic regression / fixed 0.225 decision threshold",
        "purpose": "Audit generalization across stations, train categories, and score levels",
        "target": "latest explicitly recorded timetable arrival update >= 6 minutes after schedule",
        "target_is_verified_physical_arrival": False,
        "fit_n": int(len(train)),
        "training_dates": ["2026-09-01", "2026-09-16"],
        "september_holdout_dates": ["2026-09-21", "2026-09-30"],
        "october_holdout_dates": ["2026-10-01", "2026-10-31"],
        "threshold_frozen_before_holdouts": True,
        "october_has_been_inspected_before": True,
        "ready_for_passenger_delay_predictions": False,
        "needs_verified_arrival_ground_truth": True,
        "outcomes_missing_for_stops_without_explicit_updates": True,
        "notes": [
            "This does not evaluate actual arrival-time reliability.",
            "Unobserved updates cannot be assumed punctual.",
            "Cancelled and replacement services are outside this model.",
            "Station and train-type subgroup scores can differ substantially.",
            "Scores are not calibrated to physical train delays.",
            "A later untouched full month is needed for future model-promotion decisions.",
        ],
        "source": "Deutsche Bahn CC BY 4.0; historical archive by Piet Brömmel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "periods": {},
    }
    cohorts, calibration = [], []
    for period, data in [
        ("september_holdout", september_holdout), ("october_holdout", october_holdout)
    ]:
        scores = model.predict_proba(data[FEATURES])[:, 1]
        y = data["target_6plus"].astype(int)
        summary = score_summary(y, scores)
        all_groups, omitted = grouped_audit(data, scores)
        bins = calibration_bins(data, scores)
        summary["omitted_groups_below_100_records"] = omitted
        summary["reported_groups"] = len(all_groups)
        summary["calibration_bins_reported"] = len(bins)
        report["periods"][period] = summary
        cohorts.extend({"period": period, **row} for row in all_groups)
        calibration.extend({"period": period, **row} for row in bins)

    folder = Path(reports_root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_audit_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _csv(folder / "model_audit_cohorts.csv", cohorts, [
        "period", "cohort", "value", "n", "observed_6plus_update_fraction",
        "average_model_score", "score_minus_observed_fraction", "roc_auc",
        "average_precision", "brier", "precision_at_0225", "recall_at_0225",
        "true_alerts", "false_alerts", "missed_events", "false_alerts_per_1000",
    ])
    _csv(folder / "model_audit_calibration.csv", calibration, [
        "period", "bin_lower", "bin_upper", "n", "mean_model_score",
        "observed_archived_event_fraction", "absolute_gap",
    ])
    return report


def _csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_model_audit(data_root=Path("data"), reports_root=Path("reports")):
    """Use data already downloaded locally, or download from vetted archive."""
    import pandas as pd
    from .config import source_filename
    from .download import download_month
    from .pipeline import ingest_dortmund
    from .observed_updates import change_filename, download_change_month, validate_month

    root = Path(data_root)
    reports = Path(reports_root)
    frames = {}
    coverage = []
    for month in (TRAIN_MONTH, COMPARISON_MONTH):
        processed = root / "processed" / f"dortmund_{month}.parquet"
        source = root / "raw" / source_filename(month)
        if not processed.is_file():
            if not source.is_file():
                source = download_month(month, root)
            ingest_dortmund(source, root)
        changes = root / "raw" / change_filename(month)
        if not changes.is_file():
            changes = download_change_month(month, root)
        with TemporaryDirectory(prefix="railpulse_audit_") as temp_dir:
            paths, _ = validate_month(month, processed, changes, Path(temp_dir))
            for path in paths:
                if Path(path).name in ("validated_quality.csv", "validated_by_train_type.csv",
                                       "validated_by_station.csv"):
                    df = pd.read_csv(path)
                    for record in df.to_dict(orient="records"):
                        if Path(path).name != "validated_quality.csv":
                            # Suppress small cohorts to avoid fragile descriptive comparisons.
                            if int(record.get("eligible_scheduled_rows", 0)) < MIN_GROUP_SIZE:
                                continue
                        scope = Path(path).name.removeprefix("validated_").removesuffix(".csv")
                        coverage.append({"month": month, "scope": scope, **record})
        frames[month] = extract_training_rows(month, processed, changes)
    result = evaluate_fixed_archival_model(frames[TRAIN_MONTH],
                                           frames[COMPARISON_MONTH], reports)
    # Existing validated columns are aggregates; write no per-stop or model data.
    fields = ["month", "scope", "station_name", "train_type",
              "all_processed_dortmund_rows", "stations", "eligible_scheduled_arrival_rows",
              "eligible_with_any_change_snapshot", "eligible_with_explicit_arrival_update",
              "eligible_without_explicit_arrival_update", "eligible_without_change_snapshot",
              "explicit_updates_in_valid_range",
              "explicit_updates_disagreeing_with_processed_time",
              "cancelled_arrival_records", "explicit_update_coverage_pct",
              "no_explicit_update_share_pct",
              "eligible_scheduled_rows", "explicit_update_rows",
              "avg_explicit_update_delta_min", "median_explicit_update_delta_min",
              "explicit_updates_6plus_pct"]
    coverage_path = reports / "model_audit_coverage.csv"
    with coverage_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(coverage)
    result["coverage_audit"] = "model_audit_coverage.csv"
    (reports / "model_audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
