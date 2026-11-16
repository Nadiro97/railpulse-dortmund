"""Unseen-month evaluation of our locked September 2026 timetable-update classifier.

No October labels enter model fitting, model choice, or threshold selection.
Only aggregated scores are retained, never per-train or row-level data.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .config import checked_month
from .ml_baseline import CATEGORICAL, NUMERIC, extract_training_rows, month_bounds, evaluate_binary
from .ml_improved import make_models, threshold_metrics

MODEL = "logistic_regression"
ALERT_THRESHOLD = 0.225
FEATURES = CATEGORICAL + NUMERIC
MIN_PARTITION_ROWS = 100


def _dates(frame):
    return frame["planned_date"].map(
        lambda value: value.date() if hasattr(value, "date") else (
            date.fromisoformat(value) if isinstance(value, str) else value
        )
    )


def fixed_partitions(september_frame, next_month_frame, train_month: str, test_month: str):
    """Train on 1–16 of reference month; reproduce later holdout; test entire next month."""
    train_start, sept_hold_start, sept_end = month_bounds(train_month)
    oct_start, _, oct_end = month_bounds(test_month)
    if oct_start < sept_end:
        raise ValueError("Unseen test month must begin after the reference training month ends.")
    if not set(FEATURES + ["planned_date", "target_6plus"]).issubset(september_frame.columns):
        raise ValueError("Missing required reference-month model columns")
    if not set(FEATURES + ["planned_date", "target_6plus"]).issubset(next_month_frame.columns):
        raise ValueError("Missing required unseen-month model columns")

    dates = _dates(september_frame)
    test_dates = _dates(next_month_frame)
    train = september_frame.loc[
        (dates >= train_start) & (dates < date(train_start.year, train_start.month, 17))
    ].copy()
    sept_holdout = september_frame.loc[
        (dates >= sept_hold_start) & (dates < sept_end)
    ].copy()
    outside = next_month_frame.loc[
        (test_dates >= oct_start) & (test_dates < oct_end)
    ].copy()

    for label, frame in (("reference train", train), ("September holdout", sept_holdout),
                         ("unseen month", outside)):
        if len(frame) < MIN_PARTITION_ROWS:
            raise ValueError(f"{label} has fewer than {MIN_PARTITION_ROWS} eligible records.")
        if frame["target_6plus"].nunique() < 2:
            raise ValueError(f"{label} must contain both target classes.")
    return train, sept_holdout, outside


def evaluate_locked_months(
    reference_frame, unseen_frame, train_month: str = "2026-09",
    test_month: str = "2026-10", reports_root: Path = Path("reports")
) -> dict:
    """No model/threshold tuning on either held-out September or future October."""
    import numpy as np

    train, sep_test, outside = fixed_partitions(
        reference_frame, unseen_frame, train_month, test_month
    )
    model = make_models()[MODEL]
    train_labels = train["target_6plus"].astype(int)
    model.fit(train[FEATURES], train_labels)
    prevalence = float(train_labels.mean())

    def metrics(holdout):
        labels = holdout["target_6plus"].astype(int)
        probabilities = model.predict_proba(holdout[FEATURES])[:, 1]
        no_skill = np.full(len(holdout), prevalence, dtype=float)
        return {
            "sample_n": int(len(holdout)),
            "positive_fraction": round(float(labels.mean()), 6),
            "selected_model": {
                "ranking": evaluate_binary(labels, probabilities),
                "locked_alert_threshold": threshold_metrics(
                    labels, probabilities, ALERT_THRESHOLD
                ),
                "default_half_threshold": threshold_metrics(labels, probabilities, 0.5),
            },
            "constant_prevalence": {
                "ranking": evaluate_binary(labels, no_skill),
                "locked_alert_threshold": threshold_metrics(
                    labels, no_skill, ALERT_THRESHOLD
                ),
            },
        }

    sept_summary = metrics(sep_test)
    new_summary = metrics(outside)
    ref_start, _, ref_end = month_bounds(train_month)
    unseen_start, _, unseen_end = month_bounds(test_month)

    report = {
        "experiment": "Unseen-month test of fixed logistic risk model for explicit archived timetable updates >=6 minutes",
        "reference_train_month": checked_month(train_month),
        "unseen_test_month": checked_month(test_month),
        "training_dates": [
            str(ref_start),
            str(date(ref_start.year, ref_start.month, 16)),
        ],
        "reference_holdout_dates": [
            str(date(ref_start.year, ref_start.month, 21)),
            str(ref_end.fromordinal(ref_end.toordinal() - 1)),
        ],
        "unseen_test_dates": [
            str(unseen_start),
            str(unseen_end.fromordinal(unseen_end.toordinal() - 1)),
        ],
        "trained_rows": int(len(train)),
        "train_positive_fraction": round(prevalence, 6),
        "model_configuration": MODEL,
        "locked_probability_threshold": ALERT_THRESHOLD,
        "prearrived_features_only": FEATURES,
        "model_selection_was_september_validation": True,
        "untouched_test_labels_never_used_to_tune": True,
        "reference_september_holdout": sept_summary,
        "unseen_month_holdout": new_summary,
        "label_is_verified_physical_arrival": False,
        "caveat": "Retrospective API timetable updates may be forecasts, not physical arrival times. Only stops with explicit change snapshots enter evaluation; missing records are unknown.",
        "attribution": "Deutsche Bahn CC BY 4.0; historical archive by Piet Brömmel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    folder = Path(reports_root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ml_cross_month_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (folder / "ml_cross_month_comparison.csv").open("w", newline="") as handle:
        columns = [
            "period", "model", "sample_n", "positive_fraction",
            "threshold", "accuracy", "precision", "recall", "f1", "f2",
            "roc_auc", "average_precision", "brier", "tp", "fp", "fn", "tn",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for period, summary in (
            ("reference_september_holdout", sept_summary),
            ("unseen_month_holdout", new_summary),
        ):
            for name in ("selected_model", "constant_prevalence"):
                rank = summary[name]["ranking"]
                decision = summary[name]["locked_alert_threshold"]
                writer.writerow({
                    "period": period,
                    "model": MODEL if name == "selected_model" else "constant_prevalence",
                    "sample_n": summary["sample_n"],
                    "positive_fraction": summary["positive_fraction"],
                    **{key: decision[key] for key in (
                        "threshold", "accuracy", "precision", "recall", "f1", "f2",
                        "tp", "fp", "fn", "tn",
                    )},
                    **{key: rank[key] for key in ("roc_auc", "average_precision", "brier")},
                })
    return report


def run_cross_month(
    train_month: str, test_month: str, data_root: Path = Path("data"),
    reports_root: Path = Path("reports")
) -> dict:
    """Use only allowlisted historical archive months; fetch missing parsed changes."""
    from .observed_updates import change_filename, download_change_month

    checked_month(train_month)
    checked_month(test_month)
    root = Path(data_root)
    frames = []
    for month in (train_month, test_month):
        processed = root / "processed" / f"dortmund_{month}.parquet"
        if not processed.is_file():
            raise FileNotFoundError(f"Run the {month} downloader and extractor: {processed}")
        change = root / "raw" / change_filename(month)
        if not change.is_file():
            change = download_change_month(month, root)
        frames.append(extract_training_rows(month, processed, change))

    return evaluate_locked_months(
        frames[0], frames[1], train_month, test_month, reports_root
    )
