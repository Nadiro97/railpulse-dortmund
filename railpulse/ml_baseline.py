"""Leakage-aware historical baseline for EXPLICIT timetable update deviations.

Target: latest explicit updated arrival time is >=6 minutes after scheduled arrival.
This is NOT an independently verified physical train delay.
Only calendar, scheduled hour, station, and train category are model inputs.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import checked_month
from .pipeline import quoted
from .observed_updates import REQUIRED_CHANGES, REQUIRED_PROCESSED, check_schema

CATEGORICAL = ["station_name", "train_type"]
NUMERIC = ["scheduled_hour", "day_of_week"]


def month_bounds(month: str) -> tuple[date, date, date]:
    """Training days 1-20; test days 21-end (strict chronological split)."""
    year, number = map(int, checked_month(month).split("-"))
    start = date(year, number, 1)
    end = date(year + (number == 12), (number % 12) + 1, 1)
    split = date(year, number, 21)
    return start, split, end


def extract_training_rows(
    month: str, processed_file: Path, change_file: Path,
):
    """Return only eligible, explicitly changed arrivals from local files.

    The stop-level DataFrame is strictly in memory and is NEVER published.
    Large parsed snapshots are reduced to Dortmund EVA codes before joining.
    """
    import duckdb

    start, split, end = month_bounds(month)
    processed_file = Path(processed_file)
    change_file = Path(change_file)
    for f in (processed_file, change_file):
        if not f.is_file():
            raise FileNotFoundError(f)
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='4GB'")
        source = f"read_parquet({quoted(processed_file.resolve())})"
        changes = f"read_parquet({quoted(change_file.resolve())})"
        for query, expected, name in (
            (source, REQUIRED_PROCESSED, "processed"),
            (changes, REQUIRED_CHANGES, "parsed changes"),
        ):
            actual = {row[0] for row in connection.execute(
                f"DESCRIBE SELECT * FROM {query}"
            ).fetchall()}
            check_schema(actual, expected, name)
        connection.execute(f"""
            CREATE TEMP TABLE base AS
            SELECT id, eva, station_name, COALESCE(train_type, '(unknown)') AS train_type,
                arrival_planned_time
            FROM {source}
            WHERE station_name ILIKE 'Dortmund%'
              AND arrival_planned_time >= TIMESTAMP '{start}'
              AND arrival_planned_time < TIMESTAMP '{end}'
              AND arrival_is_canceled IS FALSE
              AND has_plan_record IS TRUE
              AND is_additional_stop IS FALSE
              AND is_replacement_train IS FALSE
        """)
        # Explicit change only: no COALESCE to planned times.
        connection.execute(f"""
            CREATE TEMP TABLE changed AS
            SELECT id, eva,
                arg_max(arrival_change_time, snapshot_timestamp)
                    AS explicit_updated_time
            FROM {changes}
            WHERE eva IN (SELECT DISTINCT eva FROM base)
              AND arrival_change_time IS NOT NULL
            GROUP BY id, eva
        """)
        return connection.execute("""
            SELECT b.station_name, b.train_type,
                EXTRACT(hour FROM b.arrival_planned_time)::INTEGER AS scheduled_hour,
                EXTRACT(isodow FROM b.arrival_planned_time)::INTEGER AS day_of_week,
                CAST(b.arrival_planned_time AS DATE) AS planned_date,
                CASE WHEN date_diff('second', b.arrival_planned_time,
                        c.explicit_updated_time) / 60.0 >= 6
                    THEN 1 ELSE 0 END AS target_6plus
            FROM base b JOIN changed c USING (id, eva)
            WHERE date_diff('second', b.arrival_planned_time, c.explicit_updated_time)
                    / 60.0 BETWEEN -60 AND 240
            ORDER BY planned_date
        """).df()
    finally:
        connection.close()


def evaluate_binary(y_true, probability) -> dict:
    """Metrics with a clear fixed threshold, including imbalance-aware PR AUC."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss,
        confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    )
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probability, dtype=float)
    binary = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(labels, binary)), 5),
        "average_precision": round(float(average_precision_score(labels, scores)), 5)
            if len(set(labels)) > 1 else None,
        "roc_auc": round(float(roc_auc_score(labels, scores)), 5)
            if len(set(labels)) > 1 else None,
        "brier": round(float(brier_score_loss(labels, scores)), 5),
        "precision_at_0_5": round(float(precision_score(labels, binary, zero_division=0)), 5),
        "recall_at_0_5": round(float(recall_score(labels, binary, zero_division=0)), 5),
        "f1_at_0_5": round(float(f1_score(labels, binary, zero_division=0)), 5),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_and_evaluate(
    frame, month: str, reports_root: Path = Path("reports")
) -> dict:
    """Train a historical baseline and evaluate on later calendar dates only."""
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    start, split, end = month_bounds(month)
    if frame.empty:
        raise ValueError("No explicitly updated arrival records found.")
    dates = frame["planned_date"].map(
        lambda value: value.date() if hasattr(value, "date") else
        (date.fromisoformat(value) if isinstance(value, str) else value)
    )
    train_mask = (dates >= start) & (dates < split)
    test_mask = (dates >= split) & (dates < end)
    train, test = frame.loc[train_mask].copy(), frame.loc[test_mask].copy()
    if len(train) < 100 or len(test) < 100:
        raise ValueError("At least 100 explicit-update rows per split are required.")
    if train["target_6plus"].nunique() != 2 or test["target_6plus"].nunique() != 2:
        raise ValueError("Both classes must be represented in train and test.")
    features = CATEGORICAL + NUMERIC
    if not set(features).issubset(frame.columns):
        raise ValueError("Missing scheduled-time feature columns")
    X_train, X_test = train[features], test[features]
    y_train = train["target_6plus"].astype(int)
    y_test = test["target_6plus"].astype(int)
    model = Pipeline([
        ("encode", ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("calendar", "passthrough", NUMERIC),
        ])),
        ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict_proba(X_test)[:, 1]
    no_skill = np.full(len(test), float(y_train.mean()))

    report = {
        "experiment": "scheduled-features prediction of latest explicit arrival update >= 6 min",
        "label_is_confirmed_physical_arrival": False,
        "selection": "Only stops with an explicit change snapshot and a plausible delta",
        "coverage_caution": "Exclude missing updates; test estimates do not generalize to all trains",
        "month": checked_month(month),
        "train_window": [str(start), str(split - timedelta(days=1))],
        "test_window": [str(split), str(end - timedelta(days=1))],
        "features_used": features,
        "forbidden_post_event_features": [
            "arrival_change_time", "delay_in_min", "snapshot_timestamp",
            "explicit_updated_time", "target_6plus", "arrival_is_canceled",
        ],
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "train_positive_fraction": round(float(y_train.mean()), 6),
        "test_positive_fraction": round(float(y_test.mean()), 6),
        "prevalence_baseline": evaluate_binary(y_test, no_skill),
        "logistic_regression": evaluate_binary(y_test, predicted),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Deutsche Bahn, CC BY 4.0; Piet Brömmel archive",
        "result_is_portfolio_baseline_only": True,
    }
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "ml_baseline_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (reports_root / "ml_baseline_comparison.csv").open("w", newline="") as fh:
        keys = ["model", "roc_auc", "average_precision", "brier",
                "accuracy", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5"]
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for name in ("prevalence_baseline", "logistic_regression"):
            writer.writerow({"model": name, **{
                key: report[name][key] for key in keys if key != "model"
            }})
    return report


def run_baseline(
    month: str, processed_file: Path, change_file: Path,
    reports_root: Path = Path("reports")
) -> dict:
    rows = extract_training_rows(month, processed_file, change_file)
    return fit_and_evaluate(rows, month, reports_root)
