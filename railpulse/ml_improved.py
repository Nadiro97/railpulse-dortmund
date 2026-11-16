"""Validation-tuned timetable-update classifier; train/test chronology preserved.

This is a retrospective API timetable-update task, NOT independently verified
train arrival prediction. Only aggregate metrics are written to disk.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .ml_baseline import (
    CATEGORICAL, NUMERIC, extract_training_rows, month_bounds, evaluate_binary,
)

FEATURES = CATEGORICAL + NUMERIC
CANDIDATES = ("logistic_regression", "histogram_gradient_boosting")
MIN_VALIDATION_PRECISION = 0.45
THRESHOLDS = [round(0.05 + i * 0.025, 3) for i in range(31)]


def validation_month_splits(frame, month: str):
    """Train 1-16, validate 17-20, and evaluate once on days 21-end."""
    start, test_start, end = month_bounds(month)
    validation_start = date(start.year, start.month, 17)
    dates = frame["planned_date"].map(
        lambda d: d.date() if hasattr(d, "date") else
        (date.fromisoformat(d) if isinstance(d, str) else d)
    )
    partitions = (
        frame.loc[(dates >= start) & (dates < validation_start)].copy(),
        frame.loc[(dates >= validation_start) & (dates < test_start)].copy(),
        frame.loc[(dates >= test_start) & (dates < end)].copy(),
    )
    for name, part in zip(("train", "validation", "test"), partitions):
        if len(part) < 100:
            raise ValueError(f"{name} needs at least 100 explicit-update rows.")
        if part["target_6plus"].nunique() != 2:
            raise ValueError(f"{name} requires examples of both classes.")
    if not set(FEATURES).issubset(frame.columns):
        raise ValueError("Required pre-arrival features not present")
    return partitions


def threshold_metrics(labels, probabilities, threshold: float) -> dict:
    """Explicit alert trade-off for a threshold selected on validation only."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, fbeta_score, f1_score,
        precision_score, recall_score,
    )
    labels = np.asarray(labels, dtype=int)
    alerts = (np.asarray(probabilities, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, alerts, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": round(float(accuracy_score(labels, alerts)), 5),
        "precision": round(float(precision_score(labels, alerts, zero_division=0)), 5),
        "recall": round(float(recall_score(labels, alerts, zero_division=0)), 5),
        "f1": round(float(f1_score(labels, alerts, zero_division=0)), 5),
        "f2": round(float(fbeta_score(labels, alerts, beta=2, zero_division=0)), 5),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(labels, probabilities, min_precision=MIN_VALIDATION_PRECISION):
    """Maximize F2, subject to a predeclared validation precision constraint.

    If no threshold satisfies the constraint, choose max F2 and report fallback.
    The held-out test labels/scores NEVER enter this function.
    """
    choices = [threshold_metrics(labels, probabilities, threshold)
               for threshold in THRESHOLDS]
    acceptable = [m for m in choices if m["precision"] >= min_precision
                  and m["tp"] > 0]
    eligible = acceptable or choices
    best = max(eligible, key=lambda m: (m["f2"], m["recall"], m["precision"]))
    return best, bool(acceptable)


def make_models():
    """Two fixed configurations; selection uses validation average precision."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    logistic = Pipeline([
        ("encode", ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("calendar", "passthrough", NUMERIC),
        ])),
        ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])
    boosted = Pipeline([
        ("encode", ColumnTransformer([
            ("categories", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ), CATEGORICAL),
            ("calendar", "passthrough", NUMERIC),
        ], sparse_threshold=0)),
        ("model", HistGradientBoostingClassifier(
            categorical_features=[True, True, False, False],
            max_iter=120,
            learning_rate=0.07,
            max_leaf_nodes=15,
            min_samples_leaf=75,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=42,
        )),
    ])
    return {"logistic_regression": logistic, "histogram_gradient_boosting": boosted}


def fit_improved(frame, month: str, reports_root: Path = Path("reports")) -> dict:
    """Fit on earlier dates; choose model/threshold from validation; test once."""
    import numpy as np

    train, valid, test = validation_month_splits(frame, month)
    x_train, x_valid, x_test = (part[FEATURES] for part in (train, valid, test))
    y_train, y_valid, y_test = (
        part["target_6plus"].astype(int) for part in (train, valid, test)
    )
    trained = {}
    selected_thresholds = {}
    validation_results = {}

    for name, model in make_models().items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        threshold, constraint_met = select_threshold(y_valid, probabilities)
        validation_results[name] = {
            "ranking": evaluate_binary(y_valid, probabilities),
            "threshold_per_validation": threshold,
            "precision_constraint_met": constraint_met,
        }
        selected_thresholds[name] = threshold["threshold"]
        trained[name] = model

    # Locked choice: validation average precision, with Brier as a tie breaker.
    chosen = max(
        CANDIDATES,
        key=lambda name: (
            validation_results[name]["ranking"]["average_precision"],
            -validation_results[name]["ranking"]["brier"],
        ),
    )

    # Test labels are now accessed exactly once, after all choices are frozen.
    test_results = {}
    for name, model in trained.items():
        scores = model.predict_proba(x_test)[:, 1]
        test_results[name] = {
            "ranking": evaluate_binary(y_test, scores),
            "fixed_half": threshold_metrics(y_test, scores, 0.5),
            "validation_selected_threshold": threshold_metrics(
                y_test, scores, selected_thresholds[name]
            ),
        }

    prevalence = np.full(len(test), float(y_train.mean()), dtype=float)
    start, test_start, end = month_bounds(month)
    validation_start = date(start.year, start.month, 17)
    output = {
        "experiment": "Explicit retrospective arrival-update >= 6 min",
        "label_is_confirmed_physical_arrival": False,
        "month": month,
        "features_used": FEATURES,
        "forbidden_post_event_features": [
            "arrival_change_time", "delay_in_min", "snapshot_timestamp",
            "explicit_updated_time", "arrival_is_canceled", "target_6plus",
        ],
        "train_window": [str(start), str(validation_start - timedelta(days=1))],
        "validation_window": [
            str(validation_start), str(test_start - timedelta(days=1))
        ],
        "test_window": [str(test_start), str(end - timedelta(days=1))],
        "train_n": int(len(train)),
        "validation_n": int(len(valid)),
        "test_n": int(len(test)),
        "train_positive_fraction": round(float(y_train.mean()), 6),
        "validation_positive_fraction": round(float(y_valid.mean()), 6),
        "test_positive_fraction": round(float(y_test.mean()), 6),
        "threshold_policy": (
            "Maximize validation F2 at precision >= 0.45, grid 0.05..0.80 step 0.025; "
            "fallback to max F2 if infeasible"
        ),
        "model_selection": "Highest validation average precision (Brier tie-break)",
        "selected_model": chosen,
        "selected_threshold": selected_thresholds[chosen],
        "validation": validation_results,
        "test": test_results,
        "constant_prevalence_test": {
            "ranking": evaluate_binary(y_test, prevalence),
            "fixed_half": threshold_metrics(y_test, prevalence, 0.5),
        },
        "test_never_used_for_tuning": True,
        "only_explicit_archived_updates": True,
        "generalization_warning": "One-month snapshot labels; not physical arrivals",
        "data_attribution": "Deutsche Bahn CC BY 4.0; archive by Piet Brömmel",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "ml_improved_metrics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "model", "chosen", "threshold", "roc_auc", "average_precision",
        "brier", "precision", "recall", "f1", "f2", "tp", "fp", "fn", "tn",
    ]
    with (reports_root / "ml_improved_comparison.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for name, measurement in test_results.items():
            vals = measurement["validation_selected_threshold"]
            writer.writerow({
                "model": name, "chosen": name == chosen,
                **{k: measurement["ranking"][k]
                   for k in ("roc_auc", "average_precision", "brier")},
                **{k: vals[k] for k in columns if k in vals},
            })
    return output


def run_improved(
    month: str, processed_file: Path, change_file: Path,
    reports_root: Path = Path("reports"),
) -> dict:
    rows = extract_training_rows(month, processed_file, change_file)
    return fit_improved(rows, month, reports_root)
