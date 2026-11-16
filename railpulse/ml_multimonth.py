"""Multi-month research backtest; October 2026 is an ALREADY EXAMINED benchmark.

No claim of new blind holdout: our prior September-trained model was tested on
October before this experiment. November is not yet a complete published month.
The label is an archived timetable update, never verified physical arrival.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import checked_month
from .ml_baseline import CATEGORICAL, NUMERIC, evaluate_binary, extract_training_rows
from .ml_improved import make_models, threshold_metrics
from .ml_cross_month import ALERT_THRESHOLD, MODEL

FEATURES = CATEGORICAL + NUMERIC
TRAIN_MONTHS = ("2026-07", "2026-08")
VALIDATION_MONTH = "2026-09"
BENCHMARK_MONTH = "2026-10"
TARGET_VALIDATION_PRECISION = 0.55
THRESHOLDS = tuple(round(0.10 + i * 0.025, 3) for i in range(33))


def select_reduced_alert_threshold(y, probability, min_precision=TARGET_VALIDATION_PRECISION):
    """Select threshold based ONLY on validation metrics; no test inspection.

    Maximize F1 among thresholds meeting predeclared precision requirement.
    If infeasible, return no selection (rather than silently lower target).
    """
    candidates = [threshold_metrics(y, probability, t) for t in THRESHOLDS]
    eligible = [row for row in candidates
                if row["tp"] > 0 and row["precision"] >= min_precision]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["f1"], row["precision"], -row["fp"]))


def _check_months(train_months, validation_month, benchmark_month):
    """Require strict chronological order and unique months."""
    from datetime import date
    sequence = [*train_months, validation_month, benchmark_month]
    for m in sequence:
        checked_month(m)
    if not train_months or len(set(sequence)) != len(sequence):
        raise ValueError("Training months must be nonempty and all months distinct.")
    dates = [date(*map(int, m.split("-")), 1) for m in sequence]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Months must be strictly increasing: train < validation < benchmark.")


def _validate_frame(frame, label):
    if len(frame) < 100:
        raise ValueError(f"{label}: at least 100 rows required")
    if frame["target_6plus"].nunique() != 2:
        raise ValueError(f"{label}: both target classes required")
    if not set(FEATURES).issubset(frame.columns):
        raise ValueError(f"{label}: missing scheduled features")


def _score(y, probability, threshold):
    ranking = evaluate_binary(y, probability)
    decision = threshold_metrics(y, probability, threshold)
    return {
        "ranking": ranking, "decision": decision,
        "sample_n": int(len(y)),
        "positive_rate": round(float(y.mean()), 6),
        "alert_rate": round((decision["tp"] + decision["fp"]) / len(y), 6),
        "false_alerts_per_1000": round(1000 * decision["fp"] / len(y), 2),
    }


def fit_multi_month(
    frames: dict, reports_root: Path = Path("reports"),
    train_months: tuple[str, ...] = TRAIN_MONTHS,
    validation_month: str = VALIDATION_MONTH,
    benchmark_month: str = BENCHMARK_MONTH,
):
    """Select on September, compare on October only as an inspected benchmark."""
    import numpy as np
    import pandas as pd

    _check_months(train_months, validation_month, benchmark_month)
    for month in (*train_months, validation_month, benchmark_month):
        if month not in frames:
            raise ValueError(f"Missing local {month} observations")
        _validate_frame(frames[month], month)

    training = pd.concat([frames[m] for m in train_months], ignore_index=True)
    validation = frames[validation_month]
    benchmark = frames[benchmark_month]
    _validate_frame(training, "combined training")

    X_train, y_train = training[FEATURES], training["target_6plus"].astype(int)
    X_valid, y_valid = validation[FEATURES], validation["target_6plus"].astype(int)
    X_bench, y_bench = benchmark[FEATURES], benchmark["target_6plus"].astype(int)

    validation_candidates = {}
    fitted = {}
    for model_name, model in make_models().items():
        model.fit(X_train, y_train)
        score = model.predict_proba(X_valid)[:, 1]
        selection = select_reduced_alert_threshold(y_valid, score)
        validation_candidates[model_name] = {
            "ranking": evaluate_binary(y_valid, score),
            "threshold_choice": selection,
            "meets_precision_target": selection is not None,
        }
        fitted[model_name] = model

    # No benchmark label is consulted before this choice is fixed.
    feasible = [name for name, result in validation_candidates.items()
                if result["meets_precision_target"]]
    if feasible:
        chosen = max(feasible, key=lambda name: (
            validation_candidates[name]["ranking"]["average_precision"],
            -validation_candidates[name]["ranking"]["brier"],
        ))
        selected_threshold = validation_candidates[chosen]["threshold_choice"]["threshold"]
    else:
        chosen, selected_threshold = None, None

    # Fixed, previously published reference: logistic regression trained on
    # September days 1-16 at threshold .225 (the previous portfolio setup).
    from datetime import date
    d = validation["planned_date"].map(
        lambda x: x.date() if hasattr(x, "date") else (
            date.fromisoformat(x) if isinstance(x, str) else x
        )
    )
    ref_start = date(*map(int, validation_month.split("-")), 1)
    ref_end = date(ref_start.year, ref_start.month, 17)
    baseline_train = validation.loc[(d >= ref_start) & (d < ref_end)]
    _validate_frame(baseline_train, "old September reference train")
    prior = make_models()[MODEL]
    prior.fit(baseline_train[FEATURES], baseline_train["target_6plus"].astype(int))

    reference = _score(y_bench, prior.predict_proba(X_bench)[:, 1], ALERT_THRESHOLD)
    summary = {
        "training_months": list(train_months),
        "validation_month": validation_month,
        "benchmark_month": benchmark_month,
        "warning": (
            "October was inspected in an earlier experiment, so this comparison "
            "is exploratory, NOT an independent blind test. Use a later untouched "
            "month to decide whether to promote a new model."
        ),
        "reference_model": MODEL,
        "reference_threshold": ALERT_THRESHOLD,
        "reference": reference,
        "validation_precision_target": TARGET_VALIDATION_PRECISION,
        "selection_policy": "Max validation average precision among models with >=55% validation alert precision; then max validation F1 threshold",
        "validation_candidates": validation_candidates,
        "chosen_candidate": chosen,
        "candidate_threshold": selected_threshold,
        "label_is_verified_actual_arrival": False,
        "licensed_data": "Deutsche Bahn CC BY 4.0; historical archive by Piet Brömmel",
        "candidate_promoted_to_production": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "monthly_samples": {
            month: {"n": int(len(frames[month])),
                    "positive_rate": round(float(frames[month]["target_6plus"].mean()), 6)}
            for month in (*train_months, validation_month, benchmark_month)
        },
    }

    # Report both candidates for transparency, but do not use results to
    # change the model or threshold selected on September validation.
    comparisons = []
    for name, model in fitted.items():
        threshold = validation_candidates[name]["threshold_choice"]
        if threshold is None:
            # No threshold achieves the declared September precision constraint.
            comparisons.append({"model": name, "eligible": False})
            continue
        benchmark_scores = _score(y_bench, model.predict_proba(X_bench)[:, 1],
                                  threshold["threshold"])
        comparisons.append({"model": name, "eligible": True,
                            "validation_threshold": threshold["threshold"],
                            "october_exploratory": benchmark_scores})
    summary["candidate_october_diagnostics"] = comparisons

    report_dir = Path(reports_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "ml_multimonth_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    cols = ["model", "status", "threshold", "n", "precision", "recall",
            "f1", "roc_auc", "average_precision", "brier",
            "alerts", "false_alerts", "false_alerts_per_1000"]
    with (report_dir / "ml_multimonth_comparison.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()

        def write_model(name, threshold, scored, status):
            decision, rank = scored["decision"], scored["ranking"]
            writer.writerow({
                "model": name, "status": status, "threshold": threshold,
                "n": scored["sample_n"], "precision": decision["precision"],
                "recall": decision["recall"], "f1": decision["f1"],
                "roc_auc": rank["roc_auc"],
                "average_precision": rank["average_precision"], "brier": rank["brier"],
                "alerts": decision["tp"] + decision["fp"],
                "false_alerts": decision["fp"],
                "false_alerts_per_1000": scored["false_alerts_per_1000"],
            })
        write_model("previous_september_logistic", ALERT_THRESHOLD, reference,
                    "published reference")
        for result in comparisons:
            if result["eligible"]:
                write_model(result["model"], result["validation_threshold"],
                            result["october_exploratory"],
                            "selected on September" if result["model"] == chosen else "unselected diagnostic")
    return summary


def run_multi_month(data_root=Path("data"), reports_root=Path("reports")):
    """Fetch only allowlisted months, retain all national/stop data locally."""
    from .config import source_filename
    from .download import download_month
    from .pipeline import ingest_dortmund
    from .observed_updates import change_filename, download_change_month

    root = Path(data_root)
    frames = {}
    for month in (*TRAIN_MONTHS, VALIDATION_MONTH, BENCHMARK_MONTH):
        source = root / "raw" / source_filename(month)
        processed = root / "processed" / f"dortmund_{month}.parquet"
        if not processed.is_file():
            if not source.is_file():
                source = download_month(month, root)
            ingest_dortmund(source, root)
        changes = root / "raw" / change_filename(month)
        if not changes.is_file():
            changes = download_change_month(month, root)
        frames[month] = extract_training_rows(month, processed, changes)
        del source, processed, changes
    return fit_multi_month(frames, reports_root=reports_root)
