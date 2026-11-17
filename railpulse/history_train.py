"""Chronological, rail-only multi-month backtest, not a live delay predictor.

Inventory first, download eligible monthly source triples, inspect snapshot
coverage, cache private feature rows, then split usable *whole months* 80/20.
The most recent 20% is a retrospective test (previous 2026 months inspected).
No reported result is an independently verified physical arrival/departure.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .event_data import MODES, LABEL_SCHEMA_VERSION, extract_labels, get_month_sources
from .evidence_model import (
    MODEL_ROOT, REPORT_ROOT, UNKNOWN, _pipeline, _pred, build_features,
    cohort_errors, metrics,
)
from .history_inventory import FIRST_MONTH, inventory, split_complete_months

MIN_MONTH_ROWS = 100
MIN_LABEL_COVERAGE = 0.50
# Separate diagnostic rather than silently broadening the training population.
MIN_TRAIN_ROWS = 1000
MIN_ROUTE_COUNT = 20
CANDIDATES = ("always_punctual", "training_mean", "simple_ridge",
              "route_ridge", "route_boost")


def cache_path(month, mode, root=Path("data")):
    if mode not in MODES:
        raise ValueError("Unknown Dortmund event")
    return Path(root) / "processed" / "history_rail" / f"{month}_{mode}.parquet"


def coverage_path(month, mode, root=Path("data")):
    if mode not in MODES:
        raise ValueError("Unknown Dortmund event")
    return Path(root) / "processed" / "history_rail" / f"{month}_{mode}_coverage.json"


def history_report_path(mode, root=REPORT_ROOT):
    if mode not in MODES:
        raise ValueError("Unknown Dortmund event")
    return Path(root) / f"history_rail_{mode}_evaluation.json"


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def extract_cached_month(month, mode, *, root=Path("data"), refresh=False):
    """Never cache raw national rows under public reports or tracked paths."""
    import pandas as pd

    path = cache_path(month, mode, root)
    coverage_file = coverage_path(month, mode, root)
    if path.is_file() and coverage_file.is_file() and not refresh:
        recorded = json.loads(coverage_file.read_text(encoding="utf-8"))
        if recorded.get("label_schema_version") == LABEL_SCHEMA_VERSION:
            return pd.read_parquet(path), recorded
        # Old cached results were built with a post-event-only label rule
        # and cannot be reused. Raw source archives stay cached locally.

    sources = get_month_sources(month, root)
    data, coverage = extract_labels(month, mode, sources, rail_only=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)
    _atomic_json(coverage_file, coverage)
    return data, coverage


def _usable_month(coverage, frame):
    """Mandatory quality floor, documented and applied BEFORE splitting."""
    if len(frame) < MIN_MONTH_ROWS:
        return False, "too_few_explicit_archival_updates"
    if coverage.get("label_coverage_fraction", 0.0) < MIN_LABEL_COVERAGE:
        return False, "insufficient_archived_update_coverage"
    if not {"observed_update_minutes", "planned_date"} <= set(frame.columns):
        return False, "missing_essential_columns"
    if frame["observed_update_minutes"].isna().any():
        return False, "null_target_in_labelled_rows"
    if (frame["observed_update_minutes"] < 0).any() or (frame["observed_update_minutes"] > 120).any():
        return False, "out_of_range_label"
    return True, None


def _get_frame(item):
    import pandas as pd
    return pd.read_parquet(item) if isinstance(item, (str, Path)) else item


def _month_rows(cached, months):
    import pandas as pd

    return pd.concat([_get_frame(cached[m]) for m in months], ignore_index=True)


def _constant_baselines(training):
    y = training["observed_update_minutes"]
    return {"always_punctual": None, "training_mean": float(y.mean())}


def _models(training):
    fitted = _constant_baselines(training)
    for kind in ("simple_ridge", "route_ridge", "route_boost"):
        model = _pipeline(kind)
        model.fit(training, training["observed_update_minutes"])
        fitted[kind] = model
    return fitted


def evaluate_history(cached, coverage, mode, split, *,
                     model_dir=MODEL_ROOT, reports_dir=REPORT_ROOT):
    """Fit on older months, select on development validation, test newer months.

    Retain aggregated evaluations only. Test months can never change the
    selected algorithm; they are already researched, not an untouched blind set.
    """
    import joblib

    if mode not in MODES:
        raise ValueError("Invalid Dortmund event")
    chosen_months = split["train"] + split["validation"] + split["test"]
    if any(m not in cached or m not in coverage for m in chosen_months):
        raise ValueError("Missing required month frames or coverage")
    training = build_features(_month_rows(cached, split["train"]))
    validation = build_features(_month_rows(cached, split["validation"]))
    if len(training) < MIN_TRAIN_ROWS or len(validation) < MIN_MONTH_ROWS:
        raise ValueError("Insufficient training/validation observations")
    fitted = _models(training)
    validation_score = {
        name: metrics(validation["observed_update_minutes"], _pred(estimator, validation))
        for name, estimator in fitted.items()
    }
    selected = min(
        fitted, key=lambda name: (
            validation_score[name]["weighted_mae_minutes"],
            validation_score[name]["mae_minutes"],
            list(fitted).index(name),
        )
    )
    # Never refit on validation data to preserve the exact evaluation timeline.
    baseline_selection = validation_score["always_punctual"]["weighted_mae_minutes"]
    better_than_dummy = (
        selected != "always_punctual"
        and validation_score[selected]["weighted_mae_minutes"] <
            0.97 * baseline_selection
    )
    test_results = {}
    overall_score_sums = []
    for month in split["test"]:
        frame = build_features(_get_frame(cached[month]))
        model_scores = {
            name: metrics(frame["observed_update_minutes"], _pred(model, frame))
            for name, model in fitted.items()
        }
        selection_pred = _pred(fitted[selected], frame)
        test_results[month] = {
            "n": len(frame),
            "models": model_scores,
            "station_type_line_cohorts": cohort_errors(frame, selection_pred, minimum=100),
            "label_coverage_fraction": coverage[month]["label_coverage_fraction"],
        }
        overall_score_sums.append((frame["observed_update_minutes"].to_numpy(), selection_pred))
    import numpy as np
    actual = np.concatenate([pair[0] for pair in overall_score_sums])
    predicted = np.concatenate([pair[1] for pair in overall_score_sums])
    overall = metrics(actual, predicted)
    n_train = len(training)
    n_validation = len(validation)
    # Only include clearly supported station/line/route combinations; no
    # inference that the partial path is the entire passenger journey.
    service_cols = ["station_name", "train_type", "line_number",
                    "route_origin", "route_destination"]
    counts = training.groupby(service_cols, dropna=False).size()
    supported = int((counts >= MIN_ROUTE_COUNT).sum())
    report = {
        "schema": "railpulse-rail-history-backtest-v1",
        "mode": mode,
        "scope": "rail-only Dortmund scheduled station events",
        "source": "piebro/deutsche-bahn-data, DB CC BY 4.0",
        "target": "post-event explicit archive timetable changed-time minus scheduled-time, positive minutes <=120",
        "target_is_verified_physical_arrival": False,
        "split": split,
        "selected_model": selected,
        "selection_metric": "validation weighted MAE minutes; ordinary MAE tiebreak",
        "selected_beats_zero_baseline_by_at_least_3pct": bool(better_than_dummy),
        "validation_models": validation_score,
        "test_monthly": test_results,
        "test_overall": overall,
        "training_observations": n_train,
        "validation_observations": n_validation,
        "test_observations": len(actual),
        "supported_route_combinations": supported,
        "monthly_coverage": {month: coverage[month] for month in chosen_months},
        "is_true_blind_test": False,
        "is_approved_real_world_delay_prediction": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "The model is for comparative research only. Do not show Pünktlich or X minutes as confirmed actual train delay without independently verified arrivals/departures.",
    }
    path = history_report_path(mode, reports_dir)
    _atomic_json(path, report)
    # Do not override the existing evidence-model artifact or dashboard guard.
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "schema": report["schema"],
        "mode": mode,
        "model_name": selected,
        "estimator": fitted[selected],
        "train_months": list(split["train"]),
        "validation_months": list(split["validation"]),
        "not_verified_physical_delay": True,
        "not_approved_for_live_forecast": True,
    }, model_root / f"history_rail_{mode}_research.joblib")
    return report


def run_history(*, start=FIRST_MONTH, through=None, today=None,
                data_root=Path("data"), reports_dir=REPORT_ROOT,
                model_dir=MODEL_ROOT, mode="both", refresh=False,
                inventory_only=False, api=None):
    """Full audit and optional heavy multi-month training, no silent fallbacks.

    An archived source month must contain all 3 published files.
    The split is computed only from usable quality-checked months.
    """
    if mode not in (*MODES, "both"):
        raise ValueError("Mode must be arrival, departure or both")
    report = inventory(start=start, through=through, today=today,
                       directory=reports_dir, api=api)
    if inventory_only:
        return {"inventory": report, "training_performed": False}
    months = [entry["month"] for entry in report["rows"]
              if entry["complete_source_triple"]]
    if len(months) < 10:
        raise ValueError("Fewer than ten complete source months available. Check the inventory report.")
    selected_modes = MODES if mode == "both" else (mode,)
    result = {"inventory": report, "training_performed": False, "modes": {}}
    for event in selected_modes:
        cached, coverage, exclusions = {}, {}, {}
        for month in months:
            try:
                frame, qual = extract_cached_month(
                    month, event, root=data_root, refresh=refresh)
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                # Explicitly expose incomplete months. Never infer that their
                # unobserved trains were on time.
                exclusions[month] = "download_or_extract_error: " + str(exc)[:250]
                continue
            ok, reason = _usable_month(qual, frame)
            # Retain non-sensitive aggregate diagnostics, even for rejected months.
            coverage[month] = qual
            if ok:
                cached[month] = cache_path(month, event, data_root)
            else:
                exclusions[month] = reason
        usable_months = sorted(cached)
        if len(usable_months) < 10:
            summary = {
                "mode": event, "usable_months": usable_months,
                "excluded_months": exclusions,
                "monthly_quality": coverage,
                "minimum_label_coverage_fraction": MIN_LABEL_COVERAGE,
                "label_schema_version": LABEL_SCHEMA_VERSION,
                "error": "At least ten audited, labelled rail months are required.",
            }
            result["modes"][event] = summary
            _atomic_json(Path(reports_dir)/f"history_rail_{event}_quality.json", summary)
            continue
        split = split_complete_months(usable_months)
        # Months failing data quality remain excluded, not repurposed as punctual.
        selected = evaluate_history(
            cached, coverage, event, split,
            model_dir=model_dir, reports_dir=reports_dir)
        selected["excluded_months"] = exclusions
        selected["usable_months"] = usable_months
        _atomic_json(history_report_path(event, reports_dir), selected)
        result["modes"][event] = {
            "selected_model": selected["selected_model"],
            "training_observations": selected["training_observations"],
            "validation_observations": selected["validation_observations"],
            "test_observations": selected["test_observations"],
            "test_mae_minutes": selected["test_overall"]["mae_minutes"],
            "months_in_test": split["test"],
            "excluded_months": exclusions,
        }
        result["training_performed"] = True
    return result
