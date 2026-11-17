"""Train and serve a compact JSON-only logistic timetable-update demo.

Only the historical September 1-16 learning partition is used for fitting.
No pickle, no arbitrary object deserialization, no train-stop rows saved.
Scores reflect historical *updated timetables*, not actual physical arrivals.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

from .ml_baseline import CATEGORICAL, NUMERIC, extract_training_rows
from .ml_improved import make_models
from .ml_cross_month import ALERT_THRESHOLD

MODEL_SCHEMA = "railpulse-logistic-demo-v1"
MODEL_PATH = Path("models/railpulse_demo_2026-09.json")
EXPECTED_MONTH = "2026-09"
FEATURES = CATEGORICAL + NUMERIC


def fit_demo(frame, output_path: Path = MODEL_PATH, month: str = EXPECTED_MONTH) -> dict:
    """Reproduce the chosen experimental baseline with strictly earlier rows."""
    if month != EXPECTED_MONTH:
        raise ValueError("The approved demo model must train on September 2026 only.")
    if not set(FEATURES + ["planned_date", "target_6plus"]).issubset(frame.columns):
        raise ValueError("Missing training columns.")
    dates = frame["planned_date"].map(
        lambda d: d.date() if hasattr(d, "date") else
        (date.fromisoformat(d) if isinstance(d, str) else d)
    )
    train = frame.loc[
        (dates >= date(2026, 9, 1)) & (dates <= date(2026, 9, 16))
    ].copy()
    if len(train) < 100 or train["target_6plus"].nunique() != 2:
        raise ValueError("Training requires >=100 records and both outcome classes.")

    model = make_models()["logistic_regression"]
    model.fit(train[FEATURES], train["target_6plus"].astype(int))
    encoded = model.named_steps["encode"]
    classifier = model.named_steps["model"]
    onehot = encoded.named_transformers_["categories"]
    cats = [[str(v) for v in group] for group in onehot.categories_]
    coefficients = [float(v) for v in classifier.coef_[0]]
    expected_count = sum(len(c) for c in cats) + len(NUMERIC)
    if len(coefficients) != expected_count:
        raise ValueError("Unexpected logistic coefficient shape.")
    if any(not math.isfinite(x) for x in coefficients + [float(classifier.intercept_[0])]):
        raise ValueError("Non-finite model coefficients.")
    result = {
        "schema": MODEL_SCHEMA,
        "month": EXPECTED_MONTH,
        "fit_dates": ["2026-09-01", "2026-09-16"],
        "n_train": int(len(train)),
        "intercept": float(classifier.intercept_[0]),
        "coefficient_vector": coefficients,
        "categories": dict(zip(CATEGORICAL, cats)),
        "numerical_fields": NUMERIC,
        "threshold": ALERT_THRESHOLD,
        "features": FEATURES,
        "label": "latest explicitly recorded timetable update >= 6 min after schedule",
        "is_live_prediction": False,
        "is_verified_actual_delay": False,
        "source_attribution": "Deutsche Bahn CC BY 4.0; archive by Piet Brömmel",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def validate_artifact(artifact: dict) -> dict:
    """Reject corrupt, out-of-scope, or non-finite local artifacts."""
    if not isinstance(artifact, dict) or artifact.get("schema") != MODEL_SCHEMA:
        raise ValueError("Unsupported demo artifact schema.")
    if artifact.get("month") != EXPECTED_MONTH or artifact.get("fit_dates") != [
        "2026-09-01", "2026-09-16"
    ]:
        raise ValueError("Model training dates do not match the reviewed experiment.")
    if artifact.get("threshold") != ALERT_THRESHOLD:
        raise ValueError("Unexpected decision threshold.")
    if artifact.get("is_verified_actual_delay") is not False or artifact.get("is_live_prediction") is not False:
        raise ValueError("Misleading model metadata.")
    if artifact.get("features") != FEATURES or artifact.get("numerical_fields") != NUMERIC:
        raise ValueError("Unexpected features.")
    categories = artifact.get("categories")
    if not isinstance(categories, dict) or set(categories) != set(CATEGORICAL):
        raise ValueError("Missing feature categories.")
    for key in CATEGORICAL:
        values = categories[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)) or not all(
            isinstance(v, str) and v for v in values
        ):
            raise ValueError(f"Invalid {key} categories.")
    expected = sum(len(categories[k]) for k in CATEGORICAL) + len(NUMERIC)
    coefficients = artifact.get("coefficient_vector")
    if not isinstance(coefficients, list) or len(coefficients) != expected:
        raise ValueError("Model coefficient vector length mismatch.")
    if not isinstance(artifact.get("n_train"), int) or artifact["n_train"] < 100:
        raise ValueError("Invalid training sample count.")
    values = coefficients + [artifact.get("intercept")]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
               for x in values):
        raise ValueError("Model coefficients must be finite numbers.")
    return artifact


def load_artifact(path: Path = MODEL_PATH) -> dict:
    """Read only JSON with a bounded file size, not executable model objects."""
    path = Path(path)
    if path.stat().st_size > 512_000:
        raise ValueError("Model artifact is unexpectedly large.")
    return validate_artifact(json.loads(path.read_text(encoding="utf-8")))


def predict_probability(artifact: dict, station_name: str, train_type: str,
                        scheduled_hour: int, weekday: int) -> float:
    """Return a model score for explicit historical timetable-update risk."""
    validate_artifact(artifact)
    categories = artifact["categories"]
    for feature, value in (("station_name", station_name), ("train_type", train_type)):
        if value not in categories[feature]:
            raise ValueError(f"Unknown {feature}: {value!r}")
    if isinstance(scheduled_hour, bool) or not isinstance(scheduled_hour, int) or not 0 <= scheduled_hour <= 23:
        raise ValueError("Scheduled hour must be an integer in 0..23.")
    if isinstance(weekday, bool) or not isinstance(weekday, int) or not 1 <= weekday <= 7:
        raise ValueError("Weekday must be an integer in 1..7.")
    weights = artifact["coefficient_vector"]
    idx = 0
    score = float(artifact["intercept"])
    for key, value in (("station_name", station_name), ("train_type", train_type)):
        score += weights[idx + categories[key].index(value)]
        idx += len(categories[key])
    score += weights[idx] * scheduled_hour + weights[idx + 1] * weekday
    # Stable logistic transform with extreme values clamped to avoid overflow.
    if score >= 0:
        probability = 1.0 / (1.0 + math.exp(-score))
    else:
        v = math.exp(score)
        probability = v / (1.0 + v)
    return float(probability)


def train_local_model(data_root: Path = Path("data"), output_path: Path = MODEL_PATH) -> dict:
    """Run on a local archive only; never commit trained model files."""
    from .config import source_filename
    from .download import download_month
    from .pipeline import ingest_dortmund
    from .observed_updates import change_filename, download_change_month
    root = Path(data_root)
    processed = root / "processed" / "dortmund_2026-09.parquet"
    source = root / "raw" / source_filename(EXPECTED_MONTH)
    if not processed.is_file():
        if not source.is_file():
            source = download_month(EXPECTED_MONTH, root)
        ingest_dortmund(source, root)
    changes = root / "raw" / change_filename(EXPECTED_MONTH)
    if not changes.is_file():
        changes = download_change_month(EXPECTED_MONTH, root)
    rows = extract_training_rows(EXPECTED_MONTH, processed, changes)
    return fit_demo(rows, output_path, EXPECTED_MONTH)
