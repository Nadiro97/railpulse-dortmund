"""Discover and audit the historical archive without downloading national Parquet.

Split by COMPLETE calendar months, not random stops. The final chronological
fifth is an already-researched backtest, NOT a previously untouched blind test.
"""
from __future__ import annotations

import calendar
import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from .config import DATASET_REPO, checked_month

FIRST_MONTH = "2024-07"
FOLDERS = {
    "processed": "monthly_processed_data",
    "plan": "monthly_processed_data_plan",
    "change": "monthly_processed_data_change",
}
FILE_PATTERN = re.compile(r"^data-(20\d{2}-(?:0[1-9]|1[0-2]))\.parquet$")
PROFILES = ("arrival", "departure")


def month_range(start: str, end: str):
    """Inclusive first-of-month sequence, validated and increasing."""
    start, end = checked_month(start), checked_month(end)
    a, b = map(int, start.split("-"))
    y, m = a, b
    limit = tuple(map(int, end.split("-")))
    if (y, m) > limit:
        raise ValueError("start month must not be later than end month")
    while (y, m) <= limit:
        yield f"{y:04d}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def last_complete_month(today: date):
    """The current calendar month is incomplete, even if some data appeared."""
    if not isinstance(today, date):
        raise TypeError("today must be a date")
    year, month = today.year, today.month
    return f"{year-1:04d}-12" if month == 1 else f"{year:04d}-{month-1:02d}"


def parse_archive_paths(paths, *, start=FIRST_MONTH, today=None, through=None):
    """Monthly manifest from published file PATHS, not from inferred month names."""
    if today is None:
        today = date.today()
    max_month = last_complete_month(today)
    end = min(checked_month(through), max_month) if through else max_month
    if end < start:
        raise ValueError("No complete months since requested start")
    found = {key: set() for key in FOLDERS}
    prefixes = {folder + "/": key for key, folder in FOLDERS.items()}
    for raw in paths:
        if not isinstance(raw, str):
            continue
        for prefix, key in prefixes.items():
            if not raw.startswith(prefix):
                continue
            m = FILE_PATTERN.fullmatch(raw[len(prefix):])
            if m and start <= m.group(1) <= end:
                found[key].add(m.group(1))
            break
    rows = []
    for month in month_range(start, end):
        entry = {"month": month, **{f"has_{name}": month in found[name] for name in FOLDERS}}
        entry["complete_source_triple"] = all(entry[f"has_{name}"] for name in FOLDERS)
        entry["missing_sources"] = ",".join(
            name for name in FOLDERS if not entry[f"has_{name}"]
        )
        rows.append(entry)
    return rows


def list_source_paths(api=None):
    """Metadata-only Hugging Face lookup; avoids downloading huge Parquet files."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    paths = []
    for folder in FOLDERS.values():
        # Only inspect each known monthly top-level folder; there may be many
        # other (hourly, zipped) datasets elsewhere in the repository.
        for node in api.list_repo_tree(
            repo_id=DATASET_REPO, repo_type="dataset",
            path_in_repo=folder, recursive=False,
        ):
            path = getattr(node, "path", None)
            if isinstance(path, str):
                paths.append(path)
    return paths


def split_complete_months(months, *, test_fraction=0.20, validation_min=2):
    """Chronological 80% development / 20% newer testing by MONTH count.

    Within the development period, the latest max(2,~10% of development)
    months provide validation. The first months provide fit data. The newest
    test months are NOT used for fitting or picking a candidate.
    """
    import math

    months = list(months)
    if months != sorted(set(months)) or len(months) < 10:
        raise ValueError("Require at least ten unique ascending available months")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    count = len(months)
    test_n = math.ceil(count * test_fraction)
    development_n = count - test_n
    validation_n = max(validation_min, round(development_n * 0.10))
    if development_n - validation_n < 4:
        raise ValueError("Insufficient months before the validation and test periods")
    train = months[:development_n-validation_n]
    validate = months[development_n-validation_n:development_n]
    test = months[development_n:]
    if not (max(train) < min(validate) < max(validate) < min(test)):
        raise ValueError("Data split must have distinct chronological months")
    return {"train": train, "validation": validate, "test": test,
            "count": count, "test_fraction_by_month": test_n/count,
            "test_is_untouched_blind_holdout": False,
            "note": "Historical months have already been explored; test is retrospective, not untouched blind evaluation."}


def manifest_from_paths(paths, *, start=FIRST_MONTH, today=None, through=None):
    rows = parse_archive_paths(paths, start=start, today=today, through=through)
    months = [r["month"] for r in rows if r["complete_source_triple"]]
    split = split_complete_months(months) if len(months) >= 10 else None
    by_month = {}
    if split:
        for part in ("train", "validation", "test"):
            by_month.update({month: part for month in split[part]})
    for r in rows:
        r["split"] = by_month.get(r["month"], "unavailable")
    return {
        "schema": "railpulse-history-month-inventory-v1",
        "source": DATASET_REPO,
        "first_requested_month": start,
        "last_complete_month": rows[-1]["month"],
        "months_catalogued": len(rows),
        "months_with_all_sources": len(months),
        "rows": rows,
        "split": split,
        "is_verified_physical_arrival_data": False,
        "is_final_blind_test": False,
        "source_coverage_caveat": "Older collection covered fewer stations and months may have missing hours; each month needs data-quality inspection.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(report, directory=Path("reports")):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"history_month_inventory.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    fields = [
        "month", "has_processed", "has_plan", "has_change",
        "complete_source_triple", "missing_sources", "split",
    ]
    with (folder/"history_month_inventory.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(report["rows"])
    return folder/"history_month_inventory.json"


def inventory(*, start=FIRST_MONTH, through=None, today=None,
              directory=Path("reports"), api=None):
    paths = list_source_paths(api=api)
    info = manifest_from_paths(paths, start=start, through=through, today=today)
    write_manifest(info, directory)
    return info
