"""One reviewed open dataset source. No arbitrary URLs accepted."""
from pathlib import Path
import re

DATASET_REPO = "piebro/deutsche-bahn-data"
DATASET_PATH_PREFIX = "monthly_processed_data/"
DEFAULT_MONTH = "2026-09"
DEFAULT_ROOT = Path("data")
MONTH_PATTERN = re.compile(r"20\d{2}-(?:0[1-9]|1[0-2])\Z")

def checked_month(month: str) -> str:
    if not MONTH_PATTERN.fullmatch(month):
        raise ValueError("Month must be YYYY-MM (year 2000–2099).")
    return month

def source_filename(month: str) -> str:
    return f"{DATASET_PATH_PREFIX}data-{checked_month(month)}.parquet"
