import pytest
from railpulse.config import checked_month, source_filename, DATASET_REPO

def test_approved_source():
    assert DATASET_REPO == "piebro/deutsche-bahn-data"
    assert source_filename("2026-09") == "monthly_processed_data/data-2026-09.parquet"

@pytest.mark.parametrize("bad", ["2026-13", "2026-00", "2026-9", "../../secret", "2026-09/*.parquet", "", "1900-09"])
def test_invalid_months_rejected(bad):
    with pytest.raises(ValueError):
        checked_month(bad)
