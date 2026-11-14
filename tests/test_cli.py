import pytest
from railpulse.cli import main

def test_cli_rejects_bad_month():
    with pytest.raises(ValueError):
        main(["download", "--month", "2026-00"])
