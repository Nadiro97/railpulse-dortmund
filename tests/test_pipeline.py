from pathlib import Path
import pytest
from railpulse.pipeline import quoted, validate_columns, selected_expressions, REQUIRED

def test_sql_quote():
    assert quoted(Path("/tmp/a'b.parquet")) == "'/tmp/a''b.parquet'"

def test_missing_column_errors():
    with pytest.raises(ValueError, match="arrival_change_time"):
        validate_columns(REQUIRED - {"arrival_change_time"})

def test_optional_is_null_not_zero():
    statements = " ".join(selected_expressions(REQUIRED))
    assert 'CAST(NULL AS DOUBLE) AS "delay_in_min"' in statements
    assert "date_diff('second', arrival_planned_time, arrival_change_time) / 60.0" in statements
    assert "ELSE NULL END AS changed_arrival_delta_min" in statements

def test_required_columns():
    assert validate_columns(REQUIRED) is None
