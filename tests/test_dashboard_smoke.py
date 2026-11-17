"""Dashboard smoke test: the demo loads without private model or CSV files."""
from pathlib import Path
import pytest


def test_streamlit_dashboard_loads_without_real_data(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # A fresh GitHub checkout has no private model data and no real report CSV.
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"))
    app.run(timeout=20)
    assert not app.exception, [str(e.message) for e in app.exception]
    assert any("Voraussichtliche Verspätung am Dortmunder Halt" in str(x.value) for x in app.subheader)
    assert len(app.info) >= 1, "Expected guidance for a checkout without trained artifacts"


def test_demo_cli_help_exists():
    from railpulse.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["train-demo", "--help"])
    assert exc.value.code == 0
