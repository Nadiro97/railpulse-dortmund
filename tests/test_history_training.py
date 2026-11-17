"""Synthetic historical inventory and time-split tests. No real DB records committed."""
from datetime import date, timedelta
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from railpulse.event_data import RAIL_TYPES
from railpulse.history_inventory import (
    FOLDERS, last_complete_month, manifest_from_paths, month_range,
    split_complete_months, inventory,
)
from railpulse.history_train import (
    _usable_month, cache_path, coverage_path, extract_cached_month, evaluate_history, run_history,
)


def fake_paths(end="2026-10", absent=None):
    absent = set(absent or [])
    return [
        f"{folder}/data-{m}.parquet"
        for m in month_range("2024-07", end)
        for kind, folder in FOLDERS.items()
        if (m, kind) not in absent
    ]


class FakeApi:
    def __init__(self, paths):
        self.paths = paths
        self.calls = []

    def list_repo_tree(self, *, repo_id, repo_type, path_in_repo, recursive):
        self.calls.append((repo_id, repo_type, path_in_repo, recursive))
        return [SimpleNamespace(path=p) for p in self.paths
                if p.startswith(path_in_repo + "/")]


def test_last_complete_month_and_month_boundaries():
    assert last_complete_month(date(2026, 11, 17)) == "2026-10"
    assert last_complete_month(date(2027, 1, 1)) == "2026-12"
    assert list(month_range("2026-11", "2027-02")) == [
        "2026-11", "2026-12", "2027-01", "2027-02",
    ]


def test_manifest_only_counts_published_source_triples_and_older_missing(tmp_path):
    fake = FakeApi(fake_paths(absent={("2025-03", "change")}))
    result = inventory(
        today=date(2026, 11, 17), api=fake, directory=tmp_path,
    )
    assert result["months_catalogued"] == 28
    assert result["months_with_all_sources"] == 27
    missing = next(x for x in result["rows"] if x["month"] == "2025-03")
    assert missing["split"] == "unavailable"
    assert missing["missing_sources"] == "change"
    assert result["split"]["count"] == 27
    assert result["split"]["test"] == [
        "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"
    ]
    assert result["split"]["test_is_untouched_blind_holdout"] is False
    assert len(fake.calls) == 3
    assert all(call[3] is False for call in fake.calls)
    assert (tmp_path / "history_month_inventory.csv").is_file()
    assert (tmp_path / "history_month_inventory.json").is_file()


def test_split_does_not_shuffle_months_or_promote_recent_data_to_training():
    months = list(month_range("2024-07", "2026-10"))
    split = split_complete_months(months)
    assert len(split["train"]) == 20
    assert len(split["validation"]) == 2
    assert len(split["test"]) == 6
    assert split["train"][-1] == "2026-02"
    assert split["validation"] == ["2026-03", "2026-04"]
    assert split["test"][0] == "2026-05"
    assert split["test"][-1] == "2026-10"
    with pytest.raises(ValueError):
        split_complete_months(months[::-1])


def test_inventory_only_does_not_download_or_train(tmp_path):
    result = run_history(today=date(2026, 11, 17),
                         data_root=tmp_path / "data",
                         reports_dir=tmp_path / "reports",
                         inventory_only=True,
                         api=FakeApi(fake_paths()))
    assert not result["training_performed"]
    assert result["inventory"]["months_with_all_sources"] == 28
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "models").exists()


def test_rail_categories_includes_local_and_long_distance_but_not_bus():
    for code in ("RE", "RB", "S", "ICE", "IC", "EC", "ERB", "NX"):
        assert code in RAIL_TYPES
    for code in ("Bus", "BUS", "TRAM", ""):
        assert code not in RAIL_TYPES


def test_month_quality_floor_requires_observed_updates_not_imputed_zero():
    frame = pd.DataFrame({
        "observed_update_minutes": [0.0] * 120,
        "planned_date": [date(2026, 1, 1)] * 120,
    })
    assert _usable_month({"label_coverage_fraction": .9}, frame)[0]
    ok, reason = _usable_month({"label_coverage_fraction": .1}, frame)
    assert not ok and "coverage" in reason
    assert reason == "insufficient_archived_update_coverage"
    frame.loc[0, "observed_update_minutes"] = float("nan")
    assert not _usable_month({"label_coverage_fraction": .9}, frame)[0]


def fake_history_frame(month, n=115):
    year, m = map(int, month.split("-"))
    start = date(year, m, 1)
    records = []
    for i in range(n):
        d = start + timedelta(days=i % 25)
        j = i % 4
        delay = float([0., 4., 9., 16.][(i + m) % 4])
        records.append({
            "station_name": "Dortmund Hbf",
            "train_type": ["S", "RE", "ICE", "RB"][j],
            "line_number": ["S1", "RE1", "(unknown)", "RB59"][j],
            "train_number": str(100 + j),
            "route_origin": ["Hamm Hbf", "Bochum Hbf"][j % 2],
            "route_destination": ["Aachen Hbf", "Düsseldorf Hbf"][j % 2],
            "planned_date": d,
            "scheduled_hour": 5 + (i % 18),
            "weekday": d.isoweekday(),
            "route_stop_index": float(j),
            "hops_before": float(j),
            "hops_after": 12.0 - j,
            "observed_update_minutes": delay,
        })
    return pd.DataFrame(records)


def test_retrospective_evaluation_reports_month_error_not_percent(tmp_path):
    pytest.importorskip("sklearn")
    months = list(month_range("2024-07", "2026-10"))
    split = split_complete_months(months)
    frames = {}
    for month in months:
        path = tmp_path / "local" / (month + ".parquet")
        path.parent.mkdir(exist_ok=True)
        fake_history_frame(month).to_parquet(path)
        frames[month] = path
    cover = {m: {"label_coverage_fraction": 0.90} for m in months}
    result = evaluate_history(frames, cover, "arrival", split,
                              model_dir=tmp_path / "models",
                              reports_dir=tmp_path / "reports")
    assert result["training_observations"] == 20 * 115
    assert result["test_observations"] == 6 * 115
    assert result["selected_model"] in ("always_punctual", "training_mean",
                                        "simple_ridge", "route_ridge", "route_boost")
    assert set(result["test_monthly"]) == set(split["test"])
    assert result["test_overall"]["mae_minutes"] >= 0
    assert result["test_overall"]["weighted_mae_minutes"] >= 0
    assert not result["is_true_blind_test"]
    assert not result["is_approved_real_world_delay_prediction"]
    assert not result["target_is_verified_physical_arrival"]
    report_file = tmp_path / "reports" / "history_rail_arrival_evaluation.json"
    contents = report_file.read_text(encoding="utf-8")
    assert '"planned_date"' not in contents
    assert '"train_number"' not in contents
    assert (tmp_path / "models" / "history_rail_arrival_research.joblib").is_file()


def test_cache_uses_private_data_directory_not_public_reports():
    assert cache_path("2025-01", "arrival").as_posix().startswith(
        "data/processed/history_rail/"
    )



def test_old_post_event_cache_is_automatically_reprocessed(monkeypatch, tmp_path):
    """Previously rejected months must not stay rejected after the label repair."""
    from railpulse.event_data import LABEL_SCHEMA_VERSION

    local = tmp_path / "data"
    old_path = cache_path("2026-09", "departure", local)
    metadata = coverage_path("2026-09", "departure", local)
    old_path.parent.mkdir(parents=True)
    fake_history_frame("2026-09").to_parquet(old_path)
    metadata.write_text(
        '{"label_coverage_fraction": 0.001, "month": "2026-09"}',
        encoding="utf-8",
    )
    events = []

    def files(month, root):
        events.append(("fetch", month))
        return {"fake": "local-snapshot-files"}

    def extract(month, mode, sources, *, rail_only=False):
        events.append(("extract", mode, rail_only))
        return fake_history_frame(month), {
            "label_coverage_fraction": .88,
            "post_event_explicit_update_fraction": .01,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "post_event_observation_is_required": False,
        }

    monkeypatch.setattr("railpulse.history_train.get_month_sources", files)
    monkeypatch.setattr("railpulse.history_train.extract_labels", extract)
    frame, metrics = extract_cached_month("2026-09", "departure", root=local)
    assert len(frame) == 115
    assert metrics["label_coverage_fraction"] == .88
    assert events == [("fetch", "2026-09"), ("extract", "departure", True)]
    assert json.loads(metadata.read_text())["label_schema_version"] == LABEL_SCHEMA_VERSION
    again, _ = extract_cached_month("2026-09", "departure", root=local)
    assert len(again) == 115
    assert len(events) == 2, "Updated cache should be reused without re-extraction"


def test_quality_reports_preserve_monthly_diagnostics_for_rejected_months(tmp_path, monkeypatch):
    from railpulse.event_data import LABEL_SCHEMA_VERSION
    import json

    def synthetic_inventory(**kw):
        return {
            "rows": [
                {"month": month, "complete_source_triple": True}
                for month in month_range("2024-07", "2025-04")
            ],
            "months_with_all_sources": 10,
            "months_catalogued": 10,
            "split": {},
        }

    def incomplete(month, event, *, root, refresh):
        return fake_history_frame(month), {
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "label_coverage_fraction": .01,
            "known_any_explicit_change": 42,
            "known_post_event_explicit_change": 0,
            "scheduled_eligible": 1000,
        }

    monkeypatch.setattr("railpulse.history_train.inventory", synthetic_inventory)
    monkeypatch.setattr("railpulse.history_train.extract_cached_month", incomplete)
    result = run_history(
        reports_dir=tmp_path / "reports",
        model_dir=tmp_path / "models", data_root=tmp_path / "data",
        mode="departure",
    )
    assert not result["training_performed"]
    report = json.loads(
        (tmp_path/"reports/history_rail_departure_quality.json").read_text()
    )
    assert report["monthly_quality"]["2024-07"]["known_any_explicit_change"] == 42
    assert report["monthly_quality"]["2024-07"]["label_coverage_fraction"] == .01
    assert report["excluded_months"]["2024-07"] == "insufficient_archived_update_coverage"
