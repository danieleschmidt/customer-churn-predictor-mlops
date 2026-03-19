"""Tests for the JSON experiment tracker."""
import json
import tempfile
from pathlib import Path
import pytest
from churn.tracker import ExperimentTracker


def _make_tracker(tmp_path):
    return ExperimentTracker(tmp_path / "runs.json")


def test_log_run_creates_file(tmp_path):
    t = _make_tracker(tmp_path)
    t.log_run("MyModel", {"lr": 0.01}, {"roc_auc": 0.85})
    assert (tmp_path / "runs.json").exists()


def test_log_run_stores_data(tmp_path):
    t = _make_tracker(tmp_path)
    run_id = t.log_run("MyModel", {"lr": 0.01}, {"roc_auc": 0.85, "f1": 0.72})
    runs = t.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == run_id
    assert r["model"] == "MyModel"
    assert r["metrics"]["roc_auc"] == pytest.approx(0.85, rel=1e-4)


def test_multiple_runs(tmp_path):
    t = _make_tracker(tmp_path)
    t.log_run("A", {}, {"roc_auc": 0.80})
    t.log_run("B", {}, {"roc_auc": 0.88})
    t.log_run("C", {}, {"roc_auc": 0.75})
    assert len(t.list_runs()) == 3


def test_best_run(tmp_path):
    t = _make_tracker(tmp_path)
    t.log_run("A", {}, {"roc_auc": 0.80})
    t.log_run("B", {}, {"roc_auc": 0.88})
    t.log_run("C", {}, {"roc_auc": 0.75})
    best = t.best_run("roc_auc")
    assert best["model"] == "B"


def test_persistence(tmp_path):
    """Runs should survive creating a new tracker instance."""
    path = tmp_path / "runs.json"
    t1 = ExperimentTracker(path)
    t1.log_run("Model1", {}, {"roc_auc": 0.9})
    t2 = ExperimentTracker(path)
    assert len(t2.list_runs()) == 1


def test_summary_format(tmp_path):
    t = _make_tracker(tmp_path)
    t.log_run("LR", {"C": 1.0}, {"roc_auc": 0.78, "f1": 0.65})
    s = t.summary()
    assert "LR" in s
    assert "0.7800" in s


def test_best_run_no_data(tmp_path):
    t = _make_tracker(tmp_path)
    assert t.best_run("roc_auc") is None
