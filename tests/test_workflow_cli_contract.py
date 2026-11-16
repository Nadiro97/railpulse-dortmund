"""CI contract: every workflow command must be supported by the installed CLI.

Catches regressions like starting 'ml-multimonth' before CLI registration.
No external datasets or networks are used.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CLI_PATTERN = re.compile(r"(?<![\w.])python(?:3)?\s+-m\s+railpulse\s+([a-z][a-z0-9-]*)")
REQUIRED_WORKFLOW_COMMANDS = {
    "run", "download", "extract", "validate-updates", "ml-baseline",
    "ml-improve", "ml-cross-month", "ml-multimonth",
}


def workflow_paths():
    return sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])


def workflow_document(path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_all_workflows_parse_and_have_jobs():
    paths = workflow_paths()
    assert paths, "No GitHub Actions workflows found"
    for path in paths:
        workflow = workflow_document(path)
        assert isinstance(workflow, dict), path.name
        assert isinstance(workflow.get("on"), dict), f"{path.name}: missing triggers"
        assert isinstance(workflow.get("jobs"), dict) and workflow["jobs"], path.name
        for name, job in workflow["jobs"].items():
            assert isinstance(job.get("steps"), list) and job["steps"], (path.name, name)


def test_tests_run_on_every_push_and_heavy_runs_are_manual():
    paths = {p.name: p for p in workflow_paths()}
    assert "tests.yml" in paths
    ci = workflow_document(paths["tests.yml"])["on"]
    assert "push" in ci and "pull_request" in ci
    for name, path in paths.items():
        if name == "tests.yml":
            continue
        triggers = workflow_document(path)["on"]
        assert "workflow_dispatch" in triggers, f"{name} is not manually runnable"
        assert "push" not in triggers, f"{name} unexpectedly downloads data on every push"


def test_all_workflow_cli_commands_are_real():
    commands = set()
    for path in workflow_paths():
        workflow = workflow_document(path)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                commands.update(CLI_PATTERN.findall(step.get("run", "")))
    assert REQUIRED_WORKFLOW_COMMANDS <= commands, (
        f"Expected workflow CLI commands missing: {sorted(REQUIRED_WORKFLOW_COMMANDS-commands)}"
    )
    for command in sorted(commands):
        execution = subprocess.run(
            [sys.executable, "-m", "railpulse", command, "--help"],
            cwd=ROOT, text=True, capture_output=True, timeout=15, check=False
        )
        assert execution.returncode == 0, (
            f"Workflow invokes unsupported CLI subcommand {command}: "
            f"{execution.stderr or execution.stdout}"
        )
        assert "usage:" in execution.stdout.lower()


def test_tests_workflow_enforces_cli_contract():
    ci = workflow_document(WORKFLOWS / "tests.yml")
    run_steps = [
        step.get("run", "")
        for job in ci["jobs"].values()
        for step in job.get("steps", [])
    ]
    assert any("pytest -q" in line for line in run_steps)
    assert any("compileall" in line for line in run_steps)
    assert any("pip check" in line for line in run_steps)
