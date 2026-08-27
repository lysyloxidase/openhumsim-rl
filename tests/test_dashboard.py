from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

import examples.dashboard_server as dashboard_server
from examples.dashboard_server import (
    DASHBOARD_HTML,
    DashboardSession,
    RevisionConflictError,
    dashboard_meta,
)


def _stage_dashboard_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    release: dict[str, object] | None = None,
    validation: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    source_release = json.loads(
        dashboard_server.RELEASE_MANIFEST.read_text(encoding="utf-8")
    )
    source_validation = json.loads(
        dashboard_server.VALIDATION_RESULTS.read_text(encoding="utf-8")
    )
    release = source_release if release is None else release
    validation = source_validation if validation is None else validation

    validation_path = tmp_path / "validation" / "results.json"
    validation_path.parent.mkdir()
    validation_bytes = (
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    validation_path.write_bytes(validation_bytes)

    gate = release["focused_integrity_gate"]
    assert isinstance(gate, dict)
    gate["results_path"] = "validation/results.json"
    gate["results_sha256"] = sha256(validation_bytes).hexdigest()
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard_server, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_server, "RELEASE_MANIFEST", release_path)
    monkeypatch.setattr(dashboard_server, "VALIDATION_RESULTS", validation_path)
    return release_path, validation_path


def test_dashboard_is_self_contained_and_has_research_boundary() -> None:
    html = DASHBOARD_HTML.read_text(encoding="utf-8")

    assert '<html lang="pl">' in html
    assert 'name="viewport"' in html
    assert "RESEARCH SOFTWARE ONLY" in html
    assert "Latent model state · system balances" in html
    assert "SYNTHETIC<br>LATENT MODEL STATE" not in html
    assert "/api/step" in html
    assert "/api/history" in html
    assert "X-OpenHumSim-Session" in html
    assert "openhumsim.dashboard.export.v2" in html
    assert "openhumsim.dashboard.recovered-export.v1" in html
    assert "openhumsim.experiment-manifest.v1" in html
    assert "experiment_manifest" in html
    assert "recoveredExportButton" in html
    assert "csvExportButton" in html
    assert "measurementInspectorTable" in html
    assert "runSummaryStrip" in html
    assert "observationSummaryTable" in html
    assert "actionSummaryTable" in html
    assert "rewardSummaryTable" in html
    assert "manifestSummary" in html
    assert 'ci.python_versions.join("/")' in html
    assert "text/csv" in html
    assert "complete_from_reset:false" in html
    assert 'observation:"offline-preview"' in html
    assert 'offline_preview:true' in html
    assert 'mode:"step"' in html
    assert "Auto-kroki" in html
    assert "aria-valuetext" in html
    assert "measurement_channels?.[spec.key]" in html
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert '<script src=' not in html
    assert '<link rel="stylesheet"' not in html


def test_dashboard_meta_locks_action_and_observation_contracts() -> None:
    meta = dashboard_meta()

    assert meta["schema"] == "openhumsim.dashboard.meta.v2"
    assert meta["model_version"] == "0.23.1"
    assert meta["action_names"] == (
        "insulin",
        "oral_carbs",
        "exercise",
        "saline",
        "oxygen",
        "ventilation_pressure_assist",
        "oral_water",
        "oral_probe_compound",
    )
    assert [item["name"] for item in meta["actions"]] == list(meta["action_names"])
    assert meta["observation_contract"]["clinical_count"] == 54
    assert meta["observation_contract"]["full_count"] == 138
    assert meta["availability"] == {
        "release_manifest": True,
        "validation_results": True,
        "ci_evidence": True,
    }
    assert meta["validation"] == {
        "passed": 11,
        "total": 11,
        "executed_regression_cases": 103,
    }
    assert meta["full_test_suite"]["status"] == "passed"
    assert meta["full_test_suite"]["passed"] == 315
    assert meta["full_test_suite"]["total"] == 315
    assert meta["supported_python_ci"]["status"] == "passed"
    assert meta["supported_python_ci"]["conclusion"] == "success"
    assert meta["supported_python_ci"]["python_versions"] == [
        "3.10",
        "3.12",
        "3.14",
    ]
    assert meta["observation_contract"]["reward_profile"] == (
        "latent_research_v0.23"
    )
    assert meta["observation_contract"]["benchmark_reward_profile"] == (
        "observable_benchmark_v0.23"
    )


def test_dashboard_meta_rejects_validation_with_wrong_sha256(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, validation_path = _stage_dashboard_evidence(monkeypatch, tmp_path)
    validation_path.write_bytes(validation_path.read_bytes() + b" ")

    meta = dashboard_server.dashboard_meta()

    assert meta["availability"]["release_manifest"] is True
    assert meta["availability"]["validation_results"] is False
    assert meta["validation"] is None


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema", "openhumsim.validation-results.v999"),
        ("version", "999.0.0"),
    ),
)
def test_dashboard_meta_rejects_digest_locked_but_incompatible_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    invalid_value: str,
) -> None:
    validation = json.loads(
        dashboard_server.VALIDATION_RESULTS.read_text(encoding="utf-8")
    )
    validation[field] = invalid_value
    _stage_dashboard_evidence(
        monkeypatch,
        tmp_path,
        validation=validation,
    )

    meta = dashboard_server.dashboard_meta()

    assert meta["availability"]["validation_results"] is False
    assert meta["validation"] is None


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema", "openhumsim.release.v999"),
        ("version", "999.0.0"),
    ),
)
def test_dashboard_meta_rejects_incompatible_release_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    invalid_value: str,
) -> None:
    release = json.loads(
        dashboard_server.RELEASE_MANIFEST.read_text(encoding="utf-8")
    )
    release[field] = invalid_value
    _stage_dashboard_evidence(
        monkeypatch,
        tmp_path,
        release=release,
    )

    meta = dashboard_server.dashboard_meta()

    assert meta["availability"]["release_manifest"] is False
    assert meta["availability"]["validation_results"] is False
    assert meta["validation"] is None


def test_dashboard_meta_rejects_internally_inconsistent_pass_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validation = json.loads(
        dashboard_server.VALIDATION_RESULTS.read_text(encoding="utf-8")
    )
    validation["checks"][0]["passed"] = False
    _stage_dashboard_evidence(
        monkeypatch,
        tmp_path,
        validation=validation,
    )

    meta = dashboard_server.dashboard_meta()

    assert meta["availability"]["validation_results"] is False
    assert meta["validation"] is None


def test_dashboard_ci_evidence_requires_the_exact_clean_candidate(
    tmp_path: Path,
) -> None:
    release = json.loads(
        dashboard_server.RELEASE_MANIFEST.read_text(encoding="utf-8")
    )
    release["focused_integrity_gate"]["git_commit"] = "a" * 40
    release["focused_integrity_gate"]["git_worktree_dirty"] = False
    evidence = {
        "schema": dashboard_server.CI_EVIDENCE_SCHEMA,
        "latest_successful_run": {
            "conclusion": "success",
            "commit_sha": "a" * 40,
            "python_versions": ["3.10", "3.12", "3.14"],
            "scientific_validation": "success",
            "package_smoke_test": "success",
            "run_url": "https://example.invalid/run/1",
            "completed_at": "2026-08-25T00:00:00Z",
        },
    }
    evidence_path = tmp_path / "ci.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    verified = dashboard_server._verified_ci_evidence(evidence_path, release)
    assert verified is not None
    assert verified["status"] == "passed"
    assert verified["commit_sha"] == "a" * 40

    release["focused_integrity_gate"]["git_worktree_dirty"] = True
    assert dashboard_server._verified_ci_evidence(evidence_path, release) is None
    release["focused_integrity_gate"]["git_worktree_dirty"] = False
    evidence["latest_successful_run"]["commit_sha"] = "b" * 40
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert dashboard_server._verified_ci_evidence(evidence_path, release) is None


def test_dashboard_ci_evidence_rejects_partial_python_matrix(tmp_path: Path) -> None:
    release = json.loads(
        dashboard_server.RELEASE_MANIFEST.read_text(encoding="utf-8")
    )
    release["focused_integrity_gate"]["git_commit"] = "a" * 40
    release["focused_integrity_gate"]["git_worktree_dirty"] = False
    evidence = {
        "schema": dashboard_server.CI_EVIDENCE_SCHEMA,
        "latest_successful_run": {
            "conclusion": "success",
            "commit_sha": "a" * 40,
            "python_versions": ["3.10", "3.12"],
            "scientific_validation": "success",
            "package_smoke_test": "success",
        },
    }
    evidence_path = tmp_path / "ci.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    assert dashboard_server._verified_ci_evidence(evidence_path, release) is None


@pytest.mark.parametrize(
    "failed_job",
    ["scientific_validation", "package_smoke_test"],
)
def test_dashboard_ci_evidence_rejects_failed_required_job(
    tmp_path: Path,
    failed_job: str,
) -> None:
    release = json.loads(
        dashboard_server.RELEASE_MANIFEST.read_text(encoding="utf-8")
    )
    release["focused_integrity_gate"]["git_commit"] = "a" * 40
    release["focused_integrity_gate"]["git_worktree_dirty"] = False
    evidence = {
        "schema": dashboard_server.CI_EVIDENCE_SCHEMA,
        "latest_successful_run": {
            "conclusion": "success",
            "commit_sha": "a" * 40,
            "python_versions": ["3.10", "3.12", "3.14"],
            "scientific_validation": "success",
            "package_smoke_test": "success",
        },
    }
    evidence["latest_successful_run"][failed_job] = "failure"
    evidence_path = tmp_path / "ci.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    assert dashboard_server._verified_ci_evidence(evidence_path, release) is None


def test_dashboard_session_exposes_measurements_separately_from_debug_truth() -> None:
    session = DashboardSession()
    frame = session.snapshot()

    assert frame["schema"] == "openhumsim.dashboard.v2"
    assert frame["data_profile"] == {
        "observation": "clinical",
        "measurement": "realistic",
        "debug_truth_included": True,
        "clinical_values_are_raw": True,
    }
    assert len(frame["measurements"]) == 54
    assert set(frame["measurement_ages_min"]) == {
        "cgm_measurement_age_min",
        "monitor_measurement_age_min",
        "blood_gas_measurement_age_min",
        "chemistry_measurement_age_min",
        "hemodynamic_measurement_age_min",
    }
    assert "glucose_mg_dl" in frame["vitals"]
    assert "sensor_glucose_mg_dl" in frame["measurements"]
    assert frame["measurements"]["sensor_glucose_mg_dl"] != frame["vitals"]["glucose_mg_dl"]
    assert frame["seed"] == 42
    assert isinstance(frame["run_id"], str)
    assert frame["revision"] == 0
    assert frame["action"] is None
    assert frame["intervention"] is None
    json.dumps(frame, allow_nan=False)


def test_dashboard_step_uses_real_environment_and_validates_actions() -> None:
    session = DashboardSession()
    initial = session.snapshot()
    stepped = session.step(
        {},
        expected_run_id=initial["run_id"],
        expected_revision=initial["revision"],
    )

    assert stepped["time_min"] == pytest.approx(initial["time_min"] + 5.0)
    assert stepped["status"]["needs_reset"] is False
    assert stepped["cumulative_reward"] == pytest.approx(stepped["reward"])
    assert stepped["seed"] == 42
    assert stepped["run_id"] == initial["run_id"]
    assert stepped["revision"] == 1
    assert stepped["action"] == {
        name: 0.0 for name in dashboard_meta()["action_names"]
    }
    assert stepped["intervention"]["fio2"] == pytest.approx(0.21)

    with pytest.raises(ValueError, match="unknown action"):
        session.step({"not_an_action": 1.0})
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        session.step({"exercise": 1.1})
    with pytest.raises(ValueError, match="not booleans"):
        session.step({"exercise": True})
    with pytest.raises(ValueError, match="finite numbers"):
        session.step({"exercise": 10**10000})
    with pytest.raises(ValueError, match="unsupported dashboard scenario"):
        session.reset("not_a_scenario", 42)
    with pytest.raises(RevisionConflictError, match="stale dashboard state"):
        session.step(
            {},
            expected_run_id=initial["run_id"],
            expected_revision=initial["revision"],
        )

    reset = session.reset(
        "baseline",
        77,
        expected_run_id=stepped["run_id"],
        expected_revision=stepped["revision"],
    )
    assert reset["seed"] == 77
    assert reset["action"] is None
    assert reset["run_id"] != stepped["run_id"]
    assert reset["revision"] == 0
    history = session.history_envelope()
    assert history["schema"] == "openhumsim.dashboard.history.v2"
    assert history["complete_from_reset"] is True
    assert history["frames"] == [reset]
    manifest = history["experiment_manifest"]
    assert manifest["schema"] == "openhumsim.experiment-manifest.v1"
    assert manifest["randomness"]["reset_seed"] == 77
    assert manifest["randomness"]["physiology_and_measurement"][
        "independent_streams"
    ] is True
    assert len(manifest["observation_catalog"]) == 54


def test_dashboard_documentation_targets_existing_files() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    dashboard_doc = (root / "docs" / "dashboard.md").read_text(encoding="utf-8")

    assert "PYTHONPATH=src python3 examples/dashboard_server.py" in readme
    assert "docs/dashboard.md" in readme
    assert "experiment-manifest.v1" in dashboard_doc
    assert "tidy CSV" in dashboard_doc
    assert "SeedSequence.spawn(2)" in dashboard_doc
    assert (root / "examples" / "dashboard_server.py").is_file()
    assert (root / "dashboard" / "index.html").is_file()
