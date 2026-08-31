from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

import pytest

from validation.verify_release_evidence import (
    ReleaseEvidenceError,
    verify_release_evidence,
)
from validation.release_contract_v0232 import (
    ACTION_COUNT,
    ACTION_SHA256,
    BENCHMARK_REWARD_PROFILE,
    CLINICAL_OBSERVATION_COUNT,
    CLINICAL_OBSERVATION_SHA256,
    DEBUG_REWARD_PROFILE,
    EXPLICIT_SOURCE_RELATIVE_PATHS,
    FULL_OBSERVATION_COUNT,
    FULL_OBSERVATION_SHA256,
    PYTEST_CONTRACTS,
    STATE_SCHEMA_VERSION,
    pytest_source_relative_paths,
    release_source_paths,
)


VERSION = "0.23.2"
CANDIDATE_TIMESTAMP = "2026-08-30T08:00:00+00:00"
CI_TIMESTAMP = "2026-08-30T08:30:00Z"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "--no-gpg-sign", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_records(root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in release_source_paths(root)
    ]


def _source_fingerprint(records: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


@pytest.fixture
def evidence_repo(tmp_path: Path) -> Path:
    root = tmp_path / "release-repository"
    for relative in (
        *EXPLICIT_SOURCE_RELATIVE_PATHS,
        *pytest_source_relative_paths(),
    ):
        if relative == "pyproject.toml":
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"synthetic-openhumsim\"\nversion = \"0.23.2\"\n",
        encoding="utf-8",
    )
    model_path = root / "src" / "openhumsim_rl" / "example_model.py"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        "MODEL_CONSTANT = 23\n",
        encoding="utf-8",
    )
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Release verifier test")
    _git(root, "config", "user.email", "release-verifier@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    candidate_commit = _commit(root, "candidate source")

    records = _source_records(root)
    fingerprint = _source_fingerprint(records)
    checks: list[dict[str, Any]] = [
        {
            "name": "exact_release_version",
            "passed": True,
            "values": {"version": VERSION},
        },
        {
            "name": "state_schema_and_reward_profiles",
            "passed": True,
            "values": {
                "state_schema_version": STATE_SCHEMA_VERSION,
                "debug_reward_profile": DEBUG_REWARD_PROFILE,
                "benchmark_reward_profile": BENCHMARK_REWARD_PROFILE,
            },
        },
    ]
    for name, targets in PYTEST_CONTRACTS:
        values: dict[str, Any] = {
            "test_paths": list(targets),
            "returncode": 0,
            "executed_cases": 1,
            "stdout": "1 passed",
            "stderr": "",
        }
        if len(targets) == 1:
            values["test_path"] = targets[0]
        checks.append({"name": name, "passed": True, "values": values})
    checks.append(
        {
            "name": "source_snapshot_stable_during_gate",
            "passed": True,
            "values": {
                "before_sha256": fingerprint,
                "after_sha256": fingerprint,
            },
        }
    )
    results = {
        "schema": "openhumsim.validation-results.v1",
        "version": VERSION,
        "state_schema_version": "0.22",
        "scope": "synthetic release-verifier fixture",
        "executed_at_utc": CANDIDATE_TIMESTAMP,
        "runtime": {
            "python": "3.12.13",
            "implementation": "CPython",
            "platform": "synthetic",
            "packages": {},
        },
        "source_provenance": {
            "fingerprint_schema": "openhumsim.source-fingerprint.v1",
            "sha256": fingerprint,
            "file_count": len(records),
            "files": records,
            "git_commit": candidate_commit,
            "git_worktree_dirty": False,
        },
        "summary": {
            "passed": len(checks),
            "total": len(checks),
            "executed_regression_cases": len(PYTEST_CONTRACTS),
        },
        "checks": checks,
    }
    results_path = root / "validation" / f"validation_results_v{VERSION}.json"
    results_bytes = _write_json(results_path, results)

    ci_run = {
        "workflow": ".github/workflows/ci.yml",
        "run_id": 123456,
        "run_url": "https://github.com/example/project/actions/runs/123456",
        "commit_sha": candidate_commit,
        "completed_at": CI_TIMESTAMP,
        "conclusion": "success",
        "python_versions": ["3.10", "3.12", "3.14"],
        "scientific_validation": "success",
        "package_smoke_test": "success",
    }
    release = {
        "schema": "openhumsim.release.v1",
        "status": "released",
        "version": VERSION,
        "author": "lysyloxidase",
        "license": "Apache-2.0",
        "state_schema_version": STATE_SCHEMA_VERSION,
        "reward_profile": DEBUG_REWARD_PROFILE,
        "default_debug_reward_profile": DEBUG_REWARD_PROFILE,
        "benchmark_reward_profile": BENCHMARK_REWARD_PROFILE,
        "clinical_observation_count": CLINICAL_OBSERVATION_COUNT,
        "clinical_observation_sha256": CLINICAL_OBSERVATION_SHA256,
        "full_observation_count": FULL_OBSERVATION_COUNT,
        "full_observation_sha256": FULL_OBSERVATION_SHA256,
        "action_count": ACTION_COUNT,
        "action_sha256": ACTION_SHA256,
        "default_observation_profile": "clinical",
        "default_measurement_profile": "realistic",
        "default_info_profile": "debug",
        "strict_benchmark_info_profile": "benchmark",
        "checkpoint_basename": "openhumsim_ppo_v0232_smoke",
        "focused_integrity_gate": {
            "status": "passed",
            "passed": len(checks),
            "total": len(checks),
            "executed_regression_cases": len(PYTEST_CONTRACTS),
            "executed_at_utc": CANDIDATE_TIMESTAMP,
            "host_python": "3.12.13",
            "results_path": f"validation/validation_results_v{VERSION}.json",
            "results_sha256": sha256(results_bytes).hexdigest(),
            "source_fingerprint_sha256": fingerprint,
            "source_file_count": len(records),
            "git_commit": candidate_commit,
            "git_worktree_dirty": False,
        },
        "full_test_suite": {
            "status": "passed",
            "execution": "local",
            "passed": 400,
            "total": 400,
            "warnings_as_errors": True,
            "executed_at_utc": CANDIDATE_TIMESTAMP,
            "duration_seconds": 123.5,
            "host_python": "3.12.13",
            "command": "python -m pytest -q -ra -W error -p no:cacheprovider",
        },
        "package_build": {
            "status": "passed",
            "execution": "local",
            "host_python": "3.12.13",
            "wheel": "openhumsim_rl-0.23.2-py3-none-any.whl",
            "sdist": "openhumsim_rl-0.23.2.tar.gz",
            "twine_check": "passed",
            "isolated_wheel_install": "passed",
            "cli_smoke": "version, doctor and baseline demo passed",
            "checkout_only_validation_guard": "passed",
        },
        "supported_interpreter_ci": {
            "status": "passed",
            "execution": "github-actions",
            **ci_run,
            "codeql_run_id": 123457,
            "codeql_conclusion": "success",
        },
        "historical_rl_checkpoints_compatible": False,
        "full_observation_claimed_markov": False,
        "independent_external_validation_bundled": False,
        "clinical_use_supported": False,
    }
    _write_json(root / f"RELEASE_v{VERSION}.json", release)
    _write_json(
        root / "CI_EVIDENCE.json",
        {
            "schema": "openhumsim.ci-evidence.v1",
            "latest_successful_run": ci_run,
            "records": f"RELEASE_v{VERSION}.json.supported_interpreter_ci",
        },
    )
    _commit(root, "release evidence")
    return root


def _release_path(root: Path) -> Path:
    return root / f"RELEASE_v{VERSION}.json"


def _results_path(root: Path) -> Path:
    return root / "validation" / f"validation_results_v{VERSION}.json"


def _rewrite_and_commit(
    root: Path,
    path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _read_json(path)
    mutate(value)
    _write_json(path, value)
    _commit(root, message)


def _resign_results(root: Path) -> None:
    release = _read_json(_release_path(root))
    release["focused_integrity_gate"]["results_sha256"] = sha256(
        _results_path(root).read_bytes()
    ).hexdigest()
    _write_json(_release_path(root), release)


def test_release_evidence_accepts_one_clean_consistent_candidate(
    evidence_repo: Path,
) -> None:
    report = verify_release_evidence(evidence_repo)

    assert report["schema"] == "openhumsim.release-evidence-verification.v1"
    assert report["status"] == "passed"
    assert report["version"] == VERSION
    assert report["focused_checks"] == len(PYTEST_CONTRACTS) + 3
    assert report["executed_regression_cases"] == len(PYTEST_CONTRACTS)
    assert report["candidate_commit"] == _git(
        evidence_repo,
        "rev-parse",
        "HEAD^",
    )
    assert report["head_commit"] == _git(evidence_repo, "rev-parse", "HEAD")


def test_release_evidence_rejects_changed_results_bytes(
    evidence_repo: Path,
) -> None:
    _rewrite_and_commit(
        evidence_repo,
        _results_path(evidence_repo),
        lambda result: result.__setitem__("scope", "tampered after validation"),
        "tamper result",
    )

    with pytest.raises(ReleaseEvidenceError, match="results SHA-256 mismatch"):
        verify_release_evidence(evidence_repo)


def test_release_evidence_recomputes_each_source_file(
    evidence_repo: Path,
) -> None:
    (evidence_repo / "src" / "openhumsim_rl" / "example_model.py").write_text(
        "MODEL_CONSTANT = 999\n",
        encoding="utf-8",
    )
    _commit(evidence_repo, "change source after candidate")

    with pytest.raises(ReleaseEvidenceError, match="source file hash mismatch"):
        verify_release_evidence(evidence_repo)


def test_release_evidence_rejects_incomplete_source_subject_set(
    evidence_repo: Path,
) -> None:
    results = _read_json(_results_path(evidence_repo))
    records = results["source_provenance"]["files"]
    records.pop()
    fingerprint = _source_fingerprint(records)
    results["source_provenance"]["file_count"] = len(records)
    results["source_provenance"]["sha256"] = fingerprint
    results["checks"][-1]["values"] = {
        "before_sha256": fingerprint,
        "after_sha256": fingerprint,
    }
    _write_json(_results_path(evidence_repo), results)

    release = _read_json(_release_path(evidence_repo))
    release["focused_integrity_gate"]["source_file_count"] = len(records)
    release["focused_integrity_gate"]["source_fingerprint_sha256"] = fingerprint
    release["focused_integrity_gate"]["results_sha256"] = sha256(
        _results_path(evidence_repo).read_bytes()
    ).hexdigest()
    _write_json(_release_path(evidence_repo), release)
    _commit(evidence_repo, "remove fingerprint subject")

    with pytest.raises(ReleaseEvidenceError, match="subject set"):
        verify_release_evidence(evidence_repo)


def test_release_evidence_rejects_non_evidence_changes_after_candidate(
    evidence_repo: Path,
) -> None:
    (evidence_repo / "SECURITY.md").write_text(
        "unexpected post-candidate change\n",
        encoding="utf-8",
    )
    _commit(evidence_repo, "change non-evidence file")

    with pytest.raises(ReleaseEvidenceError, match="non-evidence change"):
        verify_release_evidence(evidence_repo)


def test_release_evidence_rejects_inconsistent_summary_and_checks(
    evidence_repo: Path,
) -> None:
    results = _read_json(_results_path(evidence_repo))
    results["checks"][0]["passed"] = False
    _write_json(_results_path(evidence_repo), results)
    _resign_results(evidence_repo)
    _commit(evidence_repo, "tamper check summary")

    with pytest.raises(ReleaseEvidenceError, match="summary passed does not match"):
        verify_release_evidence(evidence_repo)


def test_release_evidence_rejects_source_and_gate_commit_mismatch(
    evidence_repo: Path,
) -> None:
    results = _read_json(_results_path(evidence_repo))
    results["source_provenance"]["git_commit"] = _git(
        evidence_repo,
        "rev-parse",
        "HEAD",
    )
    _write_json(_results_path(evidence_repo), results)
    _resign_results(evidence_repo)
    _commit(evidence_repo, "tamper source commit")

    with pytest.raises(ReleaseEvidenceError, match="source commit does not match"):
        verify_release_evidence(evidence_repo)


@pytest.mark.parametrize(
    ("mutate_release", "mutate_ci", "message"),
    (
        (
            lambda release: release["supported_interpreter_ci"].__setitem__(
                "python_versions",
                ["3.12"],
            ),
            lambda evidence: evidence["latest_successful_run"].__setitem__(
                "python_versions",
                ["3.12"],
            ),
            "Python versions",
        ),
        (
            lambda release: release["supported_interpreter_ci"].__setitem__(
                "scientific_validation",
                "failure",
            ),
            lambda evidence: evidence["latest_successful_run"].__setitem__(
                "scientific_validation",
                "failure",
            ),
            "scientific-validation job",
        ),
        (
            lambda release: None,
            lambda evidence: evidence["latest_successful_run"].__setitem__(
                "commit_sha",
                "f" * 40,
            ),
            "field commit_sha does not match",
        ),
    ),
)
def test_release_evidence_rejects_wrong_ci_matrix_jobs_and_commit(
    evidence_repo: Path,
    mutate_release: Callable[[dict[str, Any]], None],
    mutate_ci: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    release = _read_json(_release_path(evidence_repo))
    evidence = _read_json(evidence_repo / "CI_EVIDENCE.json")
    mutate_release(release)
    mutate_ci(evidence)
    _write_json(_release_path(evidence_repo), release)
    _write_json(evidence_repo / "CI_EVIDENCE.json", evidence)
    _commit(evidence_repo, "tamper CI evidence")

    with pytest.raises(ReleaseEvidenceError, match=message):
        verify_release_evidence(evidence_repo)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    (
        ("full_test_suite", "status", "full test suite did not pass"),
        ("package_build", "isolated_wheel_install", "isolated_wheel_install did not pass"),
    ),
)
def test_release_evidence_rejects_failed_full_suite_or_package_build(
    evidence_repo: Path,
    section: str,
    field: str,
    message: str,
) -> None:
    def fail_check(release: dict[str, Any]) -> None:
        release[section][field] = "failed"

    _rewrite_and_commit(
        evidence_repo,
        _release_path(evidence_repo),
        fail_check,
        "tamper release status",
    )

    with pytest.raises(ReleaseEvidenceError, match=message):
        verify_release_evidence(evidence_repo)


def test_release_evidence_rejects_declared_dirty_candidate(
    evidence_repo: Path,
) -> None:
    _rewrite_and_commit(
        evidence_repo,
        _release_path(evidence_repo),
        lambda release: release["focused_integrity_gate"].__setitem__(
            "git_worktree_dirty",
            True,
        ),
        "mark candidate dirty",
    )

    with pytest.raises(ReleaseEvidenceError, match="record a clean candidate"):
        verify_release_evidence(evidence_repo)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("author", "someone-else", "release author"),
        ("license", "Proprietary", "release license"),
    ),
)
def test_release_evidence_locks_author_and_license(
    evidence_repo: Path,
    field: str,
    invalid: str,
    message: str,
) -> None:
    _rewrite_and_commit(
        evidence_repo,
        _release_path(evidence_repo),
        lambda release: release.__setitem__(field, invalid),
        f"tamper release {field}",
    )

    with pytest.raises(ReleaseEvidenceError, match=message):
        verify_release_evidence(evidence_repo)


def test_release_evidence_requires_utc_timestamps(evidence_repo: Path) -> None:
    _rewrite_and_commit(
        evidence_repo,
        _release_path(evidence_repo),
        lambda release: release["full_test_suite"].__setitem__(
            "executed_at_utc",
            "2026-08-30T10:00:00+02:00",
        ),
        "tamper suite timezone",
    )

    with pytest.raises(ReleaseEvidenceError, match="expressed in UTC"):
        verify_release_evidence(evidence_repo)


def test_release_evidence_rejects_current_dirty_worktree(
    evidence_repo: Path,
) -> None:
    (evidence_repo / "untracked-release-note.txt").write_text(
        "not committed\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseEvidenceError, match="current Git worktree is dirty"):
        verify_release_evidence(evidence_repo)


@pytest.mark.parametrize(
    "relative_path",
    (
        f"RELEASE_v{VERSION}.json",
        f"validation/validation_results_v{VERSION}.json",
        "CI_EVIDENCE.json",
    ),
)
def test_release_evidence_fails_closed_when_an_artifact_is_missing(
    evidence_repo: Path,
    relative_path: str,
) -> None:
    (evidence_repo / relative_path).unlink()

    with pytest.raises(ReleaseEvidenceError, match="missing or unreadable"):
        verify_release_evidence(evidence_repo)
