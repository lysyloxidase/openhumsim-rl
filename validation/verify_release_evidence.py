"""Fail-closed verification of OpenHumSim release evidence.

The verifier does not create or update evidence.  It checks that the release
manifest, focused validation result, Git source snapshot and CI record all
refer to the same clean candidate commit and that the recorded pass summaries
are internally consistent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
from hmac import compare_digest
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping

try:
    from .release_contract_v0232 import (
        ACTION_COUNT,
        ACTION_SHA256,
        BENCHMARK_REWARD_PROFILE,
        CLINICAL_OBSERVATION_COUNT,
        CLINICAL_OBSERVATION_SHA256,
        DEBUG_REWARD_PROFILE,
        EVIDENCE_ONLY_PATHS,
        EXPECTED_CHECK_NAMES,
        FULL_OBSERVATION_COUNT,
        FULL_OBSERVATION_SHA256,
        PYTEST_CONTRACTS,
        RESULTS_RELATIVE_PATH,
        STATE_SCHEMA_VERSION,
        VERSION,
        release_source_paths,
    )
except ImportError:  # direct execution from the validation directory
    from release_contract_v0232 import (
        ACTION_COUNT,
        ACTION_SHA256,
        BENCHMARK_REWARD_PROFILE,
        CLINICAL_OBSERVATION_COUNT,
        CLINICAL_OBSERVATION_SHA256,
        DEBUG_REWARD_PROFILE,
        EVIDENCE_ONLY_PATHS,
        EXPECTED_CHECK_NAMES,
        FULL_OBSERVATION_COUNT,
        FULL_OBSERVATION_SHA256,
        PYTEST_CONTRACTS,
        RESULTS_RELATIVE_PATH,
        STATE_SCHEMA_VERSION,
        VERSION,
        release_source_paths,
    )

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on supported 3.10
    tomllib = None  # type: ignore[assignment]


RELEASE_SCHEMA = "openhumsim.release.v1"
VALIDATION_SCHEMA = "openhumsim.validation-results.v1"
SOURCE_FINGERPRINT_SCHEMA = "openhumsim.source-fingerprint.v1"
CI_EVIDENCE_SCHEMA = "openhumsim.ci-evidence.v1"
REPORT_SCHEMA = "openhumsim.release-evidence-verification.v1"
EXPECTED_CI_PYTHON_VERSIONS = ("3.10", "3.12", "3.14")
EXPECTED_CI_WORKFLOW = ".github/workflows/ci.yml"
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_LOWER_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOWER_HEX_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class ReleaseEvidenceError(RuntimeError):
    """Release evidence is missing, malformed or internally inconsistent."""


def _fail(message: str) -> None:
    raise ReleaseEvidenceError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a nonempty string")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _commit_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_GIT_COMMIT.fullmatch(value) is None:
        _fail(f"{label} must be a full lowercase Git commit id")
    return value


def _timestamp(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseEvidenceError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        _fail(f"{label} must include a timezone")
    if parsed.utcoffset() != timedelta(0):
        _fail(f"{label} must be expressed in UTC")
    return text


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_float(token: str) -> float:
    value = float(token)
    if not isfinite(value):
        raise ValueError(f"non-finite JSON number {token!r}")
    return value


def _load_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReleaseEvidenceError(f"missing or unreadable {label}: {path}") from exc
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")
            ),
            parse_float=_parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReleaseEvidenceError(f"invalid {label}: {path}: {exc}") from exc
    return _mapping(value, label), payload


def _read_project_version(root: Path) -> str:
    path = root / "pyproject.toml"
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReleaseEvidenceError(f"missing or unreadable pyproject.toml: {path}") from exc

    if tomllib is not None:
        try:
            project = tomllib.loads(payload.decode("utf-8")).get("project")
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ReleaseEvidenceError(f"invalid pyproject.toml: {exc}") from exc
        version = _mapping(project, "pyproject [project]").get("version")
    else:
        # Python 3.10 has no stdlib TOML parser.  Accept the project's simple,
        # static PEP 621 version declaration and reject ambiguous forms.
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReleaseEvidenceError("pyproject.toml is not UTF-8") from exc
        in_project = False
        matches: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project = line == "[project]"
                continue
            if in_project:
                match = re.fullmatch(
                    r"version\s*=\s*([\"'])([^\"']+)\1\s*(?:#.*)?",
                    line,
                )
                if match is not None:
                    matches.append(match.group(2))
        if len(matches) != 1:
            _fail("pyproject [project].version must be one static string")
        version = matches[0]

    if not isinstance(version, str) or _SAFE_VERSION.fullmatch(version) is None:
        _fail("pyproject [project].version is missing or unsafe")
    return version


def _canonical_source_fingerprint(files: list[dict[str, str]]) -> str:
    payload = json.dumps(
        files,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _git(
    root: Path,
    *arguments: str,
    text: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[Any]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=text,
            check=False,
        )
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot execute Git: {exc}") from exc
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip() if text else completed.stderr.decode(
            "utf-8", errors="replace"
        ).strip()
        _fail(f"Git {' '.join(arguments)} failed: {stderr or 'unknown error'}")
    return completed


def _verify_git_repository(
    root: Path,
    candidate_commit: str,
    *,
    version: str,
) -> str:
    top_level = _git(root, "rev-parse", "--show-toplevel").stdout.strip()
    _require(
        Path(top_level).resolve() == root.resolve(),
        "verification root must be the Git worktree root",
    )
    resolved_candidate = _git(
        root,
        "rev-parse",
        f"{candidate_commit}^{{commit}}",
    ).stdout.strip()
    _require(
        resolved_candidate == candidate_commit,
        "candidate commit does not resolve to the declared full commit id",
    )
    head_commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    ancestor = _git(
        root,
        "merge-base",
        "--is-ancestor",
        candidate_commit,
        head_commit,
        check=False,
    )
    _require(
        ancestor.returncode == 0,
        "candidate commit is not an ancestor of HEAD",
    )
    _require(version == VERSION, "release verifier contract version mismatch")
    changed = _git(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        f"{candidate_commit}..{head_commit}",
        "--",
    ).stdout
    allowed = set(EVIDENCE_ONLY_PATHS)
    for line in changed.splitlines():
        parts = line.split("\t")
        _require(len(parts) == 2, "candidate-to-release diff is malformed")
        status_code, path = parts
        _require(
            status_code in {"A", "M"} and path in allowed,
            f"non-evidence change after candidate commit: {status_code} {path}",
        )
    status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).stdout
    _require(not status.strip(), "current Git worktree is dirty")
    return head_commit


def _safe_source_path(root: Path, raw_path: Any) -> tuple[str, Path]:
    path_text = _nonempty_string(raw_path, "source file path")
    pure = PurePosixPath(path_text)
    _require(
        not pure.is_absolute()
        and path_text == pure.as_posix()
        and "." not in pure.parts
        and ".." not in pure.parts
        and "\\" not in path_text
        and ":" not in path_text
        and "\n" not in path_text
        and "\r" not in path_text,
        f"unsafe source file path: {path_text!r}",
    )
    workspace_entry = root / path_text
    _require(
        not workspace_entry.is_symlink(),
        f"source file must not be a symlink: {path_text}",
    )
    candidate = workspace_entry.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        _fail(f"source file escapes repository root: {path_text!r}")
    _require(candidate.is_file(), f"source file is missing: {path_text}")
    return path_text, candidate


def _verify_source_provenance(
    root: Path,
    provenance_value: Any,
    *,
    candidate_commit: str,
) -> tuple[str, int]:
    provenance = _mapping(provenance_value, "validation source_provenance")
    _require(
        provenance.get("fingerprint_schema") == SOURCE_FINGERPRINT_SCHEMA,
        "unsupported source fingerprint schema",
    )
    _require(
        provenance.get("git_worktree_dirty") is False,
        "validation source provenance must record a clean candidate",
    )
    _require(
        provenance.get("git_commit") == candidate_commit,
        "validation source commit does not match the release gate",
    )
    declared_count = _exact_int(
        provenance.get("file_count"),
        "source file_count",
        minimum=1,
    )
    raw_files = provenance.get("files")
    if not isinstance(raw_files, list):
        _fail("validation source files must be a JSON array")
    _require(len(raw_files) == declared_count, "source file_count does not match files")

    records: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, raw_record in enumerate(raw_files):
        record = _mapping(raw_record, f"source file record {index}")
        path_text, workspace_path = _safe_source_path(root, record.get("path"))
        _require(path_text not in seen_paths, f"duplicate source file path: {path_text}")
        seen_paths.add(path_text)
        declared_sha = _sha256_hex(
            record.get("sha256"),
            f"source file {path_text} sha256",
        )
        workspace_sha = sha256(workspace_path.read_bytes()).hexdigest()
        _require(
            compare_digest(workspace_sha, declared_sha),
            f"source file hash mismatch: {path_text}",
        )
        blob = _git(
            root,
            "show",
            f"{candidate_commit}:{path_text}",
            text=False,
        ).stdout
        candidate_sha = sha256(blob).hexdigest()
        _require(
            compare_digest(candidate_sha, declared_sha),
            f"candidate commit source hash mismatch: {path_text}",
        )
        records.append({"path": path_text, "sha256": declared_sha})

    expected_paths = {
        path.relative_to(root).as_posix()
        for path in release_source_paths(root)
    }
    _require(
        seen_paths == expected_paths,
        "source fingerprint subject set does not match the release contract",
    )

    _require(
        records == sorted(records, key=lambda item: item["path"]),
        "source fingerprint files must be sorted by path",
    )
    declared_fingerprint = _sha256_hex(
        provenance.get("sha256"),
        "source fingerprint sha256",
    )
    computed_fingerprint = _canonical_source_fingerprint(records)
    _require(
        compare_digest(computed_fingerprint, declared_fingerprint),
        "source fingerprint SHA-256 does not match its file records",
    )
    return declared_fingerprint, declared_count


def _verify_result_summary(results: Mapping[str, Any]) -> tuple[int, int, int]:
    summary = _mapping(results.get("summary"), "validation summary")
    checks = results.get("checks")
    if not isinstance(checks, list):
        _fail("validation checks must be a JSON array")
    passed = _exact_int(summary.get("passed"), "validation summary passed")
    total = _exact_int(summary.get("total"), "validation summary total", minimum=1)
    executed = _exact_int(
        summary.get("executed_regression_cases"),
        "validation executed_regression_cases",
        minimum=1,
    )
    _require(len(checks) == total, "validation summary total does not match checks")

    actual_passed = 0
    actual_executed = 0
    names: set[str] = set()
    ordered_names: list[str] = []
    checks_by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw_check in enumerate(checks):
        check = _mapping(raw_check, f"validation check {index}")
        name = _nonempty_string(check.get("name"), f"validation check {index} name")
        _require(name not in names, f"duplicate validation check name: {name}")
        names.add(name)
        ordered_names.append(name)
        checks_by_name[name] = check
        check_passed = check.get("passed")
        _require(type(check_passed) is bool, f"validation check {name} passed must be boolean")
        actual_passed += int(check_passed)
        values = _mapping(check.get("values"), f"validation check {name} values")
        if "returncode" in values:
            _require(
                type(values["returncode"]) is int and values["returncode"] == 0,
                f"validation check {name} returncode is not zero",
            )
        if "executed_cases" in values:
            actual_executed += _exact_int(
                values["executed_cases"],
                f"validation check {name} executed_cases",
                minimum=1,
            )

    _require(actual_passed == passed, "validation summary passed does not match checks")
    _require(passed == total, "not every focused validation check passed")
    _require(
        actual_executed == executed,
        "validation executed_regression_cases does not match checks",
    )
    _require(
        tuple(ordered_names) == EXPECTED_CHECK_NAMES,
        "validation check names or order do not match the release contract",
    )
    for name, targets in PYTEST_CONTRACTS:
        values = _mapping(
            checks_by_name[name].get("values"),
            f"validation check {name} values",
        )
        _require(
            values.get("test_paths") == list(targets),
            f"validation check {name} pytest targets do not match the release contract",
        )
        if len(targets) == 1:
            _require(
                values.get("test_path") == targets[0],
                f"validation check {name} single pytest target is inconsistent",
            )

    version_values = _mapping(
        checks_by_name["exact_release_version"].get("values"),
        "exact_release_version values",
    )
    _require(
        version_values.get("version") == VERSION,
        "exact release version check does not match the release contract",
    )
    profile_values = _mapping(
        checks_by_name["state_schema_and_reward_profiles"].get("values"),
        "state schema and reward profile values",
    )
    _require(
        profile_values
        == {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "debug_reward_profile": DEBUG_REWARD_PROFILE,
            "benchmark_reward_profile": BENCHMARK_REWARD_PROFILE,
        },
        "state schema or reward profile check does not match the release contract",
    )
    provenance = _mapping(results.get("source_provenance"), "source provenance")
    source_values = _mapping(
        checks_by_name["source_snapshot_stable_during_gate"].get("values"),
        "source snapshot stability values",
    )
    _require(
        source_values.get("before_sha256") == provenance.get("sha256")
        and source_values.get("after_sha256") == provenance.get("sha256"),
        "source snapshot stability check does not match provenance",
    )
    return passed, total, executed


def _verify_full_suite(release: Mapping[str, Any]) -> None:
    suite = _mapping(release.get("full_test_suite"), "release full_test_suite")
    _require(suite.get("status") == "passed", "full test suite did not pass")
    passed = _exact_int(suite.get("passed"), "full suite passed", minimum=1)
    total = _exact_int(suite.get("total"), "full suite total", minimum=1)
    _require(passed == total, "full test suite is not completely passing")
    _require(suite.get("warnings_as_errors") is True, "full suite did not use warnings as errors")
    duration = suite.get("duration_seconds")
    _require(
        type(duration) in (int, float)
        and isfinite(float(duration))
        and float(duration) > 0.0,
        "full suite duration_seconds must be positive and finite",
    )
    _timestamp(suite.get("executed_at_utc"), "full suite executed_at_utc")
    _nonempty_string(suite.get("host_python"), "full suite host_python")
    command = _nonempty_string(suite.get("command"), "full suite command")
    _require(
        "-W error" in command or "-Werror" in command,
        "full suite command does not enable warnings as errors",
    )


def _verify_package_build(release: Mapping[str, Any], version: str) -> None:
    package = _mapping(release.get("package_build"), "release package_build")
    _require(package.get("status") == "passed", "package build did not pass")
    _require(
        package.get("wheel") == f"openhumsim_rl-{version}-py3-none-any.whl",
        "package wheel filename does not match the project version",
    )
    _require(
        package.get("sdist") == f"openhumsim_rl-{version}.tar.gz",
        "package sdist filename does not match the project version",
    )
    for field in (
        "twine_check",
        "isolated_wheel_install",
        "checkout_only_validation_guard",
    ):
        _require(package.get(field) == "passed", f"package build {field} did not pass")
    cli_smoke = _nonempty_string(package.get("cli_smoke"), "package build cli_smoke")
    _require(
        cli_smoke == "passed"
        or (cli_smoke.endswith(" passed") and "failed" not in cli_smoke.lower()),
        "package build CLI smoke did not pass",
    )
    if "dashboard_smoke" in package:
        _require(package.get("dashboard_smoke") == "passed", "dashboard smoke did not pass")


def _verify_release_contract(release: Mapping[str, Any]) -> None:
    expected = {
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
        "historical_rl_checkpoints_compatible": False,
        "full_observation_claimed_markov": False,
        "independent_external_validation_bundled": False,
        "clinical_use_supported": False,
    }
    for field, value in expected.items():
        _require(
            release.get(field) == value,
            f"release contract field {field} does not match v0.23.2",
        )


def _verify_ci(
    release: Mapping[str, Any],
    ci_evidence: Mapping[str, Any],
    *,
    version: str,
    candidate_commit: str,
) -> None:
    _require(ci_evidence.get("schema") == CI_EVIDENCE_SCHEMA, "unsupported CI evidence schema")
    _require(
        ci_evidence.get("records")
        == f"RELEASE_v{version}.json.supported_interpreter_ci",
        "CI evidence records field does not target the current release",
    )
    release_ci = _mapping(
        release.get("supported_interpreter_ci"),
        "release supported_interpreter_ci",
    )
    run = _mapping(ci_evidence.get("latest_successful_run"), "CI latest_successful_run")

    _require(release_ci.get("status") == "passed", "supported-interpreter CI did not pass")
    _require(release_ci.get("execution") == "github-actions", "CI execution is not github-actions")
    _require(release_ci.get("workflow") == EXPECTED_CI_WORKFLOW, "release CI workflow is unexpected")
    _require(release_ci.get("conclusion") == "success", "release CI conclusion is not success")
    _require(
        release_ci.get("python_versions") == list(EXPECTED_CI_PYTHON_VERSIONS),
        "release CI Python versions do not match the supported matrix",
    )
    _require(
        release_ci.get("scientific_validation") == "success",
        "release CI scientific-validation job did not succeed",
    )
    _require(
        release_ci.get("package_smoke_test") == "success",
        "release CI package-smoke job did not succeed",
    )
    _require(release_ci.get("codeql_conclusion") == "success", "release CodeQL did not succeed")
    _exact_int(release_ci.get("codeql_run_id"), "release CodeQL run_id", minimum=1)
    _exact_int(release_ci.get("run_id"), "release CI run_id", minimum=1)
    _timestamp(release_ci.get("completed_at"), "release CI completed_at")
    run_url = _nonempty_string(release_ci.get("run_url"), "release CI run_url")
    _require(run_url.startswith("https://"), "release CI run_url must use HTTPS")
    _require(
        release_ci.get("commit_sha") == candidate_commit,
        "release CI commit does not match the candidate commit",
    )

    for field in (
        "workflow",
        "run_id",
        "run_url",
        "commit_sha",
        "completed_at",
        "conclusion",
        "python_versions",
        "scientific_validation",
        "package_smoke_test",
    ):
        _require(
            run.get(field) == release_ci.get(field),
            f"CI evidence field {field} does not match the release manifest",
        )
    _require(run.get("workflow") == EXPECTED_CI_WORKFLOW, "CI evidence workflow is unexpected")
    _require(run.get("commit_sha") == candidate_commit, "CI evidence commit is not the candidate")
    _require(run.get("conclusion") == "success", "CI evidence conclusion is not success")
    _require(
        run.get("python_versions") == list(EXPECTED_CI_PYTHON_VERSIONS),
        "CI evidence Python versions do not match the supported matrix",
    )
    _require(
        run.get("scientific_validation") == "success",
        "CI evidence scientific-validation job did not succeed",
    )
    _require(
        run.get("package_smoke_test") == "success",
        "CI evidence package-smoke job did not succeed",
    )


def verify_release_evidence(root: Path | str) -> dict[str, Any]:
    """Verify release evidence rooted at *root* or raise ReleaseEvidenceError."""

    repository = Path(root).resolve()
    _require(repository.is_dir(), f"verification root is not a directory: {repository}")
    version = _read_project_version(repository)
    _require(version == VERSION, "project version does not match the verifier contract")
    release_name = f"RELEASE_v{version}.json"
    results_name = f"validation/validation_results_v{version}.json"

    release, _ = _load_json(repository / release_name, "release manifest")
    results, results_bytes = _load_json(
        repository / results_name,
        "focused validation results",
    )
    ci_evidence, _ = _load_json(repository / "CI_EVIDENCE.json", "CI evidence")

    _require(release.get("schema") == RELEASE_SCHEMA, "unsupported release manifest schema")
    _require(release.get("status") == "released", "release manifest status is not released")
    _require(release.get("version") == version, "release manifest version does not match pyproject")
    _require(release.get("author") == "lysyloxidase", "release author is not lysyloxidase")
    _require(release.get("license") == "Apache-2.0", "release license is not Apache-2.0")
    _verify_release_contract(release)
    _require(results.get("schema") == VALIDATION_SCHEMA, "unsupported validation results schema")
    _require(results.get("version") == version, "validation results version does not match pyproject")
    release_state_schema = _nonempty_string(
        release.get("state_schema_version"),
        "release state_schema_version",
    )
    _require(
        results.get("state_schema_version") == release_state_schema,
        "validation and release state schema versions differ",
    )

    gate = _mapping(release.get("focused_integrity_gate"), "release focused_integrity_gate")
    _require(gate.get("status") == "passed", "focused integrity gate status is not passed")
    _require(gate.get("results_path") == results_name, "focused results_path is not canonical")
    declared_results_sha = _sha256_hex(gate.get("results_sha256"), "focused results_sha256")
    actual_results_sha = sha256(results_bytes).hexdigest()
    _require(
        compare_digest(actual_results_sha, declared_results_sha),
        "focused validation results SHA-256 mismatch",
    )

    passed, total, executed = _verify_result_summary(results)
    for field, expected in (
        ("passed", passed),
        ("total", total),
        ("executed_regression_cases", executed),
    ):
        gate_value = _exact_int(
            gate.get(field),
            f"focused gate {field}",
            minimum=1,
        )
        _require(gate_value == expected, f"focused gate {field} does not match results")
    _require(passed == total, "focused integrity gate is not completely passing")
    _require(
        gate.get("executed_at_utc") == results.get("executed_at_utc"),
        "focused gate execution time does not match results",
    )
    _timestamp(results.get("executed_at_utc"), "validation executed_at_utc")
    gate_python = _nonempty_string(
        gate.get("host_python"),
        "focused gate host_python",
    )
    runtime = _mapping(results.get("runtime"), "validation runtime")
    _require(
        runtime.get("python") == gate_python,
        "focused gate host Python does not match validation runtime",
    )
    _require(
        gate.get("git_worktree_dirty") is False,
        "focused gate must record a clean candidate",
    )
    candidate_commit = _commit_hex(gate.get("git_commit"), "focused gate git_commit")

    fingerprint_sha, fingerprint_count = _verify_source_provenance(
        repository,
        results.get("source_provenance"),
        candidate_commit=candidate_commit,
    )
    gate_fingerprint_sha = _sha256_hex(
        gate.get("source_fingerprint_sha256"),
        "focused gate source_fingerprint_sha256",
    )
    _require(
        gate_fingerprint_sha == fingerprint_sha,
        "focused gate source fingerprint does not match results",
    )
    gate_source_count = _exact_int(
        gate.get("source_file_count"),
        "focused gate source_file_count",
        minimum=1,
    )
    _require(
        gate_source_count == fingerprint_count,
        "focused gate source file count does not match results",
    )

    _verify_full_suite(release)
    _verify_package_build(release, version)
    _verify_ci(
        release,
        ci_evidence,
        version=version,
        candidate_commit=candidate_commit,
    )
    head_commit = _verify_git_repository(
        repository,
        candidate_commit,
        version=version,
    )

    return {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "version": version,
        "release_manifest": release_name,
        "validation_results": results_name,
        "validation_results_sha256": actual_results_sha,
        "source_fingerprint_sha256": fingerprint_sha,
        "source_file_count": fingerprint_count,
        "focused_checks": total,
        "executed_regression_cases": executed,
        "candidate_commit": candidate_commit,
        "head_commit": head_commit,
        "ci_python_versions": list(EXPECTED_CI_PYTHON_VERSIONS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify fail-closed OpenHumSim release evidence",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: checkout containing this script)",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_release_evidence(args.root)
    except ReleaseEvidenceError as exc:
        print(f"release evidence verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
