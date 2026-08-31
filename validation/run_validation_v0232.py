"""Focused v0.23.2 release-candidate integrity gate.

This script carries forward the v0.23.1 solver, mechanics, temporal,
transactional, policy-manifest, snapshot and history contracts.  It adds the
v0.23.2 pulmonary/ventilator regressions plus the policy-preprocessing,
command-line and packaged-dashboard contracts changed by this patch release.
It records only checks executed by this invocation, fingerprints the exact
source and test subjects, and is not external clinical validation.

Release-evidence integration tests that consume ``RELEASE_v0.23.2.json`` and
this script's output remain part of the complete repository suite.  They cannot
be inputs to the gate that creates that output without introducing a circular
dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historical_version_guard import require_exact_version

require_exact_version("0.23.2")

from openhumsim_rl import HumanHomeostasisEnv, __version__
from openhumsim_rl.env import (
    LATENT_REWARD_PROFILE,
    OBSERVABLE_REWARD_PROFILE,
)
from openhumsim_rl.physiology import STATE_SCHEMA_VERSION


RESULTS_PATH = ROOT / "validation" / "validation_results_v0.23.2.json"
checks: list[dict[str, object]] = []

# A contract may use several pytest targets so related release behavior remains
# one top-level gate check while every executed case is still counted.
PYTEST_CONTRACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "patch_release_regressions_v0231",
        ("tests/test_patch_regressions_v0231.py",),
    ),
    (
        "solver_and_config_regressions",
        ("tests/test_solver_config_hardening.py",),
    ),
    (
        "mechanics_continuity_and_total_peep",
        ("tests/test_mechanics_continuity_v023.py",),
    ),
    (
        "temporal_reward_and_measurement_contracts",
        ("tests/test_temporal_contract_v023.py",),
    ),
    (
        "transactional_step_contracts",
        ("tests/test_step_transaction_v023.py",),
    ),
    (
        "policy_manifest_and_checkpoint_contracts",
        ("tests/test_policy_manifest_v023.py",),
    ),
    (
        "environment_snapshot_contracts",
        ("tests/test_environment_snapshot_v023.py",),
    ),
    (
        "observation_history_and_baseline_harness_contracts",
        ("tests/test_history_wrapper_v023.py",),
    ),
    (
        "biophysics_and_result_only_pulmonary_contracts",
        (
            "tests/test_biophysics_regressions.py",
            (
                "tests/test_physics_regressions_v021.py::"
                "test_hpv_fixed_point_advances_kinetics_only_once"
            ),
        ),
    ),
    (
        "cli_policy_metadata_and_packaged_dashboard_contracts",
        (
            "tests/test_cli_contract.py",
            "tests/test_config_manifest_v022.py",
            "tests/test_dashboard_http.py",
            (
                "tests/test_dashboard.py::"
                "test_dashboard_is_self_contained_and_has_research_boundary"
            ),
            (
                "tests/test_dashboard.py::"
                "test_packaged_dashboard_is_canonical_and_legacy_import_is_compatible"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_session_exposes_measurements_separately_from_debug_truth"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_step_uses_real_environment_and_validates_actions"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_step_rolls_back_after_post_transition_frame_failure"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_reset_rolls_back_after_frame_failure"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_documentation_targets_existing_files"
            ),
        ),
    ),
    (
        "release_evidence_verifier_contracts",
        ("tests/test_release_evidence.py",),
    ),
)


def add(name: str, passed: bool, values: dict[str, object]) -> None:
    checks.append({"name": name, "passed": bool(passed), "values": values})


def run_pytest_contract(name: str, test_targets: tuple[str, ...]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-W",
        "error",
        *test_targets,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    pytest_output = "\n".join((completed.stdout, completed.stderr))
    count_match = re.search(r"(?:^|\s)(\d+) passed(?:[,\s]|$)", pytest_output)
    values: dict[str, object] = {
        "test_paths": list(test_targets),
        "returncode": completed.returncode,
        "executed_cases": (
            int(count_match.group(1)) if count_match is not None else None
        ),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if len(test_targets) == 1:
        # Retain the single-path convenience field used by earlier v0.23
        # validation records while supporting grouped targets above.
        values["test_path"] = test_targets[0]
    add(name, completed.returncode == 0, values)


def _pytest_source_paths() -> tuple[Path, ...]:
    paths = {
        ROOT / target.split("::", maxsplit=1)[0]
        for _, targets in PYTEST_CONTRACTS
        for target in targets
    }
    return tuple(sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix()))


def _source_paths() -> tuple[Path, ...]:
    paths = list((ROOT / "src" / "openhumsim_rl").glob("**/*.py"))
    paths.extend(_pytest_source_paths())
    paths.extend(
        (
            Path(__file__).resolve(),
            ROOT / "validation" / "historical_version_guard.py",
            ROOT / "validation" / "verify_release_evidence.py",
            ROOT / "examples" / "train_ppo.py",
            ROOT / "examples" / "dashboard_server.py",
            ROOT / "src" / "openhumsim_rl" / "dashboard" / "index.html",
            ROOT / "dashboard" / "index.html",
            ROOT / "validation" / "rl_benchmark_v0.23.py",
            ROOT / "README.md",
            ROOT / "docs" / "dashboard.md",
            ROOT / "LICENSE",
            ROOT / "NOTICE",
            ROOT / "CITATION.cff",
            ROOT / ".github" / "workflows" / "ci.yml",
            ROOT / ".github" / "workflows" / "release.yml",
            ROOT / "pyproject.toml",
        )
    )
    return tuple(
        sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())
    )


def _source_snapshot() -> dict[str, object]:
    records = []
    for path in _source_paths():
        relative = path.relative_to(ROOT).as_posix()
        records.append(
            {
                "path": relative,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    canonical = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {
        "fingerprint_schema": "openhumsim.source-fingerprint.v1",
        "sha256": sha256(canonical).hexdigest(),
        "file_count": len(records),
        "files": records,
        "git_commit": commit,
        "git_worktree_dirty": dirty,
    }


def _runtime_packages() -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for distribution in ("numpy", "pytest", "gymnasium"):
        try:
            resolved[distribution] = package_version(distribution)
        except PackageNotFoundError:
            resolved[distribution] = None
    return resolved


source_snapshot_before = _source_snapshot()


add(
    "exact_release_version",
    __version__ == "0.23.2",
    {"version": __version__},
)

debug_env = HumanHomeostasisEnv(info_profile="debug")
benchmark_env = HumanHomeostasisEnv(info_profile="benchmark")
_, debug_info = debug_env.reset(seed=23002)
_, benchmark_info = benchmark_env.reset(seed=23002)
add(
    "state_schema_and_reward_profiles",
    STATE_SCHEMA_VERSION == "0.22"
    and debug_env.reward_profile == LATENT_REWARD_PROFILE
    and benchmark_env.reward_profile == OBSERVABLE_REWARD_PROFILE
    and debug_info["environment_semantics"]["reward_observability"]
    == "latent_mechanistic_state"
    and benchmark_info["environment_semantics"]["reward_observability"]
    == "public_observation_action_and_transition_events",
    {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "debug_reward_profile": debug_env.reward_profile,
        "benchmark_reward_profile": benchmark_env.reward_profile,
    },
)

for check_name, targets in PYTEST_CONTRACTS:
    run_pytest_contract(check_name, targets)

source_snapshot_after = _source_snapshot()
add(
    "source_snapshot_stable_during_gate",
    source_snapshot_after["sha256"] == source_snapshot_before["sha256"],
    {
        "before_sha256": source_snapshot_before["sha256"],
        "after_sha256": source_snapshot_after["sha256"],
    },
)

payload = {
    "schema": "openhumsim.validation-results.v1",
    "version": "0.23.2",
    "state_schema_version": "0.22",
    "scope": (
        "internal v0.23.2 pulmonary and respiratory mechanics, solver/configuration, "
        "temporal reward and measurement, transactional-step, policy-observation "
        "and checkpoint manifest, environment snapshot, observation history, CLI "
        "and packaged-dashboard invariants; not external clinical validation"
    ),
    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    "runtime": {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": _runtime_packages(),
    },
    "source_provenance": source_snapshot_before,
    "summary": {
        "passed": sum(bool(check["passed"]) for check in checks),
        "total": len(checks),
        "executed_regression_cases": sum(
            int(check["values"].get("executed_cases") or 0)
            for check in checks
        ),
    },
    "checks": checks,
}

# Write only after every check above has actually executed. The atomic replace
# prevents a partial JSON document from masquerading as validation evidence.
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with NamedTemporaryFile(
    "w",
    encoding="utf-8",
    dir=RESULTS_PATH.parent,
    prefix=f".{RESULTS_PATH.name}.",
    suffix=".tmp",
    delete=False,
) as handle:
    temporary_path = Path(handle.name)
    json.dump(payload, handle, indent=2, allow_nan=False)
    handle.write("\n")
temporary_path.replace(RESULTS_PATH)

print(json.dumps(payload, indent=2, allow_nan=False))
raise SystemExit(
    0 if payload["summary"]["passed"] == payload["summary"]["total"] else 1
)
