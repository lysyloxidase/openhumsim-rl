"""Focused v0.23 release-candidate integrity gate.

This script executes the solver/configuration, respiratory-mechanics,
temporal/reward/measurement, transactional-step, policy-manifest,
environment-snapshot and observation-history regressions. It records only the
checks run by this invocation, fingerprints their source files, and is not
external clinical validation.
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

require_exact_version("0.23.0")

from openhumsim_rl import HumanHomeostasisEnv, __version__
from openhumsim_rl.env import (
    LATENT_REWARD_PROFILE,
    OBSERVABLE_REWARD_PROFILE,
)
from openhumsim_rl.physiology import STATE_SCHEMA_VERSION


RESULTS_PATH = ROOT / "validation" / "validation_results_v0.23.json"
checks: list[dict[str, object]] = []

PYTEST_CONTRACTS = (
    ("solver_and_config_regressions", "tests/test_solver_config_hardening.py"),
    (
        "mechanics_continuity_and_total_peep",
        "tests/test_mechanics_continuity_v023.py",
    ),
    (
        "temporal_reward_and_measurement_contracts",
        "tests/test_temporal_contract_v023.py",
    ),
    (
        "transactional_step_contracts",
        "tests/test_step_transaction_v023.py",
    ),
    (
        "policy_manifest_and_checkpoint_contracts",
        "tests/test_policy_manifest_v023.py",
    ),
    (
        "environment_snapshot_contracts",
        "tests/test_environment_snapshot_v023.py",
    ),
    (
        "observation_history_and_baseline_harness_contracts",
        "tests/test_history_wrapper_v023.py",
    ),
)


def add(name: str, passed: bool, values: dict[str, object]) -> None:
    checks.append({"name": name, "passed": bool(passed), "values": values})


def run_pytest_contract(name: str, test_path: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-W",
        "error",
        test_path,
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
    add(
        name,
        completed.returncode == 0,
        {
            "test_path": test_path,
            "returncode": completed.returncode,
            "executed_cases": (
                int(count_match.group(1)) if count_match is not None else None
            ),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        },
    )


def _source_paths() -> tuple[Path, ...]:
    paths = list((ROOT / "src" / "openhumsim_rl").glob("**/*.py"))
    paths.extend(ROOT / test_path for _, test_path in PYTEST_CONTRACTS)
    paths.extend(
        (
            Path(__file__).resolve(),
            ROOT / "examples" / "train_ppo.py",
            ROOT / "validation" / "rl_benchmark_v0.23.py",
            ROOT / "pyproject.toml",
        )
    )
    return tuple(sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix()))


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
    __version__ == "0.23.0",
    {"version": __version__},
)

debug_env = HumanHomeostasisEnv(info_profile="debug")
benchmark_env = HumanHomeostasisEnv(info_profile="benchmark")
_, debug_info = debug_env.reset(seed=23001)
_, benchmark_info = benchmark_env.reset(seed=23001)
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

for check_name, path in PYTEST_CONTRACTS:
    run_pytest_contract(check_name, path)

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
    "version": "0.23.0",
    "state_schema_version": "0.22",
    "scope": (
        "internal v0.23 solver/configuration, respiratory mechanics, temporal, "
        "reward, measurement, transactional-step, checkpoint-manifest, "
        "environment-snapshot and observation-history invariants; not external "
        "clinical validation"
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
raise SystemExit(0 if payload["summary"]["passed"] == payload["summary"]["total"] else 1)
