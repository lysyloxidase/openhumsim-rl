"""Optional Stable-Baselines3 smoke example.

Install:
    pip install -e ".[rl]"

The native one-sided [0,1] action space is used because all eight actuators are
nonnegative physical interventions and 0 is the natural no-intervention point.
The benchmark interface is deliberately strict. This feed-forward MLP is only a
smoke baseline for a POMDP; serious comparisons should use observation history
or a recurrent/belief-state policy.

The checkpoint is accompanied by a deterministic JSON manifest.  Manifest
construction deliberately has no Stable-Baselines3 dependency, so it can be
used by evaluation and provenance tooling without the optional ``rl`` extra.
"""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as package_version
from hashlib import sha256
import os
import platform
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, __version__
from openhumsim_rl.env import (
    ACTION_NAMES,
    CLINICAL_OBSERVATION_NAMES,
    OBSERVABLE_REWARD_PROFILE,
    _normalization_for,
)
from openhumsim_rl.measurement import ClinicalMeasurementConfig
from openhumsim_rl.policy_manifest import (
    POLICY_MANIFEST_CONTRACT_KEYS,
    PolicyCompatibilityError,
    build_policy_manifest,
    normalize_json_types,
    ordered_contract_sha256,
    validate_policy_manifest,
    write_policy_manifest,
)
from openhumsim_rl.units import ACTION_SEMANTICS


CHECKPOINT_BASENAME = "openhumsim_ppo_v0231_smoke"
MANIFEST_FILENAME = f"{CHECKPOINT_BASENAME}.manifest.json"
CHECKPOINT_FILENAME = f"{CHECKPOINT_BASENAME}.zip"
STATE_SCHEMA_VERSION = "0.22"
REWARD_PROFILE = OBSERVABLE_REWARD_PROFILE
SCENARIO = "oral_glucose_75g"
OBSERVATION_PROFILE = "clinical"
MEASUREMENT_PROFILE = "realistic"
INFO_PROFILE = "benchmark"
TRAINING_SEED = 22022
TOTAL_TIMESTEPS = 50_000
ALGORITHM = "stable_baselines3.PPO"
PPO_HYPERPARAMETERS: dict[str, Any] = {
    "policy": "MlpPolicy",
    "n_steps": 256,
    "batch_size": 64,
    "learning_rate": 3e-4,
}


def observation_names_sha256(observation_names: Sequence[str]) -> str:
    """Hash the ordered observation contract, not an unordered set of names."""

    return ordered_contract_sha256(observation_names)


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _source_commit_sha() -> str | None:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        value = github_sha.strip().lower()
        if len(value) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RuntimeError("GITHUB_SHA is not a full hexadecimal Git object ID")
        return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    if not value:
        return None
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError("git rev-parse HEAD returned an invalid object ID")
    return value


def _source_tree_dirty() -> bool | None:
    """Report dirtiness only for executable sources covered by the fingerprint."""

    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src/openhumsim_rl",
                "examples/train_ppo.py",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _source_fingerprint_sha256() -> str:
    """Hash executable model and training-example sources, including dirty edits."""

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src" / "openhumsim_rl").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    digest = sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _runtime_contract() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "stable_baselines3": _installed_version("stable-baselines3"),
        "torch": _installed_version("torch"),
    }


def _space_contract(space, *, width: int, label: str) -> dict[str, Any]:
    shape = tuple(getattr(space, "shape", ()))
    if shape != (int(width),):
        raise ValueError(f"{label} space shape must be {(int(width),)}, got {shape}")
    dtype = np.dtype(getattr(space, "dtype", None))
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"{label} space dtype must be floating point")
    low = np.asarray(getattr(space, "low", None), dtype=float)
    high = np.asarray(getattr(space, "high", None), dtype=float)
    if low.shape != shape or high.shape != shape:
        raise ValueError(f"{label} space bounds must match shape {shape}")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError(f"{label} space bounds must be finite")
    if np.any(low > high):
        raise ValueError(f"{label} space lower bounds must not exceed upper bounds")
    return normalize_json_types(
        {
            "shape": list(shape),
            "dtype": dtype.name,
            "low": low.tolist(),
            "high": high.tolist(),
        }
    )


def capture_training_provenance() -> dict[str, Any]:
    """Capture source and runtime identity before a training run starts."""

    return normalize_json_types({
        "source_commit_sha": _source_commit_sha(),
        "source_tree_dirty": _source_tree_dirty(),
        "source_fingerprint_sha256": _source_fingerprint_sha256(),
        "runtime": _runtime_contract(),
    })


def assert_training_provenance_unchanged(
    captured: dict[str, Any],
) -> None:
    """Reject a checkpoint if its executable source/runtime changed mid-run."""

    expected = normalize_json_types(captured)
    current = capture_training_provenance()
    if current != expected:
        changed = sorted(
            key for key in set(expected) | set(current)
            if expected.get(key) != current.get(key)
        )
        raise RuntimeError(
            "training provenance changed before checkpoint finalization: "
            f"{changed}"
        )


def build_training_manifest(
    *,
    environment: HumanHomeostasisEnv | None = None,
    config: HumanConfig | None = None,
    observation_names: Sequence[str] | None = None,
    observation_centers: Sequence[float] | None = None,
    observation_scales: Sequence[float] | None = None,
    scenario: str = SCENARIO,
    observation_profile: str = OBSERVATION_PROFILE,
    measurement_profile: str = MEASUREMENT_PROFILE,
    measurement_config: ClinicalMeasurementConfig | dict[str, Any] | None = None,
    info_profile: str = INFO_PROFILE,
    reward_profile: str = REWARD_PROFILE,
    checkpoint_path: str | Path = CHECKPOINT_FILENAME,
    require_checkpoint: bool = False,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build reproducible checkpoint metadata without importing SB3."""

    if environment is not None:
        effective_config = environment.config
        ordered_names = tuple(environment.observation_names)
        center = np.asarray(environment._obs_center, dtype=float)
        scale = np.asarray(environment._obs_scale, dtype=float)
        scenario = str(environment.active_scenario)
        observation_profile = str(environment.observation_profile)
        measurement_profile = str(environment.measurement_profile)
        info_profile = str(environment.info_profile)
        reward_profile = str(environment.reward_profile)
        measurement_config_payload = (
            None
            if environment.measurement_model is None
            else asdict(environment.measurement_model.config)
        )
        observation_space_contract = _space_contract(
            environment.observation_space,
            width=len(ordered_names),
            label="observation",
        )
        action_space_contract = _space_contract(
            environment.action_space,
            width=len(ACTION_NAMES),
            label="action",
        )
    else:
        effective_config = HumanConfig() if config is None else config
        ordered_names = tuple(
            CLINICAL_OBSERVATION_NAMES
            if observation_names is None
            else observation_names
        )
        if (observation_centers is None) != (observation_scales is None):
            raise ValueError(
                "observation_centers and observation_scales must be supplied together"
            )
        if observation_centers is None:
            center, scale = _normalization_for(ordered_names)
        else:
            center = np.asarray(observation_centers, dtype=float)
            scale = np.asarray(observation_scales, dtype=float)
        if measurement_profile == "realistic":
            resolved_measurement_config = (
                ClinicalMeasurementConfig()
                if measurement_config is None
                else measurement_config
            )
            measurement_config_payload = (
                asdict(resolved_measurement_config)
                if isinstance(resolved_measurement_config, ClinicalMeasurementConfig)
                else dict(resolved_measurement_config)
            )
        else:
            if measurement_config is not None:
                raise ValueError(
                    "measurement_config is only valid for measurement_profile='realistic'"
                )
            measurement_config_payload = None
        observation_space_contract = {
            "shape": [len(ordered_names)],
            "dtype": "float32",
            "low": [-1.0] * len(ordered_names),
            "high": [1.0] * len(ordered_names),
        }
        action_space_contract = {
            "shape": [len(ACTION_NAMES)],
            "dtype": "float32",
            "low": [0.0] * len(ACTION_NAMES),
            "high": [1.0] * len(ACTION_NAMES),
        }

    captured = (
        capture_training_provenance()
        if provenance is None
        else normalize_json_types(provenance)
    )
    required_provenance = {
        "source_commit_sha",
        "source_tree_dirty",
        "source_fingerprint_sha256",
        "runtime",
    }
    if set(captured) != required_provenance:
        raise ValueError(
            "provenance keys must be exactly "
            f"{sorted(required_provenance)}, got {sorted(captured)}"
        )
    return build_policy_manifest(
        checkpoint_path=checkpoint_path,
        openhumsim_version=__version__,
        state_schema_version=STATE_SCHEMA_VERSION,
        reward_profile=reward_profile,
        scenario=scenario,
        observation_profile=observation_profile,
        measurement_profile=measurement_profile,
        measurement_config=measurement_config_payload,
        info_profile=info_profile,
        observation_names=ordered_names,
        observation_centers=center,
        observation_scales=scale,
        observation_space_contract=observation_space_contract,
        action_names=ACTION_NAMES,
        action_semantics=ACTION_SEMANTICS,
        action_space_contract=action_space_contract,
        config=effective_config,
        algorithm=ALGORITHM,
        algorithm_hyperparameters=PPO_HYPERPARAMETERS,
        total_timesteps=TOTAL_TIMESTEPS,
        training_seed=TRAINING_SEED,
        source_commit_sha=captured["source_commit_sha"],
        source_tree_dirty=captured["source_tree_dirty"],
        source_fingerprint_sha256=captured["source_fingerprint_sha256"],
        runtime=captured["runtime"],
        require_checkpoint=require_checkpoint,
    )


def write_training_manifest(
    path: str | Path = MANIFEST_FILENAME,
    *,
    environment: HumanHomeostasisEnv | None = None,
    config: HumanConfig | None = None,
    observation_names: Sequence[str] | None = None,
    checkpoint_path: str | Path = CHECKPOINT_FILENAME,
    require_checkpoint: bool = False,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a byte-for-byte deterministic training manifest."""

    if require_checkpoint and provenance is None:
        raise ValueError(
            "a checkpoint manifest requires provenance captured before training"
        )
    output_path = Path(path)
    manifest = build_training_manifest(
        environment=environment,
        config=config,
        observation_names=observation_names,
        checkpoint_path=checkpoint_path,
        require_checkpoint=require_checkpoint,
        provenance=provenance,
    )
    return write_policy_manifest(output_path, manifest)


def load_policy_checked(
    checkpoint_path: str | Path = CHECKPOINT_FILENAME,
    manifest_path: str | Path = MANIFEST_FILENAME,
    *,
    env: HumanHomeostasisEnv | None = None,
):
    """Load PPO only after exact artifact and environment compatibility checks."""

    from stable_baselines3 import PPO

    evaluation_env = env or HumanHomeostasisEnv(
        config=HumanConfig(),
        scenario=SCENARIO,
        observation_profile=OBSERVATION_PROFILE,
        measurement_profile=MEASUREMENT_PROFILE,
        info_profile=INFO_PROFILE,
    )
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise PolicyCompatibilityError(f"checkpoint does not exist: {checkpoint}")

    # Validate and load the same private snapshot. A concurrent replacement of
    # the caller's path can no longer change the bytes after hashing.
    with TemporaryDirectory(prefix="openhumsim-policy-") as temporary_directory:
        snapshot = Path(temporary_directory) / checkpoint.name
        shutil.copyfile(checkpoint, snapshot)
        current = build_training_manifest(
            environment=evaluation_env,
            checkpoint_path=snapshot,
            require_checkpoint=True,
        )
        validate_policy_manifest(
            manifest_path,
            checkpoint_path=snapshot,
            expected={key: current[key] for key in POLICY_MANIFEST_CONTRACT_KEYS},
        )
        return PPO.load(snapshot, env=evaluation_env)


def main() -> None:
    # Keep the optional dependency local: importing this module to generate or
    # inspect a manifest must work without Stable-Baselines3 installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    config = HumanConfig()
    env = HumanHomeostasisEnv(
        config=config,
        scenario=SCENARIO,
        observation_profile=OBSERVATION_PROFILE,
        measurement_profile=MEASUREMENT_PROFILE,
        info_profile=INFO_PROFILE,
    )
    training_provenance = capture_training_provenance()
    check_env(env, warn=True)

    model = PPO(
        PPO_HYPERPARAMETERS["policy"],
        env,
        verbose=1,
        n_steps=PPO_HYPERPARAMETERS["n_steps"],
        batch_size=PPO_HYPERPARAMETERS["batch_size"],
        learning_rate=PPO_HYPERPARAMETERS["learning_rate"],
        seed=TRAINING_SEED,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(CHECKPOINT_BASENAME)
    assert_training_provenance_unchanged(training_provenance)
    write_training_manifest(
        environment=env,
        checkpoint_path=CHECKPOINT_FILENAME,
        require_checkpoint=True,
        provenance=training_provenance,
    )


if __name__ == "__main__":
    main()
