from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import HumanConfig


POLICY_MANIFEST_SCHEMA = "openhumsim.policy-manifest.v1"

# Every field that can change policy inputs, outputs, execution, or provenance
# must be supplied independently by a validator. Derived hashes are checked
# against their payloads below and therefore do not need a second expected copy.
POLICY_MANIFEST_CONTRACT_KEYS = (
    "openhumsim_version",
    "state_schema_version",
    "reward_profile",
    "scenario",
    "observation_profile",
    "measurement_profile",
    "measurement_config",
    "info_profile",
    "observation_names",
    "observation_normalization",
    "observation_space",
    "action_contract",
    "config",
    "training",
    "source_commit_sha",
    "source_tree_dirty",
    "source_fingerprint_sha256",
    "runtime",
)


class PolicyCompatibilityError(ValueError):
    """Raised when a policy artifact does not match its declared environment."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by policy contracts."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def normalize_json_types(value: Any) -> Any:
    """Return the exact finite JSON value used on disk.

    A canonical round trip prevents an in-memory tuple from mismatching the
    equivalent list read from a sidecar during policy loading.
    """

    return json.loads(canonical_json(value))


def ordered_contract_sha256(values: Sequence[Any]) -> str:
    """Hash an ordered contract; order is part of policy compatibility."""

    return sha256(canonical_json(list(values)).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a checkpoint artifact."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_space_contract(
    value: Mapping[str, Any],
    *,
    width: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "shape",
        "dtype",
        "low",
        "high",
    }:
        raise ValueError(
            f"{label} space contract must contain shape, dtype, low and high"
        )
    shape_raw = value["shape"]
    if not isinstance(shape_raw, (list, tuple)) or any(
        isinstance(size, bool) or not isinstance(size, (int, np.integer))
        for size in shape_raw
    ):
        raise ValueError(f"{label} space shape must contain only integers")
    shape = tuple(int(size) for size in shape_raw)
    if shape != (int(width),):
        raise ValueError(
            f"{label} space shape must be {(int(width),)}, got {shape}"
        )
    try:
        dtype = np.dtype(value["dtype"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} space dtype is invalid") from exc
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"{label} space dtype must be floating point")
    try:
        low = np.asarray(value["low"], dtype=float)
        high = np.asarray(value["high"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} space bounds must be numeric") from exc
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


def build_policy_manifest(
    *,
    checkpoint_path: str | Path,
    openhumsim_version: str,
    state_schema_version: str,
    reward_profile: str,
    scenario: str,
    observation_profile: str,
    measurement_profile: str,
    measurement_config: Mapping[str, Any] | None,
    info_profile: str,
    observation_names: Sequence[str],
    observation_centers: Sequence[float],
    observation_scales: Sequence[float],
    observation_space_contract: Mapping[str, Any],
    action_names: Sequence[str],
    action_semantics: Mapping[str, Any],
    action_space_contract: Mapping[str, Any],
    config: HumanConfig,
    algorithm: str,
    algorithm_hyperparameters: Mapping[str, Any],
    total_timesteps: int,
    training_seed: int,
    source_commit_sha: str | None,
    source_tree_dirty: bool | None,
    source_fingerprint_sha256: str,
    runtime: Mapping[str, Any],
    require_checkpoint: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed policy sidecar with artifact and interface hashes."""

    checkpoint = Path(checkpoint_path)
    if require_checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    names = [str(value) for value in observation_names]
    centers = [float(value) for value in observation_centers]
    scales = [float(value) for value in observation_scales]
    actions = [str(value) for value in action_names]
    if not (len(names) == len(centers) == len(scales)):
        raise ValueError("observation names, centers and scales must have equal length")
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("observation scales must be positive")
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")

    observation_space = _normalize_space_contract(
        observation_space_contract,
        width=len(names),
        label="observation",
    )
    action_space = _normalize_space_contract(
        action_space_contract,
        width=len(actions),
        label="action",
    )
    normalization = normalize_json_types({
        "transform": "tanh((raw-center)/scale)",
        "centers": centers,
        "scales": scales,
    })
    action_contract = normalize_json_types({
        "names": actions,
        "names_sha256": ordered_contract_sha256(actions),
        "semantics": {name: action_semantics[name] for name in actions},
        "agent_step_min": float(config.agent_step_min),
        "space": action_space,
    })
    artifact_sha = file_sha256(checkpoint) if checkpoint.is_file() else None
    manifest = {
        "schema": POLICY_MANIFEST_SCHEMA,
        # Compatibility aliases retained for the established example API.
        "manifest_schema_version": 1,
        "checkpoint_basename": checkpoint.stem,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": artifact_sha,
        "openhumsim_version": str(openhumsim_version),
        "state_schema_version": str(state_schema_version),
        "reward_profile": str(reward_profile),
        "scenario": str(scenario),
        "observation_profile": str(observation_profile),
        "measurement_profile": str(measurement_profile),
        "measurement_config": (
            None
            if measurement_config is None
            else normalize_json_types(dict(measurement_config))
        ),
        "info_profile": str(info_profile),
        "observation_names": names,
        "observation_names_sha256": ordered_contract_sha256(names),
        "observation_names_hash_format": "sha256:canonical-json-array:utf-8",
        "observation_normalization": normalization,
        "observation_normalization_sha256": sha256(
            canonical_json(normalization).encode("utf-8")
        ).hexdigest(),
        "observation_space": observation_space,
        "observation_space_sha256": sha256(
            canonical_json(observation_space).encode("utf-8")
        ).hexdigest(),
        "action_contract": action_contract,
        "action_contract_sha256": sha256(
            canonical_json(action_contract).encode("utf-8")
        ).hexdigest(),
        "config": normalize_json_types(asdict(config)),
        "training": {
            "algorithm": str(algorithm),
            "hyperparameters": normalize_json_types(dict(algorithm_hyperparameters)),
            "total_timesteps": int(total_timesteps),
            "seed": int(training_seed),
        },
        "source_commit_sha": source_commit_sha,
        "source_tree_dirty": source_tree_dirty,
        "source_fingerprint_sha256": str(source_fingerprint_sha256),
        "runtime": normalize_json_types(dict(runtime)),
    }
    return normalize_json_types(manifest)


def write_policy_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a canonical, finite JSON policy sidecar."""

    output = Path(path)
    output.write_text(
        json.dumps(
            dict(manifest),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def validate_policy_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    checkpoint_path: str | Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a policy manifest and checkpoint, rejecting every mismatch.

    ``expected`` is deliberately explicit and must contain every key in
    ``POLICY_MANIFEST_CONTRACT_KEYS``. This prevents a caller from accidentally
    omitting a compatibility or provenance field.
    """

    try:
        if isinstance(manifest_or_path, Mapping):
            manifest = normalize_json_types(dict(manifest_or_path))
        else:
            manifest = json.loads(Path(manifest_or_path).read_text(encoding="utf-8"))
        expected_normalized = {
            str(key): normalize_json_types(value) for key, value in expected.items()
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PolicyCompatibilityError(f"policy manifest is not finite JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PolicyCompatibilityError("policy manifest root must be a JSON object")
    if manifest.get("schema") != POLICY_MANIFEST_SCHEMA:
        raise PolicyCompatibilityError(
            f"unsupported policy manifest schema {manifest.get('schema')!r}"
        )
    missing_manifest = sorted(
        set(POLICY_MANIFEST_CONTRACT_KEYS) - set(manifest)
    )
    if missing_manifest:
        raise PolicyCompatibilityError(
            f"policy manifest contract is incomplete: missing={missing_manifest}"
        )

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise PolicyCompatibilityError(f"checkpoint does not exist: {checkpoint}")
    if manifest.get("checkpoint_filename") != checkpoint.name:
        raise PolicyCompatibilityError("checkpoint filename does not match manifest")
    if manifest.get("checkpoint_basename") != checkpoint.stem:
        raise PolicyCompatibilityError("checkpoint basename does not match manifest")
    declared_digest = manifest.get("checkpoint_sha256")
    if not declared_digest or declared_digest != file_sha256(checkpoint):
        raise PolicyCompatibilityError("checkpoint SHA-256 does not match manifest")

    if manifest.get("manifest_schema_version") != 1:
        raise PolicyCompatibilityError("manifest schema compatibility alias is invalid")
    source_commit = manifest.get("source_commit_sha")
    if source_commit is not None and (
        not isinstance(source_commit, str)
        or len(source_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise PolicyCompatibilityError("source_commit_sha is not a full Git object ID")
    source_fingerprint = manifest.get("source_fingerprint_sha256")
    if (
        not isinstance(source_fingerprint, str)
        or len(source_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in source_fingerprint)
    ):
        raise PolicyCompatibilityError("source_fingerprint_sha256 is invalid")
    source_tree_dirty = manifest.get("source_tree_dirty")
    if source_tree_dirty is not None and not isinstance(source_tree_dirty, bool):
        raise PolicyCompatibilityError("source_tree_dirty must be boolean or null")
    if manifest.get("observation_names_hash_format") != (
        "sha256:canonical-json-array:utf-8"
    ):
        raise PolicyCompatibilityError("observation-name hash format is invalid")
    observation_names = manifest.get("observation_names")
    if not isinstance(observation_names, list):
        raise PolicyCompatibilityError("observation_names must be a JSON array")
    if manifest.get("observation_names_sha256") != ordered_contract_sha256(
        observation_names
    ):
        raise PolicyCompatibilityError("observation-name hash is internally inconsistent")
    normalization = manifest.get("observation_normalization")
    if not isinstance(normalization, dict):
        raise PolicyCompatibilityError("observation_normalization must be a JSON object")
    if manifest.get("observation_normalization_sha256") != sha256(
        canonical_json(normalization).encode("utf-8")
    ).hexdigest():
        raise PolicyCompatibilityError(
            "observation-normalization hash is internally inconsistent"
        )
    observation_space = manifest.get("observation_space")
    if not isinstance(observation_space, dict):
        raise PolicyCompatibilityError("observation_space must be a JSON object")
    if manifest.get("observation_space_sha256") != sha256(
        canonical_json(observation_space).encode("utf-8")
    ).hexdigest():
        raise PolicyCompatibilityError(
            "observation-space hash is internally inconsistent"
        )
    try:
        normalized_observation_space = _normalize_space_contract(
            observation_space,
            width=len(observation_names),
            label="observation",
        )
    except ValueError as exc:
        raise PolicyCompatibilityError(str(exc)) from exc
    if normalized_observation_space != observation_space:
        raise PolicyCompatibilityError("observation_space is not canonical")
    action_contract = manifest.get("action_contract")
    if not isinstance(action_contract, dict) or not isinstance(
        action_contract.get("names"), list
    ):
        raise PolicyCompatibilityError("action_contract is malformed")
    if action_contract.get("names_sha256") != ordered_contract_sha256(
        action_contract["names"]
    ):
        raise PolicyCompatibilityError("action-name hash is internally inconsistent")
    if manifest.get("action_contract_sha256") != sha256(
        canonical_json(action_contract).encode("utf-8")
    ).hexdigest():
        raise PolicyCompatibilityError("action-contract hash is internally inconsistent")
    action_space = action_contract.get("space")
    if not isinstance(action_space, dict):
        raise PolicyCompatibilityError("action_contract space is malformed")
    try:
        normalized_action_space = _normalize_space_contract(
            action_space,
            width=len(action_contract["names"]),
            label="action",
        )
    except ValueError as exc:
        raise PolicyCompatibilityError(str(exc)) from exc
    if normalized_action_space != action_space:
        raise PolicyCompatibilityError("action space contract is not canonical")

    missing_expected = sorted(
        set(POLICY_MANIFEST_CONTRACT_KEYS) - set(expected_normalized)
    )
    if missing_expected:
        raise PolicyCompatibilityError(
            f"expected policy contract is incomplete: missing={missing_expected}"
        )
    for key, value in expected_normalized.items():
        if manifest.get(key) != value:
            raise PolicyCompatibilityError(
                f"policy contract mismatch for {key!r}: "
                f"manifest={manifest.get(key)!r}, expected={value!r}"
            )
    return manifest
