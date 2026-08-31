from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import HumanConfig


POLICY_MANIFEST_SCHEMA = "openhumsim.policy-manifest.v1"
POLICY_OBSERVATION_CONTRACT_SCHEMA = (
    "openhumsim.policy-observation-contract.v1"
)
POLICY_ACTION_CONTRACT_SCHEMA = "openhumsim.policy-action-contract.v1"
POLICY_ACTION_MAPPING_SCHEMA = "openhumsim.policy-action-mapping.v1"
OBSERVATION_HISTORY_PREPROCESSING_SCHEMA = (
    "openhumsim.observation-history-preprocessing.v1"
)

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


def _normalize_policy_action_contract(
    value: Mapping[str, Any],
    *,
    action_names: Sequence[str],
    policy_space: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the policy-facing action mapping and its native target."""

    expected_keys = {
        "schema",
        "interface_id",
        "action_names",
        "policy_space",
        "native_space",
        "mapping",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("policy action contract has an invalid key set")
    normalized = normalize_json_types(dict(value))
    if normalized["schema"] != POLICY_ACTION_CONTRACT_SCHEMA:
        raise ValueError("policy action contract schema is unsupported")
    names = [str(name) for name in action_names]
    if normalized["action_names"] != names:
        raise ValueError("policy action names do not match the action contract")
    normalized_policy_space = _normalize_space_contract(
        normalized["policy_space"],
        width=len(names),
        label="policy action",
    )
    expected_policy_space = _normalize_space_contract(
        policy_space,
        width=len(names),
        label="action",
    )
    if normalized_policy_space != expected_policy_space:
        raise ValueError(
            "policy action space does not match the environment action space"
        )
    native_space = _normalize_space_contract(
        normalized["native_space"],
        width=len(names),
        label="native action",
    )

    mapping = normalized["mapping"]
    mapping_keys = {
        "schema",
        "transform",
        "expression",
        "componentwise",
        "negative_policy_values",
        "policy_zero_is_no_intervention",
    }
    if not isinstance(mapping, dict) or set(mapping) != mapping_keys:
        raise ValueError("policy action mapping has an invalid key set")
    if mapping["schema"] != POLICY_ACTION_MAPPING_SCHEMA:
        raise ValueError("policy action mapping schema is unsupported")
    if mapping["componentwise"] is not True:
        raise ValueError("policy action mapping must be componentwise")
    if mapping["policy_zero_is_no_intervention"] is not True:
        raise ValueError("policy action zero must mean no intervention")

    interface_id = normalized["interface_id"]
    if interface_id == "native_one_sided_v1":
        expected_mapping = {
            "schema": POLICY_ACTION_MAPPING_SCHEMA,
            "transform": "identity",
            "expression": "native=policy_action",
            "componentwise": True,
            "negative_policy_values": "not_applicable",
            "policy_zero_is_no_intervention": True,
        }
        if mapping != expected_mapping or native_space != normalized_policy_space:
            raise ValueError("native action mapping is internally inconsistent")
    elif interface_id == "symmetric_positive_part_v1":
        expected_mapping = {
            "schema": POLICY_ACTION_MAPPING_SCHEMA,
            "transform": "componentwise_positive_part",
            "expression": "native=max(policy_action,0)",
            "componentwise": True,
            "negative_policy_values": "mapped_to_zero",
            "policy_zero_is_no_intervention": True,
        }
        policy_low = np.asarray(normalized_policy_space["low"], dtype=float)
        policy_high = np.asarray(normalized_policy_space["high"], dtype=float)
        native_low = np.asarray(native_space["low"], dtype=float)
        native_high = np.asarray(native_space["high"], dtype=float)
        if (
            mapping != expected_mapping
            or not np.array_equal(policy_low, -np.ones(len(names)))
            or not np.array_equal(policy_high, np.ones(len(names)))
            or not np.array_equal(native_low, np.zeros(len(names)))
            or not np.array_equal(native_high, np.ones(len(names)))
        ):
            raise ValueError(
                "symmetric positive-part action mapping is internally inconsistent"
            )
    else:
        raise ValueError("policy action interface is unsupported")

    normalized["policy_space"] = normalized_policy_space
    normalized["native_space"] = native_space
    return normalize_json_types(normalized)


def _normalize_observation_preprocessing(
    value: Mapping[str, Any],
    *,
    width: int,
) -> dict[str, Any]:
    """Validate and canonicalize a policy-facing observation transformation."""

    if not isinstance(value, Mapping):
        raise ValueError("observation preprocessing must be a mapping")
    normalized = normalize_json_types(dict(value))
    legacy_keys = {"transform", "centers", "scales"}
    if set(normalized) == legacy_keys:
        if normalized["transform"] != "tanh((raw-center)/scale)":
            raise ValueError("base observation transform is unsupported")
        centers_raw = normalized["centers"]
        scales_raw = normalized["scales"]
        if not isinstance(centers_raw, list) or not isinstance(scales_raw, list):
            raise ValueError("observation centers and scales must be arrays")
        if len(centers_raw) != width or len(scales_raw) != width:
            raise ValueError(
                "observation preprocessing width does not match observation names"
            )
        try:
            centers = np.asarray(centers_raw, dtype=float)
            scales = np.asarray(scales_raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation centers and scales must be numeric") from exc
        if (
            centers.shape != (width,)
            or scales.shape != (width,)
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0.0)
        ):
            raise ValueError("observation centers and scales are invalid")
        return normalize_json_types(
            {
                "transform": "tanh((raw-center)/scale)",
                "centers": centers.tolist(),
                "scales": scales.tolist(),
            }
        )

    history_keys = {
        "schema",
        "transform",
        "history_length",
        "base_observation_size",
        "output_size",
        "layout",
        "base_observation_contract",
        "padding_value",
        "valid_history_mask",
    }
    if set(normalized) != history_keys:
        raise ValueError("observation preprocessing fields are unsupported")
    if normalized["schema"] != OBSERVATION_HISTORY_PREPROCESSING_SCHEMA:
        raise ValueError("observation-history preprocessing schema is unsupported")
    if normalized["transform"] != "masked_history_concatenation":
        raise ValueError("observation-history transform is unsupported")
    history_length = normalized["history_length"]
    base_size = normalized["base_observation_size"]
    output_size = normalized["output_size"]
    if (
        type(history_length) is not int
        or history_length < 1
        or type(base_size) is not int
        or base_size < 1
        or type(output_size) is not int
    ):
        raise ValueError("observation-history dimensions must be positive integers")
    expected_output_size = history_length * base_size + history_length
    if output_size != expected_output_size or width != expected_output_size:
        raise ValueError(
            "observation-history dimensions do not match the policy observation"
        )

    values_stop = history_length * base_size
    expected_layout = {
        "order": "oldest_to_newest_then_valid_history_mask",
        "history_shape": [history_length, base_size],
        "history_values_slice": [0, values_stop],
        "valid_history_mask_slice": [values_stop, output_size],
        "latest_observation_slice": [values_stop - base_size, values_stop],
    }
    if normalized["layout"] != expected_layout:
        raise ValueError("observation-history layout is internally inconsistent")
    padding_value = normalized["padding_value"]
    if (
        isinstance(padding_value, bool)
        or not isinstance(padding_value, (int, float))
        or float(padding_value) != 0.0
    ):
        raise ValueError("observation-history padding value must be zero")
    expected_mask = {
        "transform": "identity",
        "length": history_length,
        "values": [0.0, 1.0],
        "semantics": "0=zero_padding,1=valid_observation",
    }
    if normalized["valid_history_mask"] != expected_mask:
        raise ValueError("observation-history validity-mask contract is invalid")

    base_contract = normalized["base_observation_contract"]
    expected_base_keys = {
        "schema",
        "observation_names",
        "observation_space",
        "preprocessing",
    }
    if not isinstance(base_contract, dict) or set(base_contract) != expected_base_keys:
        raise ValueError("base policy observation contract is malformed")
    if base_contract["schema"] != POLICY_OBSERVATION_CONTRACT_SCHEMA:
        raise ValueError("base policy observation contract schema is unsupported")
    base_names = base_contract["observation_names"]
    if (
        not isinstance(base_names, list)
        or len(base_names) != base_size
        or any(not isinstance(name, str) for name in base_names)
        or len(set(base_names)) != len(base_names)
    ):
        raise ValueError("base policy observation names are invalid")
    base_space = _normalize_space_contract(
        base_contract["observation_space"],
        width=base_size,
        label="base observation",
    )
    base_preprocessing = _normalize_observation_preprocessing(
        base_contract["preprocessing"],
        width=base_size,
    )
    normalized["base_observation_contract"] = {
        "schema": POLICY_OBSERVATION_CONTRACT_SCHEMA,
        "observation_names": list(base_names),
        "observation_space": base_space,
        "preprocessing": base_preprocessing,
    }
    normalized["padding_value"] = 0.0
    return normalize_json_types(normalized)


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
    observation_centers: Sequence[float] | None,
    observation_scales: Sequence[float] | None,
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
    observation_preprocessing: Mapping[str, Any] | None = None,
    policy_action_contract: Mapping[str, Any] | None = None,
    require_checkpoint: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed policy sidecar with artifact and interface hashes."""

    checkpoint = Path(checkpoint_path)
    if require_checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    names = [str(value) for value in observation_names]
    actions = [str(value) for value in action_names]
    if len(set(names)) != len(names):
        raise ValueError("observation names must be unique")
    if observation_preprocessing is None:
        if observation_centers is None or observation_scales is None:
            raise ValueError(
                "observation centers and scales are required for the base transform"
            )
        preprocessing_payload: Mapping[str, Any] = {
            "transform": "tanh((raw-center)/scale)",
            "centers": [float(value) for value in observation_centers],
            "scales": [float(value) for value in observation_scales],
        }
    else:
        if observation_centers is not None or observation_scales is not None:
            raise ValueError(
                "supply either observation preprocessing or centers/scales, not both"
            )
        preprocessing_payload = observation_preprocessing
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
    normalization = _normalize_observation_preprocessing(
        preprocessing_payload,
        width=len(names),
    )
    if policy_action_contract is None:
        policy_action_contract = {
            "schema": POLICY_ACTION_CONTRACT_SCHEMA,
            "interface_id": "native_one_sided_v1",
            "action_names": actions,
            "policy_space": action_space,
            "native_space": action_space,
            "mapping": {
                "schema": POLICY_ACTION_MAPPING_SCHEMA,
                "transform": "identity",
                "expression": "native=policy_action",
                "componentwise": True,
                "negative_policy_values": "not_applicable",
                "policy_zero_is_no_intervention": True,
            },
        }
    action_interface = _normalize_policy_action_contract(
        policy_action_contract,
        action_names=actions,
        policy_space=action_space,
    )
    action_contract = normalize_json_types({
        "names": actions,
        "names_sha256": ordered_contract_sha256(actions),
        "semantics": {name: action_semantics[name] for name in actions},
        "agent_step_min": float(config.agent_step_min),
        "space": action_space,
        "policy_interface": action_interface,
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
    try:
        normalized_preprocessing = _normalize_observation_preprocessing(
            normalization,
            width=len(observation_names),
        )
    except ValueError as exc:
        raise PolicyCompatibilityError(str(exc)) from exc
    if normalized_preprocessing != normalization:
        raise PolicyCompatibilityError(
            "observation_normalization is not canonical"
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
    if (
        not isinstance(action_contract, dict)
        or set(action_contract)
        != {
            "names",
            "names_sha256",
            "semantics",
            "agent_step_min",
            "space",
            "policy_interface",
        }
        or not isinstance(action_contract.get("names"), list)
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
    try:
        normalized_action_interface = _normalize_policy_action_contract(
            action_contract["policy_interface"],
            action_names=action_contract["names"],
            policy_space=action_space,
        )
    except ValueError as exc:
        raise PolicyCompatibilityError(str(exc)) from exc
    if normalized_action_interface != action_contract["policy_interface"]:
        raise PolicyCompatibilityError(
            "policy action contract is not canonical"
        )

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
