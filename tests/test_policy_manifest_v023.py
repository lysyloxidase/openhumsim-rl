from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from openhumsim_rl import (
    HumanConfig,
    HumanHomeostasisEnv,
    ObservationHistoryWrapper,
    SymmetricActionHumanEnv,
)
from openhumsim_rl.env import LATENT_REWARD_PROFILE, OBSERVABLE_REWARD_PROFILE
from openhumsim_rl.measurement import ClinicalMeasurementConfig
from openhumsim_rl.policy_manifest import (
    POLICY_MANIFEST_CONTRACT_KEYS,
    POLICY_MANIFEST_SCHEMA,
    PolicyCompatibilityError,
    canonical_json,
    validate_policy_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _training_module():
    path = ROOT / "examples" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("policy_manifest_v023", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compatibility_contract(manifest: dict) -> dict:
    return {key: manifest[key] for key in POLICY_MANIFEST_CONTRACT_KEYS}


def _benchmark_env(
    *, measurement_config: ClinicalMeasurementConfig | None = None
) -> HumanHomeostasisEnv:
    return HumanHomeostasisEnv(
        scenario="oral_glucose_75g",
        observation_profile="clinical",
        measurement_profile="realistic",
        measurement_config=measurement_config,
        info_profile="benchmark",
    )


def _install_fake_sb3(monkeypatch: pytest.MonkeyPatch) -> type:
    class FakePPO:
        @classmethod
        def load(cls, path, env=None):
            checkpoint = Path(path)
            return {
                "path": checkpoint,
                "payload": checkpoint.read_bytes(),
                "env": env,
            }

    module = ModuleType("stable_baselines3")
    module.PPO = FakePPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", module)
    return FakePPO


def test_policy_manifest_locks_training_artifact_and_complete_contract(tmp_path: Path):
    training = _training_module()
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"synthetic-stable-baselines-checkpoint")
    manifest_path = tmp_path / training.MANIFEST_FILENAME
    provenance = training.capture_training_provenance()

    training.write_training_manifest(
        manifest_path,
        config=HumanConfig(),
        checkpoint_path=checkpoint,
        require_checkpoint=True,
        provenance=provenance,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"] == POLICY_MANIFEST_SCHEMA
    assert len(manifest["checkpoint_sha256"]) == 64
    assert manifest["training"]["seed"] == training.TRAINING_SEED
    assert manifest["training"]["total_timesteps"] == training.TOTAL_TIMESTEPS
    assert manifest["training"]["algorithm"] == training.ALGORITHM
    assert manifest["training"]["hyperparameters"] == training.PPO_HYPERPARAMETERS
    assert manifest["action_contract"]["agent_step_min"] == 5.0
    assert manifest["action_contract"]["space"] == {
        "shape": [8],
        "dtype": "float32",
        "low": [0.0] * 8,
        "high": [1.0] * 8,
    }
    assert manifest["action_contract"]["policy_interface"] == {
        "schema": "openhumsim.policy-action-contract.v1",
        "interface_id": "native_one_sided_v1",
        "action_names": list(manifest["action_contract"]["names"]),
        "policy_space": manifest["action_contract"]["space"],
        "native_space": manifest["action_contract"]["space"],
        "mapping": {
            "schema": "openhumsim.policy-action-mapping.v1",
            "transform": "identity",
            "expression": "native=policy_action",
            "componentwise": True,
            "negative_policy_values": "not_applicable",
            "policy_zero_is_no_intervention": True,
        },
    }
    assert manifest["observation_space"] == {
        "shape": [54],
        "dtype": "float32",
        "low": [-1.0] * 54,
        "high": [1.0] * 54,
    }
    assert manifest["measurement_config"]["abg_result_delay_min"] == 7.0
    assert len(manifest["action_contract"]["names_sha256"]) == 64
    assert len(manifest["action_contract_sha256"]) == 64
    assert len(manifest["observation_normalization_sha256"]) == 64
    assert len(manifest["observation_space_sha256"]) == 64
    assert len(manifest["source_fingerprint_sha256"]) == 64
    assert set(manifest["runtime"]) == {
        "python", "numpy", "stable_baselines3", "torch"
    }

    validated = validate_policy_manifest(
        manifest_path,
        checkpoint_path=checkpoint,
        expected=_compatibility_contract(manifest),
    )
    assert validated == manifest

    with pytest.raises(PolicyCompatibilityError, match="incomplete"):
        validate_policy_manifest(
            manifest_path,
            checkpoint_path=checkpoint,
            expected={},
        )


def test_policy_manifest_fails_closed_for_artifact_or_environment_mismatch(
    tmp_path: Path,
):
    training = _training_module()
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"checkpoint-v1")
    manifest_path = tmp_path / training.MANIFEST_FILENAME
    provenance = training.capture_training_provenance()
    training.write_training_manifest(
        manifest_path,
        checkpoint_path=checkpoint,
        require_checkpoint=True,
        provenance=provenance,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _compatibility_contract(manifest)

    bad_expected = dict(expected)
    bad_expected["reward_profile"] = "different-reward"
    with pytest.raises(PolicyCompatibilityError, match="reward_profile"):
        validate_policy_manifest(
            manifest_path,
            checkpoint_path=checkpoint,
            expected=bad_expected,
        )

    checkpoint.write_bytes(b"checkpoint-v2")
    with pytest.raises(PolicyCompatibilityError, match="SHA-256"):
        validate_policy_manifest(
            manifest_path,
            checkpoint_path=checkpoint,
            expected=expected,
        )


def test_policy_manifest_requires_the_declared_checkpoint(tmp_path: Path):
    training = _training_module()
    missing = tmp_path / "missing.zip"
    with pytest.raises(FileNotFoundError):
        training.build_training_manifest(
            checkpoint_path=missing,
            require_checkpoint=True,
        )

    missing.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="captured before training"):
        training.write_training_manifest(
            tmp_path / "manifest.json",
            checkpoint_path=missing,
            require_checkpoint=True,
        )


def test_loader_uses_actual_environment_measurement_and_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    training = _training_module()
    _install_fake_sb3(monkeypatch)
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"valid-policy-snapshot")
    manifest_path = tmp_path / training.MANIFEST_FILENAME
    source_env = _benchmark_env()
    provenance = training.capture_training_provenance()
    training.write_training_manifest(
        manifest_path,
        environment=source_env,
        checkpoint_path=checkpoint,
        require_checkpoint=True,
        provenance=provenance,
    )

    loaded = training.load_policy_checked(
        checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        env=_benchmark_env(),
    )
    assert loaded["payload"] == b"valid-policy-snapshot"

    wrong_scenario = HumanHomeostasisEnv(
        scenario="baseline",
        observation_profile="clinical",
        measurement_profile="realistic",
        info_profile="benchmark",
    )
    with pytest.raises(PolicyCompatibilityError, match="scenario"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_scenario,
        )

    wrong_measurement_profile = HumanHomeostasisEnv(
        scenario="oral_glucose_75g",
        observation_profile="clinical",
        measurement_profile="ideal",
        info_profile="benchmark",
    )
    with pytest.raises(PolicyCompatibilityError, match="measurement_profile"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_measurement_profile,
        )

    wrong_info_profile = HumanHomeostasisEnv(
        scenario="oral_glucose_75g",
        observation_profile="clinical",
        measurement_profile="realistic",
        info_profile="debug",
        reward_profile=OBSERVABLE_REWARD_PROFILE,
    )
    with pytest.raises(PolicyCompatibilityError, match="info_profile"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_info_profile,
        )

    wrong_reward_profile = _benchmark_env()
    wrong_reward_profile.reward_profile = LATENT_REWARD_PROFILE
    with pytest.raises(PolicyCompatibilityError, match="reward_profile"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_reward_profile,
        )

    wrong_measurement = _benchmark_env(
        measurement_config=ClinicalMeasurementConfig(noise_multiplier=2.0)
    )
    with pytest.raises(PolicyCompatibilityError, match="measurement_config"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_measurement,
        )

    wrong_normalization = _benchmark_env()
    wrong_normalization._obs_scale = np.array(
        wrong_normalization._obs_scale, copy=True
    )
    wrong_normalization._obs_scale[0] *= 2.0
    with pytest.raises(PolicyCompatibilityError, match="observation_normalization"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_normalization,
        )

    wrong_observation_space = _benchmark_env()
    observation_space_type = type(wrong_observation_space.observation_space)
    wrong_observation_space.observation_space = observation_space_type(
        low=np.full(54, -5.0, dtype=np.float64),
        high=np.full(54, 5.0, dtype=np.float64),
        dtype=np.float64,
    )
    with pytest.raises(PolicyCompatibilityError, match="observation_space"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_observation_space,
        )

    wrong_action_space = _benchmark_env()
    action_space_type = type(wrong_action_space.action_space)
    wrong_action_space.action_space = action_space_type(
        low=np.full(8, -1.0, dtype=np.float64),
        high=np.full(8, 2.0, dtype=np.float64),
        dtype=np.float64,
    )
    with pytest.raises(PolicyCompatibilityError, match="action_contract"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_action_space,
        )


def test_history_wrapper_manifest_locks_layout_and_loader_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _training_module()
    _install_fake_sb3(monkeypatch)
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"history-policy-checkpoint")
    manifest_path = tmp_path / training.MANIFEST_FILENAME
    source = ObservationHistoryWrapper(_benchmark_env(), history_length=3)
    provenance = training.capture_training_provenance()

    training.write_training_manifest(
        manifest_path,
        environment=source,
        checkpoint_path=checkpoint,
        require_checkpoint=True,
        provenance=provenance,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preprocessing = manifest["observation_normalization"]

    assert manifest["observation_space"]["shape"] == [165]
    assert len(manifest["observation_names"]) == 165
    assert preprocessing["history_length"] == 3
    assert preprocessing["base_observation_size"] == 54
    assert preprocessing["output_size"] == 165
    assert preprocessing["layout"]["valid_history_mask_slice"] == [162, 165]
    assert preprocessing["valid_history_mask"] == {
        "transform": "identity",
        "length": 3,
        "values": [0.0, 1.0],
        "semantics": "0=zero_padding,1=valid_observation",
    }

    malformed = deepcopy(manifest)
    malformed["observation_normalization"]["layout"][
        "latest_observation_slice"
    ][0] -= 1
    malformed["observation_normalization_sha256"] = sha256(
        canonical_json(malformed["observation_normalization"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PolicyCompatibilityError, match="layout"):
        validate_policy_manifest(
            malformed,
            checkpoint_path=checkpoint,
            expected=_compatibility_contract(malformed),
        )

    matching = ObservationHistoryWrapper(_benchmark_env(), history_length=3)
    loaded = training.load_policy_checked(
        checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        env=matching,
    )
    assert loaded["payload"] == b"history-policy-checkpoint"
    assert loaded["env"] is matching

    wrong_length = ObservationHistoryWrapper(_benchmark_env(), history_length=2)
    with pytest.raises(PolicyCompatibilityError, match="policy contract mismatch"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=wrong_length,
        )


def test_symmetric_action_manifest_locks_positive_part_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _training_module()
    _install_fake_sb3(monkeypatch)
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"symmetric-policy-checkpoint")
    manifest_path = tmp_path / training.MANIFEST_FILENAME
    source = SymmetricActionHumanEnv(
        scenario="oral_glucose_75g",
        observation_profile="clinical",
        measurement_profile="realistic",
        info_profile="benchmark",
    )
    provenance = training.capture_training_provenance()
    training.write_training_manifest(
        manifest_path,
        environment=source,
        checkpoint_path=checkpoint,
        require_checkpoint=True,
        provenance=provenance,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interface = manifest["action_contract"]["policy_interface"]

    assert manifest["action_contract"]["space"]["low"] == [-1.0] * 8
    assert interface["interface_id"] == "symmetric_positive_part_v1"
    assert interface["mapping"] == {
        "schema": "openhumsim.policy-action-mapping.v1",
        "transform": "componentwise_positive_part",
        "expression": "native=max(policy_action,0)",
        "componentwise": True,
        "negative_policy_values": "mapped_to_zero",
        "policy_zero_is_no_intervention": True,
    }
    assert interface["native_space"]["low"] == [0.0] * 8
    assert interface["native_space"]["high"] == [1.0] * 8

    matching = SymmetricActionHumanEnv(
        scenario="oral_glucose_75g",
        observation_profile="clinical",
        measurement_profile="realistic",
        info_profile="benchmark",
    )
    loaded = training.load_policy_checked(
        checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        env=matching,
    )
    assert loaded["payload"] == b"symmetric-policy-checkpoint"

    class SameBoxDifferentMapping(SymmetricActionHumanEnv):
        def policy_action_contract(self):
            contract = super().policy_action_contract()
            contract["interface_id"] = "native_one_sided_v1"
            contract["native_space"] = contract["policy_space"]
            contract["mapping"] = {
                "schema": "openhumsim.policy-action-mapping.v1",
                "transform": "identity",
                "expression": "native=policy_action",
                "componentwise": True,
                "negative_policy_values": "not_applicable",
                "policy_zero_is_no_intervention": True,
            }
            return contract

    different_mapping = SameBoxDifferentMapping(
        scenario="oral_glucose_75g",
        observation_profile="clinical",
        measurement_profile="realistic",
        info_profile="benchmark",
    )
    assert different_mapping.action_space.shape == matching.action_space.shape
    assert np.array_equal(
        different_mapping.action_space.low,
        matching.action_space.low,
    )
    with pytest.raises(PolicyCompatibilityError, match="action_contract"):
        training.load_policy_checked(
            checkpoint_path=checkpoint,
            manifest_path=manifest_path,
            env=different_mapping,
        )


def test_provenance_and_derived_contract_mutations_fail_closed(tmp_path: Path):
    training = _training_module()
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"checkpoint")
    manifest = training.build_training_manifest(
        environment=_benchmark_env(),
        checkpoint_path=checkpoint,
        require_checkpoint=True,
    )
    expected = _compatibility_contract(manifest)

    mutations = []

    changed = deepcopy(manifest)
    changed["training"]["seed"] += 1
    mutations.append((changed, "training"))

    changed = deepcopy(manifest)
    changed["source_commit_sha"] = "0" * 40
    mutations.append((changed, "source_commit_sha"))

    changed = deepcopy(manifest)
    del changed["source_commit_sha"]
    mutations.append((changed, "incomplete"))

    changed = deepcopy(manifest)
    changed["source_tree_dirty"] = not bool(changed["source_tree_dirty"])
    mutations.append((changed, "source_tree_dirty"))

    changed = deepcopy(manifest)
    changed["source_fingerprint_sha256"] = "f" * 64
    mutations.append((changed, "source_fingerprint_sha256"))

    changed = deepcopy(manifest)
    changed["runtime"]["python"] = "0.0.0"
    mutations.append((changed, "runtime"))

    changed = deepcopy(manifest)
    changed["measurement_config"]["noise_multiplier"] = 99.0
    mutations.append((changed, "measurement_config"))

    changed = deepcopy(manifest)
    changed["observation_normalization"]["centers"][0] += 1.0
    mutations.append((changed, "normalization"))

    changed = deepcopy(manifest)
    changed["observation_space"]["low"][0] -= 1.0
    mutations.append((changed, "observation-space hash"))

    changed = deepcopy(manifest)
    changed["action_contract"]["space"]["high"][0] += 1.0
    mutations.append((changed, "action-contract hash"))

    for changed_manifest, match in mutations:
        with pytest.raises(PolicyCompatibilityError, match=match):
            validate_policy_manifest(
                changed_manifest,
                checkpoint_path=checkpoint,
                expected=expected,
            )


def test_training_provenance_is_captured_before_and_rechecked(
    monkeypatch: pytest.MonkeyPatch,
):
    training = _training_module()
    captured = training.capture_training_provenance()
    training.assert_training_provenance_unchanged(captured)

    changed = deepcopy(captured)
    changed["source_fingerprint_sha256"] = "f" * 64
    monkeypatch.setattr(training, "capture_training_provenance", lambda: changed)
    with pytest.raises(RuntimeError, match="source_fingerprint_sha256"):
        training.assert_training_provenance_unchanged(captured)


def test_loader_hashes_and_loads_the_same_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    training = _training_module()
    _install_fake_sb3(monkeypatch)
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    original_payload = b"validated-checkpoint"
    checkpoint.write_bytes(original_payload)
    manifest_path = tmp_path / training.MANIFEST_FILENAME
    provenance = training.capture_training_provenance()
    training.write_training_manifest(
        manifest_path,
        environment=_benchmark_env(),
        checkpoint_path=checkpoint,
        require_checkpoint=True,
        provenance=provenance,
    )

    original_validate = training.validate_policy_manifest

    def validate_then_replace(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        checkpoint.write_bytes(b"replacement-after-validation")
        return result

    monkeypatch.setattr(training, "validate_policy_manifest", validate_then_replace)
    loaded = training.load_policy_checked(
        checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        env=_benchmark_env(),
    )

    assert loaded["payload"] == original_payload
    assert loaded["path"] != checkpoint
    assert not loaded["path"].exists()
    assert checkpoint.read_bytes() == b"replacement-after-validation"


def test_json_tuple_types_round_trip_without_loader_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    training = _training_module()
    monkeypatch.setitem(
        training.PPO_HYPERPARAMETERS,
        "policy_kwargs",
        {"net_arch": (64, 64)},
    )
    checkpoint = tmp_path / training.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"checkpoint")
    manifest_path = tmp_path / training.MANIFEST_FILENAME
    provenance = training.capture_training_provenance()
    training.write_training_manifest(
        manifest_path,
        environment=_benchmark_env(),
        checkpoint_path=checkpoint,
        require_checkpoint=True,
        provenance=provenance,
    )
    from_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    in_memory = training.build_training_manifest(
        environment=_benchmark_env(),
        checkpoint_path=checkpoint,
        require_checkpoint=True,
    )

    assert from_disk["training"]["hyperparameters"]["policy_kwargs"][
        "net_arch"
    ] == [64, 64]
    assert from_disk == in_memory
    validate_policy_manifest(
        manifest_path,
        checkpoint_path=checkpoint,
        expected=_compatibility_contract(in_memory),
    )
