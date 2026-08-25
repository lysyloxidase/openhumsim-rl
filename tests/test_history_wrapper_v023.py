from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from openhumsim_rl import (
    HumanConfig,
    HumanHomeostasisEnv,
    ObservationHistoryWrapper,
)
from openhumsim_rl.env import (
    ACTION_NAMES,
    BENCHMARK_INFO_KEYS,
    LATENT_REWARD_PROFILE,
    OBSERVABLE_REWARD_PROFILE,
)
from openhumsim_rl.measurement import ClinicalMeasurementConfig


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "validation" / "rl_benchmark_v0.23.py"


def _strict_env(*, episode_minutes: float = 3.0) -> HumanHomeostasisEnv:
    return HumanHomeostasisEnv(
        config=HumanConfig(
            agent_step_min=1.0,
            integration_step_min=0.25,
            episode_minutes=episode_minutes,
        ),
        observation_profile="clinical",
        measurement_profile="realistic",
        info_profile="benchmark",
    )


def _benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "openhumsim_rl_benchmark_v023", BENCHMARK_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_sha256(value) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def test_history_layout_mask_space_and_dtype() -> None:
    env = ObservationHistoryWrapper(_strict_env(), history_length=3)
    observation, info = env.reset(seed=2301)
    width = env.base_observation_size

    assert observation.shape == (3 * width + 3,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert np.array_equal(observation[:2 * width], np.zeros(2 * width))
    assert np.array_equal(
        observation[env.valid_history_mask_slice],
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    first = observation[env.latest_observation_slice].copy()

    observation, _, terminated, truncated, step_info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert not terminated and not truncated
    assert env.observation_space.contains(observation)
    assert np.array_equal(observation[width:2 * width], first)
    assert np.array_equal(
        observation[env.valid_history_mask_slice],
        np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
    )

    observation, _, terminated, truncated, _ = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert not terminated and not truncated
    assert np.array_equal(
        observation[env.valid_history_mask_slice],
        np.ones(3, dtype=np.float32),
    )
    assert info["info_profile"] == step_info["info_profile"] == "benchmark"
    assert len(env.observation_names) == env.observation_space.shape[0]


def test_history_wrapper_is_seed_deterministic_and_preserves_strict_info() -> None:
    first = ObservationHistoryWrapper(_strict_env(), history_length=2)
    second = ObservationHistoryWrapper(_strict_env(), history_length=2)
    first_observation, first_info = first.reset(seed=2302)
    second_observation, second_info = second.reset(seed=2302)
    assert np.array_equal(first_observation, second_observation)
    assert first_info == second_info

    action = np.zeros(first.action_space.shape, dtype=np.float32)
    action[4] = 0.10
    first_step = first.step(action)
    second_step = second.step(action)
    assert np.array_equal(first_step[0], second_step[0])
    assert first_step[1:4] == second_step[1:4]
    assert first_step[4] == second_step[4]

    for info in (first_info, first_step[4]):
        assert set(info).issubset(BENCHMARK_INFO_KEYS)
        assert "state" not in info
        assert "time_min" not in info
        assert "reward_terms" not in info


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_history_wrapper_rejects_invalid_length(invalid) -> None:
    with pytest.raises(ValueError, match="history_length"):
        ObservationHistoryWrapper(_strict_env(), history_length=invalid)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            HumanHomeostasisEnv(
                observation_profile="full",
                measurement_profile="ideal",
                info_profile="benchmark",
            ),
            "observation_profile='clinical'",
        ),
        (
            HumanHomeostasisEnv(
                observation_profile="clinical",
                measurement_profile="ideal",
                info_profile="benchmark",
            ),
            "measurement_profile='realistic'",
        ),
        (HumanHomeostasisEnv(), "info_profile='benchmark'"),
    ],
)
def test_history_wrapper_rejects_non_strict_profiles(
    environment: HumanHomeostasisEnv,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ObservationHistoryWrapper(environment, history_length=2)


def test_history_wrapper_rejects_latent_reward_profile() -> None:
    environment = _strict_env()
    assert environment.reward_profile == OBSERVABLE_REWARD_PROFILE
    environment.reward_profile = LATENT_REWARD_PROFILE

    with pytest.raises(
        ValueError,
        match=f"reward_profile={OBSERVABLE_REWARD_PROFILE!r}",
    ):
        ObservationHistoryWrapper(environment, history_length=2)


def test_history_length_one_keeps_current_observation_and_valid_mask() -> None:
    env = ObservationHistoryWrapper(_strict_env(), history_length=1)
    observation, _ = env.reset(seed=2304)
    assert observation.shape == (env.base_observation_size + 1,)
    assert observation[-1] == 1.0
    assert env.observation_space.contains(observation)


def test_history_runtime_snapshot_json_round_trip_and_resume() -> None:
    source = ObservationHistoryWrapper(_strict_env(episode_minutes=4.0), history_length=3)
    source.reset(seed=2303)
    first_action = np.zeros(source.action_space.shape, dtype=np.float32)
    first_action[4] = 0.05
    source.step(first_action)

    snapshot = source.runtime_snapshot()
    restored_payload = json.loads(json.dumps(snapshot, allow_nan=False))
    assert restored_payload == snapshot
    assert snapshot["base_environment_sha256"] == _canonical_sha256(
        source.env.to_versioned_snapshot()
    )
    binding_payload = dict(snapshot)
    binding_sha256 = binding_payload.pop("snapshot_sha256")
    assert binding_sha256 == _canonical_sha256(binding_payload)

    target_base = _strict_env(episode_minutes=4.0)
    target_base.reset(seed=2303)
    target_base.step(first_action)
    target = ObservationHistoryWrapper(target_base, history_length=3)
    target.restore_runtime_snapshot(restored_payload)

    next_action = np.zeros(source.action_space.shape, dtype=np.float32)
    next_action[5] = 0.05
    source_step = source.step(next_action)
    target_step = target.step(next_action)
    assert np.array_equal(source_step[0], target_step[0])
    assert source_step[1:4] == target_step[1:4]
    assert source_step[4] == target_step[4]

    wrong_length = ObservationHistoryWrapper(
        _strict_env(episode_minutes=4.0), history_length=2
    )
    with pytest.raises(ValueError, match="length"):
        wrong_length.restore_runtime_snapshot(restored_payload)

    changed_normalization_base = _strict_env(episode_minutes=4.0)
    changed_normalization_base._obs_center = (
        changed_normalization_base._obs_center.copy()
    )
    changed_normalization_base._obs_center[0] += 0.01
    changed_normalization = ObservationHistoryWrapper(
        changed_normalization_base, history_length=3
    )
    with pytest.raises(ValueError, match="base contract"):
        changed_normalization.restore_runtime_snapshot(restored_payload)


def test_history_runtime_snapshot_rejects_cross_run_and_corruption_atomically() -> None:
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action[4] = 0.05

    source = ObservationHistoryWrapper(
        _strict_env(episode_minutes=4.0), history_length=3
    )
    source.reset(seed=2313)
    source.step(action)
    source_snapshot = source.runtime_snapshot()

    target = ObservationHistoryWrapper(
        _strict_env(episode_minutes=4.0), history_length=3
    )
    target.reset(seed=2314)
    target.step(action)
    before = target.runtime_snapshot()
    with pytest.raises(ValueError, match="current base environment"):
        target.restore_runtime_snapshot(source_snapshot)
    assert target.runtime_snapshot() == before

    matching = ObservationHistoryWrapper(
        _strict_env(episode_minutes=4.0), history_length=3
    )
    matching.reset(seed=2313)
    matching.step(action)
    before = matching.runtime_snapshot()

    corrupted = json.loads(json.dumps(source_snapshot))
    corrupted["history"][-2][0] += 1e-4
    with pytest.raises(ValueError, match="digest does not match"):
        matching.restore_runtime_snapshot(corrupted)
    assert matching.runtime_snapshot() == before

    wrong_latest = json.loads(json.dumps(source_snapshot))
    wrong_latest["history"][-1][0] += 1e-4
    unsigned = dict(wrong_latest)
    unsigned.pop("snapshot_sha256")
    wrong_latest["snapshot_sha256"] = _canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="latest observation does not match"):
        matching.restore_runtime_snapshot(wrong_latest)
    assert matching.runtime_snapshot() == before

    invalid_mask = json.loads(json.dumps(source_snapshot))
    invalid_mask["history"][-1] = [0.0] * matching.base_observation_size
    invalid_mask["valid_history_mask"][-1] = 0.0
    unsigned = dict(invalid_mask)
    unsigned.pop("snapshot_sha256")
    invalid_mask["snapshot_sha256"] = _canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="latest observation must be valid"):
        matching.restore_runtime_snapshot(invalid_mask)
    assert matching.runtime_snapshot() == before


def test_short_benchmark_writes_hashed_strict_manifest(tmp_path: Path) -> None:
    benchmark = _benchmark_module()
    output = tmp_path / "benchmark.json"
    result = benchmark.run_benchmark(
        output,
        seeds=(2311, 2312),
        scenarios=("respiratory_acidosis",),
        history_length=2,
        episode_minutes=1.0,
    )
    repeated = benchmark.build_benchmark_result(
        seeds=(2311, 2312),
        scenarios=("respiratory_acidosis",),
        history_length=2,
        episode_minutes=1.0,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert repeated == result
    second = tmp_path / "benchmark-second.json"
    benchmark.write_benchmark_result(second, result)
    assert second.read_bytes() == output.read_bytes()

    assert result["schema"] == benchmark.BENCHMARK_SCHEMA
    assert result["training"] == {
        "performed": False,
        "learned_policy": None,
        "training_results_reported": False,
    }
    assert result["clinical_claims"] is False
    assert result["profiles"]["info"] == "benchmark"
    assert result["measurement_config"] == asdict(ClinicalMeasurementConfig())
    assert result["rng"] == {
        "root_seeds": [2311, 2312],
        "episode_root_seed": (
            "the recorded episode seed is passed unchanged to env.reset"
        ),
        "seed_sequence": "numpy.random.SeedSequence(root_seed).spawn(2)",
        "child_stream_order": ["physiology", "measurement"],
        "child_spawn_keys": {"physiology": [0], "measurement": [1]},
        "bit_generator": "PCG64",
        "action_space_seed": "root_seed + 1",
        "seed_none_used": False,
        "policy_randomness": None,
    }
    assert result["evaluation"]["seeds"] == [2311, 2312]
    assert result["evaluation"]["held_out_scenarios"] == [
        "respiratory_acidosis"
    ]
    assert set(result["policies"]) == {"no_op", "observable_heuristic"}

    observation_contract = dict(result["observation_contract"])
    observation_digest = observation_contract.pop("sha256")
    assert observation_digest == benchmark.canonical_sha256(observation_contract)
    action_contract = dict(result["action_contract"])
    action_digest = action_contract.pop("sha256")
    assert action_digest == benchmark.canonical_sha256(action_contract)

    for policy in result["policies"].values():
        assert len(policy["episodes"]) == 2
        assert policy["summary"]["episode_count"] == 2
        for episode in policy["episodes"]:
            assert set(episode) == {
                "scenario",
                "seed",
                "return",
                "steps",
                "duration_min",
                "terminated",
                "truncated",
                "termination_reason",
                "nonzero_action_steps",
            }
            assert np.isfinite(episode["return"])
            assert episode["terminated"] or episode["truncated"]
