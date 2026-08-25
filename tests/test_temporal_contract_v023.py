from __future__ import annotations

import json

import numpy as np
import pytest

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.env import (
    ACTION_NAMES,
    CLINICAL_OBSERVATION_NAMES,
    LATENT_REWARD_PROFILE,
    OBSERVABLE_REWARD_PROFILE,
)
from openhumsim_rl.measurement import (
    ClinicalMeasurementConfig,
    ClinicalMeasurementModel,
)
from openhumsim_rl.physiology import HumanState


def _continuous_rollout(agent_step_min: float) -> dict[str, object]:
    episode_minutes = 10.0
    env = HumanHomeostasisEnv(
        config=HumanConfig(
            agent_step_min=agent_step_min,
            integration_step_min=0.25,
            episode_minutes=episode_minutes,
        ),
        observation_profile="full",
        info_profile="debug",
        reward_profile=LATENT_REWARD_PROFILE,
    )
    env.reset(seed=2301)
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action[2] = 0.8  # continuous exercise intensity, not a per-decision bolus

    total_reward = 0.0
    total_intervention_cost = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        total_intervention_cost += env._last_reward_terms["intervention_cost"]
        assert not terminated
        if truncated:
            break

    _, continuous_cost = env._intervention_cost_components(action)
    return {
        "reward": total_reward,
        "intervention_cost": total_intervention_cost,
        "expected_intervention_cost": -continuous_cost
        * episode_minutes
        / 5.0,
        "state": env.state.as_dict(),
    }


def test_continuous_intervention_return_is_agent_step_invariant():
    five_min = _continuous_rollout(5.0)
    two_and_half_min = _continuous_rollout(2.5)

    assert five_min["intervention_cost"] == pytest.approx(
        five_min["expected_intervention_cost"], abs=1e-12
    )
    assert two_and_half_min["intervention_cost"] == pytest.approx(
        two_and_half_min["expected_intervention_cost"], abs=1e-12
    )
    assert five_min["reward"] == pytest.approx(
        two_and_half_min["reward"], abs=5e-4
    )


def _cgm_delivery_count(agent_step_min: float) -> tuple[int, int]:
    env = HumanHomeostasisEnv(
        config=HumanConfig(
            agent_step_min=agent_step_min,
            integration_step_min=0.25,
            episode_minutes=10.0,
        ),
        measurement_config=ClinicalMeasurementConfig(
            monitor_dropout_probability=0.0,
            cgm_dropout_probability=0.0,
            cgm_sample_interval_min=5.0,
            cgm_relative_noise_sd=0.0,
            noise_multiplier=0.0,
        ),
        info_profile="benchmark",
    )
    observation, _ = env.reset(seed=2302)
    assert observation.shape == (len(CLINICAL_OBSERVATION_NAMES),) == (54,)
    zero_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    while True:
        _, _, terminated, truncated, _ = env.step(zero_action)
        assert not terminated
        if truncated:
            break
    assert env.measurement_model is not None
    cgm = env.measurement_model.diagnostics()["groups"]["cgm"]
    return int(cgm["delivered"]), int(cgm["dropped"])


def test_cgm_delivery_count_uses_sensor_cadence_not_agent_step():
    assert _cgm_delivery_count(5.0) == (3, 0)
    assert _cgm_delivery_count(2.5) == (3, 0)


def test_benchmark_reward_is_a_function_of_public_observation_not_hidden_state():
    config = HumanConfig(
        agent_step_min=1.0,
        integration_step_min=0.25,
        episode_minutes=1.0,
    )
    first = HumanHomeostasisEnv(config=config, info_profile="benchmark")
    second = HumanHomeostasisEnv(config=config, info_profile="benchmark")
    first.reset(seed=2306)
    second.reset(seed=2306)
    second.state.oxygen_debt_ml_min += 800.0
    second.state.pulmonary_overdistension_fraction = 0.75
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)

    np.testing.assert_array_equal(first._get_obs(), second._get_obs())
    assert first._reward(action)[0] != pytest.approx(second._reward(action)[0])
    exercise_action = action.copy()
    exercise_action[2] = 1.0
    assert first._observable_reward(action)[0] != pytest.approx(
        first._observable_reward(exercise_action)[0]
    )

    def hold_state(state, intervention, duration_min):
        return state

    first.model.integrate = hold_state
    second.model.integrate = hold_state
    first_obs, first_reward, _, first_truncated, first_info = first.step(action)
    second_obs, second_reward, _, second_truncated, second_info = second.step(action)

    np.testing.assert_array_equal(first_obs, second_obs)
    assert first_truncated and second_truncated
    assert first_reward == second_reward
    assert (
        first_info["environment_semantics"]["reward_profile"]
        == second_info["environment_semantics"]["reward_profile"]
        == OBSERVABLE_REWARD_PROFILE
    )
    assert (
        first_info["environment_semantics"]["reward_observability"]
        == second_info["environment_semantics"]["reward_observability"]
        == "public_observation_action_and_transition_events"
    )


def test_benchmark_info_fails_closed_for_latent_reward_profile():
    with pytest.raises(ValueError, match="observable benchmark reward"):
        HumanHomeostasisEnv(
            info_profile="benchmark",
            reward_profile=LATENT_REWARD_PROFILE,
        )


class _AlternatingPanelRng:
    def __init__(self):
        self._values = iter((0.1, 0.9, 0.9, 0.1))

    def random(self) -> float:
        return float(next(self._values))

    def normal(self, _mean: float, _sd: float) -> float:
        return 0.0


def test_abg_and_chemistry_use_coherent_panel_dropout_events():
    state = HumanState()
    model = ClinicalMeasurementModel(
        ClinicalMeasurementConfig(
            monitor_dropout_probability=0.0,
            cgm_dropout_probability=0.0,
            cgm_sample_interval_min=5.0,
            abg_interval_min=5.0,
            abg_result_delay_min=0.0,
            abg_dropout_probability=0.5,
            chemistry_interval_min=5.0,
            chemistry_result_delay_min=0.0,
            chemistry_dropout_probability=0.5,
            cgm_relative_noise_sd=0.0,
            noise_multiplier=0.0,
        )
    )
    rng = _AlternatingPanelRng()
    model.initialize(state, rng)  # type: ignore[arg-type]

    model.advance(state, time_min=5.0, dt_min=5.0, rng=rng)  # type: ignore[arg-type]
    channels = model.diagnostics()["channels"]
    abg = [item for item in channels.values() if item["group"] == "abg"]
    chemistry = [
        item for item in channels.values() if item["group"] == "chemistry"
    ]
    assert {(item["dropped"], item["delivered"]) for item in abg} == {(1, 1)}
    assert {
        (item["dropped"], item["delivered"], item["sample_time_min"])
        for item in chemistry
    } == {(0, 2, 5.0)}

    model.advance(state, time_min=10.0, dt_min=5.0, rng=rng)  # type: ignore[arg-type]
    channels = model.diagnostics()["channels"]
    for group in ("abg", "chemistry"):
        group_items = [
            item for item in channels.values() if item["group"] == group
        ]
        assert len(
            {
                (
                    item["dropped"],
                    item["delivered"],
                    item["sample_time_min"],
                )
                for item in group_items
            }
        ) == 1


def test_measurement_runtime_snapshot_is_json_round_trippable_and_exact():
    state = HumanState()
    model = ClinicalMeasurementModel(
        ClinicalMeasurementConfig(
            monitor_dropout_probability=0.0,
            cgm_dropout_probability=0.0,
            cgm_sample_interval_min=5.0,
            abg_interval_min=5.0,
            abg_result_delay_min=7.0,
            chemistry_interval_min=5.0,
            chemistry_result_delay_min=12.0,
            cgm_relative_noise_sd=0.0,
            noise_multiplier=0.0,
        )
    )
    rng = np.random.default_rng(2303)
    model.initialize(state, rng)
    model.advance(state, time_min=5.0, dt_min=5.0, rng=rng)
    snapshot = json.loads(json.dumps(model.runtime_snapshot()))
    assert any(
        item["pending_results"] for item in snapshot["channels"].values()
    )

    model.advance(state, time_min=10.0, dt_min=5.0, rng=rng)
    model.restore_runtime_snapshot(snapshot)
    assert model.runtime_snapshot() == snapshot


def test_measurement_draws_do_not_perturb_future_reset_jitter():
    config = HumanConfig(episode_minutes=5.0)
    realistic = HumanHomeostasisEnv(config=config)
    ideal = HumanHomeostasisEnv(
        config=config,
        observation_profile="full",
        measurement_profile="ideal",
    )

    realistic.reset(seed=2304)
    ideal.reset(seed=2304)
    assert realistic.state.as_dict() == ideal.state.as_dict()

    # Realistic initialization consumes many measurement draws. A subsequent
    # unseeded reset must still derive identical physiology jitter from the
    # untouched Gym master stream.
    realistic.reset()
    ideal.reset()
    assert realistic.state.as_dict() == ideal.state.as_dict()


def test_failed_first_substep_rolls_back_bolus_cost_and_terminal_is_one_event():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=2305)
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action[3] = 0.5

    def fail_first_substep(state, intervention, duration_min):
        raise FloatingPointError("synthetic first-substep failure")

    env.model.integrate = fail_first_substep
    _, reward, terminated, truncated, _ = env.step(action)

    assert terminated and not truncated
    assert env._last_reward_terms["intervention_cost"] == 0.0
    assert env._last_reward_terms["terminal"] == -10.0
    assert reward == pytest.approx(-10.0)


def test_pk_target_reward_terms_remain_stable_when_thresholds_are_crossed():
    config = HumanConfig(
        agent_step_min=0.25,
        integration_step_min=0.25,
        episode_minutes=0.25,
    )
    env = HumanHomeostasisEnv(
        config=config,
        scenario="pk_target",
        observation_profile="full",
        info_profile="debug",
        reward_profile=LATENT_REWARD_PROFILE,
    )
    env.reset(seed=2307)
    env.state.probe_effect_site_mg_l = config.pbpk_target_effect_site_mg_l
    env.state.probe_plasma_mg_l = 0.0
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)

    start_terms = env._reward(action)[1]
    assert start_terms["pk_in_target"] == pytest.approx(0.40)
    assert start_terms["pk_high_exposure"] == pytest.approx(0.0)

    def cross_both_thresholds(state, intervention, duration_min):
        del intervention, duration_min
        state.probe_effect_site_mg_l = 2.0 * config.pbpk_target_effect_site_mg_l
        state.probe_plasma_mg_l = 2.0 * config.pbpk_high_exposure_mg_l
        return state

    env.model.integrate = cross_both_thresholds
    _, reward, terminated, truncated, _ = env.step(action)

    assert np.isfinite(reward)
    assert not terminated and truncated
    assert env.elapsed_minutes == pytest.approx(config.episode_minutes)
    assert env._last_reward_terms["pk_in_target"] == pytest.approx(0.01)
    assert env._last_reward_terms["pk_high_exposure"] < 0.0
