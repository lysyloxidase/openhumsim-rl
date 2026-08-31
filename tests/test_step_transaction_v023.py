from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.env import ACTION_NAMES


ZERO_ACTION = np.zeros(len(ACTION_NAMES), dtype=np.float32)


def _realistic_environment() -> HumanHomeostasisEnv:
    config = HumanConfig(agent_step_min=0.25, integration_step_min=0.25)
    env = HumanHomeostasisEnv(
        config=config,
        observation_profile="clinical",
        measurement_profile="realistic",
    )
    env.reset(seed=23001)
    return env


def _assert_substep_runtime_rolled_back(
    before: dict,
    after: dict,
) -> None:
    before_runtime = before["runtime"]
    after_runtime = after["runtime"]
    assert after_runtime["state"] == before_runtime["state"]
    assert after_runtime["physiology"] == before_runtime["physiology"]
    assert after_runtime["measurement"] == before_runtime["measurement"]
    assert after_runtime["rng"] == before_runtime["rng"]
    assert after_runtime["elapsed_minutes"] == before_runtime["elapsed_minutes"]


def test_unexpected_integration_exception_rolls_back_before_reraising() -> None:
    env = _realistic_environment()
    before = deepcopy(env.to_versioned_snapshot())

    def fail_after_partial_mutation(state, intervention, duration_min):
        del intervention, duration_min
        state.glucose_mg_dl += 123.0
        env.model.cardiovascular._cardiac_phase += 0.25
        raise RuntimeError("synthetic integration programming failure")

    env.model.integrate = fail_after_partial_mutation
    with pytest.raises(RuntimeError, match="integration programming failure"):
        env.step(ZERO_ACTION)

    assert env.to_versioned_snapshot() == before


def test_late_unexpected_exception_rolls_back_the_complete_agent_step() -> None:
    env = HumanHomeostasisEnv(
        config=HumanConfig(agent_step_min=0.5, integration_step_min=0.25),
        observation_profile="clinical",
        measurement_profile="realistic",
    )
    env.reset(seed=23003)
    before = deepcopy(env.to_versioned_snapshot())
    action = ZERO_ACTION.copy()
    action[3] = 0.5
    action[7] = 0.5
    original_integrate = env.model.integrate
    call_count = 0

    def fail_during_second_substep(state, intervention, duration_min):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            state.glucose_mg_dl += 123.0
            env.model.cardiovascular._cardiac_phase += 0.25
            raise RuntimeError("late integration programming failure")
        return original_integrate(state, intervention, duration_min)

    env.model.integrate = fail_during_second_substep
    with pytest.raises(RuntimeError, match="late integration programming failure"):
        env.step(action)

    assert call_count == 2
    assert env.to_versioned_snapshot() == before


def test_output_construction_exception_rolls_back_the_complete_agent_step() -> None:
    env = HumanHomeostasisEnv(
        config=HumanConfig(agent_step_min=0.5, integration_step_min=0.25),
        observation_profile="clinical",
        measurement_profile="realistic",
    )
    env.reset(seed=23004)
    before = deepcopy(env.to_versioned_snapshot())
    action = ZERO_ACTION.copy()
    action[3] = 0.5

    def fail_info(**kwargs):
        del kwargs
        raise RuntimeError("transition info programming failure")

    env._get_info = fail_info
    with pytest.raises(RuntimeError, match="transition info programming failure"):
        env.step(action)

    assert env.to_versioned_snapshot() == before


def test_nonfinite_private_integration_runtime_is_a_controlled_failure() -> None:
    env = _realistic_environment()
    before = deepcopy(env.to_versioned_snapshot())

    def poison_private_runtime(state, intervention, duration_min):
        del intervention, duration_min
        env.model.cardiovascular._cardiac_phase = float("nan")
        return state

    env.model.integrate = poison_private_runtime
    observation, reward, terminated, truncated, info = env.step(ZERO_ACTION)

    assert terminated and not truncated
    assert info["termination_reason"] == "numerical_failure_nonfinite_state"
    assert reward == pytest.approx(-10.0)
    assert np.all(np.isfinite(observation))
    after = env.to_versioned_snapshot()
    _assert_substep_runtime_rolled_back(before, after)
    assert after["runtime"]["needs_reset"] is True


def test_measurement_exception_rolls_back_clock_runtime_rng_and_pending_bolus() -> None:
    env = _realistic_environment()
    assert env.measurement_model is not None
    before = deepcopy(env.to_versioned_snapshot())
    action = ZERO_ACTION.copy()
    action[3] = 0.5
    action[7] = 0.5

    def fail_measurement(state, time_min, dt_min, rng, **kwargs):
        del state, dt_min, kwargs
        env.measurement_model.current_time_min = float(time_min)
        first_channel = next(iter(env.measurement_model.channels.values()))
        first_channel.value += 123.0
        rng.random()
        raise RuntimeError("synthetic measurement programming failure")

    env.measurement_model.advance = fail_measurement
    with pytest.raises(RuntimeError, match="measurement programming failure"):
        env.step(action)

    assert env.to_versioned_snapshot() == before


def test_finite_state_reward_overflow_rolls_back_as_a_controlled_failure() -> None:
    env = HumanHomeostasisEnv(
        config=HumanConfig(agent_step_min=0.25, integration_step_min=0.25),
        observation_profile="full",
    )
    env.reset(seed=23002)
    before = deepcopy(env.to_versioned_snapshot())

    def create_finite_reward_overflow(state, intervention, duration_min):
        del intervention, duration_min
        state.glucose_mg_dl = 1e308
        return state

    env.model.integrate = create_finite_reward_overflow
    observation, reward, terminated, truncated, info = env.step(ZERO_ACTION)

    assert terminated and not truncated
    assert info["termination_reason"] == "numerical_failure_nonfinite_state"
    assert reward == pytest.approx(-10.0)
    assert np.all(np.isfinite(observation))
    after = env.to_versioned_snapshot()
    _assert_substep_runtime_rolled_back(before, after)
    assert after["runtime"]["needs_reset"] is True
