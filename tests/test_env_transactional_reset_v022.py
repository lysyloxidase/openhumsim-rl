from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.env import ACTION_NAMES


ZERO_ACTION = np.zeros(len(ACTION_NAMES), dtype=np.float32)


def _plain_runtime(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tolist())
    if isinstance(value, dict):
        return {key: _plain_runtime(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_runtime(item) for item in value)
    return value


def test_reset_closes_existing_carbon_pool_and_seeds_physical_elimination_endpoint():
    config = replace(
        HumanConfig(), baseline_fio2=0.15, max_fio2=0.60
    )
    env = HumanHomeostasisEnv(
        config=config,
        scenario="respiratory_acidosis",
        observation_profile="full",
    )
    env.reset(seed=77)
    state = env.state

    residual = (
        env.model.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
            state, fio2=config.baseline_fio2, exercise=0.0
        )
    )
    coupled_tolerance = max(
        1e-6, 10.0 * config.co2_pool_solver_tolerance_mmol_l
    )
    assert abs(residual) <= coupled_tolerance
    assert state.co2_final_gas_closure_residual_mmol_l == pytest.approx(
        residual, abs=1e-12
    )

    alveolar_dead = float(np.clip(
        state.pulmonary_alveolar_dead_space_fraction, 0.0, 0.80
    ))
    effective_va = state.alveolar_ventilation_l_min * (1.0 - alveolar_dead)
    assert state.effective_co2_ventilation_l_min == pytest.approx(effective_va)
    assert state.vco2_elimination_ml_min == pytest.approx(
        state.paco2_mmHg * effective_va / 0.863
    )
    pulmonary_check = env.model.pulmonary_exchange.estimate_arterial_oxygen(
        state,
        pco2_mmHg=state.paco2_mmHg,
        fio2=config.baseline_fio2,
        exercise=0.0,
        dt_min=0.0,
        apply=False,
    )
    assert pulmonary_check.pao2_mmHg == pytest.approx(
        state.pao2_mmHg, abs=1e-4
    )
    assert pulmonary_check.effective_respiratory_exchange_ratio == pytest.approx(
        np.clip(state.vco2_elimination_ml_min / state.vo2_ml_min, 0.50, 2.00)
    )

    # Reset reconciliation is equilibrium-only: no flux ledger may advance.
    assert state.exchangeable_co2_pool_mmol == pytest.approx(
        state.initial_exchangeable_co2_pool_mmol
    )
    assert state.co2_generated_mmol == 0.0
    assert state.co2_eliminated_mmol == 0.0
    assert state.co2_urinary_bicarbonate_loss_mmol == 0.0
    assert state.co2_mass_balance_error_mmol == 0.0


@pytest.mark.parametrize(
    "failure_mode", ["raise", "nonfinite", "runtime_nonfinite"]
)
def test_instant_intervention_numerical_failure_rolls_back_state_and_runtime(
    failure_mode: str,
):
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=2205)
    state_before = env.state.as_dict()
    runtime_before = _plain_runtime(env.model.runtime_snapshot())

    def fail_instantly(state, intervention):
        state.glucose_mg_dl += 123.0
        env.model.cardiovascular._cardiac_phase += 0.25
        env.model.respiratory_cycle._last_end_expiratory_volume_l += 2.0
        if failure_mode == "raise":
            raise FloatingPointError("synthetic instant-action failure")
        if failure_mode == "nonfinite":
            state.sodium_mmol_l = float("nan")
        else:
            env.model.cardiovascular._cardiac_phase = float("nan")
        return state

    env.model.apply_instant_intervention = fail_instantly
    obs, reward, terminated, truncated, info = env.step(ZERO_ACTION)

    assert terminated and not truncated
    assert info["termination_reason"] == "numerical_failure_nonfinite_state"
    assert env.elapsed_minutes == 0.0
    assert env.state.as_dict() == state_before
    assert _plain_runtime(env.model.runtime_snapshot()) == runtime_before
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(ZERO_ACTION)


def test_unexpected_instant_intervention_exception_rolls_back_before_reraising():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=2206)
    state_before = env.state.as_dict()
    runtime_before = _plain_runtime(env.model.runtime_snapshot())

    def fail_instantly(state, intervention):
        state.glucose_mg_dl += 123.0
        env.model.cardiovascular._cardiac_phase += 0.25
        env.model.respiratory_cycle._last_end_expiratory_volume_l += 2.0
        raise RuntimeError("synthetic programming failure")

    env.model.apply_instant_intervention = fail_instantly
    with pytest.raises(RuntimeError, match="synthetic programming failure"):
        env.step(ZERO_ACTION)

    assert env.elapsed_minutes == 0.0
    assert env.state.as_dict() == state_before
    assert _plain_runtime(env.model.runtime_snapshot()) == runtime_before
    assert not env._needs_reset


def test_first_integration_failure_rolls_back_the_pending_instant_action():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=2208)
    state_before = env.state.as_dict()
    runtime_before = _plain_runtime(env.model.runtime_snapshot())
    action = ZERO_ACTION.copy()
    action[3] = 0.5  # saline bolus
    action[7] = 0.5  # oral PBPK dose

    def fail_first_substep(state, intervention, duration_min):
        state.glucose_mg_dl += 123.0
        env.model.cardiovascular._cardiac_phase += 0.25
        env.model.respiratory_cycle._last_end_expiratory_volume_l += 2.0
        raise FloatingPointError("synthetic first-substep failure")

    env.model.integrate = fail_first_substep
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated and not truncated
    assert info["termination_reason"] == "numerical_failure_nonfinite_state"
    assert env.elapsed_minutes == 0.0
    assert env.state.as_dict() == state_before
    assert _plain_runtime(env.model.runtime_snapshot()) == runtime_before
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


def test_finite_terminal_state_from_instant_action_skips_integration_safely():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=2207)

    def cross_threshold(state, intervention):
        state.glucose_mg_dl = env.config.glucose_max_terminate + 1.0
        return state

    def integration_must_not_run(state, intervention, duration_min):
        raise AssertionError("integration ran after an instant terminal state")

    env.model.apply_instant_intervention = cross_threshold
    env.model.integrate = integration_must_not_run
    obs, reward, terminated, truncated, info = env.step(ZERO_ACTION)

    assert terminated and not truncated
    assert info["termination_reason"] == "extreme_hyperglycemia"
    assert env.elapsed_minutes == 0.0
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
