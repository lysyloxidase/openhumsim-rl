from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.env import ACTION_NAMES, CLINICAL_OBSERVATION_NAMES, OBSERVATION_NAMES
from openhumsim_rl.physiology import HumanPhysiology, HumanState, Intervention
from openhumsim_rl.respiratory import RespiratoryModel


ZERO_ACTION = np.zeros(len(ACTION_NAMES), dtype=np.float32)


def test_lactate_amount_is_the_source_of_truth_and_fluid_only_dilutes_it():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=2201)
    amount_before = env.state.lactate_amount_mmol
    concentration_before = env.state.lactate_mmol_l
    generated_before = env.state.lactate_generated_mmol
    cleared_before = env.state.lactate_cleared_mmol

    env.model.apply_instant_intervention(
        env.state,
        Intervention(saline_ml=1_000.0, fio2=env.config.baseline_fio2),
    )

    assert env.state.lactate_amount_mmol == pytest.approx(amount_before)
    assert env.state.lactate_generated_mmol == generated_before
    assert env.state.lactate_cleared_mmol == cleared_before
    assert env.state.lactate_mmol_l < concentration_before
    assert env.state.lactate_mmol_l == pytest.approx(
        amount_before / env.state.lactate_distribution_volume_l
    )
    assert abs(env.state.lactate_mass_balance_error_mmol) < 1e-12


def test_lactate_generation_clearance_and_mass_ledger_close_without_clipping():
    config = HumanConfig()
    model = HumanPhysiology(config)
    state = HumanState()
    model.initialize_state(state)
    nonvolatile_before = state.nonvolatile_strong_anion_mEq
    state.vo2_demand_ml_min = 500.0
    state.oxygen_debt_ml_min = 300.0

    for _ in range(20):
        model.energy_metabolism.step_lactate(
            state, exercise=0.8, dt_min=config.integration_step_min
        )

    assert state.lactate_generated_mmol > 0.0
    assert state.lactate_cleared_mmol > 0.0
    assert state.exercise_lactate_production_mmol_min > 0.0
    assert state.hypoxic_lactate_production_mmol_min > 0.0
    assert state.lactate_amount_mmol > state.initial_lactate_amount_mmol
    assert state.lactate_amount_mmol == pytest.approx(
        state.initial_lactate_amount_mmol
        + state.lactate_generated_mmol
        - state.lactate_cleared_mmol,
        abs=1e-10,
    )
    assert abs(state.lactate_mass_balance_error_mmol) < 1e-10
    # Lactate affects acid-base once through its strong-anion concentration;
    # the energy ledger itself must not also add it to the UMA pool.
    assert state.nonvolatile_strong_anion_mEq == nonvolatile_before


def test_oxidative_vco2_uses_achieved_vo2_and_not_unmet_demand():
    config = HumanConfig()
    model = RespiratoryModel(config)
    state = HumanState(
        pao2_mmHg=35.0,
        paco2_mmHg=40.0,
        ph_arterial=7.40,
        cardiac_output_l_min=1.5,
        vo2_demand_ml_min=1_200.0,
    )

    model.update_oxygen_transport(state)
    model.update_metabolic_gas_production(state, exercise=1.0)

    assert state.oxygen_debt_ml_min > 0.0
    assert state.vco2_ml_min == pytest.approx(
        state.metabolic_respiratory_quotient * state.vo2_ml_min
    )
    assert state.oxidative_vco2_ml_min == state.vco2_ml_min
    assert state.vco2_ml_min < state.vco2_demand_ml_min
    assert 0.60 <= state.metabolic_respiratory_quotient <= 1.0

    state.vo2_ml_min = 0.0
    model.update_metabolic_gas_production(state, exercise=1.0)
    assert state.oxidative_vco2_ml_min == 0.0


def test_accumulated_oxygen_deficit_is_a_monotonic_trapezoidal_integral():
    config = HumanConfig()
    model = HumanPhysiology(config)
    state = HumanState()
    model.energy_metabolism.initialize_state(state)

    state.oxygen_debt_ml_min = 100.0
    model.energy_metabolism.accumulate_oxygen_deficit(
        state, previous_deficit_ml_min=0.0, dt_min=0.25
    )
    assert state.cumulative_oxygen_deficit_ml == pytest.approx(12.5)
    first = state.cumulative_oxygen_deficit_ml

    state.oxygen_debt_ml_min = 0.0
    model.energy_metabolism.accumulate_oxygen_deficit(
        state, previous_deficit_ml_min=100.0, dt_min=0.25
    )
    assert state.cumulative_oxygen_deficit_ml == pytest.approx(25.0)
    assert state.cumulative_oxygen_deficit_ml >= first
    # This is exposure accumulated over the episode, not a repayable EPOC pool.
    model.energy_metabolism.accumulate_oxygen_deficit(
        state, previous_deficit_ml_min=0.0, dt_min=10.0
    )
    assert state.cumulative_oxygen_deficit_ml == pytest.approx(25.0)


@pytest.mark.parametrize("fio2", [0.15, 0.21, 0.60])
def test_final_haldane_state_closes_against_the_conserved_carbon_pool(fio2: float):
    config = replace(
        HumanConfig(), baseline_fio2=fio2, max_fio2=max(fio2, 0.60), agent_step_min=1.0
    )
    env = HumanHomeostasisEnv(config=config, observation_profile="full")
    env.reset(seed=2202)
    _, _, terminated, truncated, _ = env.step(ZERO_ACTION)

    assert not terminated and not truncated
    residual = env.model.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
        env.state, fio2=fio2, exercise=0.0
    )
    assert abs(residual) <= max(
        1e-6, 10.0 * config.co2_pool_solver_tolerance_mmol_l
    )
    assert env.state.co2_final_gas_closure_residual_mmol_l == pytest.approx(
        residual
    )
    assert abs(env.state.co2_mass_balance_error_mmol) < 1e-10


def test_energy_carbon_rollout_converges_with_outer_step_refinement():
    def rollout(dt_min: float) -> dict[str, float]:
        config = replace(
            HumanConfig(),
            agent_step_min=5.0,
            integration_step_min=dt_min,
            episode_minutes=10.0,
        )
        env = HumanHomeostasisEnv(config=config, observation_profile="full")
        env.reset(seed=2203)
        action = ZERO_ACTION.copy()
        action[2] = 1.0
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        return info["state"]

    coarse = rollout(0.25)
    fine = rollout(0.05)

    assert coarse["co2_generated_mmol"] == pytest.approx(
        fine["co2_generated_mmol"], rel=0.01
    )
    assert coarse["co2_eliminated_mmol"] == pytest.approx(
        fine["co2_eliminated_mmol"], rel=0.02
    )
    assert coarse["lactate_amount_mmol"] == pytest.approx(
        fine["lactate_amount_mmol"], rel=0.005
    )
    assert coarse["cumulative_oxygen_deficit_ml"] == pytest.approx(
        fine["cumulative_oxygen_deficit_ml"], rel=0.03, abs=1.0
    )
    assert coarse["paco2_mmHg"] == pytest.approx(fine["paco2_mmHg"], abs=0.5)
    assert coarse["ph_arterial"] == pytest.approx(fine["ph_arterial"], abs=0.01)


@pytest.mark.parametrize("hemoglobin_g_dl", [7.0, 3.0])
def test_supply_limited_carbon_ledger_uses_final_vco2_and_elimination_endpoints(
    hemoglobin_g_dl: float,
):
    """A severe O2 limitation must not book the optimistic VO2 predictor."""
    config = HumanConfig()
    model = HumanPhysiology(config)
    state = HumanState()
    model.initialize_state(state)
    blood_volume_l = state.plasma_volume_l + state.rbc_volume_l
    state.hemoglobin_mass_g = hemoglobin_g_dl * blood_volume_l * 10.0
    model._update_blood_composition(state)

    start_vco2 = state.vco2_ml_min
    generated_before = state.co2_generated_mmol
    model.integrate(
        state,
        Intervention(exercise_intensity=1.0, fio2=0.15),
        config.integration_step_min,
    )

    generated = state.co2_generated_mmol - generated_before
    expected_generated = (
        0.5 * (start_vco2 + state.vco2_ml_min)
        * model.blood_gas.gas_mmol_per_ml_stpd
        * config.integration_step_min
    )
    expected_elimination_endpoint = (
        state.paco2_mmHg * state.effective_co2_ventilation_l_min / 0.863
    )
    assert state.oxygen_debt_ml_min > 0.0
    assert generated == pytest.approx(expected_generated, rel=2e-5, abs=1e-8)
    assert state.vco2_elimination_ml_min == pytest.approx(
        expected_elimination_endpoint, rel=2e-5, abs=1e-6
    )
    assert abs(state.co2_mass_balance_error_mmol) < 1e-10
    assert abs(state.co2_final_gas_closure_residual_mmol_l) < 1e-6


def test_v022_reset_contract_and_failure_rollback_cover_energy_ledgers():
    env = HumanHomeostasisEnv(observation_profile="full", info_profile="debug")
    full_obs, first_info = env.reset(seed=2204)
    first_amount = env.state.lactate_amount_mmol
    env.reset(seed=2204)

    assert env.state.lactate_amount_mmol == pytest.approx(first_amount)
    assert env.state.lactate_generated_mmol == 0.0
    assert env.state.lactate_cleared_mmol == 0.0
    assert env.state.cumulative_oxygen_deficit_ml == 0.0
    assert first_info["environment_semantics"]["state_schema_version"] == "0.22"
    assert (
        first_info["environment_semantics"]["reward_profile"]
        == "latent_research_v0.23"
    )
    assert "energy_metabolism" in first_info
    assert full_obs.shape == (len(OBSERVATION_NAMES),)
    assert len(CLINICAL_OBSERVATION_NAMES) == 54
    assert "lactate_amount_mmol" not in CLINICAL_OBSERVATION_NAMES

    state_before = deepcopy(env.state.as_dict())
    energy_fields = (
        "lactate_amount_mmol",
        "lactate_generated_mmol",
        "lactate_cleared_mmol",
        "lactate_mass_balance_error_mmol",
        "instantaneous_oxygen_deficit_ml_min",
        "cumulative_oxygen_deficit_ml",
        "oxidative_vco2_ml_min",
        "vco2_generation_interval_average_ml_min",
    )
    integrate = env.model.integrate

    def fail_after_energy(state, intervention, duration_min):
        integrate(state, intervention, duration_min)
        state.lactate_amount_mmol += 123.0
        state.lactate_generated_mmol += 123.0
        raise FloatingPointError("synthetic post-energy failure")

    env.model.integrate = fail_after_energy
    _, _, terminated, truncated, info = env.step(ZERO_ACTION)
    assert terminated and not truncated
    assert info["termination_reason"] == "numerical_failure_nonfinite_state"
    # The integration snapshot is taken after the zero-valued instant-action
    # refresh; check exact rollback of every newly introduced energy state.
    for name in energy_fields:
        assert getattr(env.state, name) == state_before[name]
    # The instant intervention and first substep are one transaction, so every
    # public state field returns exactly to the pre-action snapshot.
    assert env.state.as_dict() == state_before

    strict = HumanHomeostasisEnv(info_profile="benchmark")
    _, strict_info = strict.reset(seed=2204)
    assert "energy_metabolism" not in strict_info
    assert "mass_balance" not in strict_info
