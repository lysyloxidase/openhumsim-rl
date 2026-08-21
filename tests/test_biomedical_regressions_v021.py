from __future__ import annotations

import numpy as np
import pytest

from openhumsim_rl import HumanConfig
from openhumsim_rl.pbpk import ReferencePBPKModel
from openhumsim_rl.physiology import HumanPhysiology, HumanState
from openhumsim_rl.respiratory import RespiratoryIntervention, RespiratoryModel


def _initialized_physiology() -> tuple[HumanConfig, HumanPhysiology, HumanState]:
    config = HumanConfig()
    model = HumanPhysiology(config)
    state = HumanState()
    model.initialize_state(state)
    return config, model, state


def test_renal_acid_compensation_is_not_counted_again_as_chloride_loss():
    config, model, state = _initialized_physiology()
    state.ph_arterial = 7.20
    dt = 0.25
    acid_pool_before = state.nonvolatile_strong_anion_mEq
    generated_before = state.nonvolatile_acid_generated_mEq
    excreted_before = state.nonvolatile_acid_excreted_mEq

    model.renal.step(state, exercise=0.0, dt=dt)

    gross_acid_excretion = (
        state.urine_ammonium_mmol_min
        + state.urine_titratable_acid_mmol_min
    )
    expected_chloride = np.clip(
        config.baseline_urine_chloride_mmol_min
        * state.urine_sodium_mmol_min
        / config.baseline_urine_sodium_mmol_min,
        0.002,
        0.90,
    )
    generated = config.endogenous_acid_production_mmol_min * dt
    removed = min(acid_pool_before + generated, gross_acid_excretion * dt)

    assert state.urine_ammonium_mmol_min > (
        config.baseline_net_acid_excretion_mmol_min
        * config.baseline_ammonium_fraction_of_nae
    )
    assert state.urine_chloride_mmol_min == pytest.approx(expected_chloride)
    assert state.nonvolatile_acid_generated_mEq - generated_before == pytest.approx(
        generated
    )
    assert state.nonvolatile_acid_excreted_mEq - excreted_before == pytest.approx(
        removed
    )
    assert state.nonvolatile_strong_anion_mEq == pytest.approx(
        acid_pool_before + generated - removed
    )


def test_bicarbonaturia_and_nonvolatile_acid_use_distinct_ledgers():
    config, model, state = _initialized_physiology()
    state.ph_arterial = 7.60
    dt = 0.25
    acid_pool_before = state.nonvolatile_strong_anion_mEq
    acid_excreted_before = state.nonvolatile_acid_excreted_mEq
    co2_bicarbonate_loss_before = state.co2_urinary_bicarbonate_loss_mmol

    model.renal.step(state, exercise=0.0, dt=dt)
    gross_acid_excretion = (
        state.urine_ammonium_mmol_min
        + state.urine_titratable_acid_mmol_min
    )
    urinary_bicarbonate = state.urine_bicarbonate_mmol_min

    assert urinary_bicarbonate > 0.0
    assert state.renal_acid_excretion_mmol_min == pytest.approx(
        gross_acid_excretion - urinary_bicarbonate
    )
    assert state.nonvolatile_acid_excreted_mEq - acid_excreted_before == pytest.approx(
        min(
            acid_pool_before + config.endogenous_acid_production_mmol_min * dt,
            gross_acid_excretion * dt,
        )
    )

    model.blood_gas.step_arterial_carbon_balance(
        state, fio2=config.baseline_fio2, exercise=0.0, dt_min=dt
    )
    assert (
        state.co2_urinary_bicarbonate_loss_mmol
        - co2_bicarbonate_loss_before
    ) == pytest.approx(urinary_bicarbonate * dt)


def test_arterial_co2_upper_bound_keeps_hh_snapshot_consistent_after_clipping():
    config, model, state = _initialized_physiology()
    # Force a carbon content above the representable range so the root finder
    # selects its upper PaCO2 boundary.
    state.exchangeable_co2_pool_mmol = (
        1_000.0 * model.blood_gas.exchangeable_volume_l(state)
    )
    model.blood_gas.step_arterial_carbon_balance(
        state, fio2=config.baseline_fio2, exercise=0.0, dt_min=0.0
    )
    assert state.paco2_mmHg == pytest.approx(
        model.blood_gas.ARTERIAL_PCO2_MAX_MMHG
    )

    HumanPhysiology._clip_state(state)
    expected_bicarbonate = model.acid_base.bicarbonate_from_ph_pco2(
        state.ph_arterial, state.paco2_mmHg
    )
    assert state.bicarbonate_mmol_l == pytest.approx(expected_bicarbonate)
    assert abs(state.henderson_hasselbalch_residual) < 1e-10


def test_oxygen_supply_transition_is_continuous_and_respects_both_limits():
    ceiling = 500.0
    width = HumanConfig().oxygen_supply_transition_width_fraction * ceiling
    smooth = RespiratoryModel._smooth_supply_limited_consumption

    assert smooth(ceiling - 2.0 * width, ceiling, width) == pytest.approx(
        ceiling - 2.0 * width
    )
    assert smooth(ceiling + 2.0 * width, ceiling, width) == pytest.approx(ceiling)

    at_switch = smooth(ceiling, ceiling, width)
    assert 0.0 < at_switch < ceiling
    epsilon = 1e-4
    left_slope = (
        at_switch - smooth(ceiling - epsilon, ceiling, width)
    ) / epsilon
    right_slope = (
        smooth(ceiling + epsilon, ceiling, width) - at_switch
    ) / epsilon
    assert left_slope == pytest.approx(right_slope, abs=1e-5)

    for demand in np.linspace(0.0, 2.0 * ceiling, 101):
        achieved = smooth(demand, ceiling, width)
        assert 0.0 <= achieved <= min(demand, ceiling) + 1e-12

    state = HumanState(
        pao2_mmHg=45.0,
        paco2_mmHg=40.0,
        ph_arterial=7.40,
        cardiac_output_l_min=2.0,
        vo2_demand_ml_min=500.0,
    )
    model = RespiratoryModel(HumanConfig())
    model.update_oxygen_transport(state)
    expected_margin = (
        model.cfg.oxygen_max_extraction_fraction
        * state.oxygen_delivery_ml_min
        - state.vo2_demand_ml_min
    )
    assert state.oxygen_supply_margin_ml_min == pytest.approx(expected_margin)
    assert state.oxygen_supply_margin_ml_min < 0.0
    assert state.oxygen_debt_ml_min > 0.0


def test_respiratory_drive_adds_hypoxic_stimulation_and_hypocapnic_suppression():
    config = HumanConfig()
    model = RespiratoryModel(config)
    intervention = RespiratoryIntervention(
        fio2=config.baseline_fio2,
        ventilation_support_l_min=0.0,
    )

    def response(pao2: float, paco2: float, ph: float) -> tuple[float, float]:
        state = HumanState(
            pao2_mmHg=pao2,
            paco2_mmHg=paco2,
            ph_arterial=ph,
            respiratory_rate_bpm=config.baseline_rr_bpm,
            respiratory_drive_target_tidal_volume_l=0.50,
        )
        model.update_mechanics(
            state,
            intervention,
            exercise=0.0,
            dt=config.respiratory_tau_min,
        )
        return (
            state.respiratory_rate_bpm,
            state.respiratory_drive_target_tidal_volume_l,
        )

    baseline = response(95.0, 40.0, 7.40)
    hypoxemic = response(45.0, 40.0, 7.40)
    hypocapnic_alkalemic = response(95.0, 25.0, 7.55)

    assert hypoxemic[0] > baseline[0]
    assert hypoxemic[1] > baseline[1]
    assert hypocapnic_alkalemic[0] < baseline[0]
    assert hypocapnic_alkalemic[1] < baseline[1]


def test_pbpk_systemic_hepatic_elimination_is_withdrawn_from_liver():
    config = HumanConfig()
    model = ReferencePBPKModel(config)
    state = HumanState(gfr_ml_min=0.0)
    equilibrium_concentration = 0.50
    state.probe_plasma_mg = equilibrium_concentration * state.plasma_volume_l
    for _, amount_attr, volume_attr, _, kp_attr in model._TISSUES:
        setattr(
            state,
            amount_attr,
            equilibrium_concentration
            * getattr(config, volume_attr)
            * getattr(config, kp_attr),
        )
    state.probe_administered_mg = sum(
        getattr(state, attr)
        for attr in (
            "probe_plasma_mg",
            "probe_liver_mg",
            "probe_kidney_mg",
            "probe_muscle_mg",
            "probe_adipose_mg",
            "probe_rest_mg",
        )
    )

    plasma_before = state.probe_plasma_mg
    liver_before = state.probe_liver_mg
    dt = config.pbpk_internal_step_min
    model.step(state, exercise=0.0, dt=dt)

    expected_hepatic_loss = (
        config.pbpk_fraction_unbound
        * config.pbpk_hepatic_clint_l_min
        * equilibrium_concentration
        * dt
    )
    assert state.probe_eliminated_hepatic_mg == pytest.approx(
        expected_hepatic_loss
    )
    assert state.probe_liver_mg == pytest.approx(
        liver_before - expected_hepatic_loss
    )
    assert state.probe_plasma_mg == pytest.approx(plasma_before)
    assert state.probe_mass_balance_error_mg == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_clip_state_rejects_nonfinite_numerical_state(nonfinite: float):
    state = HumanState()
    state.paco2_mmHg = nonfinite
    with pytest.raises(FloatingPointError, match="paco2_mmHg"):
        HumanPhysiology._clip_state(state)
