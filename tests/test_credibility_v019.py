from dataclasses import replace
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.physiology import Intervention


ZERO = np.zeros(8, dtype=np.float32)


def test_sc_insulin_is_delayed_and_mass_conserving():
    env = HumanHomeostasisEnv()
    env.reset(seed=1)
    s = env.state
    i0 = s.insulin_uU_ml
    env.model.metabolism.add_exogenous_insulin(s, 2.0)
    assert abs(s.insulin_uU_ml - i0) < 1e-12
    rates = []
    for minute in range(1, 181):
        env.model.metabolism.step(s, exercise=0.0, dt_min=1.0)
        rates.append(s.sc_insulin_absorption_model_units_min)
    peak_min = int(np.argmax(rates)) + 1
    assert 80 <= peak_min <= 100
    assert abs(s.sc_insulin_mass_balance_error_model_units) < 1e-10
    assert s.sc_insulin_absorbed_model_units > 0.5


def _hypoglycemia_run(counterreg_max):
    cfg = replace(
        HumanConfig(),
        glucagon_counterreg_egp_max_mg_kg_min=counterreg_max,
    )
    env = HumanHomeostasisEnv(config=cfg)
    env.reset(seed=2)
    p = env.model.metabolism.p
    env.state.dalla_gp_mg_kg = 50.0 * p.VG_dl_kg
    env.model.metabolism._refresh_outputs(env.state)
    glycogen0 = env.state.liver_glycogen_g
    for _ in range(6):
        env.model.integrate(env.state, env._decode_action(ZERO), 5.0)
    return env.state, glycogen0


def test_glucagon_counterregulation_raises_egp_and_uses_glycogen():
    off, _ = _hypoglycemia_run(0.0)
    on, glycogen0 = _hypoglycemia_run(1.0)
    assert on.glucose_mg_dl > off.glucose_mg_dl + 1.0
    assert on.glucagon_counterregulatory_egp_mg_kg_min > 0.05
    assert on.liver_glycogen_g < glycogen0
    assert on.glucagon_counterregulatory_glucose_released_mg > 0.0


def test_free_water_moves_into_both_compartments_and_conserves_tbw():
    env = HumanHomeostasisEnv()
    env.reset(seed=3)
    s = env.state
    tbw0, ecf0, icf0, na0 = s.total_body_water_l, s.ecf_volume_l, s.icf_volume_l, s.sodium_mmol_l
    env.model.apply_instant_intervention(s, Intervention(oral_water_ml=900.0))
    assert abs(s.total_body_water_l - (tbw0 + 0.9)) < 1e-10
    assert s.ecf_volume_l > ecf0
    assert s.icf_volume_l > icf0
    assert s.sodium_mmol_l < na0
    assert abs(s.ecf_effective_tonicity_mOsm_l - s.icf_effective_tonicity_mOsm_l) < 1e-8


def test_isotonic_saline_stays_predominantly_ecf():
    env = HumanHomeostasisEnv()
    env.reset(seed=4)
    s = env.state
    ecf0, icf0 = s.ecf_volume_l, s.icf_volume_l
    env.model.apply_instant_intervention(s, Intervention(saline_ml=900.0))
    ecf_gain = s.ecf_volume_l - ecf0
    icf_gain = s.icf_volume_l - icf0
    assert ecf_gain > 0.75
    assert abs(icf_gain) < 0.20
    assert abs(s.ecf_effective_tonicity_mOsm_l - s.icf_effective_tonicity_mOsm_l) < 1e-8


def _k_shift(insulin=6.0, ph=7.40, exercise=0.0):
    env = HumanHomeostasisEnv()
    env.reset(seed=5)
    s = env.state
    total0 = s.ecf_potassium_mmol + s.icf_potassium_mmol
    s.insulin_uU_ml = insulin
    s.ph_arterial = ph
    env.model.renal._transcellular_potassium_step(s, exercise=exercise, dt=10.0)
    env.model.renal._update_derived_concentrations(s)
    total1 = s.ecf_potassium_mmol + s.icf_potassium_mmol
    return s, total0, total1


def test_transcellular_k_direction_and_conservation():
    basal, t0, t1 = _k_shift()
    high_insulin, ti0, ti1 = _k_shift(insulin=80.0)
    acid, ta0, ta1 = _k_shift(ph=7.20)
    exercise, te0, te1 = _k_shift(exercise=1.0)

    assert high_insulin.potassium_mmol_l < basal.potassium_mmol_l
    assert acid.potassium_mmol_l > basal.potassium_mmol_l
    assert exercise.potassium_mmol_l > basal.potassium_mmol_l
    for a, b in [(t0,t1),(ti0,ti1),(ta0,ta1),(te0,te1)]:
        assert abs(a-b) < 1e-10


def test_default_clinical_observation_profile_does_not_expose_latent_p1_states():
    env = HumanHomeostasisEnv()
    assert "sc_insulin_depot1_model_units" not in env.observation_names
    assert "potassium_transcellular_target_mmol_l" not in env.observation_names
    full = HumanHomeostasisEnv(observation_profile="full")
    assert "sc_insulin_depot1_model_units" in full.observation_names
    assert "potassium_transcellular_target_mmol_l" in full.observation_names


def test_scenario_initial_conditions_do_not_double_count_fluid_administration():
    env = HumanHomeostasisEnv(scenario="saline_challenge_30ml_kg")
    _, info = env.reset(seed=9)
    mb = info["mass_balance"]
    assert abs(mb["water_mass_balance_error_l"]) < 1e-12
    assert abs(mb["sodium_mass_balance_error_mmol"]) < 1e-12
    assert abs(mb["chloride_mass_balance_error_mmol"]) < 1e-12
    assert abs(mb["water_partition_residual_l"]) < 1e-12
