from __future__ import annotations

import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.env import CLINICAL_OBSERVATION_NAMES, OBSERVATION_NAMES
from openhumsim_rl.oxygen_binding import OxygenBindingModel
from openhumsim_rl.physiology import HumanPhysiology, HumanState

ZERO = np.zeros(8, dtype=np.float32)


def test_ventilation_action_must_pass_through_pressure_flow_mechanics():
    control = HumanHomeostasisEnv(scenario="hypoventilation", observation_profile="full")
    assist = HumanHomeostasisEnv(scenario="hypoventilation", observation_profile="full")
    control.reset(seed=42); assist.reset(seed=42)
    a = ZERO.copy(); a[5] = 1.0
    _, _, _, _, ic = control.step(ZERO)
    _, _, _, _, ia = assist.step(a)
    sc, sa = ic["state"], ia["state"]
    assert ia["intervention"]["legacy_ventilation_support_l_min"] == 0.0
    assert ia["intervention"]["ventilation_pressure_assist_cmH2O"] > 0.0
    assert sa["respiratory_cycle_peak_airway_pressure_cmH2O"] > sc["respiratory_cycle_peak_airway_pressure_cmH2O"] + 5.0
    assert sa["respiratory_cycle_ventilator_work_j_breath"] > 0.05
    assert sa["pulmonary_intrathoracic_pressure_delta_cmH2O"] > 0.1
    assert sa["tidal_volume_l"] > sc["tidal_volume_l"] + 0.05
    # No hidden L/min source: the reported VA is explained by actual RR, VT,
    # anatomical dead space and ventilation efficiency.
    expected = sa["respiratory_rate_bpm"] * max(0.05, sa["tidal_volume_l"] - assist.config.dead_space_l) * sa["ventilation_efficiency"]
    assert abs(sa["alveolar_ventilation_l_min"] - expected) < 1e-8


def test_bohr_shift_changes_saturation_and_p50_in_correct_direction():
    m = OxygenBindingModel(HumanConfig())
    acid = m.saturation(40.0, 7.20, 40.0)
    normal = m.saturation(40.0, 7.40, 40.0)
    alk = m.saturation(40.0, 7.60, 40.0)
    assert acid.saturation_fraction < normal.saturation_fraction < alk.saturation_fraction
    assert acid.p50_mmHg > normal.p50_mmHg > alk.p50_mmHg
    assert 25.5 < normal.p50_mmHg < 27.5


def test_oxygen_consumption_becomes_supply_dependent_and_exposes_debt():
    c = HumanConfig()
    model = HumanPhysiology(c)
    s = HumanState()
    model.initialize_state(s)
    s.cardiac_output_l_min = 1.5
    s.pao2_mmHg = 40.0
    s.ph_arterial = 7.25
    s.paco2_mmHg = 55.0
    s.vo2_demand_ml_min = 600.0
    model.respiratory.update_oxygen_transport(s)
    assert s.vo2_ml_min <= c.oxygen_max_extraction_fraction * s.oxygen_delivery_ml_min + 1e-8
    assert s.oxygen_debt_ml_min > 100.0
    assert s.aerobic_fraction < 0.5
    assert abs(s.oxygen_extraction_ratio - c.oxygen_max_extraction_fraction) < 1e-9


def test_persistent_oxygen_debt_drives_lactate_upward():
    env = HumanHomeostasisEnv(scenario="baseline", observation_profile="full")
    env.reset(seed=10)
    # Set a previous-step debt to test the explicitly operator-split hypoxic
    # lactate source without claiming a disease-specific lactate model.
    env.state.vo2_demand_ml_min = 500.0
    env.state.oxygen_debt_ml_min = 300.0
    l0 = env.state.lactate_mmol_l
    env.model._substep(env.state, env._decode_action(ZERO), env.config.integration_step_min)
    assert env.state.hypoxic_lactate_production_mmol_l_min > 0.0
    assert env.state.lactate_mmol_l > l0


def test_regional_dead_space_reduces_effective_co2_ventilation():
    base = HumanHomeostasisEnv(scenario="baseline", observation_profile="full")
    vq = HumanHomeostasisEnv(scenario="vq_mismatch", observation_profile="full")
    base.reset(seed=42); vq.reset(seed=42)
    ib = iv = None
    for _ in range(6):
        _, _, tb, trb, ib = base.step(ZERO)
        _, _, tv, trv, iv = vq.step(ZERO)
        assert not (tb or trb or tv or trv)
    sb, sv = ib["state"], iv["state"]
    assert sv["pulmonary_alveolar_dead_space_fraction"] > sb["pulmonary_alveolar_dead_space_fraction"]
    assert sv["effective_co2_ventilation_l_min"] < sv["alveolar_ventilation_l_min"]
    assert sv["paco2_mmHg"] > sb["paco2_mmHg"] + 0.1


def test_fluid_shift_changes_hct_hb_but_conserves_hb_mass():
    base = HumanHomeostasisEnv(scenario="baseline", observation_profile="full")
    saline = HumanHomeostasisEnv(scenario="saline_challenge_30ml_kg", observation_profile="full")
    dry = HumanHomeostasisEnv(scenario="dehydrated", observation_profile="full")
    _, ib = base.reset(seed=1); _, isal = saline.reset(seed=1); _, idry = dry.reset(seed=1)
    b, s, d = ib["state"], isal["state"], idry["state"]
    assert abs(s["hemoglobin_mass_g"] - b["hemoglobin_mass_g"]) < 1e-9
    assert abs(d["hemoglobin_mass_g"] - b["hemoglobin_mass_g"]) < 1e-9
    assert s["hemoglobin_g_dl"] < b["hemoglobin_g_dl"] < d["hemoglobin_g_dl"]
    assert s["hematocrit_fraction"] < b["hematocrit_fraction"] < d["hematocrit_fraction"]
    assert s["arterial_o2_content_ml_dl"] < b["arterial_o2_content_ml_dl"]


def test_reduced_charge_closure_counts_divalent_carbonate():
    env = HumanHomeostasisEnv(observation_profile="full")
    _, info = env.reset(seed=3)
    s = info["state"]
    expected = (
        s["bicarbonate_mmol_l"]
        + 2.0 * s["carbonate_mmol_l"]
        + s["albumin_charge_mEq_l"]
        + s["phosphate_charge_mEq_l"]
    )
    assert abs(s["strong_ion_difference_effective_mEq_l"] - expected) < 1e-8
    assert abs(s["charge_balance_residual_mEq_l"]) < 1e-6


def test_default_policy_observation_hides_latent_mechanistic_state():
    measurement_only = {
        "sensor_glucose_mg_dl",
        "cgm_measurement_age_min", "monitor_measurement_age_min",
        "blood_gas_measurement_age_min", "chemistry_measurement_age_min",
        "hemodynamic_measurement_age_min",
    }
    assert (set(CLINICAL_OBSERVATION_NAMES) - measurement_only).issubset(set(OBSERVATION_NAMES))
    for hidden in (
        "pulmonary_vq_log_sd",
        "pulmonary_hpv_resistance_multiplier",
        "dalla_egp_mg_kg_min",
        "probe_effect_site_mg_l",
        "oxygen_debt_ml_min",
    ):
        assert hidden not in CLINICAL_OBSERVATION_NAMES
    env = HumanHomeostasisEnv()
    obs, info = env.reset(seed=5)
    assert obs.shape == (len(CLINICAL_OBSERVATION_NAMES),)
    assert info["observation_profile"] == "clinical"


def test_reward_explicitly_penalizes_resp_irregularity_and_tissue_oxygen_debt():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=5)
    _, clean = env._reward(ZERO)
    env.state.respiratory_ventilator_asynchrony_index_pct = 60.0
    env.state.respiratory_cycle_auto_peep_cmH2O = 8.0
    env.state.pulmonary_overdistension_fraction = 0.75
    env.state.oxygen_debt_ml_min = 250.0
    bad_total, bad = env._reward(ZERO)
    clean_total = sum(clean.values())
    assert bad["ventilator_asynchrony"] < clean["ventilator_asynchrony"]
    assert bad["auto_peep"] < clean["auto_peep"]
    assert bad["overdistension"] < clean["overdistension"]
    assert bad["oxygen_debt"] < clean["oxygen_debt"]
    assert bad_total < clean_total
