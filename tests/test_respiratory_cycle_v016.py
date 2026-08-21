from __future__ import annotations

import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv

ZERO = np.zeros(8, dtype=np.float32)


def rollout(scenario: str, steps: int = 6, seed: int = 42, config: HumanConfig | None = None):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    env.reset(seed=seed)
    info = None
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(ZERO)
        if terminated or truncated:
            break
    return env, info["state"]


def test_baseline_dynamic_cycle_is_physically_closed():
    env, s = rollout("baseline")
    assert 0.40 <= s["tidal_volume_l"] <= 0.65
    assert 8.0 <= s["respiratory_rate_bpm"] <= 18.0
    assert s["respiratory_cycle_auto_peep_cmH2O"] < 0.15
    assert abs(s["respiratory_cycle_equation_residual_cmH2O"]) < 1e-9
    assert 35.0 <= s["paco2_mmHg"] <= 48.0
    assert len(env.model.respiratory_cycle.last_trace["time_s"]) > 100


def test_obstruction_increases_time_constant_resistive_work_and_auto_peep():
    _, b = rollout("baseline")
    _, o = rollout("airway_obstruction")
    assert o["respiratory_cycle_time_constant_s"] > 4.0 * b["respiratory_cycle_time_constant_s"]
    assert o["respiratory_cycle_auto_peep_cmH2O"] > 0.5
    assert o["respiratory_cycle_resistive_work_j_breath"] > 2.0 * b["respiratory_cycle_resistive_work_j_breath"]
    assert o["respiratory_cycle_muscle_work_j_breath"] > 1.4 * b["respiratory_cycle_muscle_work_j_breath"]
    assert o["respiratory_cycle_expiratory_flow_limited_fraction"] > 0.0
    assert o["respiratory_cycle_flow_limiting_pressure_cmH2O"] > 0.5


def test_tachypnea_worsens_dynamic_hyperinflation_in_same_obstruction():
    _, o = rollout("airway_obstruction")
    _, t = rollout("tachypnea_airway_obstruction")
    assert t["respiratory_rate_bpm"] >= 23.0
    assert t["respiratory_cycle_auto_peep_cmH2O"] > o["respiratory_cycle_auto_peep_cmH2O"] + 1.0
    assert t["respiratory_cycle_dynamic_hyperinflation_l"] > o["respiratory_cycle_dynamic_hyperinflation_l"] + 0.08


def test_copd_scale_peepi_anchor_is_reproduced_by_obstruction_challenge():
    # Dal Vecchio et al. 1990: stable COPD PEEPi 2.4 +/- 1.6 cmH2O.
    _, s = rollout("tachypnea_airway_obstruction")
    assert 0.8 <= s["respiratory_cycle_auto_peep_cmH2O"] <= 4.0


def test_pressure_control_transfers_work_from_muscle_to_ventilator():
    _, s = rollout("pressure_control_ventilation")
    assert s["respiratory_cycle_peak_muscle_pressure_cmH2O"] < 1e-6
    assert 14.0 <= s["respiratory_cycle_peak_airway_pressure_cmH2O"] <= 16.0
    assert s["respiratory_cycle_ventilator_work_j_breath"] > 0.20
    assert s["respiratory_cycle_auto_peep_cmH2O"] < 0.25


def test_pressure_control_obstruction_can_still_generate_auto_peep():
    _, s = rollout("pressure_control_obstruction")
    assert s["respiratory_cycle_auto_peep_cmH2O"] > 1.0
    assert s["paco2_mmHg"] > 45.0


def test_intrinsic_peep_feeds_end_expiratory_transpulmonary_pressure():
    _, b = rollout("baseline")
    _, o = rollout("tachypnea_airway_obstruction")
    delta_pl = o["pulmonary_transpulmonary_pressure_end_exp_cmH2O"] - b["pulmonary_transpulmonary_pressure_end_exp_cmH2O"]
    assert abs(delta_pl - o["respiratory_cycle_auto_peep_cmH2O"]) < 0.15
    assert abs(o["pulmonary_pressure_identity_residual_cmH2O"]) < 1e-9


def test_pv_loop_has_nonzero_hysteretic_area():
    _, b = rollout("baseline")
    _, o = rollout("airway_obstruction")
    assert b["respiratory_cycle_pv_hysteresis_j_breath"] > 0.0
    assert o["respiratory_cycle_pv_hysteresis_j_breath"] > b["respiratory_cycle_pv_hysteresis_j_breath"]


def test_cycle_timestep_convergence():
    coarse = HumanConfig(respiratory_cycle_dt_s=0.01)
    fine = HumanConfig(respiratory_cycle_dt_s=0.005)
    _, a = rollout("tachypnea_airway_obstruction", config=coarse)
    _, b = rollout("tachypnea_airway_obstruction", config=fine)
    assert abs(a["tidal_volume_l"] - b["tidal_volume_l"]) < 0.03
    assert abs(a["respiratory_cycle_auto_peep_cmH2O"] - b["respiratory_cycle_auto_peep_cmH2O"]) < 0.20
    assert abs(a["respiratory_cycle_muscle_work_j_breath"] - b["respiratory_cycle_muscle_work_j_breath"]) < 0.08


def test_conservation_regression_with_dynamic_cycle():
    _, s = rollout("pressure_control_obstruction")
    assert abs(s["co2_mass_balance_error_mmol"]) < 1e-7
    assert abs(s["charge_balance_residual_mEq_l"]) < 1e-6
    assert abs(s["cv_blood_volume_error_ml"]) < 1e-6
