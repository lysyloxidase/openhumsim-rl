from __future__ import annotations
import numpy as np
from openhumsim_rl import HumanHomeostasisEnv, HumanConfig

ZERO = np.zeros(8, dtype=np.float32)

def roll(scenario: str, steps: int = 2, seed: int = 2, config=None):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    env.reset(seed=seed)
    info = None
    for _ in range(steps):
        _, _, term, trunc, info = env.step(ZERO)
        if term or trunc:
            break
    return env, info["state"]

def test_synchronous_pressure_support_has_low_asynchrony():
    _, s = roll("pressure_support_synchronous")
    assert s["respiratory_ventilator_asynchrony_index_pct"] < 10.0
    assert s["respiratory_ventilator_mean_trigger_delay_s"] < 0.10
    assert 30.0 <= s["paco2_mmHg"] <= 50.0

def test_intrinsic_peep_can_cause_ineffective_triggering_and_external_peep_unloads_it():
    # Compare the same first 5-min analysis window. With the v0.22 coupled
    # alveolar-gas/RER model this deliberately severe obstruction can cross the
    # hypoxemia safety boundary during a second window; a partial terminal
    # window is not a comparable breath-asynchrony denominator.
    _, bad = roll("pressure_support_ineffective_trigger", steps=1)
    _, peep = roll("pressure_support_ineffective_trigger_peep", steps=1)
    assert bad["respiratory_ventilator_ineffective_trigger_fraction"] > 0.30
    assert peep["respiratory_ventilator_ineffective_trigger_fraction"] < 0.10
    assert peep["respiratory_ventilator_asynchrony_index_pct"] < bad["respiratory_ventilator_asynchrony_index_pct"]

def test_higher_cycling_fraction_reduces_delayed_cycling_and_auto_peep():
    _, late = roll("pressure_support_delayed_cycling")
    _, opt = roll("pressure_support_delayed_cycling_optimized")
    assert late["respiratory_ventilator_mean_cycling_delay_s"] > 0.50
    assert opt["respiratory_ventilator_mean_cycling_delay_s"] < late["respiratory_ventilator_mean_cycling_delay_s"] - 0.50
    assert opt["respiratory_cycle_auto_peep_cmH2O"] < late["respiratory_cycle_auto_peep_cmH2O"]

def test_premature_cycling_is_detected():
    _, s = roll("pressure_support_premature_cycling")
    assert s["respiratory_ventilator_premature_cycling_fraction"] > 0.50
    assert s["respiratory_ventilator_mean_cycling_delay_s"] < -0.50

def test_double_triggering_creates_more_ventilator_breaths_than_efforts():
    _, s = roll("pressure_support_double_trigger")
    assert s["respiratory_ventilator_double_trigger_fraction"] > 0.40
    assert s["respiratory_ventilator_breaths_per_min"] > s["respiratory_ventilator_patient_efforts_per_min"]

def test_leak_can_autotrigger_ventilator():
    _, s = roll("pressure_support_autotrigger_leak")
    assert s["respiratory_ventilator_autotrigger_fraction"] > 0.40
    assert s["respiratory_ventilator_breaths_per_min"] > s["respiratory_ventilator_patient_efforts_per_min"]

def test_waveform_trace_contains_patient_and_ventilator_state():
    env, _ = roll("pressure_support_delayed_cycling")
    tr = env.model.respiratory_cycle.last_trace
    assert "neural_inspiration" in tr
    assert "ventilator_active" in tr
    assert "ventilator_support_cmH2O" in tr
    assert np.any(tr["ventilator_active"] > tr["neural_inspiration"])

def test_v017_timestep_convergence():
    c1 = HumanConfig(respiratory_cycle_dt_s=0.010)
    c2 = HumanConfig(respiratory_cycle_dt_s=0.005)
    _, a = roll("pressure_support_delayed_cycling", config=c1)
    _, b = roll("pressure_support_delayed_cycling", config=c2)
    assert abs(a["respiratory_ventilator_mean_cycling_delay_s"] - b["respiratory_ventilator_mean_cycling_delay_s"]) < 0.12
    assert abs(a["tidal_volume_l"] - b["tidal_volume_l"]) < 0.06
    assert abs(a["respiratory_cycle_auto_peep_cmH2O"] - b["respiratory_cycle_auto_peep_cmH2O"]) < 0.25

def test_conservation_survives_pressure_support():
    _, s = roll("pressure_support_synchronous")
    assert abs(s["co2_mass_balance_error_mmol"]) < 1e-7
    assert abs(s["charge_balance_residual_mEq_l"]) < 1e-6
    assert abs(s["cv_blood_volume_error_ml"]) < 1e-6
