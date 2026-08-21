from __future__ import annotations

"""Focused v0.21 numerical- and benchmark-integrity gate.

This is verification against invariants and regression cases.  It is not an
external clinical validation and must not be presented as one.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historical_version_guard import require_exact_version

require_exact_version("0.21.0")

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.calibration import reference_outputs
from openhumsim_rl.cgm import CGMObservationConfig, blood_to_cgm_trace
from openhumsim_rl.env import ACTION_NAMES, CLINICAL_OBSERVATION_NAMES
from openhumsim_rl.measurement import ClinicalMeasurementConfig, ClinicalMeasurementModel
from openhumsim_rl.pbpk import ReferencePBPKModel
from openhumsim_rl.physiology import HumanPhysiology, HumanState
from openhumsim_rl.population import DEFAULT_PARAMETER_SPECS, LockedCohortManifest
from openhumsim_rl.respiratory import RespiratoryModel


ZERO = np.zeros(len(ACTION_NAMES), dtype=np.float32)
checks: list[dict] = []


def add(name: str, passed: bool, values: dict) -> None:
    checks.append({"name": name, "passed": bool(passed), "values": values})


# 1. Benchmark metadata is a positive allowlist, not a blacklist that can miss
# newly introduced mechanistic keys.
strict = HumanHomeostasisEnv(info_profile="benchmark")
_, strict_info = strict.reset(seed=2100)
expected_info = {
    "observation_names",
    "observation_profile",
    "measurement_profile",
    "info_profile",
    "action_names",
    "gymnasium_installed",
    "environment_semantics",
}
forbidden_info = {
    "state", "reward_terms", "scenario", "measurement", "time_min",
    "blood_gas", "blood_gas_carbon", "pbpk", "action", "intervention",
}
add(
    "benchmark_info_exact_allowlist",
    set(strict_info) == expected_info and forbidden_info.isdisjoint(strict_info),
    {"keys": sorted(strict_info)},
)


# 2. A non-divisible episode ends at its declared horizon, not one full policy
# step later.
horizon = HumanHomeostasisEnv(
    config=HumanConfig(agent_step_min=5.0, integration_step_min=0.25, episode_minutes=12.0),
    observation_profile="full",
)
horizon.reset(seed=2101)
horizon_times = []
horizon_truncated = False
for _ in range(3):
    _, _, horizon_terminated, horizon_truncated, _ = horizon.step(ZERO)
    horizon_times.append(float(horizon.elapsed_minutes))
full_intervention = horizon._decode_action(np.ones(len(ACTION_NAMES), dtype=np.float32))
add(
    "episode_horizon_no_overshoot",
    not horizon_terminated
    and horizon_truncated
    and np.allclose(horizon_times, [5.0, 10.0, 12.0])
    and full_intervention.insulin_model_units
    == horizon.config.max_insulin_model_units_per_step,
    {
        "times_min": horizon_times,
        "full_bolus_model_units": full_intervention.insulin_model_units,
    },
)


# 3. Cardiorespiratory outputs must converge when the outer operator-splitting
# interval becomes shorter than a breath or a heartbeat.
def rollout_with_outer_step(dt_min: float) -> dict:
    env = HumanHomeostasisEnv(
        config=replace(HumanConfig(), integration_step_min=dt_min),
        scenario="oral_glucose_75g",
    )
    env.reset(seed=2102)
    info = None
    for _ in range(2):
        _, _, terminated, truncated, info = env.step(ZERO)
        if terminated or truncated:
            break
    assert info is not None
    return info["state"]


coarse = rollout_with_outer_step(0.25)
fine = rollout_with_outer_step(0.05)
convergence_delta = {
    name: abs(float(coarse[name]) - float(fine[name]))
    for name in (
        "tidal_volume_l", "paco2_mmHg", "pao2_mmHg",
        "map_mmHg", "cardiac_output_l_min",
    )
}
add(
    "cardiorespiratory_outer_step_convergence",
    convergence_delta["tidal_volume_l"] < 0.01
    and convergence_delta["paco2_mmHg"] < 0.25
    and convergence_delta["pao2_mmHg"] < 0.25
    and convergence_delta["map_mmHg"] < 0.5
    and convergence_delta["cardiac_output_l_min"] < 0.10,
    convergence_delta,
)


# 4. A sub-cycle fragment is retained internally and cannot redefine VT/work.
cycle_env = HumanHomeostasisEnv()
cycle_env.reset(seed=2103)
cycle = cycle_env.model.respiratory_cycle
cycle.initialize_state(cycle_env.state)
vt_before = float(cycle_env.state.tidal_volume_l)
cycle.step(cycle_env.state, dt_min=0.025)
add(
    "incomplete_interval_is_not_a_breath",
    cycle_env.state.tidal_volume_l == vt_before
    and cycle_env.state.respiratory_cycle_total_work_j_breath == 0.0
    and cycle.last_trace == {},
    {
        "held_tidal_volume_l": float(cycle_env.state.tidal_volume_l),
        "phase_s": float(cycle_env.state.respiratory_cycle_phase_s),
    },
)


# 5. Renal acid removal is represented once: higher ammonium/TA removes UMA,
# while chloride excretion retains its independent sodium-linked ledger.
renal_cfg = HumanConfig()
renal_model = HumanPhysiology(renal_cfg)
renal_state = HumanState()
renal_model.initialize_state(renal_state)
renal_state.ph_arterial = 7.20
renal_model.renal.step(renal_state, exercise=0.0, dt=0.25)
expected_chloride = np.clip(
    renal_cfg.baseline_urine_chloride_mmol_min
    * renal_state.urine_sodium_mmol_min
    / renal_cfg.baseline_urine_sodium_mmol_min,
    0.002,
    0.90,
)
add(
    "renal_acid_effect_not_duplicated_in_chloride",
    renal_state.urine_ammonium_mmol_min
    > renal_cfg.baseline_net_acid_excretion_mmol_min
    * renal_cfg.baseline_ammonium_fraction_of_nae
    and abs(renal_state.urine_chloride_mmol_min - expected_chloride) < 1e-12,
    {
        "urine_ammonium_mmol_min": renal_state.urine_ammonium_mmol_min,
        "urine_chloride_mmol_min": renal_state.urine_chloride_mmol_min,
    },
)


# 6. Oxygen reserve must use the same extraction-limited supply ceiling as VO2.
oxygen_state = HumanState(
    pao2_mmHg=45.0,
    paco2_mmHg=40.0,
    ph_arterial=7.40,
    cardiac_output_l_min=2.0,
    vo2_demand_ml_min=500.0,
)
oxygen_model = RespiratoryModel(HumanConfig())
oxygen_model.update_oxygen_transport(oxygen_state)
expected_oxygen_margin = (
    oxygen_model.cfg.oxygen_max_extraction_fraction
    * oxygen_state.oxygen_delivery_ml_min
    - oxygen_state.vo2_demand_ml_min
)
add(
    "oxygen_reserve_matches_extraction_limited_supply",
    abs(oxygen_state.oxygen_supply_margin_ml_min - expected_oxygen_margin) < 1e-12
    and oxygen_state.oxygen_supply_margin_ml_min < 0.0
    and oxygen_state.oxygen_debt_ml_min > 0.0,
    {
        "supply_margin_ml_min": oxygen_state.oxygen_supply_margin_ml_min,
        "instantaneous_deficit_ml_min": oxygen_state.oxygen_debt_ml_min,
    },
)


# 7. Hepatic PBPK elimination must leave the liver compartment and the global
# mass balance must still close.
pbpk_cfg = HumanConfig()
pbpk = ReferencePBPKModel(pbpk_cfg)
pbpk_state = HumanState(gfr_ml_min=0.0)
equilibrium = 0.50
pbpk_state.probe_plasma_mg = equilibrium * pbpk_state.plasma_volume_l
for _, amount_attr, volume_attr, _, kp_attr in pbpk._TISSUES:
    setattr(
        pbpk_state,
        amount_attr,
        equilibrium * getattr(pbpk_cfg, volume_attr) * getattr(pbpk_cfg, kp_attr),
    )
pbpk_state.probe_administered_mg = sum(
    getattr(pbpk_state, attr)
    for attr in (
        "probe_plasma_mg", "probe_liver_mg", "probe_kidney_mg",
        "probe_muscle_mg", "probe_adipose_mg", "probe_rest_mg",
    )
)
liver_before = pbpk_state.probe_liver_mg
pbpk.step(pbpk_state, exercise=0.0, dt=pbpk_cfg.pbpk_internal_step_min)
hepatic_loss = pbpk_state.probe_eliminated_hepatic_mg
add(
    "pbpk_hepatic_elimination_withdrawn_from_liver",
    hepatic_loss > 0.0
    and abs((liver_before - pbpk_state.probe_liver_mg) - hepatic_loss) < 1e-12
    and abs(pbpk_state.probe_mass_balance_error_mg) < 1e-12,
    {
        "hepatic_loss_mg": hepatic_loss,
        "mass_balance_error_mg": pbpk_state.probe_mass_balance_error_mg,
    },
)


# 8. Delays longer than sampling cadence produce a FIFO of pending ABGs rather
# than replacing every sample before any result becomes available.
measurement_state = HumanState(paco2_mmHg=40.0)
measurement_cfg = ClinicalMeasurementConfig(
    abg_interval_min=10.0,
    abg_result_delay_min=25.0,
    monitor_dropout_probability=0.0,
    cgm_dropout_probability=0.0,
    cgm_relative_noise_sd=0.0,
    noise_multiplier=0.0,
)
measurement = ClinicalMeasurementModel(measurement_cfg)
measurement_rng = np.random.default_rng(2104)
measurement.initialize(measurement_state, measurement_rng)
initial_paco2 = measurement.measurement_value("paco2_mmHg", measurement_state)
for minute, value in ((10.0, 50.0), (20.0, 60.0), (30.0, 70.0)):
    measurement_state.paco2_mmHg = value
    measurement.advance(measurement_state, minute, 10.0, measurement_rng)
pending_before = measurement.diagnostics()["channels"]["paco2_mmHg"]["pending"]
measurement.advance(measurement_state, 35.0, 5.0, measurement_rng)
reported = measurement.measurement_value("paco2_mmHg", measurement_state)
add(
    "measurement_fifo_when_delay_exceeds_interval",
    initial_paco2 == 40.0
    and pending_before == 3
    and reported == 50.0
    and "pulmonary_aa_gradient_mmHg" not in CLINICAL_OBSERVATION_NAMES
    and "pulmonary_enghoff_dead_space_fraction" not in CLINICAL_OBSERVATION_NAMES,
    {"pending_before_delivery": pending_before, "first_reported_paco2_mmHg": reported},
)


# 9. A fixed seed controls the initial CGM noise as well as later points.
cgm_values = np.asarray([100.0, 110.0, 120.0, 105.0])
cgm_cfg = CGMObservationConfig(relative_noise_sd=0.10)
cgm_a = blood_to_cgm_trace(cgm_values, 5.0, config=cgm_cfg, seed=2105)
cgm_b = blood_to_cgm_trace(cgm_values, 5.0, config=cgm_cfg, seed=2105)
cgm_c = blood_to_cgm_trace(cgm_values, 5.0, config=cgm_cfg, seed=2106)
add(
    "cgm_initial_sample_seeded",
    np.array_equal(cgm_a, cgm_b) and cgm_a[0] != cgm_c[0],
    {"seeded_first_value": float(cgm_a[0]), "other_seed_first_value": float(cgm_c[0])},
)


# 10. The virtual-patient prior must vary a parameter that is active in the
# integrated pulmonary solve.
spec_names = {spec.name for spec in DEFAULT_PARAMETER_SPECS}
vq_low = reference_outputs(
    replace(HumanConfig(), pulmonary_baseline_vq_log_sd=0.08), seed=2107
)
vq_high = reference_outputs(
    replace(HumanConfig(), pulmonary_baseline_vq_log_sd=0.30), seed=2107
)
add(
    "virtual_population_uses_active_vq_parameter",
    "baseline_aa_gradient_mmHg" not in spec_names
    and "pulmonary_baseline_vq_log_sd" in spec_names
    and vq_low["pao2_mmHg"] > vq_high["pao2_mmHg"] + 10.0,
    {"pao2_low_vq_sd": vq_low["pao2_mmHg"], "pao2_high_vq_sd": vq_high["pao2_mmHg"]},
)


# 11. The holdout lock covers source identity, both sides of the split and seed.
manifest = LockedCohortManifest.create(
    [f"S{i:03d}" for i in range(20)],
    "v021-lock-check",
    seed=2108,
    dataset_fingerprint="sha256:v021-check",
)
tampered = replace(manifest, split_seed=manifest.split_seed + 1)
add(
    "validation_manifest_full_split_lock",
    manifest.verify_lock() and not tampered.verify_lock(),
    {"lock": manifest.validation_lock_sha256},
)


# 12. Undefined numerical configurations fail at construction time.
invalid_rejected = False
try:
    HumanConfig(integration_step_min=0.0)
except ValueError:
    invalid_rejected = True
add("invalid_numerical_config_rejected", invalid_rejected, {})


# 13. Non-finite mechanistic state cannot silently pass threshold comparisons.
nonfinite_state = HumanState(paco2_mmHg=float("nan"))
nonfinite_rejected = False
try:
    HumanPhysiology._clip_state(nonfinite_state)
except FloatingPointError:
    nonfinite_rejected = True
add("nonfinite_state_rejected", nonfinite_rejected, {})


payload = {
    "version": "0.21.0",
    "scope": "focused numerical and benchmark integrity verification; not clinical validation",
    "summary": {
        "passed": sum(check["passed"] for check in checks),
        "total": len(checks),
        "all_passed": all(check["passed"] for check in checks),
    },
    "checks": checks,
}
output = ROOT / "validation" / "validation_results_v0.21.json"
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload["summary"], indent=2))
if not payload["summary"]["all_passed"]:
    raise SystemExit(1)
