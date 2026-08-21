from __future__ import annotations

"""Focused v0.22 energy/carbon conservation gate.

These are executable internal invariants and numerical regressions. They are
not protocol-matched external clinical validation.
"""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historical_version_guard import require_exact_version

require_exact_version("0.22.0")

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, __version__
from openhumsim_rl.env import (
    ACTION_NAMES,
    CLINICAL_OBSERVATION_NAMES,
    OBSERVATION_NAMES,
)
from openhumsim_rl.physiology import HumanPhysiology, HumanState, Intervention
from openhumsim_rl.respiratory import RespiratoryModel, effective_pulmonary_rer


ZERO = np.zeros(len(ACTION_NAMES), dtype=np.float32)
checks: list[dict] = []

EXPECTED_CLINICAL_OBSERVATION_SHA256 = (
    "56770d5ea4d5ed4f81f98042bb4dcba7d0e40bfc73d109e5bdc3c2c5f5647de8"
)
EXPECTED_FULL_OBSERVATION_SHA256 = (
    "cf544ac7d1fdae6cf7b52e4320ec091409094ed43fd7bce90f4d496a854f813a"
)
EXPECTED_ACTION_SHA256 = (
    "9bc31bce6639bed396a5406518695c988e0d4f5d8740e893cec9ecfdfb985dfb"
)


def ordered_contract_sha256(names) -> str:
    encoded = json.dumps(
        list(names), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def add(name: str, passed: bool, values: dict) -> None:
    checks.append({"name": name, "passed": bool(passed), "values": values})


add("exact_release_version", __version__ == "0.22.0", {"version": __version__})


# 1. Final t=0 concentration and water volume define the conserved lactate amount.
baseline = HumanHomeostasisEnv(observation_profile="full")
_, baseline_info = baseline.reset(seed=22001)
transient = HumanHomeostasisEnv(
    scenario="transient_lactic_acidosis", observation_profile="full"
)
transient.reset(seed=22001)
add(
    "lactate_pool_initialized_after_scenario_and_volume",
    abs(
        baseline.state.lactate_amount_mmol
        - baseline.state.lactate_mmol_l
        * baseline.state.lactate_distribution_volume_l
    ) < 1e-12
    and transient.state.lactate_mmol_l == 4.0
    and transient.state.lactate_generated_mmol == 0.0
    and transient.state.lactate_cleared_mmol == 0.0,
    {
        "baseline_amount_mmol": baseline.state.lactate_amount_mmol,
        "transient_initial_concentration_mmol_l": transient.state.lactate_mmol_l,
    },
)


# 2. A water bolus changes concentration, not amount or source/sink counters.
amount_before = baseline.state.lactate_amount_mmol
concentration_before = baseline.state.lactate_mmol_l
baseline.model.apply_instant_intervention(
    baseline.state,
    Intervention(saline_ml=1_000.0, fio2=baseline.config.baseline_fio2),
)
add(
    "fluid_dilutes_without_creating_lactate",
    baseline.state.lactate_amount_mmol == amount_before
    and baseline.state.lactate_mmol_l < concentration_before
    and baseline.state.lactate_generated_mmol == 0.0
    and baseline.state.lactate_cleared_mmol == 0.0
    and abs(baseline.state.lactate_mass_balance_error_mmol) < 1e-12,
    {
        "amount_mmol": baseline.state.lactate_amount_mmol,
        "concentration_before_mmol_l": concentration_before,
        "concentration_after_mmol_l": baseline.state.lactate_mmol_l,
    },
)


# 3. Gross appearance/disposal are explicit and close the amount ledger.
energy_cfg = HumanConfig()
energy_model = HumanPhysiology(energy_cfg)
energy_state = HumanState()
energy_model.initialize_state(energy_state)
uma_before = energy_state.nonvolatile_strong_anion_mEq
energy_state.vo2_demand_ml_min = 500.0
energy_state.oxygen_debt_ml_min = 300.0
for _ in range(20):
    energy_model.energy_metabolism.step_lactate(
        energy_state, exercise=0.8, dt_min=energy_cfg.integration_step_min
    )
add(
    "lactate_amount_ledger_closes_once",
    energy_state.lactate_generated_mmol > 0.0
    and energy_state.lactate_cleared_mmol > 0.0
    and abs(energy_state.lactate_mass_balance_error_mmol) < 1e-10
    and energy_state.nonvolatile_strong_anion_mEq == uma_before,
    {
        "amount_mmol": energy_state.lactate_amount_mmol,
        "generated_mmol": energy_state.lactate_generated_mmol,
        "cleared_mmol": energy_state.lactate_cleared_mmol,
        "mass_error_mmol": energy_state.lactate_mass_balance_error_mmol,
    },
)


# 4. Oxidative VCO2 follows achieved VO2 and collapses under supply limitation.
gas_model = RespiratoryModel(HumanConfig())
gas_state = HumanState(
    pao2_mmHg=35.0,
    paco2_mmHg=40.0,
    ph_arterial=7.40,
    cardiac_output_l_min=1.5,
    vo2_demand_ml_min=1_200.0,
)
gas_model.update_oxygen_transport(gas_state)
gas_model.update_metabolic_gas_production(gas_state, exercise=1.0)
add(
    "oxidative_vco2_uses_achieved_vo2",
    gas_state.oxygen_debt_ml_min > 0.0
    and abs(
        gas_state.vco2_ml_min
        - gas_state.metabolic_respiratory_quotient * gas_state.vo2_ml_min
    ) < 1e-12
    and gas_state.vco2_ml_min < gas_state.vco2_demand_ml_min,
    {
        "vo2_demand_ml_min": gas_state.vo2_demand_ml_min,
        "vo2_achieved_ml_min": gas_state.vo2_ml_min,
        "oxidative_vco2_ml_min": gas_state.vco2_ml_min,
        "metabolic_rq": gas_state.metabolic_respiratory_quotient,
    },
)


# 5. The new stock is a monotonic integral, not a repayable "debt" state.
integral_state = HumanState()
energy_model.energy_metabolism.initialize_state(integral_state)
integral_state.oxygen_debt_ml_min = 100.0
energy_model.energy_metabolism.accumulate_oxygen_deficit(
    integral_state, previous_deficit_ml_min=0.0, dt_min=0.25
)
first_integral = integral_state.cumulative_oxygen_deficit_ml
integral_state.oxygen_debt_ml_min = 0.0
energy_model.energy_metabolism.accumulate_oxygen_deficit(
    integral_state, previous_deficit_ml_min=100.0, dt_min=0.25
)
add(
    "oxygen_deficit_integral_has_correct_units_and_semantics",
    first_integral == 12.5
    and integral_state.cumulative_oxygen_deficit_ml == 25.0,
    {
        "after_rise_ml": first_integral,
        "after_fall_ml": integral_state.cumulative_oxygen_deficit_ml,
    },
)


# 6. End-state oxygenation/Haldane chemistry closes against the same carbon pool.
closure_values = {}
closure_ok = True
for fio2 in (0.15, 0.21, 0.60):
    cfg = replace(
        HumanConfig(), baseline_fio2=fio2, max_fio2=max(0.60, fio2), agent_step_min=1.0
    )
    env = HumanHomeostasisEnv(config=cfg, observation_profile="full")
    env.reset(seed=22002)
    _, _, terminated, truncated, _ = env.step(ZERO)
    residual = env.model.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
        env.state, fio2=fio2, exercise=0.0
    )
    closure_values[str(fio2)] = residual
    closure_ok &= (
        not terminated
        and not truncated
        and abs(residual) <= max(1e-6, 10.0 * cfg.co2_pool_solver_tolerance_mmol_l)
        and abs(env.state.co2_mass_balance_error_mmol) < 1e-10
    )
add("final_haldane_carbon_pool_closure", closure_ok, closure_values)


# 7a. Challenged reset is a joint carbon/O2/RER equilibrium with zero fluxes.
reset_config = replace(HumanConfig(), baseline_fio2=0.15, max_fio2=0.60)
reset_challenge = HumanHomeostasisEnv(
    config=reset_config,
    scenario="respiratory_acidosis",
    observation_profile="full",
)
reset_challenge.reset(seed=22005)
reset_state = reset_challenge.state
reset_carbon_residual = (
    reset_challenge.model.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
        reset_state, fio2=reset_config.baseline_fio2, exercise=0.0
    )
)
reset_pulmonary = (
    reset_challenge.model.pulmonary_exchange.estimate_arterial_oxygen(
        reset_state,
        pco2_mmHg=reset_state.paco2_mmHg,
        fio2=reset_config.baseline_fio2,
        exercise=0.0,
        dt_min=0.0,
        apply=False,
    )
)
reset_expected_elimination = (
    reset_state.paco2_mmHg
    * reset_state.effective_co2_ventilation_l_min
    / 0.863
)
add(
    "reset_joint_carbon_oxygen_rer_fixed_point",
    abs(reset_carbon_residual) < 1e-6
    and abs(reset_pulmonary.pao2_mmHg - reset_state.pao2_mmHg) < 1e-4
    and abs(reset_expected_elimination - reset_state.vco2_elimination_ml_min) < 1e-8
    and reset_state.co2_generated_mmol == 0.0
    and reset_state.co2_eliminated_mmol == 0.0,
    {
        "carbon_residual_mmol_l": reset_carbon_residual,
        "pulmonary_o2_residual_mmHg": (
            reset_pulmonary.pao2_mmHg - reset_state.pao2_mmHg
        ),
        "elimination_endpoint_residual_ml_min": (
            reset_expected_elimination - reset_state.vco2_elimination_ml_min
        ),
        "effective_pulmonary_rer": reset_pulmonary.effective_respiratory_exchange_ratio,
    },
)


# 7b. Supply-limited integration must book final, not optimistic predictor, VCO2.
supply_model = HumanPhysiology(HumanConfig())
supply_state = HumanState()
supply_model.initialize_state(supply_state)
supply_blood_volume_l = supply_state.plasma_volume_l + supply_state.rbc_volume_l
supply_state.hemoglobin_mass_g = 7.0 * supply_blood_volume_l * 10.0
supply_model._update_blood_composition(supply_state)
supply_start_vco2 = supply_state.vco2_ml_min
supply_generated_before = supply_state.co2_generated_mmol
supply_model.integrate(
    supply_state,
    Intervention(exercise_intensity=1.0, fio2=0.15),
    supply_model.cfg.integration_step_min,
)
supply_generated = supply_state.co2_generated_mmol - supply_generated_before
supply_expected_generated = (
    0.5 * (supply_start_vco2 + supply_state.vco2_ml_min)
    * supply_model.blood_gas.gas_mmol_per_ml_stpd
    * supply_model.cfg.integration_step_min
)
supply_expected_elimination = (
    supply_state.paco2_mmHg
    * supply_state.effective_co2_ventilation_l_min
    / 0.863
)
add(
    "supply_limited_final_vco2_and_elimination_endpoints",
    supply_state.oxygen_debt_ml_min > 0.0
    and abs(supply_generated - supply_expected_generated)
    <= 2e-5 * max(1.0, abs(supply_expected_generated))
    and abs(supply_state.vco2_elimination_ml_min - supply_expected_elimination)
    <= 2e-5 * max(1.0, abs(supply_expected_elimination)),
    {
        "oxygen_deficit_ml_min": supply_state.oxygen_debt_ml_min,
        "generation_residual_mmol": supply_generated - supply_expected_generated,
        "elimination_endpoint_residual_ml_min": (
            supply_state.vco2_elimination_ml_min - supply_expected_elimination
        ),
    },
)


# 7c. Pulmonary RER follows lung fluxes and remains distinct from metabolic RQ.
rer_model = RespiratoryModel(HumanConfig())
rer_state = HumanState(vo2_ml_min=1500.0, vo2_demand_ml_min=1500.0)
rer_model.update_metabolic_gas_production(rer_state, exercise=1.0)
rer_state.vco2_elimination_ml_min = rer_state.oxidative_vco2_ml_min
steady_rer = effective_pulmonary_rer(rer_state, rer_model.cfg)
rer_state.vco2_elimination_ml_min = 1800.0
transient_rer = effective_pulmonary_rer(rer_state, rer_model.cfg)
add(
    "pulmonary_rer_uses_lung_flux_not_cellular_rq",
    abs(steady_rer - rer_state.metabolic_respiratory_quotient) < 1e-12
    and transient_rer != rer_state.metabolic_respiratory_quotient
    and abs(transient_rer - 1.2) < 1e-12,
    {
        "metabolic_rq": rer_state.metabolic_respiratory_quotient,
        "steady_pulmonary_rer": steady_rer,
        "transient_pulmonary_rer": transient_rer,
    },
)


# 7. Production uses the interval average; elimination uses endpoint trapezoid.
carbon_model = HumanPhysiology(HumanConfig())
carbon_state = HumanState()
carbon_model.initialize_state(carbon_state)
carbon_state.vco2_generation_interval_average_ml_min = 321.0
dt = 0.25
generated_before = carbon_state.co2_generated_mmol
eliminated_before = carbon_state.co2_eliminated_mmol
elimination_start = carbon_state.vco2_elimination_ml_min
carbon_model.blood_gas.step_arterial_carbon_balance(
    carbon_state, fio2=0.21, exercise=0.0, dt_min=dt
)
expected_generated = 321.0 * carbon_model.blood_gas.gas_mmol_per_ml_stpd * dt
expected_eliminated = (
    0.5 * (elimination_start + carbon_state.vco2_elimination_ml_min)
    * carbon_model.blood_gas.gas_mmol_per_ml_stpd
    * dt
)
add(
    "carbon_fluxes_use_interval_consistent_quadrature",
    abs(
        carbon_state.co2_generated_mmol - generated_before - expected_generated
    ) < 1e-12
    and abs(
        carbon_state.co2_eliminated_mmol - eliminated_before - expected_eliminated
    ) < 1e-12,
    {
        "generated_mmol": carbon_state.co2_generated_mmol - generated_before,
        "eliminated_mmol": carbon_state.co2_eliminated_mmol - eliminated_before,
    },
)


# 8. Coupled energy/carbon trajectories are stable under outer-step refinement.
def exercise_rollout(dt_min: float) -> dict:
    cfg = replace(
        HumanConfig(), agent_step_min=5.0, integration_step_min=dt_min, episode_minutes=10.0
    )
    env = HumanHomeostasisEnv(config=cfg, observation_profile="full")
    env.reset(seed=22003)
    action = ZERO.copy()
    action[2] = 1.0
    _, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        raise RuntimeError("unexpected termination in v0.22 convergence probe")
    return info["state"]


coarse = exercise_rollout(0.25)
fine = exercise_rollout(0.05)
relative = lambda a, b: abs(float(a) - float(b)) / max(1e-12, abs(float(b)))
convergence = {
    "co2_generated_relative": relative(coarse["co2_generated_mmol"], fine["co2_generated_mmol"]),
    "co2_eliminated_relative": relative(coarse["co2_eliminated_mmol"], fine["co2_eliminated_mmol"]),
    "lactate_amount_relative": relative(coarse["lactate_amount_mmol"], fine["lactate_amount_mmol"]),
    "oxygen_deficit_relative": relative(coarse["cumulative_oxygen_deficit_ml"], fine["cumulative_oxygen_deficit_ml"]),
    "paco2_absolute_mmHg": abs(coarse["paco2_mmHg"] - fine["paco2_mmHg"]),
    "ph_absolute": abs(coarse["ph_arterial"] - fine["ph_arterial"]),
}
add(
    "energy_carbon_outer_step_convergence",
    convergence["co2_generated_relative"] < 0.01
    and convergence["co2_eliminated_relative"] < 0.02
    and convergence["lactate_amount_relative"] < 0.005
    and convergence["oxygen_deficit_relative"] < 0.03
    and convergence["paco2_absolute_mmHg"] < 0.5
    and convergence["ph_absolute"] < 0.01,
    convergence,
)


# 9. New exact ledgers stay hidden from policy-facing telemetry.
strict = HumanHomeostasisEnv(info_profile="benchmark")
_, strict_info = strict.reset(seed=22004)
clinical_hash = ordered_contract_sha256(CLINICAL_OBSERVATION_NAMES)
full_hash = ordered_contract_sha256(OBSERVATION_NAMES)
action_hash = ordered_contract_sha256(ACTION_NAMES)
expected_strict = {
    "observation_names",
    "observation_profile",
    "measurement_profile",
    "info_profile",
    "action_names",
    "gymnasium_installed",
    "environment_semantics",
}
add(
    "observation_and_benchmark_contract_unchanged",
    len(CLINICAL_OBSERVATION_NAMES) == 54
    and len(OBSERVATION_NAMES) == 138
    and clinical_hash == EXPECTED_CLINICAL_OBSERVATION_SHA256
    and full_hash == EXPECTED_FULL_OBSERVATION_SHA256
    and action_hash == EXPECTED_ACTION_SHA256
    and "lactate_amount_mmol" not in CLINICAL_OBSERVATION_NAMES
    and "cumulative_oxygen_deficit_ml" not in CLINICAL_OBSERVATION_NAMES
    and set(strict_info) == expected_strict
    and "energy_metabolism" not in strict_info,
    {
        "clinical_dim": len(CLINICAL_OBSERVATION_NAMES),
        "full_dim": len(OBSERVATION_NAMES),
        "clinical_observation_sha256": clinical_hash,
        "full_observation_sha256": full_hash,
        "action_sha256": action_hash,
        "strict_info_keys": sorted(strict_info),
    },
)


# 10. Configs that imply a non-metabolic RQ are rejected rather than clipped.
invalid_rq_rejected = False
try:
    HumanConfig(baseline_vo2_ml_min=100.0, baseline_vco2_ml_min=150.0)
except ValueError:
    invalid_rq_rejected = True
add("invalid_metabolic_rq_config_rejected", invalid_rq_rejected, {})


# 11. Persistence is explicitly schema-versioned; legacy lactate semantics fail loud.
schema_state = HumanState(lactate_mmol_l=4.0, lactate_amount_mmol=168.0)
schema_payload = schema_state.to_versioned_payload()
schema_roundtrip = HumanState.from_versioned_payload(schema_payload)
legacy_state_rejected = False
try:
    HumanState.from_versioned_payload(
        {
            "state_schema_version": "0.21",
            "state": {"lactate_mmol_l": 4.0, "total_body_water_l": 30.0},
        }
    )
except ValueError:
    legacy_state_rejected = True
invalid_action_scale_rejected = False
try:
    HumanConfig(max_saline_ml_per_step=float("inf"))
except ValueError:
    invalid_action_scale_rejected = True
add(
    "versioned_state_and_finite_action_config_contract",
    schema_roundtrip.as_dict() == schema_state.as_dict()
    and legacy_state_rejected
    and invalid_action_scale_rejected,
    {
        "state_schema_version": schema_payload["state_schema_version"],
        "legacy_state_rejected": legacy_state_rejected,
        "invalid_action_scale_rejected": invalid_action_scale_rejected,
    },
)


payload = {
    "version": "0.22.0",
    "scope": (
        "internal energy, lactate and coupled O2/CO2 conservation invariants; "
        "not external clinical validation"
    ),
    "summary": {
        "passed": sum(check["passed"] for check in checks),
        "total": len(checks),
    },
    "checks": checks,
}
output = ROOT / "validation" / "validation_results_v0.22.json"
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if payload["summary"]["passed"] == payload["summary"]["total"] else 1)
