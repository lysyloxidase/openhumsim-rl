from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.16.0")

import json
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv

ZERO = np.zeros(8, dtype=np.float32)


def roll(scenario: str, steps: int = 6, seed: int = 42, config: HumanConfig | None = None):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    env.reset(seed=seed)
    info = None
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(ZERO)
        if terminated or truncated:
            break
    return info["state"]


checks = []

def add(name, passed, values):
    checks.append({"name": name, "passed": bool(passed), "values": values})

base = roll("baseline")
obs = roll("airway_obstruction")
tach = roll("tachypnea_airway_obstruction")
pc = roll("pressure_control_ventilation")
pco = roll("pressure_control_obstruction")

add(
    "equation_of_motion_numerical_closure",
    max(abs(base["respiratory_cycle_equation_residual_cmH2O"]), abs(obs["respiratory_cycle_equation_residual_cmH2O"])) < 1e-9,
    {
        "baseline_residual_cmH2O": base["respiratory_cycle_equation_residual_cmH2O"],
        "obstruction_residual_cmH2O": obs["respiratory_cycle_equation_residual_cmH2O"]
    },
)

add(
    "healthy_baseline_cycle",
    0.40 <= base["tidal_volume_l"] <= 0.65 and base["respiratory_cycle_auto_peep_cmH2O"] < 0.15 and 35 <= base["paco2_mmHg"] <= 48,
    {
        "VT_L": base["tidal_volume_l"],
        "RR_bpm": base["respiratory_rate_bpm"],
        "auto_PEEP_cmH2O": base["respiratory_cycle_auto_peep_cmH2O"],
        "PaCO2_mmHg": base["paco2_mmHg"],
        "time_constant_s": base["respiratory_cycle_time_constant_s"]
    },
)

add(
    "obstruction_mechanistic_direction",
    obs["respiratory_cycle_time_constant_s"] > 4 * base["respiratory_cycle_time_constant_s"] and obs["respiratory_cycle_resistive_work_j_breath"] > 2 * base["respiratory_cycle_resistive_work_j_breath"] and obs["respiratory_cycle_auto_peep_cmH2O"] > 0.5,
    {
        "baseline_tau_s": base["respiratory_cycle_time_constant_s"],
        "obstructed_tau_s": obs["respiratory_cycle_time_constant_s"],
        "baseline_resistive_work_J": base["respiratory_cycle_resistive_work_j_breath"],
        "obstructed_resistive_work_J": obs["respiratory_cycle_resistive_work_j_breath"],
        "obstructed_auto_PEEP_cmH2O": obs["respiratory_cycle_auto_peep_cmH2O"]
    },
)

add(
    "human_copd_peepi_scale_anchor",
    0.8 <= tach["respiratory_cycle_auto_peep_cmH2O"] <= 4.0,
    {
        "model_auto_PEEP_cmH2O": tach["respiratory_cycle_auto_peep_cmH2O"],
        "Dal_Vecchio_1990_mean_cmH2O": 2.4,
        "Dal_Vecchio_1990_sd_cmH2O": 1.6
    },
)

add(
    "tachypnea_dynamic_hyperinflation",
    tach["respiratory_cycle_auto_peep_cmH2O"] > obs["respiratory_cycle_auto_peep_cmH2O"] + 1.0 and tach["respiratory_cycle_dynamic_hyperinflation_l"] > obs["respiratory_cycle_dynamic_hyperinflation_l"] + 0.08,
    {
        "obstruction_auto_PEEP": obs["respiratory_cycle_auto_peep_cmH2O"],
        "tachypnea_auto_PEEP": tach["respiratory_cycle_auto_peep_cmH2O"],
        "obstruction_DH_L": obs["respiratory_cycle_dynamic_hyperinflation_l"],
        "tachypnea_DH_L": tach["respiratory_cycle_dynamic_hyperinflation_l"]
    },
)

add(
    "pressure_control_work_partition",
    pc["respiratory_cycle_peak_muscle_pressure_cmH2O"] < 1e-6 and pc["respiratory_cycle_ventilator_work_j_breath"] > 0.2 and 14 <= pc["respiratory_cycle_peak_airway_pressure_cmH2O"] <= 16,
    {
        "Pmus_peak_cmH2O": pc["respiratory_cycle_peak_muscle_pressure_cmH2O"],
        "Paw_peak_cmH2O": pc["respiratory_cycle_peak_airway_pressure_cmH2O"],
        "ventilator_work_J_breath": pc["respiratory_cycle_ventilator_work_j_breath"]
    },
)

add(
    "pressure_control_does_not_abolish_obstructive_auto_peep",
    pco["respiratory_cycle_auto_peep_cmH2O"] > 1.0 and pco["paco2_mmHg"] > 45,
    {"auto_PEEP_cmH2O": pco["respiratory_cycle_auto_peep_cmH2O"], "PaCO2_mmHg": pco["paco2_mmHg"]},
)

add(
    "pv_hysteresis_and_intrinsic_peep_coupling",
    obs["respiratory_cycle_pv_hysteresis_j_breath"] > base["respiratory_cycle_pv_hysteresis_j_breath"] and abs((tach["pulmonary_transpulmonary_pressure_end_exp_cmH2O"] - base["pulmonary_transpulmonary_pressure_end_exp_cmH2O"]) - tach["respiratory_cycle_auto_peep_cmH2O"]) < 0.15,
    {
        "baseline_hysteresis_J": base["respiratory_cycle_pv_hysteresis_j_breath"],
        "obstruction_hysteresis_J": obs["respiratory_cycle_pv_hysteresis_j_breath"],
        "tachypnea_auto_PEEP": tach["respiratory_cycle_auto_peep_cmH2O"],
        "tachypnea_PL_endexp": tach["pulmonary_transpulmonary_pressure_end_exp_cmH2O"]
    },
)

coarse = roll("tachypnea_airway_obstruction", config=HumanConfig(respiratory_cycle_dt_s=0.01))
fine = roll("tachypnea_airway_obstruction", config=HumanConfig(respiratory_cycle_dt_s=0.005))
add(
    "within_breath_timestep_convergence",
    abs(coarse["tidal_volume_l"] - fine["tidal_volume_l"]) < 0.03 and abs(coarse["respiratory_cycle_auto_peep_cmH2O"] - fine["respiratory_cycle_auto_peep_cmH2O"]) < 0.20 and abs(coarse["respiratory_cycle_muscle_work_j_breath"] - fine["respiratory_cycle_muscle_work_j_breath"]) < 0.08,
    {
        "d_VT_L": abs(coarse["tidal_volume_l"] - fine["tidal_volume_l"]),
        "d_auto_PEEP_cmH2O": abs(coarse["respiratory_cycle_auto_peep_cmH2O"] - fine["respiratory_cycle_auto_peep_cmH2O"]),
        "d_muscle_work_J": abs(coarse["respiratory_cycle_muscle_work_j_breath"] - fine["respiratory_cycle_muscle_work_j_breath"])
    },
)

add(
    "legacy_conservation_with_cycle_dynamics",
    abs(pco["co2_mass_balance_error_mmol"]) < 1e-7 and abs(pco["charge_balance_residual_mEq_l"]) < 1e-6 and abs(pco["cv_blood_volume_error_ml"]) < 1e-6,
    {
        "CO2_mass_residual_mmol": pco["co2_mass_balance_error_mmol"],
        "charge_residual_mEq_l": pco["charge_balance_residual_mEq_l"],
        "blood_volume_residual_ml": pco["cv_blood_volume_error_ml"]
    },
)

payload = {
    "version": "0.16.0",
    "summary": {
        "passed": sum(c["passed"] for c in checks),
        "total": len(checks),
        "all_passed": all(c["passed"] for c in checks)
    },
    "checks": checks
}
Path("validation/validation_results_v0.16.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload["summary"], indent=2))
if not payload["summary"]["all_passed"]:
    for c in checks:
        if not c["passed"]:
            print("FAILED", c)
    raise SystemExit(1)
