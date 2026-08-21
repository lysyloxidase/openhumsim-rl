from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.15.0")

import json
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv, HumanConfig


ZERO = np.zeros(8, dtype=np.float32)


def roll(scenario, steps=6, seed=42, config=None):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    _, info = env.reset(seed=seed)
    for _ in range(steps):
        _, _, term, trunc, info = env.step(ZERO)
        if term or trunc:
            break
    return info["state"]


results = {}
checks = []


def check(name, passed, values):
    passed = bool(passed)
    checks.append({"name": name, "passed": passed, "values": values})
    return passed


base = roll("baseline")
peep5 = roll("mechanical_ventilation_peep5")
peep12 = roll("mechanical_ventilation_peep12")
peep18 = roll("overdistension_peep18")
stiff_cw = roll("stiff_chest_wall")
stiff_lung = roll("low_lung_compliance")
collapse = roll("dependent_derecruitment")
recruit = roll("recruitment_peep")

check(
    "transpulmonary_pressure_identity",
    abs(peep5["pulmonary_pressure_identity_residual_cmH2O"]) < 1e-10,
    {"residual_cmH2O": peep5["pulmonary_pressure_identity_residual_cmH2O"]},
)

check(
    "healthy_baseline_mechanics",
    (
        base["pulmonary_recruitment_fraction"] > 0.90
        and base["pulmonary_overdistension_fraction"] < 0.10
        and 0.04 < base["pulmonary_respiratory_system_compliance_l_cmH2O"] < 0.10
    ),
    {
        "recruitment": base["pulmonary_recruitment_fraction"],
        "overdistension": base["pulmonary_overdistension_fraction"],
        "Crs_L_cmH2O": base["pulmonary_respiratory_system_compliance_l_cmH2O"],
        "PL_endexp": base["pulmonary_transpulmonary_pressure_end_exp_cmH2O"],
        "PL_endinsp": base["pulmonary_transpulmonary_pressure_end_insp_cmH2O"],
    },
)

check(
    "lung_vs_chest_wall_separation",
    (
        stiff_cw["pulmonary_airway_driving_pressure_cmH2O"] > base["pulmonary_airway_driving_pressure_cmH2O"]
        and abs(
            stiff_cw["pulmonary_transpulmonary_driving_pressure_cmH2O"]
            - base["pulmonary_transpulmonary_driving_pressure_cmH2O"]
        ) < 0.25
        and stiff_lung["pulmonary_transpulmonary_driving_pressure_cmH2O"]
        > base["pulmonary_transpulmonary_driving_pressure_cmH2O"] + 2.0
    ),
    {
        "baseline_dPaw": base["pulmonary_airway_driving_pressure_cmH2O"],
        "stiff_chest_dPaw": stiff_cw["pulmonary_airway_driving_pressure_cmH2O"],
        "baseline_dPL": base["pulmonary_transpulmonary_driving_pressure_cmH2O"],
        "stiff_chest_dPL": stiff_cw["pulmonary_transpulmonary_driving_pressure_cmH2O"],
        "stiff_lung_dPL": stiff_lung["pulmonary_transpulmonary_driving_pressure_cmH2O"],
    },
)

check(
    "peep_recruitment_tradeoff",
    (
        recruit["pulmonary_recruitment_fraction"] > collapse["pulmonary_recruitment_fraction"] + 0.50
        and recruit["pao2_mmHg"] > collapse["pao2_mmHg"] + 15.0
    ),
    {
        "collapsed_recruitment": collapse["pulmonary_recruitment_fraction"],
        "peep_recruitment": recruit["pulmonary_recruitment_fraction"],
        "collapsed_PaO2": collapse["pao2_mmHg"],
        "peep_PaO2": recruit["pao2_mmHg"],
    },
)

check(
    "high_peep_overdistension",
    (
        peep18["pulmonary_overdistension_fraction"] > peep5["pulmonary_overdistension_fraction"] + 0.30
        and peep18["pulmonary_mechanical_pvr_multiplier"] > peep5["pulmonary_mechanical_pvr_multiplier"] + 0.20
    ),
    {
        "PEEP5_overdistension": peep5["pulmonary_overdistension_fraction"],
        "PEEP18_overdistension": peep18["pulmonary_overdistension_fraction"],
        "PEEP5_mech_PVR": peep5["pulmonary_mechanical_pvr_multiplier"],
        "PEEP18_mech_PVR": peep18["pulmonary_mechanical_pvr_multiplier"],
    },
)

check(
    "positive_pressure_hemodynamic_direction",
    peep12["cardiac_output_l_min"] < base["cardiac_output_l_min"] - 0.5,
    {
        "baseline_CO": base["cardiac_output_l_min"],
        "PEEP12_CO": peep12["cardiac_output_l_min"],
        "baseline_MAP": base["map_mmHg"],
        "PEEP12_MAP": peep12["map_mmHg"],
        "intrathoracic_delta_cmH2O": peep12["pulmonary_intrathoracic_pressure_delta_cmH2O"],
    },
)

def custom(sc, peep):
    env = HumanHomeostasisEnv(scenario=sc)
    env.reset(seed=7)
    env.state.pulmonary_peep_cmH2O = peep
    env.state.pulmonary_positive_pressure_fraction = 1.0
    info = None
    for _ in range(6):
        _, _, term, trunc, info = env.step(ZERO)
        if term or trunc:
            break
    return info["state"]

norm12 = custom("baseline", 12.0)
dry12 = custom("dehydrated", 12.0)
check(
    "hypovolemia_positive_pressure_interaction",
    dry12["cardiac_output_l_min"] < norm12["cardiac_output_l_min"] and dry12["map_mmHg"] < norm12["map_mmHg"],
    {
        "normal_CO": norm12["cardiac_output_l_min"],
        "dehydrated_CO": dry12["cardiac_output_l_min"],
        "normal_MAP": norm12["map_mmHg"],
        "dehydrated_MAP": dry12["map_mmHg"],
    },
)

coarse = roll(
    "mechanical_ventilation_peep12", steps=4, seed=19,
    config=HumanConfig(integration_step_min=0.25),
)
fine = roll(
    "mechanical_ventilation_peep12", steps=4, seed=19,
    config=HumanConfig(integration_step_min=0.125),
)
check(
    "timestep_convergence",
    (
        abs(coarse["pulmonary_transpulmonary_pressure_end_insp_cmH2O"] - fine["pulmonary_transpulmonary_pressure_end_insp_cmH2O"]) < 0.5
        and abs(coarse["pulmonary_overdistension_fraction"] - fine["pulmonary_overdistension_fraction"]) < 0.03
        and abs(coarse["cardiac_output_l_min"] - fine["cardiac_output_l_min"]) < 0.25
    ),
    {
        "d_PL_endinsp": abs(coarse["pulmonary_transpulmonary_pressure_end_insp_cmH2O"] - fine["pulmonary_transpulmonary_pressure_end_insp_cmH2O"]),
        "d_overdistension": abs(coarse["pulmonary_overdistension_fraction"] - fine["pulmonary_overdistension_fraction"]),
        "d_CO": abs(coarse["cardiac_output_l_min"] - fine["cardiac_output_l_min"]),
    },
)

check(
    "legacy_conservation_under_mechanics",
    (
        abs(peep12["co2_mass_balance_error_mmol"]) < 1e-7
        and abs(peep12["charge_balance_residual_mEq_l"]) < 1e-6
        and abs(peep12["cv_blood_volume_error_ml"]) < 1e-6
    ),
    {
        "CO2_residual_mmol": peep12["co2_mass_balance_error_mmol"],
        "charge_residual_mEq_l": peep12["charge_balance_residual_mEq_l"],
        "blood_volume_residual_ml": peep12["cv_blood_volume_error_ml"],
    },
)

payload = {
    "version": "0.15.0",
    "summary": {
        "passed": sum(c["passed"] for c in checks),
        "total": len(checks),
        "all_passed": all(c["passed"] for c in checks),
    },
    "checks": checks,
}
out = Path("validation/validation_results_v0.15.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload["summary"], indent=2))
if not payload["summary"]["all_passed"]:
    raise SystemExit(1)
