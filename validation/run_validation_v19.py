from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.19.0")

from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.physiology import Intervention


checks = []

def check(name, passed, values):
    checks.append({"name": name, "passed": bool(passed), "values": values})

# 1-2 SC insulin delay, tmax and mass.
env = HumanHomeostasisEnv()
env.reset(seed=1)
s = env.state
i0 = s.insulin_uU_ml
env.model.metabolism.add_exogenous_insulin(s, 2.0)
immediate_delta = s.insulin_uU_ml - i0
rates = []
for minute in range(1, 181):
    env.model.metabolism.step(s, exercise=0.0, dt_min=1.0)
    rates.append(s.sc_insulin_absorption_model_units_min)
peak_min = int(np.argmax(rates)) + 1
check(
    "sc_insulin_not_instantaneous",
    abs(immediate_delta) < 1e-12,
    {"immediate_insulin_delta_uU_ml": immediate_delta},
)
check(
    "sc_insulin_two_depot_pk_and_mass",
    80 <= peak_min <= 100 and abs(s.sc_insulin_mass_balance_error_model_units) < 1e-10,
    {
        "absorption_peak_min": peak_min,
        "mass_balance_error_model_units": s.sc_insulin_mass_balance_error_model_units,
        "absorbed_model_units_by_180min": s.sc_insulin_absorbed_model_units,
    },
)

# 3 glucagon counterregulation.
def hypo(maxegp):
    cfg = replace(HumanConfig(), glucagon_counterreg_egp_max_mg_kg_min=maxegp)
    e = HumanHomeostasisEnv(config=cfg)
    e.reset(seed=2)
    p = e.model.metabolism.p
    e.state.dalla_gp_mg_kg = 50.0 * p.VG_dl_kg
    e.model.metabolism._refresh_outputs(e.state)
    glycogen0 = e.state.liver_glycogen_g
    z = np.zeros(8, dtype=np.float32)
    for _ in range(6):
        e.model.integrate(e.state, e._decode_action(z), 5.0)
    return e.state, glycogen0

off, _ = hypo(0.0)
on, glycogen0 = hypo(1.0)
check(
    "glucagon_counterregulatory_direction",
    (
        on.glucose_mg_dl > off.glucose_mg_dl + 1.0
        and on.glucagon_counterregulatory_egp_mg_kg_min > 0.05
        and on.liver_glycogen_g < glycogen0
    ),
    {
        "glucose_no_counterreg_mg_dl": off.glucose_mg_dl,
        "glucose_with_counterreg_mg_dl": on.glucose_mg_dl,
        "glucagon_pg_ml": on.glucagon_pg_ml,
        "counterreg_egp_mg_kg_min": on.glucagon_counterregulatory_egp_mg_kg_min,
        "glycogen_used_g": glycogen0 - on.liver_glycogen_g,
    },
)

# 4 free water.
e = HumanHomeostasisEnv()
e.reset(seed=3)
s = e.state
tbw0, ecf0, icf0, na0 = s.total_body_water_l, s.ecf_volume_l, s.icf_volume_l, s.sodium_mmol_l
e.model.apply_instant_intervention(s, Intervention(oral_water_ml=900.0))
check(
    "free_water_osmotic_redistribution",
    (
        abs(s.total_body_water_l - (tbw0 + 0.9)) < 1e-10
        and s.ecf_volume_l > ecf0 and s.icf_volume_l > icf0
        and s.sodium_mmol_l < na0
        and abs(s.ecf_effective_tonicity_mOsm_l - s.icf_effective_tonicity_mOsm_l) < 1e-8
    ),
    {
        "ecf_gain_l": s.ecf_volume_l - ecf0,
        "icf_gain_l": s.icf_volume_l - icf0,
        "sodium_before": na0,
        "sodium_after": s.sodium_mmol_l,
        "tonicity_difference_mOsm_l": s.ecf_effective_tonicity_mOsm_l - s.icf_effective_tonicity_mOsm_l,
    },
)

# 5 isotonic saline predominantly ECF.
e = HumanHomeostasisEnv()
e.reset(seed=4)
s = e.state
ecf0, icf0 = s.ecf_volume_l, s.icf_volume_l
e.model.apply_instant_intervention(s, Intervention(saline_ml=900.0))
check(
    "isotonic_saline_predominantly_ecf",
    (
        s.ecf_volume_l - ecf0 > 0.75
        and abs(s.icf_volume_l - icf0) < 0.20
        and abs(s.ecf_effective_tonicity_mOsm_l - s.icf_effective_tonicity_mOsm_l) < 1e-8
    ),
    {
        "ecf_gain_l": s.ecf_volume_l - ecf0,
        "icf_change_l": s.icf_volume_l - icf0,
        "sodium_mmol_l": s.sodium_mmol_l,
    },
)

# 6-7 K directions and mass conservation.
def kcase(insulin=6.0, ph=7.40, exercise=0.0):
    e = HumanHomeostasisEnv()
    e.reset(seed=5)
    s = e.state
    t0 = s.ecf_potassium_mmol + s.icf_potassium_mmol
    s.insulin_uU_ml = insulin
    s.ph_arterial = ph
    e.model.renal._transcellular_potassium_step(s, exercise=exercise, dt=10.0)
    e.model.renal._update_derived_concentrations(s)
    t1 = s.ecf_potassium_mmol + s.icf_potassium_mmol
    return s, t0, t1

b, b0, b1 = kcase()
ins, i0, i1 = kcase(insulin=80.0)
acid, a0, a1 = kcase(ph=7.20)
ex, x0, x1 = kcase(exercise=1.0)
check(
    "transcellular_potassium_physiologic_directions",
    ins.potassium_mmol_l < b.potassium_mmol_l < acid.potassium_mmol_l
    and ex.potassium_mmol_l > b.potassium_mmol_l,
    {
        "basal_K": b.potassium_mmol_l,
        "high_insulin_K": ins.potassium_mmol_l,
        "acidemia_K": acid.potassium_mmol_l,
        "exercise_K": ex.potassium_mmol_l,
    },
)
max_k_mass_error = max(abs(b1-b0), abs(i1-i0), abs(a1-a0), abs(x1-x0))
check(
    "transcellular_potassium_mass_conservation",
    max_k_mass_error < 1e-10,
    {"max_total_K_error_mmol": max_k_mass_error},
)

# 8 scenario initialization must not double count an initial fluid challenge.
e = HumanHomeostasisEnv(scenario="saline_challenge_30ml_kg")
_, info = e.reset(seed=9)
mb = info["mass_balance"]
check(
    "scenario_t0_fluid_ledgers_close",
    (
        abs(mb["water_mass_balance_error_l"]) < 1e-12
        and abs(mb["sodium_mass_balance_error_mmol"]) < 1e-12
        and abs(mb["chloride_mass_balance_error_mmol"]) < 1e-12
        and abs(mb["water_partition_residual_l"]) < 1e-12
    ),
    {
        "water_error_l": mb["water_mass_balance_error_l"],
        "sodium_error_mmol": mb["sodium_mass_balance_error_mmol"],
        "chloride_error_mmol": mb["chloride_mass_balance_error_mmol"],
        "water_partition_residual_l": mb["water_partition_residual_l"],
    },
)

# 9 partial observability preserved.
clinical = HumanHomeostasisEnv()
full = HumanHomeostasisEnv(observation_profile="full")
check(
    "latent_p1_states_hidden_from_default_policy",
    (
        "sc_insulin_depot1_model_units" not in clinical.observation_names
        and "potassium_transcellular_target_mmol_l" not in clinical.observation_names
        and "sc_insulin_depot1_model_units" in full.observation_names
    ),
    {
        "clinical_observation_count": len(clinical.observation_names),
        "full_observation_count": len(full.observation_names),
    },
)

payload = {
    "version": "0.19.0",
    "classification": "verification/credibility checks; not clinical validation",
    "summary": {
        "passed": sum(c["passed"] for c in checks),
        "total": len(checks),
        "all_passed": all(c["passed"] for c in checks),
    },
    "checks": checks,
}
out = Path("validation/validation_results_v0.19.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload["summary"], indent=2))
if not payload["summary"]["all_passed"]:
    raise SystemExit(1)
