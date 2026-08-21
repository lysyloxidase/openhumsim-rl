from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.18.0")

import json
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.env import CLINICAL_OBSERVATION_NAMES, OBSERVATION_NAMES
from openhumsim_rl.oxygen_binding import OxygenBindingModel
from openhumsim_rl.physiology import HumanPhysiology, HumanState

ZERO = np.zeros(8, dtype=np.float32)
checks = []

def add(name, passed, values):
    checks.append({"name": name, "passed": bool(passed), "values": values})

# 1: policy action cannot create virtual L/min ventilation.
c = HumanHomeostasisEnv(scenario="hypoventilation", observation_profile="full")
a = HumanHomeostasisEnv(scenario="hypoventilation", observation_profile="full")
c.reset(seed=42); a.reset(seed=42)
act = ZERO.copy(); act[5] = 1.0
_, _, _, _, ic = c.step(ZERO)
_, _, _, _, ia = a.step(act)
sc, sa = ic["state"], ia["state"]
expected_va = sa["respiratory_rate_bpm"] * max(0.05, sa["tidal_volume_l"] - a.config.dead_space_l) * sa["ventilation_efficiency"]
add("pressure_mediated_ventilation_action", (
    ia["intervention"]["legacy_ventilation_support_l_min"] == 0.0
    and sa["respiratory_cycle_peak_airway_pressure_cmH2O"] > 5.0
    and sa["respiratory_cycle_ventilator_work_j_breath"] > 0.05
    and abs(sa["alveolar_ventilation_l_min"] - expected_va) < 1e-8
), {"control_VA": sc["alveolar_ventilation_l_min"], "assist_VA": sa["alveolar_ventilation_l_min"], "assist_VT": sa["tidal_volume_l"], "peak_Paw": sa["respiratory_cycle_peak_airway_pressure_cmH2O"], "ventilator_work_J_breath": sa["respiratory_cycle_ventilator_work_j_breath"]})

# 2: Bohr direction / standard P50 anchor.
ob = OxygenBindingModel(HumanConfig())
acid, normal, alk = [ob.saturation(40.0, ph, 40.0) for ph in (7.20, 7.40, 7.60)]
add("bohr_shift_coupled_hbo2", acid.saturation_fraction < normal.saturation_fraction < alk.saturation_fraction and acid.p50_mmHg > normal.p50_mmHg > alk.p50_mmHg and 25.5 < normal.p50_mmHg < 27.5,
    {"sat_pH7.2": acid.saturation_fraction, "sat_pH7.4": normal.saturation_fraction, "sat_pH7.6": alk.saturation_fraction, "P50_pH7.4": normal.p50_mmHg})

# 3: finite O2 extraction + unmet demand.
hp = HumanPhysiology(HumanConfig()); hs = HumanState(); hp.initialize_state(hs)
hs.cardiac_output_l_min = 1.5; hs.pao2_mmHg = 40.0; hs.ph_arterial = 7.25; hs.paco2_mmHg = 55.0; hs.vo2_demand_ml_min = 600.0
hp.respiratory.update_oxygen_transport(hs)
add("supply_dependent_vo2", hs.oxygen_debt_ml_min > 100.0 and hs.aerobic_fraction < 0.5 and abs(hs.oxygen_extraction_ratio - hp.cfg.oxygen_max_extraction_fraction) < 1e-9,
    {"DO2": hs.oxygen_delivery_ml_min, "VO2_demand": hs.vo2_demand_ml_min, "VO2_achieved": hs.vo2_ml_min, "oxygen_debt": hs.oxygen_debt_ml_min, "aerobic_fraction": hs.aerobic_fraction})

# 4: debt creates a hypoxic lactate source (reduced-order, one-substep lag).
e = HumanHomeostasisEnv(observation_profile="full"); e.reset(seed=10)
e.state.vo2_demand_ml_min = 500.0; e.state.oxygen_debt_ml_min = 300.0
l0 = e.state.lactate_mmol_l
e.model._substep(e.state, e._decode_action(ZERO), e.config.integration_step_min)
add("oxygen_debt_to_lactate_source", e.state.hypoxic_lactate_production_mmol_l_min > 0.0 and e.state.lactate_mmol_l > l0,
    {"lactate_before": l0, "lactate_after": e.state.lactate_mmol_l, "hypoxic_source_mmol_L_min": e.state.hypoxic_lactate_production_mmol_l_min})

# 5: regional wasted ventilation affects CO2 elimination.
b = HumanHomeostasisEnv(scenario="baseline", observation_profile="full"); vq = HumanHomeostasisEnv(scenario="vq_mismatch", observation_profile="full")
b.reset(seed=42); vq.reset(seed=42)
ib = iv = None
for _ in range(4):
    _, _, _, _, ib = b.step(ZERO); _, _, _, _, iv = vq.step(ZERO)
sb, sv = ib["state"], iv["state"]
add("regional_deadspace_couples_to_co2", sv["pulmonary_alveolar_dead_space_fraction"] > sb["pulmonary_alveolar_dead_space_fraction"] and sv["effective_co2_ventilation_l_min"] < sv["alveolar_ventilation_l_min"] and sv["paco2_mmHg"] > sb["paco2_mmHg"],
    {"baseline_deadspace": sb["pulmonary_alveolar_dead_space_fraction"], "vq_deadspace": sv["pulmonary_alveolar_dead_space_fraction"], "baseline_PaCO2": sb["paco2_mmHg"], "vq_PaCO2": sv["paco2_mmHg"]})

# 6: Hb/Hct now respond to plasma-volume shifts at conserved short-horizon Hb mass.
base = HumanHomeostasisEnv(scenario="baseline", observation_profile="full"); sal = HumanHomeostasisEnv(scenario="saline_challenge_30ml_kg", observation_profile="full"); dry = HumanHomeostasisEnv(scenario="dehydrated", observation_profile="full")
_, bi = base.reset(seed=1); _, si = sal.reset(seed=1); _, di = dry.reset(seed=1)
bs, ss, ds = bi["state"], si["state"], di["state"]
add("dynamic_hct_hb_with_fluid_shift", ss["hemoglobin_g_dl"] < bs["hemoglobin_g_dl"] < ds["hemoglobin_g_dl"] and ss["hematocrit_fraction"] < bs["hematocrit_fraction"] < ds["hematocrit_fraction"] and abs(ss["hemoglobin_mass_g"] - bs["hemoglobin_mass_g"]) < 1e-9,
    {"baseline_Hb": bs["hemoglobin_g_dl"], "saline_Hb": ss["hemoglobin_g_dl"], "dehydrated_Hb": ds["hemoglobin_g_dl"], "baseline_Hct": bs["hematocrit_fraction"], "saline_Hct": ss["hematocrit_fraction"], "Hb_mass_g": bs["hemoglobin_mass_g"]})

# 7: carbonate is divalent in reduced charge closure.
ch = HumanHomeostasisEnv(observation_profile="full"); _, chi = ch.reset(seed=3); cs = chi["state"]
side_expected = cs["bicarbonate_mmol_l"] + 2.0*cs["carbonate_mmol_l"] + cs["albumin_charge_mEq_l"] + cs["phosphate_charge_mEq_l"]
add("divalent_carbonate_charge_closure", abs(cs["strong_ion_difference_effective_mEq_l"] - side_expected) < 1e-8 and abs(cs["charge_balance_residual_mEq_l"]) < 1e-6,
    {"SIDE": cs["strong_ion_difference_effective_mEq_l"], "SIDE_reconstructed": side_expected, "reduced_charge_residual": cs["charge_balance_residual_mEq_l"]})

# 8: policy profile excludes selected latent states.
hidden = {"pulmonary_vq_log_sd", "pulmonary_hpv_resistance_multiplier", "dalla_egp_mg_kg_min", "probe_effect_site_mg_l", "oxygen_debt_ml_min"}
ce = HumanHomeostasisEnv(); co, ci = ce.reset(seed=5)
measurement_only={"sensor_glucose_mg_dl","cgm_measurement_age_min","monitor_measurement_age_min","blood_gas_measurement_age_min","chemistry_measurement_age_min","hemodynamic_measurement_age_min"}
add("clinical_observation_profile_no_selected_latents", co.shape == (len(CLINICAL_OBSERVATION_NAMES),) and hidden.isdisjoint(CLINICAL_OBSERVATION_NAMES) and (set(CLINICAL_OBSERVATION_NAMES)-measurement_only).issubset(OBSERVATION_NAMES),
    {"clinical_dim": len(CLINICAL_OBSERVATION_NAMES), "full_dim": len(OBSERVATION_NAMES), "hidden_excluded": sorted(hidden)})

# 9: reward sees new physiologic harms.
r = HumanHomeostasisEnv(observation_profile="full"); r.reset(seed=5)
clean_total, clean = r._reward(ZERO)
r.state.respiratory_ventilator_asynchrony_index_pct = 60.0; r.state.respiratory_cycle_auto_peep_cmH2O = 8.0; r.state.pulmonary_overdistension_fraction = 0.75; r.state.oxygen_debt_ml_min = 250.0
bad_total, bad = r._reward(ZERO)
add("reward_penalizes_mechanistic_harms", bad_total < clean_total and bad["ventilator_asynchrony"] < clean["ventilator_asynchrony"] and bad["auto_peep"] < clean["auto_peep"] and bad["overdistension"] < clean["overdistension"] and bad["oxygen_debt"] < clean["oxygen_debt"],
    {"clean_reward": clean_total, "challenged_reward": bad_total, "challenged_terms": {k: bad[k] for k in ("ventilator_asynchrony","auto_peep","overdistension","oxygen_debt")}})

payload = {"version": "0.18.0", "release_type": "credibility_refactor", "summary": {"passed": sum(x["passed"] for x in checks), "total": len(checks), "all_passed": all(x["passed"] for x in checks)}, "checks": checks}
Path("validation/validation_results_v0.18.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload["summary"], indent=2))
if not payload["summary"]["all_passed"]:
    raise SystemExit(1)
