from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.12.0")

from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, __version__
from openhumsim_rl.acid_base import PhysicochemicalAcidBaseModel
from openhumsim_rl.blood_gas import WholeBloodGasChemistryModel
from openhumsim_rl.physiology import HumanState

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.12.json"
REF = ROOT / "validation" / "external_reference_bloodgas_v0.12.json"
ZERO = np.zeros(8, dtype=np.float32)


def check(name, value, passed, tier, note=""):
    return {"name": name, "value": value, "pass": bool(passed), "tier": tier, "note": note}


def rollout(env, minutes, action=ZERO, seed=1):
    _, info = env.reset(seed=seed)
    terminated = truncated = False
    for _ in range(int(round(minutes / env.config.agent_step_min))):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info, terminated, truncated


def standard_snapshot():
    c = HumanConfig()
    s = HumanState()
    s.paco2_mmHg = 40.0
    s.pao2_mmHg = 100.0
    s.spo2_pct = 97.2
    ab = PhysicochemicalAcidBaseModel(c)
    ab.initialize_state(s)
    d = ab.snapshot_for_pco2(s, 40.0)
    bg = WholeBloodGasChemistryModel(c, ab)
    wb = bg.snapshot(plasma_ph=d.ph, pco2_mmHg=40.0, oxygen_sat_fraction=0.972,
                     plasma_chloride_mmol_l=s.chloride_mmol_l)
    return c, s, d, bg, wb


def main():
    checks = []
    prior = json.loads((ROOT / "validation/validation_results_v0.11.json").read_text())
    checks.append(check("v0.11 frozen scientific regression", prior["summary"], prior["summary"]["failed"] == 0, "regression"))
    checks.append(check("release version", __version__, __version__ == "0.12.0", "software-verification"))

    c, s, d, bg, wb = standard_snapshot()
    ml_dl = wb.total_co2_mmol_l_blood * c.co2_gas_molar_volume_l_per_mol_stpd / 10.0
    checks.append(check(
        "O'Neill-Robbins arterial whole-blood CO2 content scale",
        {"model_ml_CO2_per_100ml": ml_dl, "external_target_ml_CO2_per_100ml": 44.6,
         "model_mmol_l": wb.total_co2_mmol_l_blood, "pH": d.ph},
        abs(ml_dl - 44.6) < 0.8,
        "external-blood-chemistry",
        "Scale benchmark at PCO2=40 mmHg and standard arterial oxygenation; not a reproduction of the full O'Neill-Robbins model."
    ))

    expected_rcl = c.funder_rcl_intercept - c.funder_rcl_ph_slope * d.ph
    expected_rh = c.funder_rh_intercept - c.funder_rh_ph_slope * d.ph
    checks.append(check(
        "Funder-Wieth Donnan + human RBC pH",
        {"plasma_pH": d.ph, "rCl": wb.donnan_rcl, "rH": wb.donnan_rh, "RBC_pH": wb.rbc_ph,
         "Kummerow_RBC_pH_mean": 7.19, "Kummerow_SD": 0.04},
        abs(wb.donnan_rcl - expected_rcl) < 1e-12 and abs(wb.donnan_rh - expected_rh) < 1e-12 and 7.15 <= wb.rbc_ph <= 7.23,
        "external-cellular-physiology"
    ))

    carb_std = bg.carbamino_fraction(40.0, 0.972)
    oxy = bg.snapshot(plasma_ph=7.40, pco2_mmHg=46.0, oxygen_sat_fraction=0.97, plasma_chloride_mmol_l=103.0)
    deoxy = bg.snapshot(plasma_ph=7.40, pco2_mmHg=46.0, oxygen_sat_fraction=0.75, plasma_chloride_mmol_l=103.0)
    checks.append(check(
        "Dash carbamino anchor + Haldane direction",
        {"standard_carbamino_fraction": carb_std, "published_standard_fraction": 0.131,
         "oxygenated_content_mmol_l": oxy.total_co2_mmol_l_blood,
         "deoxygenated_content_mmol_l": deoxy.total_co2_mmol_l_blood},
        abs(carb_std - 0.131) < 1e-12 and deoxy.total_co2_mmol_l_blood > oxy.total_co2_mmol_l_blood,
        "external-hemoglobin-biochemistry",
        "Haldane term is reduced-order; it does not reproduce the complete Dash binding equations."
    ))

    env = HumanHomeostasisEnv(scenario="baseline")
    _, base = env.reset(seed=42)
    bs = base["state"]
    av_ok = (
        bs["mixed_venous_pco2_mmHg"] > bs["paco2_mmHg"]
        and bs["mixed_venous_ph"] < bs["ph_arterial"]
        and bs["mixed_venous_total_co2_mmol_l_blood"] > bs["arterial_total_co2_mmol_l_blood"]
        and 0.5 <= bs["chloride_shift_plasma_mmol_l"] <= 3.0
        and 1.0 <= (bs["rbc_chloride_mmol_l"] - bs["mixed_venous_rbc_chloride_mmol_l"]) <= 3.0
        and bs["mixed_venous_hb_bound_chloride_gain_mmol_l_rbc"] > 0.0
        and abs(bs["chloride_shift_balance_residual_mmol_l_blood"]) < 1e-10
        and abs(bs["co2_fick_content_residual_mmol_l"]) < 1e-6
    )
    checks.append(check("arteriovenous Fick CO2 + local chloride redistribution", {
        k: bs[k] for k in ["paco2_mmHg","mixed_venous_pco2_mmHg","ph_arterial","mixed_venous_ph",
                            "arterial_total_co2_mmol_l_blood","mixed_venous_total_co2_mmol_l_blood",
                            "chloride_shift_plasma_mmol_l","rbc_chloride_mmol_l","mixed_venous_rbc_chloride_mmol_l",
                            "mixed_venous_hb_bound_chloride_gain_mmol_l_rbc","chloride_shift_balance_residual_mmol_l_blood",
                            "co2_fick_content_residual_mmol_l"]
    }, av_ok, "physical-biochemical-verification"))

    info, term, _ = rollout(HumanHomeostasisEnv(config=HumanConfig(cv_internal_step_s=0.04)), 120.0, seed=7)
    checks.append(check("exchangeable CO2 ledger conservation", {
        "carbon_mass_residual_mmol": info["mass_balance"]["co2_mass_balance_error_mmol"],
        "content_solver_residual_mmol_l": info["state"]["co2_content_solver_residual_mmol_l"],
        "generated_mmol": info["state"]["co2_generated_mmol"],
        "eliminated_mmol": info["state"]["co2_eliminated_mmol"],
    }, not term and abs(info["mass_balance"]["co2_mass_balance_error_mmol"]) < 1e-8 and abs(info["state"]["co2_content_solver_residual_mmol_l"]) < 1e-5,
    "mass-conservation-verification",
    "The conserved pool is a lumped rapidly exchangeable CO2 capacitance, not total anatomical body-carbon inventory."))

    no, tn, _ = rollout(HumanHomeostasisEnv(scenario="respiratory_acidosis"), 30.0, seed=8)
    support = ZERO.copy(); support[5] = 0.25
    yes, ty, _ = rollout(HumanHomeostasisEnv(scenario="respiratory_acidosis"), 30.0, action=support, seed=8)
    checks.append(check("ventilation support changes conserved carbon in correct direction", {
        "no_support": {k:no["state"][k] for k in ["paco2_mmHg","ph_arterial","exchangeable_co2_pool_mmol"]},
        "support": {k:yes["state"][k] for k in ["paco2_mmHg","ph_arterial","exchangeable_co2_pool_mmol"]},
    }, not tn and not ty and yes["state"]["paco2_mmHg"] < no["state"]["paco2_mmHg"] and yes["state"]["exchangeable_co2_pool_mmol"] < no["state"]["exchangeable_co2_pool_mmol"] and yes["state"]["ph_arterial"] > no["state"]["ph_arterial"],
    "multi-organ-physiology"))

    c0 = HumanConfig(cv_internal_step_s=0.04)
    i1, t1, _ = rollout(HumanHomeostasisEnv(config=replace(c0, integration_step_min=0.25)), 60.0, seed=10)
    i2, t2, _ = rollout(HumanHomeostasisEnv(config=replace(c0, integration_step_min=0.125)), 60.0, seed=10)
    dt = {"delta_PaCO2_mmHg": abs(i1["state"]["paco2_mmHg"]-i2["state"]["paco2_mmHg"]),
          "delta_pH": abs(i1["state"]["ph_arterial"]-i2["state"]["ph_arterial"])}
    checks.append(check("whole-blood carbon timestep convergence", dt, not t1 and not t2 and dt["delta_PaCO2_mmHg"] < 0.5 and dt["delta_pH"] < 0.005, "numerical-verification"))

    max_carbon=max_fick=max_cl=max_charge=0.0; unexpected=0
    for seed in range(10):
        e=HumanHomeostasisEnv(config=HumanConfig(cv_internal_step_s=0.04)); e.reset(seed=seed)
        rng=np.random.default_rng(seed)
        for _ in range(18):
            a=rng.uniform(0.0,0.20,size=8).astype(np.float32)
            _,_,t,tr,inf=e.step(a)
            max_carbon=max(max_carbon,abs(inf["mass_balance"]["co2_mass_balance_error_mmol"]))
            max_fick=max(max_fick,abs(inf["state"]["co2_fick_content_residual_mmol_l"]))
            max_cl=max(max_cl,abs(inf["state"]["chloride_shift_balance_residual_mmol_l_blood"]))
            max_charge=max(max_charge,abs(inf["acid_base"]["charge_balance_residual_mEq_l"]))
            if t: unexpected += 1; break
            if tr: break
    stress={"max_carbon_mass_residual_mmol":max_carbon,"max_fick_CO2_residual_mmol_l":max_fick,
            "max_chloride_redistribution_residual_mmol_l_blood":max_cl,"max_plasma_charge_residual_mEq_l":max_charge,
            "unexpected_terminations":unexpected}
    checks.append(check("10-seed blood-gas/chemical stress residuals", stress,
                        max_carbon<1e-8 and max_fick<1e-5 and max_cl<1e-10 and max_charge<1e-6 and unexpected==0,
                        "physical-algorithmic-verification"))

    result={
        "version":"0.12.0",
        "summary":{"passed":sum(x["pass"] for x in checks),"failed":sum(not x["pass"] for x in checks),"total":len(checks)},
        "checks":checks,
        "external_reference_file":str(REF.relative_to(ROOT)),
        "credibility_classification":{
            "plasma_acid_base":"charge-constrained Stewart-Figge closure with explicit carbonate species",
            "whole_blood_CO2":"external-scale benchmarked and carbon-mass-conserving in a reduced rapidly exchangeable pool",
            "RBC_Donnan":"empirical human Funder-Wieth equilibrium relationships",
            "hemoglobin_CO2":"standard-state carbamino anchor plus reduced Haldane dependence; not full Dash/O'Neill equations",
            "chloride_shift":"local arteriovenous redistribution is charge/mass balanced; Band-3 kinetics are not modeled",
            "clinical_decision_use":"not supported"
        },
        "remaining_limitations":[
            "effective exchangeable CO2 pool is not a full anatomical total-body carbon inventory",
            "no explicit carbonic anhydrase/Band-3 kinetic transit model",
            "no complete multi-site Hb proton/CO2/2,3-DPG/temperature binding model",
            "fixed hematocrit and simplified plasma/RBC water fractions",
            "no V/Q distribution or shunt",
            "no explicit tissue/interstitial CO2 compartments"
        ],
        "next_high_value_step":"v0.13 should add multi-compartment lung V/Q + shunt and capillary gas exchange, or first reproduce a published whole-blood CO2 dissociation dataset before further organ expansion."
    }
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2))
    for x in checks: print(('PASS' if x['pass'] else 'FAIL'),x['tier'],x['name'])

if __name__=='__main__': main()
