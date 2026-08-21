from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.6.0")

import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.metabolism_dallaman import DallaManNormalParameters
from openhumsim_rl.renal import RenalModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.6.json"
REF = json.loads((ROOT / "validation" / "reference_targets_v0.6.json").read_text())
ZERO = np.zeros(8, dtype=np.float32)


def run(env, steps, action=ZERO):
    rows=[]; total=0.0; info=None; term=False; trunc=False
    for _ in range(steps):
        _, r, term, trunc, info = env.step(action)
        total += float(r); rows.append(info)
        if term or trunc: break
    return rows, total, term, trunc, info


def check(name, value, condition, tier, note=""):
    return {"name": name, "value": value, "pass": bool(condition), "tier": tier, "note": note}


def main():
    checks=[]

    # 1) Source parameter fidelity: exact Table-I parameter implementation.
    p = asdict(DallaManNormalParameters())
    expected = {
        "VG_dl_kg":1.88,"k1_min":0.065,"k2_min":0.079,"VI_l_kg":0.05,
        "m1_min":0.190,"m2_min":0.484,"m4_min":0.194,"m5_min_kg_pmol":0.0304,
        "m6":0.6471,"HEb":0.60,"kmax_min":0.0558,"kmin_min":0.0080,
        "kabs_min":0.057,"kgri_min":0.0558,"f":0.90,"a_mg_inv":0.00013,
        "b":0.82,"c_mg_inv":0.00236,"d":0.010,"kp1_mg_kg_min":2.70,
        "kp2_min":0.0021,"kp3_mg_kg_min_per_pmol_l":0.009,
        "kp4_mg_kg_min_per_pmol_kg":0.0618,"ki_min":0.0079,
        "Fcns_mg_kg_min":1.0,"Vm0_mg_kg_min":2.50,
        "Vmx_mg_kg_min_per_pmol_l":0.047,"Km0_mg_kg":225.59,"p2U_min":0.0331,
        "K_pmol_kg_per_mg_dl":2.30,"alpha_min":0.050,
        "beta_pmol_kg_min_per_mg_dl":0.11,"gamma_min":0.5,"ke1_min":0.0005,
        "ke2_mg_kg":339.0,
    }
    max_param_error=max(abs(p[k]-v) for k,v in expected.items())
    checks.append(check("Dalla Man Table-I parameter transcription", max_param_error, max_param_error < 1e-12, "source-fidelity"))

    # 2) Basal state / challenge response (challenge is not called a validated OGTT).
    env=HumanHomeostasisEnv(scenario="oral_glucose_75g"); _, i0=env.reset(seed=1)
    rows,_,term,_,info=run(env,84)
    peak_g=max((x["state"]["glucose_mg_dl"],x["time_min"]) for x in rows)
    peak_i=max(x["state"]["insulin_uU_ml"] for x in rows)
    checks.append(check("75-g challenge produces bounded postprandial excursion", {"peak_glucose":peak_g[0],"time_min":peak_g[1],"peak_insulin_uU_ml":peak_i,"g420":info["state"]["glucose_mg_dl"]}, (140<peak_g[0]<200 and 30<=peak_g[1]<=120 and 80<info["state"]["glucose_mg_dl"]<110 and not term), "qualitative-biological", "Not an independently validated OGTT protocol."))
    checks.append(check("Dalla GI mass conservation", abs(info["metabolism"]["gi_mass_balance_error_mg"]), abs(info["metabolism"]["gi_mass_balance_error_mg"])<1e-5, "physical-invariant"))

    # 3) External hemodynamic scale against Broomé 2013 model output.
    env=HumanHomeostasisEnv(); _, ib=env.reset(seed=42); s=ib["state"]; br=REF["broome_normal_reference"]
    cv_errors={
        "CO_rel":abs(s["cardiac_output_l_min"]-br["cardiac_output_l_min"])/br["cardiac_output_l_min"],
        "SBP_abs":abs(s["systolic_pressure_mmHg"]-br["systemic_systolic_mmHg"]),
        "DBP_abs":abs(s["diastolic_pressure_mmHg"]-br["systemic_diastolic_mmHg"]),
        "RAP_abs":abs(s["central_venous_pressure_mmHg"]-br["mean_right_atrial_mmHg"]),
        "PAP_abs":abs(s["pulmonary_artery_pressure_mmHg"]-br["mean_pulmonary_artery_mmHg"]),
    }
    cv_ok=cv_errors["CO_rel"]<0.20 and cv_errors["SBP_abs"]<20 and cv_errors["DBP_abs"]<20 and cv_errors["RAP_abs"]<4 and cv_errors["PAP_abs"]<7
    checks.append(check("Closed-loop baseline vs Broomé quantitative scale", {"sim":{k:s[k] for k in ["cardiac_output_l_min","systolic_pressure_mmHg","diastolic_pressure_mmHg","central_venous_pressure_mmHg","pulmonary_artery_pressure_mmHg"]},"errors":cv_errors}, cv_ok, "external-quantitative", "Same class of model, not a parameter-identical reproduction."))

    env=HumanHomeostasisEnv(); env.reset(seed=42); a=ZERO.copy(); a[2]=1.0
    rows,_,term,_,ie=run(env,6,a); se=ie["state"]
    checks.append(check("Exercise hemodynamic scale", {"HR":se["heart_rate_bpm"],"CO":se["cardiac_output_l_min"],"MAP":se["map_mmHg"]}, (not term and se["cardiac_output_l_min"]>10.0 and 135<se["heart_rate_bpm"]<170), "external-qualitative", "Broomé exercise case reports HR 144 and CO >10 L/min."))

    # 4) Oxygen transport / Fick identity.
    oxy=ie["oxygen_transport"]
    checks.append(check("Fick oxygen identity", abs(oxy["fick_residual_ml_min"]), abs(oxy["fick_residual_ml_min"])<1e-8, "physical-invariant"))
    checks.append(check("Exercise oxygen supply remains positive", {"DO2":oxy["oxygen_delivery_ml_min"],"VO2":se["vo2_ml_min"],"OER":oxy["oxygen_extraction_ratio"],"SvO2":oxy["mixed_venous_o2_sat_pct"]}, oxy["oxygen_supply_margin_ml_min"]>0 and oxy["oxygen_extraction_ratio"]<0.90, "physiological-sanity"))

    # 5) Renal autoregulation semantics.
    plateau=[RenalModel.pressure_autoregulation_factor(x) for x in (80,100,120,140)]
    low=RenalModel.pressure_autoregulation_factor(60)
    checks.append(check("Renal pressure-autoregulation plateau", {"80_140":plateau,"at60":low}, max(plateau)-min(plateau)<1e-12 and low<0.7, "mechanistic-direction"))

    # 6) Acid-base rules encoded by scenario and chronic target slope.
    env=HumanHomeostasisEnv(scenario="respiratory_acidosis"); _, ia=env.reset(seed=1)
    sa=ia["state"]
    acute_slope=(sa["bicarbonate_mmol_l"]-24.0)/((sa["paco2_mmHg"]-40.0)/10.0)
    chronic_target=24.0 + env.config.renal_co2_compensation_gain*(60.0-40.0)
    chronic_slope=(chronic_target-24.0)/2.0
    checks.append(check("Respiratory-acidosis compensation slopes", {"acute_mmol_per_10":acute_slope,"chronic_target_mmol_per_10":chronic_slope,"renal_tau_min":env.config.renal_bicarbonate_tau_min}, abs(acute_slope-1.0)<1e-9 and abs(chronic_slope-3.5)<1e-9 and env.config.renal_bicarbonate_tau_min>=1440, "medical-rule-consistency", "Matches ATS expected slopes; dynamics remain reduced-order."))

    # 7) PBPK structural credibility: exact mass balance and renal-function direction.
    n=HumanHomeostasisEnv(scenario="pbpk_oral_dose"); n.reset(seed=4); _,_,_,_,inn=run(n,36)
    r=HumanHomeostasisEnv(scenario="reduced_renal_function"); r.reset(seed=4); ar=ZERO.copy(); ar[7]=1.0
    r.step(ar); _,_,_,_,irr=run(r,35)
    pbpk_mass=max(abs(inn["pbpk"]["mass_balance_error_mg"]),abs(irr["pbpk"]["mass_balance_error_mg"]))
    direction=irr["pbpk"]["renal_clearance_l_min"] < inn["pbpk"]["renal_clearance_l_min"]
    checks.append(check("PBPK mass conservation", pbpk_mass, pbpk_mass<1e-8, "physical-invariant"))
    checks.append(check("PBPK renal-clearance coupling", {"normal":inn["pbpk"]["renal_clearance_l_min"],"reduced":irr["pbpk"]["renal_clearance_l_min"]}, direction, "mechanistic-direction", "No real-drug PK data used; pharmacological validation remains absent."))

    # 8) Whole-body water/ion conservation under interventions.
    env=HumanHomeostasisEnv(); env.reset(seed=7); af=ZERO.copy(); af[3]=0.2; af[6]=0.2
    _,_,_,_,im=run(env,24,af); mb=im["mass_balance"]
    max_ion=max(abs(mb[k]) for k in ["sodium_mass_balance_error_mmol","chloride_mass_balance_error_mmol","potassium_mass_balance_error_mmol"])
    checks.append(check("Water/ion ledger closure", {"water_L":mb["water_mass_balance_error_l"],"max_ion_mmol":max_ion}, abs(mb["water_mass_balance_error_l"])<1e-9 and max_ion<1e-8, "chemical-physical-invariant"))

    # 9) Numerical convergence: selected outputs at smaller integration steps.
    c1=HumanConfig(cv_internal_step_s=0.02); c2=replace(c1,cv_internal_step_s=0.01)
    e1=HumanHomeostasisEnv(config=c1); e2=HumanHomeostasisEnv(config=c2); _,x1=e1.reset(seed=8); _,x2=e2.reset(seed=8)
    cv_conv={"MAP_abs":abs(x1["state"]["map_mmHg"]-x2["state"]["map_mmHg"]),"CO_abs":abs(x1["state"]["cardiac_output_l_min"]-x2["state"]["cardiac_output_l_min"])}
    checks.append(check("Cardiovascular timestep convergence", cv_conv, cv_conv["MAP_abs"]<4 and cv_conv["CO_abs"]<0.3, "numerical"))

    d1=HumanHomeostasisEnv(config=HumanConfig(dalla_internal_step_min=.05),scenario="oral_glucose_75g"); d2=HumanHomeostasisEnv(config=HumanConfig(dalla_internal_step_min=.025),scenario="oral_glucose_75g"); d1.reset(seed=9); d2.reset(seed=9)
    _,_,_,_,j1=run(d1,12); _,_,_,_,j2=run(d2,12)
    dalla_conv={"glucose_abs":abs(j1["state"]["glucose_mg_dl"]-j2["state"]["glucose_mg_dl"]),"insulin_abs":abs(j1["state"]["insulin_uU_ml"]-j2["state"]["insulin_uU_ml"])}
    checks.append(check("Dalla numerical convergence", dalla_conv, dalla_conv["glucose_abs"]<0.2 and dalla_conv["insulin_abs"]<0.2, "numerical"))

    # 10) Random-action stress, deliberately safe-amplitude but multi-organ.
    max_mass={"water":0.0,"ion":0.0,"pbpk":0.0,"fick":0.0}; terminations=[]; finite=True
    rng=np.random.default_rng(20260817)
    for seed in range(10):
        env=HumanHomeostasisEnv(config=HumanConfig(episode_minutes=120)); env.reset(seed=seed)
        for _ in range(24):
            a=rng.uniform(0,0.25,size=8).astype(np.float32)
            a[2]=rng.uniform(0,0.7); a[4]=rng.uniform(0,0.2); a[5]=rng.uniform(0,0.15)
            obs,rw,term,trunc,ii=env.step(a)
            finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(rw))
            m=ii["mass_balance"]
            max_mass["water"]=max(max_mass["water"],abs(m["water_mass_balance_error_l"]))
            max_mass["ion"]=max(max_mass["ion"],*[abs(m[k]) for k in ["sodium_mass_balance_error_mmol","chloride_mass_balance_error_mmol","potassium_mass_balance_error_mmol"]])
            max_mass["pbpk"]=max(max_mass["pbpk"],abs(ii["pbpk"]["mass_balance_error_mg"]))
            max_mass["fick"]=max(max_mass["fick"],abs(ii["oxygen_transport"]["fick_residual_ml_min"]))
            if term or trunc:
                if term: terminations.append(ii.get("termination_reason"))
                break
    stress_ok=finite and max_mass["water"]<1e-8 and max_mass["ion"]<1e-7 and max_mass["pbpk"]<1e-7 and max_mass["fick"]<1e-7
    checks.append(check("10-seed multi-organ stress test", {"finite":finite,"max_residuals":max_mass,"terminations":terminations}, stress_ok, "algorithmic-numerical"))

    # 11) Explicit epistemic failures: these are expected FAIL/UNVALIDATED, not bugs hidden as passes.
    limitations = {
        "full_electroneutrality": False,
        "total_inorganic_carbon_conservation": False,
        "real_drug_pbpk_calibration": False,
        "patient_specific_validation": False,
        "default_observation_is_markov_complete": False,
        "kdigo_aki_diagnosis": False,
        "subcutaneous_insulin_pk": False,
        "vq_distribution_and_shunt": False,
    }

    result={
        "version":"0.6.0",
        "checks":checks,
        "summary":{
            "passed":sum(c["pass"] for c in checks),
            "failed":sum(not c["pass"] for c in checks),
            "total":len(checks),
        },
        "known_unvalidated_or_missing":limitations,
        "interpretation": "Passing checks demonstrate source fidelity, internal conservation, numerical stability and selected external-scale plausibility. They do not constitute clinical validation of a whole-human simulator.",
        "references":REF["sources"],
    }
    OUT.write_text(json.dumps(result,indent=2))
    print(json.dumps(result["summary"],indent=2))
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["tier"], c["name"], c["value"])

if __name__ == "__main__":
    main()
