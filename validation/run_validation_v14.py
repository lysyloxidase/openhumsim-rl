from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.14.0")

from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, __version__

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.14.json"
REF = ROOT / "validation" / "external_reference_pulmonary_v0.14.json"
ZERO = np.zeros(8, dtype=np.float32)


def check(name, value, passed, tier, note=""):
    return {"name": name, "value": value, "pass": bool(passed), "tier": tier, "note": note}


def reset(scenario="baseline", seed=42, config=None):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    _, info = env.reset(seed=seed)
    return env, info


def main():
    checks = []
    prior = json.loads((ROOT / "validation/validation_results_v0.13.json").read_text())
    checks.append(check("v0.13 frozen scientific regression", prior["summary"], prior["summary"]["failed"] == 0, "regression"))
    checks.append(check("release version", __version__, __version__ == "0.14.0", "software-verification"))

    _, base = reset("baseline")
    bs = base["state"]
    checks.append(check(
        "healthy baseline remains recruited with little HPV redistribution",
        {
            "recruitment_fraction": bs["pulmonary_recruitment_fraction"],
            "HPV_resistance_multiplier": bs["pulmonary_hpv_resistance_multiplier"],
            "perfusion_redistribution_index": bs["pulmonary_perfusion_redistribution_index"],
            "PaO2_mmHg": bs["pao2_mmHg"],
        },
        bs["pulmonary_recruitment_fraction"] > 0.90
        and 1.0 <= bs["pulmonary_hpv_resistance_multiplier"] < 1.20
        and bs["pulmonary_perfusion_redistribution_index"] < 0.05
        and 80.0 <= bs["pao2_mmHg"] <= 105.0,
        "external-physiology-scale",
        "Healthy-rest gate; not a patient-specific reference interval."
    ))

    _, hpv = reset("vq_mismatch", seed=50)
    _, off = reset("hpv_disabled_vq_mismatch", seed=50)
    hs, os = hpv["state"], off["state"]
    checks.append(check(
        "regional HPV redistributes flow and improves oxygenation during V/Q mismatch",
        {
            "HPV_on_PaO2": hs["pao2_mmHg"],
            "HPV_off_PaO2": os["pao2_mmHg"],
            "HPV_on_redistribution": hs["pulmonary_perfusion_redistribution_index"],
            "HPV_off_redistribution": os["pulmonary_perfusion_redistribution_index"],
        },
        hs["pulmonary_perfusion_redistribution_index"] > os["pulmonary_perfusion_redistribution_index"] + 0.015
        and hs["pao2_mmHg"] > os["pao2_mmHg"] + 0.5,
        "human-mechanism-validation",
        "Direction anchored to human regional-perfusion HPV evidence (Asadi et al.); exact PaO2 gain is model-specific."
    ))

    on_cfg = replace(HumanConfig(), baseline_fio2=0.13, pulmonary_hpv_baseline_function_fraction=1.0)
    off_cfg = replace(HumanConfig(), baseline_fio2=0.13, pulmonary_hpv_baseline_function_fraction=0.0)
    on = HumanHomeostasisEnv(config=on_cfg); offe = HumanHomeostasisEnv(config=off_cfg)
    on.reset(seed=51); offe.reset(seed=51)
    ion = ioff = None
    for _ in range(4):
        _, _, ton, _, ion = on.step(ZERO)
        _, _, toff, _, ioff = offe.step(ZERO)
        if ton or toff: break
    checks.append(check(
        "global hypoxia raises pulmonary vascular resistance and pulmonary artery pressure via HPV",
        {
            "HPV_on_PVR_multiplier": ion["state"]["pulmonary_hpv_resistance_multiplier"],
            "HPV_on_PAP_mmHg": ion["state"]["pulmonary_artery_pressure_mmHg"],
            "HPV_off_PAP_mmHg": ioff["state"]["pulmonary_artery_pressure_mmHg"],
        },
        not ton and not toff
        and ion["state"]["pulmonary_hpv_resistance_multiplier"] > 1.5
        and ion["state"]["pulmonary_artery_pressure_mmHg"] > ioff["state"]["pulmonary_artery_pressure_mmHg"] + 3.0,
        "human-circulatory-mechanism-validation",
        "Direction anchored to human hypoxic exposure/transplant-lung data; v0.14 does not fit an individual PAP time course."
    ))

    _, collapsed = reset("dependent_derecruitment", seed=52)
    _, peep = reset("recruitment_peep", seed=52)
    cs, ps = collapsed["state"], peep["state"]
    checks.append(check(
        "pressure-dependent derecruitment and recruitment change gas exchange",
        {
            "low_pressure_recruitment": cs["pulmonary_recruitment_fraction"],
            "PEEP_recruitment": ps["pulmonary_recruitment_fraction"],
            "low_pressure_PaO2": cs["pao2_mmHg"],
            "PEEP_PaO2": ps["pao2_mmHg"],
            "low_pressure_Aa": cs["pulmonary_aa_gradient_mmHg"],
            "PEEP_Aa": ps["pulmonary_aa_gradient_mmHg"],
        },
        ps["pulmonary_recruitment_fraction"] > cs["pulmonary_recruitment_fraction"] + 0.25
        and ps["pao2_mmHg"] > cs["pao2_mmHg"] + 12.0
        and ps["pulmonary_aa_gradient_mmHg"] < cs["pulmonary_aa_gradient_mmHg"] - 10.0,
        "human-pressure-recruitment-mechanism",
        "Qualitative/scale gate anchored to Crotti et al.; thresholds are not patient-specific CT estimates."
    ))

    # Dynamic opening rather than scenario-only assignment.
    dyn, dinfo = reset("dependent_derecruitment", seed=53)
    r0 = dinfo["state"]["pulmonary_recruitment_fraction"]
    dyn.state.pulmonary_peep_cmH2O = dyn.config.pulmonary_recruitment_peep_cmH2O
    for _ in range(2):
        _, _, td, _, dinfo = dyn.step(ZERO)
        if td: break
    checks.append(check(
        "recruitment state evolves dynamically after pressure change",
        {"before": r0, "after_10min": dinfo["state"]["pulmonary_recruitment_fraction"]},
        not td and dinfo["state"]["pulmonary_recruitment_fraction"] > r0 + 0.20,
        "dynamic-mechanism-verification"
    ))

    # Numerical convergence.
    c = HumanConfig(cv_internal_step_s=0.04)
    e1 = HumanHomeostasisEnv(config=replace(c, integration_step_min=0.25), scenario="vq_mismatch")
    e2 = HumanHomeostasisEnv(config=replace(c, integration_step_min=0.125), scenario="vq_mismatch")
    e1.reset(seed=54); e2.reset(seed=54)
    i1=i2=None
    for _ in range(4):
        _,_,t1,_,i1=e1.step(ZERO); _,_,t2,_,i2=e2.step(ZERO)
        if t1 or t2: break
    delta={
        "PaO2_mmHg":abs(i1["state"]["pao2_mmHg"]-i2["state"]["pao2_mmHg"]),
        "HPV_R_multiplier":abs(i1["state"]["pulmonary_hpv_resistance_multiplier"]-i2["state"]["pulmonary_hpv_resistance_multiplier"]),
        "recruitment_fraction":abs(i1["state"]["pulmonary_recruitment_fraction"]-i2["state"]["pulmonary_recruitment_fraction"]),
    }
    checks.append(check(
        "HPV/recruitment timestep convergence", delta,
        not t1 and not t2 and delta["PaO2_mmHg"]<2.0 and delta["HPV_R_multiplier"]<0.08 and delta["recruitment_fraction"]<0.03,
        "numerical-verification"
    ))

    # Preserve v0.12 chemistry/conservation under new lung control.
    rng=np.random.default_rng(55); finite=True; max_co2=max_charge=0.0; unexpected=0
    for idx,scenario in enumerate(["baseline","vq_mismatch","dependent_derecruitment","recruitment_peep"]):
        env=HumanHomeostasisEnv(scenario=scenario); env.reset(seed=idx)
        for _ in range(8):
            action=rng.uniform(0.0,0.25,size=8).astype(np.float32)
            obs,reward,term,tr,info=env.step(action)
            finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(reward))
            max_co2=max(max_co2,abs(info["mass_balance"]["co2_mass_balance_error_mmol"]))
            max_charge=max(max_charge,abs(info["acid_base"]["charge_balance_residual_mEq_l"]))
            if term:
                if info.get("termination_reason") not in {"severe_hypoxemia","critical_low_pao2"}:
                    unexpected += 1
                break
            if tr: break
    stress={"finite":finite,"max_CO2_mass_residual_mmol":max_co2,"max_charge_residual_mEq_l":max_charge,"unexpected_nonpulmonary_terminations":unexpected}
    checks.append(check(
        "HPV/recruitment stress preserves chemistry and finite dynamics", stress,
        finite and max_co2<1e-8 and max_charge<1e-6 and unexpected==0,
        "physical-algorithmic-verification"
    ))

    result={
        "version":"0.14.0",
        "summary":{"passed":sum(x["pass"] for x in checks),"failed":sum(not x["pass"] for x in checks),"total":len(checks)},
        "checks":checks,
        "external_reference_file":str(REF.relative_to(ROOT)),
        "credibility_classification":{
            "HPV":"regional reduced precapillary resistance controller driven by alveolar PO2; human mechanism anchored, molecular signaling omitted",
            "regional_perfusion":"parallel conductance redistribution with whole-lung equivalent resistance coupled to the closed-loop pulmonary circulation",
            "recruitment":"six-unit pressure-threshold model with opening/closing hysteresis and finite time constants; not CT-personalized",
            "PEEP":"recruitment-state input only; not a complete mechanical ventilator or heart-lung interaction model",
            "VQ_and_diffusion":"v0.13 six-compartment forward model retained",
            "clinical_decision_use":"not supported"
        },
        "remaining_limitations":[
            "no continuous MIGET inversion or patient-specific regional V/Q distribution",
            "no pleural-pressure/gravity field or spatial lung geometry",
            "no dynamic lung/chest-wall compliance curve coupled to airway pressure and work of breathing",
            "no full PEEP effect on venous return, RV afterload and overdistension",
            "no surfactant kinetics or alveolar surface-tension model",
            "no molecular HPV signaling, endothelial modulation or disease-specific HPV impairment",
            "no ARDS/COPD/pulmonary-embolism clinical parameter estimation"
        ],
        "next_high_value_step":"v0.15 should add lung/chest-wall mechanics with pressure-volume hysteresis, overdistension and PEEP-heart interactions, then calibrate recruitment and V/Q outputs against CT/EIT/MIGET data before disease models."
    }
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["summary"],indent=2))
    for x in checks: print(("PASS" if x["pass"] else "FAIL"),x["tier"],x["name"])

if __name__ == "__main__":
    main()
