from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.13.0")

from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, __version__

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.13.json"
REF = ROOT / "validation" / "external_reference_pulmonary_v0.13.json"
ZERO = np.zeros(8, dtype=np.float32)


def check(name, value, passed, tier, note=""):
    return {"name": name, "value": value, "pass": bool(passed), "tier": tier, "note": note}


def main():
    checks = []
    prior = json.loads((ROOT / "validation/validation_results_v0.12.json").read_text())
    checks.append(check("v0.12 frozen scientific regression", prior["summary"], prior["summary"]["failed"] == 0, "regression"))
    checks.append(check("release version", __version__, __version__ == "0.13.0", "software-verification"))

    base_env = HumanHomeostasisEnv(scenario="baseline")
    _, base = base_env.reset(seed=42)
    bs = base["state"]
    checks.append(check(
        "resting pulmonary gas-exchange scale",
        {k: bs[k] for k in ["pao2_mmHg", "paco2_mmHg", "spo2_pct", "pulmonary_aa_gradient_mmHg", "pulmonary_enghoff_dead_space_fraction"]},
        80.0 <= bs["pao2_mmHg"] <= 105.0
        and 35.0 <= bs["paco2_mmHg"] <= 46.0
        and bs["spo2_pct"] >= 94.0
        and 0.0 <= bs["pulmonary_aa_gradient_mmHg"] <= 15.0
        and 0.20 <= bs["pulmonary_enghoff_dead_space_fraction"] <= 0.45,
        "external-physiology-scale",
        "Broad healthy-rest scale, not a patient-specific arterial blood-gas reference interval."
    ))

    # Warren 1991: transit ~1.05 s rest and 0.42-0.46 s during exercise; A-a widens.
    ex_env = HumanHomeostasisEnv(scenario="baseline")
    _, rest = ex_env.reset(seed=42)
    a = ZERO.copy(); a[2] = 1.0
    ex = None
    for _ in range(6):
        _, _, term, _, ex = ex_env.step(a)
        if term:
            break
    checks.append(check(
        "exercise capillary-transit scale and A-a widening",
        {
            "model_rest_transit_s": rest["state"]["pulmonary_capillary_transit_time_s"],
            "Warren_rest_transit_s": 1.05,
            "model_exercise_transit_s": ex["state"]["pulmonary_capillary_transit_time_s"],
            "Warren_exercise_range_s": [0.42, 0.46],
            "model_rest_Aa_mmHg": rest["state"]["pulmonary_aa_gradient_mmHg"],
            "model_exercise_Aa_mmHg": ex["state"]["pulmonary_aa_gradient_mmHg"],
            "Warren_heavy_exercise_Aa_mmHg": 22.3,
        },
        not term
        and abs(rest["state"]["pulmonary_capillary_transit_time_s"] - 1.05) < 0.25
        and 0.35 <= ex["state"]["pulmonary_capillary_transit_time_s"] <= 0.55
        and ex["state"]["pulmonary_aa_gradient_mmHg"] > rest["state"]["pulmonary_aa_gradient_mmHg"],
        "external-exercise-physiology",
        "Warren's PaO2=85 mmHg cohort were endurance athletes; v0.13 validates transit-time scale and A-a direction rather than forcing that exact PaO2."
    ))

    _, vq = HumanHomeostasisEnv(scenario="vq_mismatch").reset(seed=42)
    checks.append(check(
        "V/Q mismatch produces hypoxemia and wasted ventilation",
        {
            "baseline_PaO2": bs["pao2_mmHg"], "vq_PaO2": vq["state"]["pao2_mmHg"],
            "baseline_Aa": bs["pulmonary_aa_gradient_mmHg"], "vq_Aa": vq["state"]["pulmonary_aa_gradient_mmHg"],
            "baseline_VDVT": bs["pulmonary_enghoff_dead_space_fraction"], "vq_VDVT": vq["state"]["pulmonary_enghoff_dead_space_fraction"],
        },
        vq["state"]["pao2_mmHg"] < bs["pao2_mmHg"] - 15.0
        and vq["state"]["pulmonary_aa_gradient_mmHg"] > bs["pulmonary_aa_gradient_mmHg"] + 15.0
        and vq["state"]["pulmonary_enghoff_dead_space_fraction"] > bs["pulmonary_enghoff_dead_space_fraction"],
        "mechanistic-pulmonary-physiology"
    ))

    _, sh = HumanHomeostasisEnv(scenario="pulmonary_shunt").reset(seed=42)
    _, df = HumanHomeostasisEnv(scenario="diffusion_limitation").reset(seed=42)
    checks.append(check(
        "true shunt and diffusion limitation are separable mechanisms",
        {
            "shunt_PaO2": sh["state"]["pao2_mmHg"],
            "diffusion_PaO2": df["state"]["pao2_mmHg"],
            "baseline_equilibration": bs["pulmonary_diffusion_equilibration_fraction"],
            "diffusion_equilibration": df["state"]["pulmonary_diffusion_equilibration_fraction"],
        },
        sh["state"]["pao2_mmHg"] < bs["pao2_mmHg"] - 15.0
        and df["state"]["pao2_mmHg"] < bs["pao2_mmHg"] - 8.0
        and df["state"]["pulmonary_diffusion_equilibration_fraction"] < bs["pulmonary_diffusion_equilibration_fraction"] - 0.10,
        "mechanistic-pulmonary-physiology"
    ))

    # Oxygen should correct V/Q mismatch more strongly than true shunt.
    gains = {}
    for scenario in ("vq_mismatch", "pulmonary_shunt"):
        room = HumanHomeostasisEnv(scenario=scenario); room.reset(seed=12)
        oxy = HumanHomeostasisEnv(scenario=scenario); oxy.reset(seed=12)
        oa = ZERO.copy(); oa[4] = 1.0
        _, _, _, _, ir = room.step(ZERO)
        _, _, _, _, io = oxy.step(oa)
        gains[scenario] = io["state"]["pao2_mmHg"] - ir["state"]["pao2_mmHg"]
    checks.append(check(
        "supplemental-O2 response distinguishes shunt from V/Q mismatch",
        gains,
        gains["vq_mismatch"] > 2.0 * gains["pulmonary_shunt"],
        "medical-mechanism-check",
        "Relative responsiveness is tested; no treatment recommendation or clinical FiO2 target is inferred."
    ))

    # Timestep sensitivity.
    c = HumanConfig(cv_internal_step_s=0.04)
    e1 = HumanHomeostasisEnv(config=replace(c, integration_step_min=0.25), scenario="vq_mismatch")
    e2 = HumanHomeostasisEnv(config=replace(c, integration_step_min=0.125), scenario="vq_mismatch")
    e1.reset(seed=31); e2.reset(seed=31)
    i1=i2=None
    for _ in range(4):
        _,_,t1,_,i1=e1.step(ZERO); _,_,t2,_,i2=e2.step(ZERO)
        if t1 or t2: break
    delta={"PaO2_mmHg":abs(i1["state"]["pao2_mmHg"]-i2["state"]["pao2_mmHg"]),"PaCO2_mmHg":abs(i1["state"]["paco2_mmHg"]-i2["state"]["paco2_mmHg"])}
    checks.append(check("pulmonary timestep convergence", delta, not t1 and not t2 and delta["PaO2_mmHg"]<2.0 and delta["PaCO2_mmHg"]<0.75, "numerical-verification"))

    # Short multi-scenario random stress.
    rng=np.random.default_rng(13); unexpected=0; max_carbon=max_charge=0.0; finite=True
    for idx,scenario in enumerate(["baseline","vq_mismatch","pulmonary_shunt","diffusion_limitation"]):
        env=HumanHomeostasisEnv(scenario=scenario); env.reset(seed=idx)
        for _ in range(8):
            action=rng.uniform(0.0,0.25,size=8).astype(np.float32)
            obs,reward,term,tr,info=env.step(action)
            finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(reward))
            max_carbon=max(max_carbon,abs(info["mass_balance"]["co2_mass_balance_error_mmol"]))
            max_charge=max(max_charge,abs(info["acid_base"]["charge_balance_residual_mEq_l"]))
            if term:
                # Pulmonary pathology may terminate if the random controller worsens it;
                # only count non-pulmonary/unexplained reasons as unexpected.
                if info.get("termination_reason") not in {"severe_hypoxemia","critical_low_pao2"}:
                    unexpected += 1
                break
            if tr: break
    stress={"finite":finite,"max_CO2_mass_residual_mmol":max_carbon,"max_charge_residual_mEq_l":max_charge,"unexpected_nonpulmonary_terminations":unexpected}
    checks.append(check("multi-scenario pulmonary/chemical stress", stress, finite and max_carbon<1e-8 and max_charge<1e-6 and unexpected==0, "physical-algorithmic-verification"))

    result={
        "version":"0.13.0",
        "summary":{"passed":sum(x["pass"] for x in checks),"failed":sum(not x["pass"] for x in checks),"total":len(checks)},
        "checks":checks,
        "external_reference_file":str(REF.relative_to(ROOT)),
        "credibility_classification":{
            "VQ_distribution":"six-compartment reduced forward model inspired by MIGET physiology; not a MIGET inversion",
            "true_shunt":"explicit venous admixture by O2 content",
            "oxygen_diffusion":"finite equilibration driven by capillary transit time and a reduced diffusion-capacity parameter",
            "dead_space":"anatomical dead space plus reduced high-V/Q wasted-ventilation diagnostic; Enghoff-like rather than direct Bohr PACO2 measurement",
            "CO2":"v0.12 conserved whole-blood carbon pool retained; regional CO2 exchange remains reduced",
            "clinical_decision_use":"not supported"
        },
        "remaining_limitations":[
            "no spatial gravity-dependent lung geometry",
            "no dynamic alveolar recruitment/derecruitment or compliance distribution",
            "no full regional CO2 content mixing per V/Q compartment",
            "no explicit DLCO/DLNO membrane and capillary conductance decomposition",
            "no hypoxic pulmonary vasoconstriction redistribution",
            "no aerosol/airway mechanics or disease-specific COPD/ARDS parameter estimation"
        ],
        "next_high_value_step":"v0.14 should add hypoxic pulmonary vasoconstriction and dynamic regional perfusion/recruitment, then validate V/Q distributions against a published MIGET dataset before introducing ARDS/COPD disease models."
    }
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["summary"],indent=2))
    for x in checks: print(("PASS" if x["pass"] else "FAIL"),x["tier"],x["name"])

if __name__ == "__main__":
    main()
