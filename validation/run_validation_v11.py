from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.11.0")

from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, __version__

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.11.json"
REF_OUT = ROOT / "validation" / "external_reference_acidbase_v0.11.json"

ZERO = np.zeros(8, dtype=np.float32)


def check(name, value, passed, tier, note=""):
    return {"name": name, "value": value, "pass": bool(passed), "tier": tier, "note": note}


def rollout(env, minutes: float):
    _, info = env.reset(seed=1)
    steps = int(round(minutes / env.config.agent_step_min))
    terminated = truncated = False
    for _ in range(max(0, steps)):
        _, _, terminated, truncated, info = env.step(ZERO)
        if terminated or truncated:
            break
    return info, terminated, truncated


def main():
    checks = []
    prior = json.loads((ROOT / "validation/validation_results_v0.10.json").read_text())
    checks.append(check(
        "v0.10 frozen scientific regression", prior["summary"], prior["summary"]["failed"] == 0,
        "regression",
    ))
    checks.append(check("release version", __version__, __version__ == "0.11.0", "software-verification"))

    # Baseline charge/mass-action closure.
    env = HumanHomeostasisEnv()
    _, base = env.reset(seed=1)
    ab = base["acid_base"]
    closure_ok = (
        7.35 <= ab["pH"] <= 7.45
        and 20.0 <= ab["bicarbonate_mmol_l"] <= 30.0
        and abs(ab["charge_balance_residual_mEq_l"]) < 1e-6
        and abs(ab["henderson_hasselbalch_residual"]) < 1e-10
    )
    checks.append(check("physicochemical charge + HH closure", ab, closure_ok, "chemical-physical-verification"))

    # Human crystalloid benchmark: Dell'Anna et al. 2025, 30 mL/kg total saline;
    # reported pH one hour after the second bolus median 7.34 [7.32-7.36].
    senv = HumanHomeostasisEnv(scenario="saline_challenge_30ml_kg")
    _, saline = senv.reset(seed=1)
    sb, ss = base["state"], saline["state"]
    saline_result = {
        "baseline_pH": sb["ph_arterial"],
        "saline_pH_immediate_reduced_order": ss["ph_arterial"],
        "published_pH_1h_after_second_bolus_median": 7.34,
        "published_iqr": [7.32, 7.36],
        "baseline_chloride_mmol_l": sb["chloride_mmol_l"],
        "saline_chloride_mmol_l": ss["chloride_mmol_l"],
        "baseline_SIDa_mEq_l": sb["strong_ion_difference_apparent_mEq_l"],
        "saline_SIDa_mEq_l": ss["strong_ion_difference_apparent_mEq_l"],
        "albumin_after_dilution_g_dl": ss["albumin_g_dl"],
    }
    saline_ok = (
        ss["ph_arterial"] < sb["ph_arterial"] - 0.03
        and ss["chloride_mmol_l"] > sb["chloride_mmol_l"] + 4.0
        and ss["strong_ion_difference_apparent_mEq_l"] < sb["strong_ion_difference_apparent_mEq_l"] - 3.0
        and abs(ss["ph_arterial"] - 7.34) < 0.03
    )
    checks.append(check(
        "0.9% saline chloride/SID/pH human benchmark", saline_result, saline_ok,
        "external-human-physiology",
        "Timing is collapsed: this is a dose/composition response benchmark, not a reproduction of the trial time course.",
    ))

    # Lactate should act through strong-ion chemistry, without hard-coded bicarbonate.
    lenv = HumanHomeostasisEnv(scenario="transient_lactic_acidosis")
    _, lactate = lenv.reset(seed=1)
    ls = lactate["state"]
    lactate_ok = (
        ls["lactate_mmol_l"] > sb["lactate_mmol_l"] + 2.0
        and ls["strong_ion_difference_apparent_mEq_l"] < sb["strong_ion_difference_apparent_mEq_l"] - 2.0
        and ls["ph_arterial"] < sb["ph_arterial"] - 0.03
    )
    checks.append(check("lactate strong-anion acidosis direction", {
        "baseline": {"lactate": sb["lactate_mmol_l"], "pH": sb["ph_arterial"], "SIDa": sb["strong_ion_difference_apparent_mEq_l"]},
        "challenge": {"lactate": ls["lactate_mmol_l"], "pH": ls["ph_arterial"], "SIDa": ls["strong_ion_difference_apparent_mEq_l"]},
    }, lactate_ok, "biochemical-verification"))

    # Respiratory acidosis: chemistry determines the acute state, kidneys later
    # increase ammonium/acid excretion and alter strong-ion balance.
    cfg_fast = HumanConfig(cv_internal_step_s=0.04)
    renv = HumanHomeostasisEnv(config=cfg_fast, scenario="respiratory_acidosis")
    _, r0 = renv.reset(seed=1)
    initial = r0["state"].copy()
    r24, term, _ = rollout(renv, 24.0 * 60.0)
    rs = r24["state"]
    respiratory_ok = (
        not term
        and initial["paco2_mmHg"] >= 55.0
        and initial["ph_arterial"] < 7.32
        and rs["urine_ammonium_mmol_min"] > initial["urine_ammonium_mmol_min"]
        and rs["bicarbonate_mmol_l"] > initial["bicarbonate_mmol_l"]
        and rs["ph_arterial"] > initial["ph_arterial"]
    )
    checks.append(check("respiratory acidosis renal-compensation direction", {
        "initial": {k: initial[k] for k in ("paco2_mmHg","ph_arterial","bicarbonate_mmol_l","urine_ammonium_mmol_min","chloride_mmol_l")},
        "24h": {k: rs[k] for k in ("paco2_mmHg","ph_arterial","bicarbonate_mmol_l","urine_ammonium_mmol_min","chloride_mmol_l")},
    }, respiratory_ok, "multi-organ-physiology"))

    # Reduced renal function should retain nonvolatile acid / raise SIG.
    kenv = HumanHomeostasisEnv(config=cfg_fast, scenario="reduced_renal_function")
    _, k0 = kenv.reset(seed=1)
    k24, term, _ = rollout(kenv, 24.0 * 60.0)
    ks = k24["state"]
    kidney_ok = (
        not term
        and ks["renal_acid_excretion_mmol_min"] < cfg_fast.baseline_net_acid_excretion_mmol_min
        and ks["strong_ion_gap_mEq_l"] > k0["state"]["strong_ion_gap_mEq_l"] + 0.5
        and abs(k24["mass_balance"]["nonvolatile_acid_mass_balance_error_mEq"]) < 1e-8
    )
    checks.append(check("reduced renal function nonvolatile-acid retention", {
        "initial_SIG_mEq_l": k0["state"]["strong_ion_gap_mEq_l"],
        "24h_SIG_mEq_l": ks["strong_ion_gap_mEq_l"],
        "24h_pH": ks["ph_arterial"],
        "24h_NAE_mmol_min": ks["renal_acid_excretion_mmol_min"],
        "acid_mass_residual_mEq": k24["mass_balance"]["nonvolatile_acid_mass_balance_error_mEq"],
    }, kidney_ok, "renal-biochemical-verification"))

    # Integrator sensitivity for the coupled acid-base result.
    c1 = replace(HumanConfig(), integration_step_min=0.25, cv_internal_step_s=0.04)
    c2 = replace(HumanConfig(), integration_step_min=0.125, cv_internal_step_s=0.04)
    e1 = HumanHomeostasisEnv(config=c1, scenario="transient_lactic_acidosis")
    e2 = HumanHomeostasisEnv(config=c2, scenario="transient_lactic_acidosis")
    i1, _, _ = rollout(e1, 120.0)
    i2, _, _ = rollout(e2, 120.0)
    timestep = {
        "pH_dt_0.25": i1["state"]["ph_arterial"],
        "pH_dt_0.125": i2["state"]["ph_arterial"],
        "abs_delta_pH": abs(i1["state"]["ph_arterial"] - i2["state"]["ph_arterial"]),
        "delta_HCO3_mmol_l": abs(i1["state"]["bicarbonate_mmol_l"] - i2["state"]["bicarbonate_mmol_l"]),
    }
    checks.append(check("acid-base timestep convergence", timestep,
                        timestep["abs_delta_pH"] < 0.005 and timestep["delta_HCO3_mmol_l"] < 0.25,
                        "numerical-verification"))

    # Multi-seed chemical/physical residuals.
    max_charge = max_hh = max_acid_mass = 0.0
    unexpected = 0
    for seed in range(10):
        e = HumanHomeostasisEnv(config=HumanConfig(cv_internal_step_s=0.04))
        e.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(24):
            a = rng.uniform(0.0, 0.25, size=8).astype(np.float32)
            _, _, t, tr, inf = e.step(a)
            max_charge = max(max_charge, abs(inf["acid_base"]["charge_balance_residual_mEq_l"]))
            max_hh = max(max_hh, abs(inf["acid_base"]["henderson_hasselbalch_residual"]))
            max_acid_mass = max(max_acid_mass, abs(inf["mass_balance"]["nonvolatile_acid_mass_balance_error_mEq"]))
            if t:
                unexpected += 1
                break
            if tr:
                break
    stress = {
        "max_charge_balance_residual_mEq_l": max_charge,
        "max_hh_residual": max_hh,
        "max_nonvolatile_acid_mass_residual_mEq": max_acid_mass,
        "unexpected_terminations": unexpected,
    }
    checks.append(check("10-seed acid-base stress residuals", stress,
                        max_charge < 1e-6 and max_hh < 1e-10 and max_acid_mass < 1e-8 and unexpected == 0,
                        "physical-algorithmic-verification"))

    ref = {
        "version": "0.11.0",
        "primary_sources": [
            {
                "citation": "Stewart PA. Modern quantitative acid-base chemistry. Can J Physiol Pharmacol. 1983;61:1444-1461.",
                "pmid": 6423247,
                "use": "mass action, mass conservation and electroneutrality framework; SID/PCO2/weak-acid control variables",
            },
            {
                "citation": "Figge J, Rossing TH, Fencl V. The role of serum proteins in acid-base equilibria. J Lab Clin Med. 1991;117:453-467.",
                "pmid": 2045713,
                "use": "human albumin weak-acid charge model",
            },
            {
                "citation": "Figge J, Mydosh T, Fencl V. Serum proteins and acid-base equilibria: a follow-up. J Lab Clin Med. 1992;120:713-719.",
                "pmid": 1431499,
                "use": "albumin/protein acid-base refinement",
            },
            {
                "citation": "Dell'Anna AM et al. Stewart's theory and acid-base changes induced by crystalloid infusion in humans: a randomized physiological trial. Ann Intensive Care. 2025;15:54.",
                "pmid": 40263186,
                "use": "external human crystalloid benchmark",
                "saline": {"chloride_mEq_l": 154, "SID_mEq_l": 0, "total_dose_ml_kg": 30, "pH_after_second_bolus_median": 7.34, "pH_IQR": [7.32, 7.36]},
            },
        ],
        "model_scope": {
            "implemented": ["Na/K/Cl/lactate simplified SID", "PaCO2-HCO3 mass action", "albumin and phosphate weak-acid charge", "charge closure", "renal NH4/titratable-acid/bicarbonate diagnostics", "nonvolatile acid ledger"],
            "not_implemented": ["dynamic ionized Ca/Mg pools", "full carbonate species", "erythrocyte buffering/chloride shift", "hemoglobin proton buffering", "urine electroneutrality", "full nephron acid transport", "dynamic albumin synthesis/loss"],
        },
    }
    REF_OUT.write_text(json.dumps(ref, indent=2), encoding="utf-8")

    result = {
        "version": "0.11.0",
        "summary": {"passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks), "total": len(checks)},
        "checks": checks,
        "credibility_classification": {
            "acid_base_chemistry": "materially improved: charge-constrained and mass-action-constrained, but still reduced Stewart-Figge plasma chemistry",
            "saline_response": "externally direction/magnitude benchmarked to a randomized human physiology trial; timing is not reproduced",
            "renal_acid_handling": "mechanistically decomposed into NH4+, titratable acid and bicarbonaturia, but not a nephron transport model",
            "electroneutrality": "enforced for the represented plasma species plus an explicit unmeasured strong-anion pool",
            "whole_body_carbon_conservation": "not implemented; lungs control PaCO2 through gas exchange rather than a conserved total-body CO2 pool",
            "clinical_decision_use": "not supported",
        },
        "next_high_value_step": (
            "v0.12 should add full CO2/HCO3/carbonate and hemoglobin/RBC buffering plus V/Q/shunt or, alternatively, "
            "a clinically grounded creatinine/urine-output kidney module. Do not claim patient-level acid-base prediction before those are externally validated."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["tier"], c["name"])


if __name__ == "__main__":
    main()
