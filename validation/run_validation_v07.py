from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.7.0")

from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, sample_virtual_cohort

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation'/'validation_results_v0.7.json'


def check(name, value, passed, tier, note=''):
    return {'name':name,'value':value,'pass':bool(passed),'tier':tier,'note':note}


def main():
    checks=[]

    legacy=json.loads((ROOT/'validation/validation_results_v0.6.json').read_text())
    checks.append(check(
        'v0.6 scientific regression suite', legacy['summary'], legacy['summary']['failed']==0,
        'regression', 'All previously established scientific/numerical checks must remain passing.'
    ))

    # Config propagation / reset correctness introduced in v0.7.
    low=HumanHomeostasisEnv(config=HumanConfig(baseline_gfr_ml_min=95)); _,il=low.reset(seed=123)
    high=HumanHomeostasisEnv(config=HumanConfig(baseline_gfr_ml_min=125)); _,ih=high.reset(seed=123)
    checks.append(check('GFR configuration propagates into initial state',
        {'low':il['state']['gfr_ml_min'],'high':ih['state']['gfr_ml_min']},
        ih['state']['gfr_ml_min']-il['state']['gfr_ml_min']>20,'implementation-verification'))

    a=HumanHomeostasisEnv(config=HumanConfig(baseline_aa_gradient_mmHg=5)); _,ia=a.reset(seed=123)
    b=HumanHomeostasisEnv(config=HumanConfig(baseline_aa_gradient_mmHg=15)); _,ib=b.reset(seed=123)
    checks.append(check('A-a gradient configuration changes PaO2',
        {'Aa5':ia['state']['pao2_mmHg'],'Aa15':ib['state']['pao2_mmHg']},
        ia['state']['pao2_mmHg']-ib['state']['pao2_mmHg']>8,'implementation-verification'))

    vp=sample_virtual_cohort(1,seed=5)[0]
    env=HumanHomeostasisEnv(config=vp.config); _,iv=env.reset(seed=123)
    checks.append(check('Virtual-patient cardiovascular geometry conserves target blood volume',
        iv['state']['cv_blood_volume_error_ml'], abs(iv['state']['cv_blood_volume_error_ml'])<1e-6,
        'physical-invariant'))

    cal=json.loads((ROOT/'validation/calibration_v0.7.json').read_text())
    synth=cal['synthetic_parameter_recovery']
    checks.append(check('Synthetic inverse parameter recovery',
        {'max_relative_error':synth['max_relative_error'],'normalized_rmse':synth['normalized_rmse']},
        synth['success'] and synth['max_relative_error']<1e-6,'algorithmic-identifiability',
        'Synthetic identifiability only; not clinical parameter estimation.'))
    ref=cal['reference_profile_fit']
    checks.append(check('Literature-centered nominal profile fit',
        {'normalized_rmse':ref['normalized_rmse'],'fitted':ref['fitted']},
        ref['success'] and ref['normalized_rmse']<0.1,'calibration-smoke',
        'The fitted LV Emax reaching its bound is retained as evidence of structural/target tension, not hidden.'))

    uq=json.loads((ROOT/'validation/uq_virtual_cohort_v0.7.json').read_text())
    cov=uq['external_envelope_coverage']
    min_cov=min(v['fraction_within_external_envelope'] for v in cov.values())
    checks.append(check('24-patient LHS external-envelope robustness',
        {'minimum_coverage':min_cov,'coverage':cov}, min_cov>=0.95,'uncertainty-quantification',
        'Ranges were selected as plausible engineering uncertainty intervals; this is not cohort validation.'))

    recheck=json.loads((ROOT/'validation/uq_recheck_v0.7.json').read_text())
    checks.append(check('UQ coarse-solver high-fidelity recheck',
        {'max_peak_glucose_diff':recheck['max_abs_peak_glucose_diff'],'max_peak_insulin_diff':recheck['max_abs_peak_insulin_diff']},
        recheck['max_abs_peak_glucose_diff']<0.2 and recheck['max_abs_peak_insulin_diff']<0.2,
        'numerical-uq'))

    ppo=json.loads((ROOT/'validation/ppo_benchmark_v0.7.json').read_text())
    ev=ppo['evaluation']; p=ev['ppo']['mean_return']; n=ev['no_op']['mean_return']; h=ev['heuristic']['mean_return']
    checks.append(check('Held-out virtual-patient PPO benchmark',
        {'ppo':p,'no_op':n,'heuristic':h,'delta_vs_noop':p-n,'delta_vs_heuristic':p-h},
        p>n and p>=h,'rl-generalization',
        'Small advantage over heuristic; learnability/robustness smoke test only.'))

    # Default source model still exactly on published Dalla parameters: UQ multipliers are opt-in.
    cfg=HumanConfig()
    checks.append(check('Published Dalla core preserved by default UQ multipliers',
        {'insulin_sensitivity_scale':cfg.dalla_insulin_sensitivity_scale,'gastric_absorption_scale':cfg.dalla_gastric_absorption_scale},
        cfg.dalla_insulin_sensitivity_scale==1.0 and cfg.dalla_gastric_absorption_scale==1.0,
        'source-fidelity'))

    result={
        'version':'0.7.0',
        'summary':{'passed':sum(c['pass'] for c in checks),'failed':sum(not c['pass'] for c in checks),'total':len(checks)},
        'checks':checks,
        'credibility_classification':{
            'software_verification':'strong for tested paths',
            'calculation_verification':'strong reduced-order conservation + timestep checks',
            'parameter_uncertainty':'implemented as bounded LHS robustness study',
            'model_form_uncertainty':'not quantified',
            'external_validation':'partial reference-envelope comparison only',
            'patient_specific_validation':'absent',
            'clinical_decision_use':'not supported',
        },
        'known_missing_or_unvalidated':[
            'matched independent human validation cohort',
            'measurement-error model and likelihood-based/Bayesian calibration',
            'model-form discrepancy term',
            'full electroneutrality and total inorganic carbon conservation',
            'distributed V/Q and shunt',
            'real-drug PBPK calibration',
            'subcutaneous insulin PK',
            'creatinine kinetics / KDIGO AKI implementation',
            'SAC benchmark and large multi-seed neural-RL confidence intervals',
        ],
        'interpretation':'v0.7 adds parameter variability, calibration mechanics, UQ and held-out neural-RL testing. It improves model credibility evidence but does not convert OpenHumSim into a clinically validated virtual human.'
    }
    OUT.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary'],indent=2))
    for c in checks: print(('PASS' if c['pass'] else 'FAIL'),c['tier'],c['name'],c['value'])

if __name__=='__main__': main()
