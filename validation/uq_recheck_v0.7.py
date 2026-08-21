from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from openhumsim_rl import HumanHomeostasisEnv, sample_virtual_cohort
from historical_version_guard import require_exact_version

require_exact_version("0.7.0")

ROOT=Path(__file__).resolve().parents[1]
UQ=json.loads((ROOT/'validation/uq_virtual_cohort_v0.7.json').read_text())
OUT=ROOT/'validation/uq_recheck_v0.7.json'

def run(cfg):
    env=HumanHomeostasisEnv(config=cfg, scenario='oral_glucose_75g')
    _,info=env.reset(seed=123)
    peak=info['state']['glucose_mg_dl']; t=0.0; peak_i=info['state']['insulin_uU_ml']
    z=np.zeros(8,dtype=np.float32)
    for _ in range(18):
        _,_,term,trunc,info=env.step(z)
        g=info['state']['glucose_mg_dl']
        if g>peak: peak=g; t=info['time_min']
        peak_i=max(peak_i,info['state']['insulin_uU_ml'])
        if term or trunc: break
    return {'peak_glucose':float(peak),'t_peak':float(t),'peak_insulin':float(peak_i)}

cohort=sample_virtual_cohort(24,seed=20260817)
rows=UQ['patients']
order=sorted(range(len(rows)), key=lambda i: rows[i]['glucose_peak_75g_mg_dl'])
selected=sorted(set([order[0],order[1],order[-2],order[-1]]))
res=[]
for i in selected:
    vp=cohort[i]
    coarse=replace(vp.config,cv_internal_step_s=.04,cv_warmup_min=1.5)
    accurate=replace(vp.config,cv_internal_step_s=.02,cv_warmup_min=3.0)
    a=run(accurate); c=run(coarse)
    res.append({'patient_id':vp.patient_id,'accurate':a,'uq_solver':c,
                'abs_peak_glucose_diff':abs(a['peak_glucose']-c['peak_glucose']),
                'abs_peak_insulin_diff':abs(a['peak_insulin']-c['peak_insulin'])})
out={'version':'0.7.0','selected_extreme_patients':res,
     'max_abs_peak_glucose_diff':max(x['abs_peak_glucose_diff'] for x in res),
     'max_abs_peak_insulin_diff':max(x['abs_peak_insulin_diff'] for x in res),
     'interpretation':'Checks whether the faster UQ cardiovascular solver materially changes the 75-g metabolic challenge at distribution extremes.'}
OUT.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
