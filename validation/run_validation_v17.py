from __future__ import annotations
from historical_version_guard import require_exact_version
require_exact_version("0.17.0")
import json
from pathlib import Path
import numpy as np
from openhumsim_rl import HumanHomeostasisEnv, HumanConfig

ZERO=np.zeros(8,dtype=np.float32)
def roll(sc, steps=2, seed=2, config=None):
    e=HumanHomeostasisEnv(config=config, scenario=sc); e.reset(seed=seed); info=None
    for _ in range(steps):
        _,_,term,trunc,info=e.step(ZERO)
        if term or trunc: break
    return e, info['state']
checks=[]
def ck(name, passed, values): checks.append({'name':name,'passed':bool(passed),'values':values})

sync_e,sync=roll('pressure_support_synchronous')
bad_e,bad=roll('pressure_support_ineffective_trigger')
peep_e,peep=roll('pressure_support_ineffective_trigger_peep')
late_e,late=roll('pressure_support_delayed_cycling')
opt_e,opt=roll('pressure_support_delayed_cycling_optimized')
prem_e,prem=roll('pressure_support_premature_cycling')
dbl_e,dbl=roll('pressure_support_double_trigger')
auto_e,auto=roll('pressure_support_autotrigger_leak')

ck('synchronous_psv', sync['respiratory_ventilator_asynchrony_index_pct']<10 and sync['respiratory_ventilator_mean_trigger_delay_s']<0.1 and 30<=sync['paco2_mmHg']<=50, {'AI_pct':sync['respiratory_ventilator_asynchrony_index_pct'],'trigger_delay_s':sync['respiratory_ventilator_mean_trigger_delay_s'],'PaCO2':sync['paco2_mmHg']})
ck('ineffective_trigger_and_external_peep_unloading', bad['respiratory_ventilator_ineffective_trigger_fraction']>0.3 and peep['respiratory_ventilator_ineffective_trigger_fraction']<0.1 and peep['respiratory_ventilator_asynchrony_index_pct']<bad['respiratory_ventilator_asynchrony_index_pct'], {'no_PEEP_ineffective':bad['respiratory_ventilator_ineffective_trigger_fraction'],'PEEP_ineffective':peep['respiratory_ventilator_ineffective_trigger_fraction'],'no_PEEP_AI':bad['respiratory_ventilator_asynchrony_index_pct'],'PEEP_AI':peep['respiratory_ventilator_asynchrony_index_pct']})
ck('cycling_40pct_reduces_delay_and_peepi', late['respiratory_ventilator_mean_cycling_delay_s']>0.5 and opt['respiratory_ventilator_mean_cycling_delay_s']<late['respiratory_ventilator_mean_cycling_delay_s']-0.5 and opt['respiratory_cycle_auto_peep_cmH2O']<late['respiratory_cycle_auto_peep_cmH2O'], {'cycle_10_delay_s':late['respiratory_ventilator_mean_cycling_delay_s'],'cycle_40_delay_s':opt['respiratory_ventilator_mean_cycling_delay_s'],'cycle_10_PEEPi':late['respiratory_cycle_auto_peep_cmH2O'],'cycle_40_PEEPi':opt['respiratory_cycle_auto_peep_cmH2O']})
ck('premature_cycling', prem['respiratory_ventilator_premature_cycling_fraction']>0.5 and prem['respiratory_ventilator_mean_cycling_delay_s']<-0.5, {'premature_fraction':prem['respiratory_ventilator_premature_cycling_fraction'],'cycling_delay_s':prem['respiratory_ventilator_mean_cycling_delay_s']})
ck('double_triggering', dbl['respiratory_ventilator_double_trigger_fraction']>0.4 and dbl['respiratory_ventilator_breaths_per_min']>dbl['respiratory_ventilator_patient_efforts_per_min'], {'double_fraction':dbl['respiratory_ventilator_double_trigger_fraction'],'efforts_min':dbl['respiratory_ventilator_patient_efforts_per_min'],'vent_breaths_min':dbl['respiratory_ventilator_breaths_per_min']})
ck('leak_autotrigger', auto['respiratory_ventilator_autotrigger_fraction']>0.4 and auto['respiratory_ventilator_breaths_per_min']>auto['respiratory_ventilator_patient_efforts_per_min'], {'autotrigger_fraction':auto['respiratory_ventilator_autotrigger_fraction'],'efforts_min':auto['respiratory_ventilator_patient_efforts_per_min'],'vent_breaths_min':auto['respiratory_ventilator_breaths_per_min']})
tr=late_e.model.respiratory_cycle.last_trace
ck('waveform_phase_observability', all(k in tr for k in ['neural_inspiration','ventilator_active','ventilator_support_cmH2O']) and bool(np.any(tr['ventilator_active']>tr['neural_inspiration'])), {'trace_samples':len(tr['time_s'])})
_,coarse=roll('pressure_support_delayed_cycling',config=HumanConfig(respiratory_cycle_dt_s=.01)); _,fine=roll('pressure_support_delayed_cycling',config=HumanConfig(respiratory_cycle_dt_s=.005))
ck('cycle_timestep_convergence', abs(coarse['respiratory_ventilator_mean_cycling_delay_s']-fine['respiratory_ventilator_mean_cycling_delay_s'])<.12 and abs(coarse['tidal_volume_l']-fine['tidal_volume_l'])<.06 and abs(coarse['respiratory_cycle_auto_peep_cmH2O']-fine['respiratory_cycle_auto_peep_cmH2O'])<.25, {'d_cycle_delay_s':abs(coarse['respiratory_ventilator_mean_cycling_delay_s']-fine['respiratory_ventilator_mean_cycling_delay_s']),'d_VT_l':abs(coarse['tidal_volume_l']-fine['tidal_volume_l']),'d_PEEPi':abs(coarse['respiratory_cycle_auto_peep_cmH2O']-fine['respiratory_cycle_auto_peep_cmH2O'])})
ck('legacy_conservation', abs(sync['co2_mass_balance_error_mmol'])<1e-7 and abs(sync['charge_balance_residual_mEq_l'])<1e-6 and abs(sync['cv_blood_volume_error_ml'])<1e-6, {'CO2_residual':sync['co2_mass_balance_error_mmol'],'charge_residual':sync['charge_balance_residual_mEq_l'],'blood_volume_residual':sync['cv_blood_volume_error_ml']})

payload={'version':'0.17.0','summary':{'passed':sum(x['passed'] for x in checks),'total':len(checks)},'checks':checks}
payload['summary']['all_passed']=payload['summary']['passed']==payload['summary']['total']
Path('validation/validation_results_v0.17.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload['summary'],indent=2))
if not payload['summary']['all_passed']: raise SystemExit(1)
