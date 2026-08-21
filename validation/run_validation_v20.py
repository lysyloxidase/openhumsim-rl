from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv, sample_virtual_cohort
from openhumsim_rl.measurement import ClinicalMeasurementConfig
from openhumsim_rl.population import DEFAULT_PARAMETER_SPECS, LockedCohortManifest, correlated_latin_hypercube
from historical_version_guard import require_exact_version

require_exact_version("0.20.0")

ZERO = np.zeros(8, dtype=np.float32)
checks = []

def add(name, passed, values):
    checks.append({"name": name, "passed": bool(passed), "values": values})

# 1 measurement process deterministic under fixed seed.
a = HumanHomeostasisEnv(measurement_profile="realistic")
b = HumanHomeostasisEnv(measurement_profile="realistic")
o1, i1 = a.reset(seed=100)
o2, i2 = b.reset(seed=100)
for _ in range(3):
    o1, *_ = a.step(ZERO)
    o2, *_ = b.step(ZERO)
add("measurement_seed_reproducibility", np.array_equal(o1, o2), {"clinical_dim": len(o1)})

# 2 ABG sampling/delay.
e = HumanHomeostasisEnv(measurement_profile="realistic")
e.reset(seed=2); m=e.measurement_model; old=m.measurement_value("paco2_mmHg", e.state); e.state.paco2_mmHg=80.0
for minute in (5,10,15,20,25,30,35): m.advance(e.state, float(minute), 5.0, e.np_random)
held=m.measurement_value("paco2_mmHg", e.state); age35=m.group_ages()["blood_gas_measurement_age_min"]
m.advance(e.state,40.0,5.0,e.np_random); delivered=m.measurement_value("paco2_mmHg",e.state); age40=m.group_ages()["blood_gas_measurement_age_min"]
add("abg_sampling_and_result_delay", held==old and delivered>70 and abs(age40-10.0)<1e-12,
    {"initial":old,"held_t35":held,"reported_t40":delivered,"age_t35":age35,"age_t40":age40})

# 3 dropout/hold-last.
cfg=ClinicalMeasurementConfig(monitor_dropout_probability=1.0,cgm_dropout_probability=1.0)
d=HumanHomeostasisEnv(measurement_profile="realistic",measurement_config=cfg); d.reset(seed=4); md=d.measurement_model
hr0=md.measurement_value("heart_rate_bpm",d.state); d.state.heart_rate_bpm=160; md.advance(d.state,5,5,d.np_random)
add("monitor_dropout_hold_last", md.measurement_value("heart_rate_bpm",d.state)==hr0 and md.group_ages()["monitor_measurement_age_min"]>=5,
    {"held_hr":hr0,"age_min":md.group_ages()["monitor_measurement_age_min"],"dropped":md.diagnostics()["groups"]["monitor"]["dropped"]})

# 4 CGM lag.
cfg=ClinicalMeasurementConfig(cgm_dropout_probability=0.0,cgm_relative_noise_sd=0.0)
g=HumanHomeostasisEnv(measurement_profile="realistic",measurement_config=cfg); g.reset(seed=5); mg=g.measurement_model
g0=mg.measurement_value("sensor_glucose_mg_dl",g.state); g.state.glucose_mg_dl=200; mg.advance(g.state,5,5,g.np_random); g1=mg.measurement_value("sensor_glucose_mg_dl",g.state)
add("cgm_interstitial_lag", g0<g1<200, {"before":g0,"after_5min":g1,"blood":200.0})

# 5 requested rank correlations.
U=correlated_latin_hypercube(512,seed=123); names=[s.name for s in DEFAULT_PARAMETER_SPECS]
def ranks(x):
    o=np.argsort(x,kind="mergesort"); r=np.empty(len(x),float); r[o]=np.arange(len(x)); return r
def rho(a,b): return float(np.corrcoef(ranks(U[:,names.index(a)]),ranks(U[:,names.index(b)]))[0,1])
r1=rho("body_weight_kg","tbw_fraction"); r2=rho("tbw_fraction","ecf_fraction"); r3=rho("body_weight_kg","blood_volume_ml_per_kg")
add("correlated_virtual_patient_prior", r1 < -0.25 and r2 > 0.35 and r3 < -0.10, {"rho_weight_tbw_fraction":r1,"rho_tbw_ecf":r2,"rho_weight_bv_perkg":r3})

# 6 LHS marginals preserved.
n=512; lhs_ok=True
for j in range(U.shape[1]):
    lhs_ok &= len(np.unique(np.floor(U[:,j]*n).astype(int)))==n
add("correlated_prior_preserves_lhs_marginals", lhs_ok, {"n":n,"dimensions":U.shape[1]})

# 7 locked external validation cohort.
manifest=LockedCohortManifest.create([f"S{i:03d}" for i in range(40)],"external-validation-demo",seed=2020,dataset_fingerprint="sha256:demo")
add("locked_validation_no_leakage", manifest.verify_lock() and not(set(manifest.calibration_subject_ids)&set(manifest.validation_subject_ids)),
    {"n_calibration":len(manifest.calibration_subject_ids),"n_validation":len(manifest.validation_subject_ids),"lock":manifest.validation_lock_sha256})

# 8 default virtual cohort remains reproducible/physically coherent.
va=sample_virtual_cohort(24,seed=88); vb=sample_virtual_cohort(24,seed=88)
ok=[x.latent for x in va]==[x.latent for x in vb] and all(x.config.total_body_water_baseline_l>x.config.ecf_volume_baseline_l for x in va)
add("correlated_cohort_reproducibility_and_geometry",ok,{"n":24})

# 9 full profile remains explicit ideal ground truth.
f=HumanHomeostasisEnv(observation_profile="full"); fo,fi=f.reset(seed=9)
add("full_profile_remains_ideal_debug_state",fi["measurement_profile"]=="ideal" and len(fo)>len(o1),{"full_dim":len(fo),"clinical_dim":len(o1)})

# 10 strict benchmark info does not leak ground truth.
strict=HumanHomeostasisEnv(info_profile="benchmark"); so,si=strict.reset(seed=10)
add("benchmark_info_hides_mechanistic_truth", "state" not in si and "pbpk" not in si and "blood_gas" not in si and "measurement" in si,
    {"info_keys":sorted(si.keys())})

payload={"version":"0.20.0","summary":{"passed":sum(c["passed"] for c in checks),"total":len(checks),"all_passed":all(c["passed"] for c in checks)},"checks":checks}
Path("validation/validation_results_v0.20.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload["summary"],indent=2))
if not payload["summary"]["all_passed"]: raise SystemExit(1)
