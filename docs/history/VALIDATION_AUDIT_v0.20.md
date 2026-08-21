# OpenHumSim-RL v0.20 — P2 credibility audit

## Scope

v0.20 addresses observation realism, virtual-patient dependence structure and
validation-data governance. It intentionally does **not** add a new organ or
claim improved clinical predictive validity.

Credibility language follows the distinction between software/model
verification and validation for a defined context of use. Internal PASS gates
below are verification/credibility checks, not clinical validation.

## 1. Measurement model

### Implemented

The default `clinical` observation profile now passes hidden state through
`ClinicalMeasurementModel`.

Mechanisms:

- finite sampling cadence;
- finite result delay;
- Gaussian engineering measurement error;
- dropout with hold-last behavior;
- explicit measurement age;
- CGM interstitial lag via the existing v0.9 observation model.

Default cadence/delay:

```text
monitor/ventilator: 5 min / 0 min
ABG:               30 min / 7 min
chemistry:          60 min / 12 min
hemodynamic:        15 min / 2 min
CGM:                 5 min effective updates + 6 min lag
```

### External anchors

- Louie et al. 2018, PMID 29200008: stationary pulse-ox control RMSE ~1.8%,
  degraded by motion/low perfusion.
- Sediame et al. 1999, PMID 10489854: bedside blood-gas measurements showed
  non-zero imprecision versus laboratory analyzer.
- Mitchell et al., PMCID PMC9125450: standard ABG result time averaged about
  6 min 31 s in the studied ICU workflow.

The v0.20 default noise/cadence values are **engineering settings around these
scales**, not calibrated specifications for a named commercial monitor.

### Known limitations

- Gaussian noise is simplistic and mostly homoscedastic;
- pulse-ox error is not yet state-dependent on perfusion/motion/pigmentation;
- dropout events are independent Bernoulli events rather than correlated device
  failure episodes;
- monitor data are only updated at the 5-min policy step, not waveform rate;
- laboratory scheduling is fixed cadence rather than clinician-triggered;
- initial baseline results are assumed already available at t=0;
- no sample contamination/pre-analytic error model.

Status: **major POMDP realism improvement; not device-specific validation**.

## 2. Information leakage

### Observation leakage

The policy no longer sees blood glucose directly in the default clinical
profile; it sees `sensor_glucose_mg_dl` after the CGM observation model.
Intermittent laboratory values can be stale.

### `info` leakage

`info_profile="benchmark"` removes:

```text
state
oxygen_transport
mass_balance
acid_base
pulmonary_exchange
respiratory_cycle
blood_gas
pbpk
metabolism
cardiovascular
```

from Gymnasium `info`.

Status: **verified**. Hidden truth remains available only in explicit debug
mode.

## 3. Correlated virtual-patient prior

The old population prior independently LHS-sampled every latent parameter.
v0.20 uses a sparse rank-correlation matrix and an Iman-Conover-style reorder.

Properties tested:

```text
marginal LHS strata preserved exactly
weight ↔ TBW fraction       negative rank dependence
tbw fraction ↔ ECF fraction positive rank dependence
weight ↔ blood volume/kg    modest negative dependence
```

The design is reproducible under fixed seed.

### Scientific interpretation

This is **not** an empirical population covariance matrix. It is an engineering
prior motivated by the fact that body water, blood volume/Hb mass and renal
function depend on body composition/demographic covariates rather than varying
independently.

Supporting references include Chumlea et al. 2001 (PMID 11380828), Falz et al.
2019 (PMID 30791081) and Dooley et al. 2000 (PMID 11138467). The weak observed
body-size/GFR association is why v0.20 deliberately uses only a modest GFR
correlation.

Missing before population calibration:

- age;
- sex;
- height;
- BMI/body-fat/lean mass;
- ethnicity/geography where scientifically relevant;
- fitness and disease covariates;
- empirical joint distributions from a defined reference population.

Status: **better domain-randomization prior, not a human population model**.

## 4. Locked calibration/validation governance

`LockedCohortManifest` hashes the validation identifiers together with a dataset
name and optional source fingerprint.

```text
source file SHA-256
+ validation subject IDs
+ dataset identity
      ↓
validation_lock_sha256
```

A changed validation subject list or changed source fingerprint invalidates the
lock.

Jaeb `split-jaeb-cgm` now includes the local archive SHA-256 in the locked test
record.

Important distinction:

```text
Jaeb train/validation/test split = internal source holdout
independent external dataset     = still missing
```

Therefore v0.20 does not claim external validation merely because the test split
is locked.

## 5. Credibility protocol

`validation/credibility_protocol_v0.20.json` formalizes four roles:

1. population prior — domain randomization/UQ only;
2. calibration cohort — parameter/model fitting permitted;
3. locked validation cohort — no further tuning permitted;
4. independent external validation — separate data source, currently not bundled.

This is intended to prevent the common failure mode where repeated inspection
of a "test" cohort effectively turns it into additional training data.

## 6. Verification results

```text
v0.20 focused unit tests               9/9 PASS
v0.20 credibility gate               10/10 PASS
compileall                              PASS
```

Regression checks executed separately:

```text
v0.19 P1                                7 PASS
v0.18 credibility                       9 PASS
v0.17 patient-ventilator                9 PASS
v0.16 respiratory cycle                10 PASS
v0.15 respiratory mechanics             9 PASS
v0.14 pulmonary                          8 PASS
v0.13 V/Q                                8 PASS
v0.12 blood gas                          8 PASS
event/replay                             6 PASS
selected env/external-data              12 PASS
```

Long legacy tests are not collapsed into a misleading monolithic PASS when the
available release-test runtime prevents a one-shot run.

## 7. Algorithmic consequences

Any RL checkpoint trained before v0.20 is not benchmark-comparable without
retraining because the default clinical observation process changed from exact
values to noisy/delayed measurements and the observation dimension changed from
50 to 55.

Recommended benchmark environment:

```python
HumanHomeostasisEnv(
    observation_profile="clinical",
    measurement_profile="realistic",
    info_profile="benchmark",
)
```

Reward remains computed from hidden physiology. This is appropriate for a
simulator-defined training reward, but future work should distinguish training
reward from metrics available to a real controller.

## 8. What v0.20 does not solve

- independent participant-level external validation;
- empirical population calibration;
- measurement-error calibration by device and clinical context;
- state-dependent missingness/artifact models;
- clinician-driven/adaptive lab ordering;
- prospective validation;
- patient-specific inference.

## 9. Next evidence step

The next high-value release should not add more mechanisms. It should create a
**pre-registered validation harness** around at least one independent dynamic
human dataset, with frozen model/version/reward before test access, trajectory-
level error metrics, uncertainty coverage and explicit failure regions.
