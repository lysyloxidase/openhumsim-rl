# OpenHumSim-RL v0.9 — validation and credibility audit

## Scope

v0.9 addresses the next credibility bottleneck: comparing a mechanistic blood-glucose simulator with **real human interstitial-CGM data without conflating the measurement process with physiology**.

The release adds:

1. a blood-to-interstitial CGM observation model;
2. exact published human CGM reference values from Shah et al. 2019;
3. age-stratified Table-2 targets;
4. participant-level train/validation/test splitting;
5. bootstrap uncertainty on split metrics;
6. a TRAIN-only external normative-CGM reference fit;
7. held-out validation/test likelihood and predictive coverage.

## External human evidence

Primary source:

Shah VN, DuBose SN, Li Z, et al. *Continuous Glucose Monitoring Profiles in Healthy Nondiabetic Participants: A Multicenter Prospective Study.* J Clin Endocrinol Metab. 2019;104(10):4356-4364. PMID 31127824. DOI 10.1210/jc.2018-02763.

Dataset citation: Dryad DOI `10.5061/dryad.h7d11cd`. The Jaeb Center public dataset listing also exposes the `CGMND.zip` archive.

Relevant study properties:

- n=153;
- age 7–80 years;
- blinded Dexcom G6;
- 5-min sampling;
- up to 10 days wear;
- participants with <72 h total CGM were excluded unless additional wear brought them above the inclusion requirement;
- daily logs recorded meals/snacks, exercise, and sleep/wake times.

Published overall metrics encoded by v0.9:

```text
mean individual-average glucose       99 +/- 7 mg/dL
mean within-person CV                 17 +/- 3 %
median time 70-140                    96 % (IQR 93-98)
median time >140                       2.1 %
median time <70                        1.1 %
```

Exact age-stratified values from Table 2 are stored in code and `validation/external_reference_shah2019_v0.9.json`.

## Biological correctness

### Improvement

CGM is now modeled as an observation of an interstitial compartment rather than as an alias of plasma glucose. This is biologically more defensible because interstitial glucose exhibits physiological transport delay relative to blood glucose.

The default effective first-order lag constant is 6 min. Basu et al. reported a mean physiological intravascular-to-interstitial lag of approximately 5.3–6.2 min after accounting for measurement-system effects (Diabetes. 2013;62:4083-4087).

### Limitation

Interstitial glucose is not literally a pure delayed copy of blood glucose. Tissue transport, local uptake, sensor dynamics, calibration, compression artifacts and device filtering are more complex. The v0.9 first-order compartment is therefore classified as **mechanistically plausible / reduced-order**, not device-validated.

## Chemical correctness

v0.9 does not alter the core chemical conservation model. Existing Na/K/Cl/H2O, PBPK mass and Fick identities remain inherited from v0.6–v0.8.

The new CGM layer is an observation transform and introduces no conserved chemical species into the core physiology.

Known chemical gaps remain:

- full electroneutrality;
- explicit total inorganic carbon conservation;
- albumin/phosphate buffering;
- full metabolic stoichiometry/thermodynamics.

## Medical correctness

A major v0.9 safeguard is **context matching**.

The Shah cohort is free-living CGM. It cannot be treated as if it were a standardized OGTT or the Dalla-Man mixed-meal experiment. Therefore v0.9 does not directly minimize Dalla-Man trajectory error against the published 24-h mean/TIR/CV numbers.

The ≥60-y group also differs from younger strata (mean 104 mg/dL and median 93% time 70–140). OpenHumSim does not yet contain a validated age-specific glucose mechanism. That difference is retained as external evidence for future model expansion rather than fitted away with an arbitrary age coefficient.

## Physical correctness

The CGM observation step is a stable first-order linear system:

```text
dG_isf/dt = (G_blood - G_isf) / tau
```

The discrete implementation uses the exact exponential update for a constant input over the time step:

```text
alpha = 1 - exp(-dt/tau)
G_isf(t+dt) = G_isf(t) + alpha*(G_blood-G_isf)
```

The v0.9 scientific test verifies that after one time constant the step response reaches exactly `1-exp(-1)` of the final difference to numerical precision.

## Algorithmic/data-science correctness

### Participant-level splitting

The split is performed on unique participant IDs. Individual CGM readings are never independently shuffled across train and test.

Default:

```text
60% train
20% validation
20% test
```

The split is seeded and reproducible. A leakage report verifies pairwise-disjoint participant sets and complete accounting of all subjects.

### TRAIN-only fitting

The normative reference model is fitted only to TRAIN participants.

Fitted quantities:

- subject mean glucose — Normal;
- within-person CV — Normal;
- TIR 70–140 — logit-Normal;
- time >140 — logit-Normal;
- time <70 — logit-Normal.

Validation and test partitions are scored without refitting using:

- mean log-likelihood;
- 95% predictive-interval coverage.

Zero/100% bounded percentages receive a small continuity correction before the logit transform.

### What this does not prove

This statistical reference model is **not** a mechanistic patient calibration. It estimates the distribution of observed human CGM metrics. Mechanistic inference requires participant-matched exogenous inputs.

## Raw-data execution status

The public Jaeb archive is not bundled. The release environment used to build v0.9 cannot resolve/fetch the Jaeb S3 host, so no participant-level human values were fabricated.

The local workflow is complete:

```bash
uv run openhumsim data download-jaeb-cgm
uv run openhumsim data summarize-jaeb-cgm data/external/CGMND.zip
uv run openhumsim data split-jaeb-cgm data/external/CGMND.zip
uv run openhumsim data fit-jaeb-reference data/external/CGMND.zip
```

## Verification results

```text
pytest                         32 passed
Gymnasium checker               1 skipped (package absent in release container)
v0.9 targeted scientific        8 / 8 passed
v0.8 saved regression           9 / 9 passed
```

The synthetic calibration artifact `validation/cgm_reference_pipeline_synthetic_v0.9.json` exists only to regression-test splitting/fitting mechanics. It is explicitly labeled synthetic and is not human evidence.

## Credibility classification

```text
Software verification                  STRONG for tested paths
Mass/volume/Fick invariants             STRONG in inherited tested scope
CGM biological observation model       PLAUSIBLE, REDUCED-ORDER
Published human CGM reference           REAL EXTERNAL HUMAN DATA
Participant split/calibration code      VERIFIED
Participant-level Jaeb execution        PENDING LOCAL DOWNLOAD
Mechanistic parameter calibration       NOT YET ACHIEVED
Patient-specific prediction             NOT VALIDATED
Clinical decision support               NOT SUPPORTED
```

## Required v0.10 step

The scientifically correct next step is to inspect the real Jaeb archive for participant-matched:

- meal/snack start times;
- exercise timing/intensity;
- sleep/wake logs;
- demographics and any available covariates.

Then construct a locked pipeline:

```text
TRAIN
  reconstruct inputs
  infer mechanistic + observation nuisance parameters

VALIDATION
  posterior predictive checks
  model discrepancy diagnostics
  choose/freeze model form

TEST
  final locked external evaluation
  no parameter tuning
```

Until that exists, OpenHumSim should not claim participant-level calibration to real human glucose trajectories.
