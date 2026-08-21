# OpenHumSim-RL v0.10 — validation and credibility audit

## Scope

v0.10 advances the project from distribution-level CGM validation to **event-aligned free-living validation infrastructure**. It adds diary reconstruction, timestamp alignment, published inclusion rules, an external event-level reference, a separate replay-layer calibration, and participant-level held-out fitting infrastructure.

This release does **not** claim that individual Jaeb participants have been mechanistically calibrated in the release container. The actual `CGMND.zip` is not bundled and Jaeb requires user acceptance of its dataset terms before download.

## External human evidence used

### Shah et al. 2019

Used for the v0.9 normative healthy CGM distribution reference:

- 153 healthy non-diabetic participants;
- blinded Dexcom G6;
- up to 10 days;
- mean glucose around 98–99 mg/dL in most age groups;
- median 96% time 70–140 mg/dL;
- within-person CV 17±3%.

### DuBose et al. 2020/2021

Used for v0.10 event-level targets and analysis rules:

```text
exercise sessions                     451
baseline                               99 ± 12 mg/dL
nadir                                  85 ± 11 mg/dL
baseline-to-nadir change              -15 ± 18 mg/dL
median duration                        45 min (IQR 30–60)

participants in meal analysis          56
meal events                           306
premeal                                93 ± 10 mg/dL
postprandial peak                     130 ± 13 mg/dL
time to peak                           97 ± 31 min
excursion                              37 ± 15 mg/dL
```

The paper reports that participants logged exercise time/type, meal/snack start times, alcohol, and sleep/wake times. It also explicitly states that meal carbohydrate/fat quantity was not recorded and that diary omissions were possible.

## Software verification

Release test execution was split because the combined pytest invocation exceeded the available runtime in the release environment.

```text
tests/test_env.py             32 passed
tests/test_event_replay.py     6 passed
Gymnasium checker              1 optional skip (gymnasium absent in release container)
v0.10 scientific checks        9 / 9 passed
```

Both executable test modules complete successfully when run independently.

## v0.10 scientific checks

`validation/run_validation_v10.py` reports:

```text
9 passed
0 failed
```

The checks cover:

1. frozen v0.9 scientific regression;
2. release/version consistency;
3. DuBose reference transcription;
4. aggregate meal/exercise replay calibration;
5. Jaeb terms-respecting data workflow;
6. schema-adaptive CGM/diary parser;
7. published inclusion-rule alignment;
8. participant-level split with no train/validation/test leakage;
9. no raw human dataset bundled in the release.

## Aggregate event calibration result

The aggregate free-living replay layer is separate from the published Dalla Man core.

Fitted latent profile:

```text
effective meal carbohydrate       36.0 g
gastric absorption scale          0.45
representative exercise input     0.50
exercise Vmax gain                3.65
CGM lag tau                        6 min
```

High-fidelity result:

```text
                             model       DuBose mean
meal peak                   129.39       130 mg/dL
meal time to peak            95           97 min
meal excursion               36.39        37 mg/dL
exercise nadir               84.68        85 mg/dL
exercise change             -14.32       -15 mg/dL
```

Mean squared standardized residual across the five fitted endpoints:

```text
0.00207
```

### Interpretation

This is **calibration**, not independent validation. The effective meal carbohydrate and exercise extension were selected to reproduce the same aggregate targets. Their numerical values are not unique physiological estimates.

The fit is nevertheless useful because it demonstrates that the Dalla Man + CGM observation stack can reproduce the *shape/scale* of the reported healthy free-living event response without modifying the published Dalla Man parameter table itself.

## Biological audit

### Improved

- Meal and exercise timing are now represented as real external event classes rather than anonymous disturbances.
- Meal replay preserves the published Dalla Man glucose-insulin core and confines free-living adjustments to a replay profile.
- Sleep/wake timing is ingested rather than discarded.
- Participant-level calibration is separated from held-out participant evaluation.

### Still limited

- The DuBose study does not provide nutrient composition; therefore nutrient-specific physiology cannot be identified.
- Exercise type/duration is not equivalent to measured workload, VO2 or METs.
- Sleep has no validated causal effect on insulin sensitivity, hepatic glucose production or counter-regulation in the current core.
- The model still lacks incretin, mixed macronutrient, gastric nutrient-composition and circadian endocrine modules.

Classification: **biologically plausible event interface; participant-specific biological identifiability remains limited.**

## Chemical audit

v0.10 does not change the core chemical conservation architecture from v0.9.

Preserved strengths include explicit tracked pools for water/Na/K/Cl and mass-balanced glucose/PBPK subsystems on tested paths.

Remaining chemical gaps:

- no full electroneutrality constraint;
- no conserved total inorganic carbon;
- no explicit albumin/phosphate buffering system;
- no full nutrient stoichiometry for protein/fat/mixed meals;
- no molecular energy/ATP balance across the whole organism.

The new `effective_meal_carbs_g` is a **latent input parameter**, not a chemically measured meal composition.

Classification: **internally useful reduced chemistry; not a complete biochemical mass/charge model.**

## Medical audit

### Correct methodological changes

- CGM is treated as an interstitial observation rather than identical to venous/plasma glucose.
- The DuBose event rules are encoded before comparison, avoiding protocol mismatch.
- Train/validation/test splitting is participant-level, reducing leakage.
- The code explicitly refuses to label latent effective carbohydrate as observed intake.
- Jaeb download is now routed through the official terms page rather than a direct object URL.

### Not supported

- diagnosis;
- individualized dietary advice based on inferred meal grams;
- insulin/drug dosing;
- clinical exercise prescription;
- patient-specific prediction;
- causal inference about sleep from the current model.

Classification: **research validation scaffold; not a medical decision model.**

## Physical audit

No new physical conservation law is introduced in v0.10. The event layer schedules inputs into the already tested multiorgan solver.

The main physical concern is input identifiability rather than conservation: a CGM excursion cannot uniquely determine meal nutrient mass because multiple combinations of gastric kinetics, insulin sensitivity, preceding physiology and unreported food can produce similar sensor trajectories.

The event replay therefore does not force inferred input values into the core as physical ground truth.

Classification: **event scheduling is physically consistent with the solver, but external input reconstruction is underdetermined.**

## Algorithmic / ML audit

### Improvements

- participant-level leakage control;
- deterministic split seeds;
- schema discovery report before fitting;
- protocol-based event exclusion;
- TRAIN-only replay parameter fitting;
- held-out validation/test evaluation;
- separate aggregate calibration and participant-level calibration paths;
- explicit warnings around latent inputs.

### Verification

Synthetic wide-format diary + timestamped CGM fixtures verify:

```text
meal extraction
exercise extraction
duration/type association
sleep/wake extraction
CGM timestamp alignment
published meal inclusion rules
published exercise exclusion rule
participant split disjointness
held-out execution
```

### Remaining issues

- the release has not seen the actual Jaeb archive schema because user acceptance is required;
- no Bayesian hierarchical model yet for event-level latent meal input;
- the point-estimate replay profile understates parameter uncertainty;
- no model discrepancy term is yet fitted at event-trajectory level;
- missing/unreported diary events remain an unobserved-input problem.

Classification: **algorithmically sound pipeline mechanics; external real-data execution remains pending.**

## Data-governance correction

v0.9 exposed an automated direct S3 downloader. v0.10 removes that behavior. The official Jaeb page asks for identifying/contact fields, planned use and explicit agreement to terms. The CLI now prints that official workflow and accepts the locally downloaded ZIP afterward.

## What can be claimed after v0.10

Reasonable claim:

> OpenHumSim-RL has a tested event-alignment and held-out calibration pipeline that can ingest free-living CGM/diary data, reproduce published aggregate healthy meal/exercise response scales after explicit latent-input calibration, and keep calibration subjects separate from evaluation subjects.

Not reasonable:

> OpenHumSim-RL has been validated to predict individual healthy-human glucose trajectories from meals and exercise.

That second claim requires the real participant-level run plus better input identifiability.

## Future work considered for v0.11

Priority should be **identifiability**, not another organ:

1. execute v0.10 against a local Jaeb archive obtained under the dataset terms;
2. inspect actual event/data dictionary and patch any schema mismatches;
3. add hierarchical latent meal-input inference with shrinkage/uncertainty;
4. add a second external dataset/protocol where meal nutrient composition is known;
5. fit participant physiology only on TRAIN;
6. use locked validation/test trajectories;
7. add model-discrepancy and posterior-predictive diagnostics.

Only after a known-composition meal dataset is included should inferred `effective_carbs_g` be replaced by a biologically interpretable nutrient input.
