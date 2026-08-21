# OpenHumSim-RL v0.19 — biological / chemical / medical / algorithmic audit

## Classification

v0.19 is a **P1 physiological-consistency refactor**.

It is a research simulator. It is not a clinical digital twin, insulin dosing
tool, electrolyte treatment calculator, or fluid-prescription model.

The release gate is a verification/credibility gate, not independent clinical
validation.

## 1. Subcutaneous insulin pharmacokinetics

### Previous problem

Through v0.18 an insulin action could increase the Dalla Man plasma-insulin
compartment immediately. That was useful for early RL experiments but was not a
credible model of subcutaneous insulin delivery.

### v0.19 implementation

Two serial SC depots:

```text
dS1/dt = -S1/tau
dS2/dt =  S1/tau - S2/tau
absorption = S2/tau
```

The subsystem is advanced analytically inside the Dalla internal integration
step. For equal transit constants an isolated bolus has peak absorption at
approximately `tau`.

Default:

```text
tau / target tmax = 90 min
```

Verification:

```text
immediate plasma-insulin jump = 0
absorption peak              = 90 min
SC mass residual             < 1e-12 model-units
```

Primary human anchors:
- Ruan et al. 2014, PMID 25092225
- Herzig et al. 2020, PMID 31999478

Limitations:
- `model_units` are not U of clinical insulin;
- no formulation-specific hexamer/dimer chemistry;
- no site/temperature/lipohypertrophy dependence;
- no patient-specific insulin clearance;
- default is only a rapid-acting-analogue-like reference.

## 2. Glucagon counterregulation

### Previous problem

Glucagon was mostly a readout and did not affect hepatic glucose output.

### v0.19 implementation

Low glucose raises a dynamic glucagon state. A separately labelled
counterregulatory EGP extension is then added to the published Dalla Man EGP
only under hypoglycemia and is limited by remaining liver glycogen.

The glucose flux and glycogen decrement use the same RK4 quadrature, avoiding a
hidden carbon/mass mismatch inside the extension.

30-min challenge from an imposed 50 mg/dL rapid-compartment glucose state:

```text
counterreg OFF glucose   ~70.05 mg/dL
counterreg ON glucose    ~72.24 mg/dL
glucagon                 ~105.95 pg/mL
current extra EGP        ~0.142 mg/kg/min
liver glycogen used      ~0.56 g
```

This is a direction/mechanism test, not a fitted hypoglycemic-clamp prediction.

Primary human anchors:
- Rizza et al. 1979, PMID 36413
- Cryer et al. 1983, PMID 6341018

Missing:
- epinephrine;
- cortisol;
- growth hormone;
- autonomic symptoms;
- diabetes-specific loss of glucagon response;
- antecedent-hypoglycemia effects.

## 3. Transcellular potassium

### Previous problem

The renal module contained a generic controller pulling serum K toward
4.2 mmol/L. That conserved K mass but did not encode the major physiological
drivers of acute redistribution.

### v0.19 implementation

A single conserved transfer flux now responds to:
- insulin;
- arterial acid-base state;
- exercise.

Positive flux means ICF -> ECF. The transfer is bounded by available compartment
mass.

Direction test:

```text
baseline K              ~4.20 mmol/L
high-insulin challenge  ~3.86 mmol/L
acidemic challenge      ~4.95 mmol/L
exercise challenge      ~4.74 mmol/L
total-K residual        < 1e-12 mmol
```

Primary human insulin/K anchor:
- Cohen et al. 1991, PMID 1874934

Important limitation: the acid-base term is still reduced-order. Mineral,
organic and respiratory acid-base disorders do not yet have fully distinct
cellular K/H transport models. There is also no explicit beta2-adrenergic
receptor/Na-K-ATPase kinetic model.

## 4. Osmotic ECF/ICF water redistribution

### Previous problem

Oral free water was assigned directly as 1/3 ECF and 2/3 ICF.

### v0.19 implementation

The model now conserves:
- total body water;
- ECF effective osmoles;
- short-horizon intracellular effective osmoles.

The acute water partition is solved from equal effective tonicity.

For a 0.9-L free-water challenge in the deterministic verification case:

```text
ECF gain  ~0.303 L
ICF gain  ~0.597 L
Na        decreases
ECF-ICF tonicity residual ~5.7e-14 mOsm/L
```

For 0.9 L 0.9% saline:

```text
ECF gain  ~0.948 L
ICF change ~-0.048 L
```

The small ICF shift arises because 154 mmol/L NaCl is modestly hypertonic
relative to this particular seeded baseline.

The model deliberately excludes urea from acute *effective* tonicity.

Limitations:
- oral absorption is still instantaneous once the intervention is applied;
- no GI water compartment;
- no explicit aquaporin kinetics;
- intracellular osmoles are fixed on this short horizon except for future model extensions;
- no tissue-specific cell-volume model.

## 5. Scenario ledger fix

A pre-existing bug was corrected: a fluid challenge used to construct a
scenario could be counted both as part of the t=0 state and as an episode-time
administration.

v0.19 resets administration/loss ledgers after the scenario initial condition is
fully constructed.

For `saline_challenge_30ml_kg` at t=0:

```text
water mass residual      = 0
Na mass residual         = 0
Cl mass residual         = 0
TBW-(ECF+ICF) residual   = 0
```

## 6. Algorithmic implications

The default clinical-like observation remains 50 variables. New latent states,
including SC depot contents and K controller target, are visible only in
`observation_profile="full"`.

Full observation count:

```text
137
```

This preserves the POMDP-style separation introduced in v0.18.

Breaking change:

```text
older policy + v0.19 insulin semantics = invalid comparison
```

Any RL benchmark using insulin must be retrained.

## 7. Verification / regression

v0.19 focused tests:

```text
tests/test_credibility_v019.py     7 PASS
validation/run_validation_v19.py  9/9 PASS
compileall                         PASS
```

Directly affected historical layers:

```text
v0.18 credibility                 9 PASS
v0.17 patient-ventilator          9 PASS
v0.16 respiratory cycle          10 PASS
v0.15 mechanics                   9 PASS
v0.14 pulmonary                   8 PASS
v0.13 V/Q                         8 PASS
v0.12 blood gas                   8 PASS
event/replay                      6 PASS
```

`tests/test_env.py`:
- 36 shorter tests pass directly under pytest;
- the two historical tests whose names and requested iteration count imply 24 h are actually
  truncated by the environment episode horizon at 725 min (~12.1 h); this was
  verified explicitly rather than being reported as a 24-h result;
- the same state/timestep trajectory was executed in continuous checkpointed
  6-h process segments because a single local process exceeds the execution limit;
- respiratory-acidosis endpoint (~725 min): NH4, HCO3 and pH all increased from baseline;
- reduced-renal-function endpoint (~725 min): strong-ion gap increased, renal acid
  excretion remained below baseline, acid-mass residual ~1.3e-11 mEq.

Checkpointing changes only process execution, not model state or timestep. The
release therefore makes **no 24-hour compensation claim** from these tests.

Event-replay calibration was also de-duplicated: meal-grid fitting no longer
simulates the exercise branch and exercise-grid fitting no longer simulates the
meal branch. This changes runtime, not the objective or fitted mechanism.

## 8. What is still not fixed

Highest-priority remaining items:

1. lactate/ATP/substrate metabolism is still reduced-order;
2. regional V/Q is more complete for O2 than for full regional CO2 chemistry;
3. K shifts do not yet distinguish all acid species/mechanisms;
4. no explicit epinephrine/cortisol/GH counterregulation;
5. oral water lacks GI absorption kinetics;
6. virtual-patient priors are not yet a learned correlated human population;
7. clinical observations still lack realistic sampling delays/noise/missingness;
8. no prospective or patient-specific validation.

## 9. Recommended next release

v0.20 should focus on **measurement realism + correlated virtual patients**,
not additional organ equations:

```text
hidden physiological state
        ↓
sensor/lab observation model
        ├─ noise
        ├─ delay
        ├─ sampling intervals
        └─ missingness
        ↓
clinical observation
        ↓
RL policy
```

and population priors should use biologically correlated variables rather than
independent uniform draws.
