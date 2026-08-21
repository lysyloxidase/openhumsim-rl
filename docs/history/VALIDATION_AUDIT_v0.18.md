# OpenHumSim-RL v0.18 — credibility refactor audit

## Purpose

v0.18 is a corrective release. It addresses the highest-risk cross-system inconsistencies identified in the v0.17 audit rather than increasing model breadth.

## Corrected P0 issues

### 1. Ventilation action bypass — FIXED

**Before:** policy action 6 could add effective alveolar ventilation directly in L/min. This bypassed airway pressure, resistance, flow, tidal volume, work, auto-PEEP and heart-lung effects.

**Now:** action 6 maps to airway-pressure assist (`0..12 cmH2O`) and can change ventilation only through the within-breath equation of motion. A regression test reconstructs alveolar ventilation from actual RR, VT, dead space and efficiency and requires zero legacy L/min support.

Credibility: **physical/algorithmic bug fixed**. The pressure-assist controller remains reduced-order and is not a clinical ventilator.

### 2. Hb-O2 chemistry disconnected from pH/CO2 — FIXED at reduced order

A dedicated `OxygenBindingModel` now uses the Severinghaus adult standard curve plus pH-dependent Bohr shifting. PCO2 has a small residual direct affinity term because most CO2 influence is already mediated through model pH. Regional V/Q units and whole-body O2 transport use the same binding layer.

At PO2=40 mmHg, the gate requires saturation to increase from pH 7.2 -> 7.4 -> 7.6 and P50 to move in the opposite direction. Standard-condition P50 is 26.6 mmHg.

Credibility: **substantially improved but still reduced**. Temperature, 2,3-DPG, fetal Hb and dyshemoglobins remain absent; this is not the full Dash/Korman/Bassingthwaighte binding model.

### 3. Impossible imposed VO2 — FIXED at whole-body reduced order

The state now distinguishes `vo2_demand_ml_min` from achieved `vo2_ml_min`. When O2 delivery cannot meet demand at the configured extraction ceiling, achieved VO2 falls and `oxygen_debt_ml_min` increases.

The extraction ceiling (0.70) is an **engineering physiology parameter**, not a validated universal human threshold. Organ-specific extraction limits are not modeled.

### 4. Hypoxic metabolism absent — PARTIALLY FIXED

Previous-step oxygen debt now creates a reduced hypoxic lactate-production term in addition to exercise production. Clearance is first-order. This fixes the prior impossibility in which severe O2 supply failure could occur without any metabolic consequence.

Credibility: **directionally mechanistic only**. There is still no ATP/ADP, NADH, glycolytic stoichiometry, Cori cycle, liver-perfusion-dependent lactate clearance or organ-specific ischemia. Lactate must not be interpreted as a specific hypoxia biomarker.

### 5. V/Q disconnected from CO2 elimination — PARTIALLY FIXED

Regional high-V/Q wasted ventilation now contributes an alveolar-dead-space fraction. Whole-body carbon elimination uses:

```text
effective CO2 ventilation = alveolar ventilation * (1 - alveolar dead-space fraction)
```

This creates a real V/Q -> CO2 path.

Credibility: **improved but not full regional CO2 transport**. Regional CO2 content mixing is still less detailed than regional O2 mixing.

### 6. Hb/Hct fixed during saline/dehydration — FIXED for short episodes

RBC volume and Hb mass are conserved over an episode; plasma-volume changes therefore change hematocrit and [Hb]. This feeds blood O2 content and blood-gas chemistry. No erythropoiesis, hemolysis, bleeding or transfusion is modeled.

### 7. Carbonate charge — FIXED

Reduced acid-base charge closure now counts carbonate as `2 * [CO3--]`. Documentation now calls the residual **reduced charge closure**, not complete plasma electroneutrality, because Ca2+, Mg2+, sulfate and other species remain implicit.

### 8. RL privileged-state leakage — REDUCED

Default observations are now 50 clinical/monitor/lab/ventilator-like values. The 126-variable mechanistic observation remains available only with `observation_profile="full"`. Selected latent states such as exact V/Q dispersion, HPV tone, Dalla EGP, PBPK effect-site concentration and oxygen debt are excluded from the default policy input.

This is not yet a complete measurement process: laboratory delays, missingness and measurement noise remain future work.

### 9. Reward ignored newly modeled injury/asynchrony mechanisms — FIXED structurally

Reward now includes penalties for oxygen debt, ventilator asynchrony, auto-PEEP, overdistension, lung strain, mechanical power, excessive VT and low cardiac output, plus intervention cost.

Credibility caveat: reward weights are engineering choices, **not clinical utilities**. Any future RL paper must include sensitivity/ablation analysis and rule/MPC comparators.

## v0.18 validation status

New credibility tests: **9/9 PASS**.

Regression modules completed under v0.18:

```text
v0.17 patient-ventilator      9/9 PASS
v0.16 respiratory cycle     10/10 PASS
v0.15 respiratory mechanics  9/9 PASS
v0.14 pulmonary               8/8 PASS
v0.13 V/Q                     8/8 PASS
v0.12 blood gas               8/8 PASS
event/replay                  6/6 PASS
```

`tests/test_env.py` was exercised in batches because a monolithic run exceeded the available release-test runtime. All short/medium groups tested in this release passed. The 24-h respiratory-acidosis test printed PASS before the test process reached its teardown limit; the separate 24-h reduced-renal-function test did not complete inside the available runtime, so it is **not claimed as PASS for v0.18**.

## Breaking algorithmic change

All PPO checkpoints trained before v0.18 are legacy artifacts. They must not be compared directly with v0.18 policies because:

1. default policy observations changed to 50 clinical-like variables;
2. full state changed dimension as well;
3. action 6 changed physical meaning from virtual L/min support to pressure assist in cmH2O;
4. reward changed.

Retraining and new baselines are mandatory.

## Remaining high-priority gaps

Not fixed in v0.18:

- SC insulin PK and validated counter-regulatory glucagon dynamics;
- mechanistic transcellular K handling and osmotic ECF/ICF water redistribution;
- full substrate/ATP/lactate metabolism;
- full regional CO2 blood-content mixing;
- temperature/2,3-DPG/dyshemoglobin effects on Hb;
- spontaneous negative-pressure heart-lung interaction across each breath;
- correlated human virtual-patient priors;
- explicit measurement noise/delay/missingness;
- locked independent human validation cohorts;
- real-drug PBPK.

## Credibility classification after refactor

```text
software/API                         strong research scaffold
modeled-pool conservation            strong verification
ventilation action physics           substantially improved
whole-blood O2/CO2 coupling          moderate mechanistic
acid-base chemistry                  moderate reduced-order
cardiovascular 0D                    moderate mechanistic
renal/electrolyte physiology         limited-moderate
whole-body metabolism                limited
external quantitative validation     limited
clinical predictive use              unsupported
RL benchmark credibility             improved, retraining required
```

The appropriate description remains **mechanistic human-homeostasis research simulator**, not validated digital twin.
