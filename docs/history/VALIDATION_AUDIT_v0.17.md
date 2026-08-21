# OpenHumSim-RL v0.17 — patient–ventilator interaction credibility audit

## Scope

v0.17 adds an explicit **pressure-support patient–ventilator interaction layer** on top of the v0.16 within-breath equation-of-motion model.

The causal chain is now:

```text
neural respiratory timing
        ↓
      Pmus(t)
        ↓
patient-generated pressure/flow signal
        ↓
ventilator trigger
        ↓
pressure-support rise
        ↓
flow-dependent cycling
        ↓
expiration / possible retrigger
```

This permits explicit detection of:

- ineffective triggering;
- delayed cycling;
- premature cycling;
- double triggering;
- leak-driven auto-triggering.

The model remains a **research forward model**. It is not a validated ventilator, patient monitor, clinical alarm algorithm or treatment-selection system.

## 1. Physiological mechanisms

### Triggering

Patient effort must overcome residual intrinsic PEEP not counterbalanced by external PEEP plus a trigger threshold. This creates a mechanistic pathway:

```text
obstruction → dynamic hyperinflation → PEEPi
                              ↓
                    threshold load ↑
                              ↓
                    ineffective effort
```

External PEEP unloads a configurable fraction of that threshold load. This is consistent with the direction reported by Chao et al. (PMID 9404759).

Status: **mechanistically supported; magnitude not patient calibrated**.

### Pressure support and neural unloading

Pressure support does not simply add pressure on top of unchanged patient effort. v0.17 reduces the pressure demand assigned to the patient while retaining a minimum neural drive required for triggering.

Status: **physiologically preferable to additive-drive v0.16 behavior; reduced-order**.

### Cycling

Ventilator inspiration terminates when inspiratory flow falls below a configurable fraction of peak inspiratory flow or reaches a maximum inspiratory time.

In the COPD-like delayed-cycling challenge:

```text
10% cycling threshold:  delay ≈ +1.03 s; PEEPi ≈ 1.55 cmH2O
40% cycling threshold:  delay ≈ -0.11 s; PEEPi ≈ 0.54 cmH2O
```

Chiumello et al. (PMID 17893630) found in COPD that increasing cycling-off to 40% reduced delay from 0.93±0.50 to 0.52±0.25 s at PS 15 cmH2O and reduced dynamic PEEPi. v0.17 reproduces the **direction and order of magnitude**, but overcorrects to slightly early cycling in the optimized challenge and is therefore not a fitted reproduction.

Status: **mechanistically supported, quantitatively approximate**.

### Premature cycling / double triggering

A ventilator cycle ending substantially before the neural inspiration is flagged as premature. If retriggering is enabled and neural inspiration persists, a second assisted breath can occur during the same neural effort.

Status: **mechanistically supported taxonomy; not waveform-validated against a patient cohort**.

### Auto-triggering

A leak-flow bias can cross the flow-trigger threshold in the absence of neural inspiration and initiate an assisted breath.

Status: **mechanistically supported; leak circuit itself remains simplified**.

## 2. Numerical verification

v0.17 scientific gate:

```text
9 / 9 PASS
```

Checks:

1. low-asynchrony synchronized PSV;
2. ineffective trigger + external-PEEP unloading;
3. 10% vs 40% cycling-off ablation;
4. premature cycling;
5. double triggering;
6. leak auto-triggering;
7. waveform phase observability;
8. 10-ms vs 5-ms cycle timestep convergence;
9. preservation of CO2/charge/blood-volume conservation.

v0.17 module tests:

```text
patient–ventilator                 9 PASS
v0.16 dynamic respiratory cycle  10 PASS
v0.15 respiratory mechanics       9 PASS
v0.14 pulmonary control           8 PASS
v0.13 V/Q                         8 PASS
v0.12 blood gas                   8 PASS
event/replay                      6 PASS
```

`test_env.py` contains 37 tests. The monolithic invocation exceeds the available release-test runtime, so it was executed in chunks: first 20 PASS, remaining 15 short/medium PASS, plus both long v0.11 renal/acid-base tests PASS.

## 3. Representative v0.17 challenge outputs

### Synchronized pressure support

```text
trigger delay           ≈ 0.02 s
asynchrony index         0%
VT                       ≈ 0.53 L
PaCO2                    ≈ 38 mmHg
```

### Ineffective trigger challenge

Without external PEEP:

```text
ineffective fraction    ≈ 0.67–0.71
ventilator breaths       << patient efforts
```

With PEEP and earlier cycling:

```text
ineffective fraction     0
asynchrony index          0%
```

This is a deliberately severe challenge and must not be interpreted as clinical prevalence.

### Delayed cycling challenge

```text
cycling-off 10%: delay  ≈ +1.03 s
cycling-off 40%: delay  ≈ -0.11 s
```

### Double triggering challenge

```text
double-trigger fraction ≈ 0.7–0.8
ventilator rate          > neural effort rate
```

## 4. Algorithmic status

New observations expose timing and event burden, but **action space remains 8-dimensional**. The RL agent cannot yet directly manipulate pressure support, PEEP, trigger sensitivity, rise time or cycling threshold.

This is intentional. Adding those actions before verifying the forward patient–ventilator model would confound physics validation with policy optimization.

## 5. Remaining limitations

v0.17 still lacks:

- measured diaphragm electrical activity (EAdi) or neural timing data;
- expiratory muscle activity;
- true ventilator circuit compliance and compressible volume;
- detailed leak dynamics and leak compensation algorithms;
- trigger filters/device-specific signal processing;
- flow starvation under volume control;
- reverse triggering;
- multi-compartment airway closure during patient–ventilator interaction;
- validated NAVA/PAV controllers;
- patient-specific fitting of trigger/cycling delays;
- clinical outcome validation.

The asynchrony index can be interpreted only as a **model event burden**, not a bedside diagnostic measurement.

## 6. Credibility classification

```text
within-breath equation of motion        strong internal verification
trigger/cycling state machine           strong software verification
ineffective-trigger direction           moderate mechanistic credibility
cycling-threshold direction             moderate-strong; human physiologic anchor
double/premature/autotrigger taxonomy   moderate mechanistic credibility
quantitative patient-level timing       weak / not externally fitted
clinical ventilator control             absent
```

## 7. Recommended v0.18

v0.18 can now expose a **bounded ventilator-control action layer**:

```text
PEEP
pressure support
rise time
cycling threshold
trigger sensitivity
```

but only inside explicit safe engineering bounds and with separate penalties for:

```text
asynchrony
hypercapnia / hypoxemia
excess VT / strain
overdistension
high mechanical power
low cardiac output / MAP
```

The first control benchmark should compare RL against a deterministic synchrony-tuning controller, not against no-op alone.
