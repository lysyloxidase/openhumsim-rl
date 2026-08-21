# OpenHumSim-RL v0.15 — credibility audit

## Scope

v0.15 adds a reduced explicit model of lung and chest-wall mechanics and
couples positive intrathoracic pressure to the pre-existing closed-loop
circulation.

This audit distinguishes:

- **verified**: algebra / software property directly tested;
- **mechanistically supported**: direction/structure supported by human physiology;
- **plausible reduced-order**: useful engineering approximation, not externally fitted;
- **not validated**: not suitable for patient prediction.

## 1. Physical / mechanical verification

### Transpulmonary pressure

Implemented identity:

```text
PL = Paw - Ppl
```

The release gate requires residual < 1e-10 cmH2O.

Status: **verified**.

Primary mechanism anchor: Akoumianaki et al. 2014, PMID 24467647; Grieco et al.
2017, PMID 28828360. Esophageal pressure is used clinically/research-wise as a
surrogate for pleural-pressure changes and separates lung from chest-wall load.

### Separate lung and chest-wall elastance

For the same tidal volume:

```text
dP_lung = VT / C_lung
dP_cw   = VT / C_chestwall
dP_aw   = dP_lung + dP_cw
```

The stiff-chest-wall ablation raises airway driving pressure without materially
changing the lung transpulmonary driving pressure. The low-lung-compliance
ablation raises transpulmonary driving pressure.

Status: **verified internally; mechanistically supported; not patient calibrated**.

### Overdistension

A smooth high-PL penalty reduces incremental lung compliance and raises the
mechanical contribution to PVR. This prevents "more PEEP is always better."

Status: **plausible reduced-order**, not a CT/EIT-derived stress-strain model.

## 2. Physiological credibility

### Recruitment vs PEEP

The same derecruitment-prone model is compared at low pressure and higher PEEP.
Recruitment and PaO2 improve when end-expiratory transpulmonary pressure rises.

Status: **mechanistically supported**, but opening/closing thresholds are not
patient-specific.

### Heart-lung interaction

Positive intrathoracic pressure is added to intrathoracic chamber pressures in
the closed-loop circulation. This reduces the systemic venous-return gradient.
The lung-volume mechanical PVR multiplier is combined with HPV resistance.

Direction in v0.15:

```text
baseline CO > PEEP12 CO
dehydrated + PEEP12 CO < euvolemic + PEEP12 CO
```

This matches established heart-lung physiology.

Primary anchors:
- Luecke & Pelosi 2005, PMID 16356246
- Hamahata et al. 2023, PMID 37541314
- Vieillard-Baron et al. 2016, PMID 27038480

Status: **mechanistically supported; magnitude not clinically validated**.

## 3. Numerical / algorithmic verification

Scientific gate:

```text
9 / 9 PASS
```

Checks:

1. PL = Paw - Ppl identity
2. healthy baseline mechanics
3. lung-vs-chest-wall separation
4. PEEP recruitment tradeoff
5. high-PEEP overdistension
6. positive-pressure hemodynamic direction
7. hypovolemia × positive-pressure interaction
8. timestep convergence
9. conservation regressions

Timestep gate compares 0.25-min and 0.125-min outer integration steps.

A 60-min PEEP12 rollout executes in ~2.8 s in the reference environment.

## 4. Regression

Executed modularly because the monolithic suite can exceed the local
available release-test runtime:

```text
core environment                         37 PASS
v0.15 mechanics                           9 PASS
v0.14 pulmonary                           8 PASS
v0.13 pulmonary                           8 PASS
v0.12 blood gas                           8 PASS
event/replay                              6 PASS
scientific v0.15                        9/9 PASS
```

The historical v0.14 scientific script has 8 mechanistic PASS results and one
expected version-string mismatch after upgrading to 0.15.0.

## 5. What v0.15 still does NOT validate

- no patient-specific esophageal-pressure calibration;
- no measured flow-pressure airway resistance curve;
- no dynamic airway closure;
- no regional chest-wall mechanics or gravitational pleural-pressure field;
- no viscoelastic stress relaxation;
- no full inspiratory/expiratory P-V loop;
- no ventilator waveform, inspiratory time, flow-control or plateau maneuver;
- no spontaneous respiratory-muscle pressure model;
- no diaphragm/abdominal mechanics;
- no validated mechanical-power/VILI risk model;
- no patient-level PEEP recommendation.

Therefore v0.15 must **not** be used to choose ventilator settings.

## 6. Credibility classification

```text
software/API                     strong
mass/charge conservation         strong within modeled pools
PL pressure identity             strong
lung/chest-wall load separation  moderate-strong mechanistic
PEEP recruitment direction       moderate mechanistic
heart-lung hemodynamic direction moderate mechanistic
overdistension law               reduced-order heuristic
patient-specific mechanics       absent
clinical ventilator validation   absent
```

## 7. Recommended next release

v0.16 should add dynamic airway resistance and a real respiratory-cycle model:

```text
Pmus / ventilator waveform
        ↓
airway resistance + inertance
        ↓
flow(t)
        ↓
volume(t)
        ↓
Paw(t), Ppl(t), PL(t)
        ↓
work of breathing / P-V hysteresis
```

Only after that should PEEP/VT become RL treatment actions.
