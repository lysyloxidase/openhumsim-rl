# OpenHumSim-RL v0.14 — credibility audit

## Scope

v0.14 adds **regional pulmonary vascular control and aeration dynamics** to the v0.13 six-unit V/Q forward model. The release is intended to improve physiological and physical credibility before disease-specific ARDS/COPD/PE models are introduced.

## Implemented mechanisms

### 1. Hypoxic pulmonary vasoconstriction

Each regional unit has a persistent HPV tone. Lower local alveolar PO2 increases a precapillary resistance multiplier. Regional flow is solved from six parallel vascular conductances and then normalized to the non-shunt pulmonary flow. The equivalent parallel-resistance multiplier feeds the closed-loop pulmonary vascular resistance on the following operator-split step.

This means two separable effects emerge:

- heterogeneous hypoxia -> perfusion redistribution and improved V/Q matching;
- global hypoxia -> little useful redistribution but higher whole-lung pulmonary vascular resistance/PAP.

The implementation is a vascular-control surrogate. It does **not** model PASMC mitochondrial sensing, ROS, Ca2+, endothelium, NO, AMPK or other molecular HPV pathways.

### 2. Recruitment/derecruitment

Each of the six units has a recruitment fraction in [0,1]. A reduced mean distending pressure is computed from PEEP plus a fraction of VT/compliance-derived driving pressure. Unit-specific closing thresholds follow a dependent-region ordering. Opening uses a higher threshold than remaining open, creating reduced hysteresis, with distinct opening/closing time constants.

Derecruited units receive very little ventilation but are not automatically assigned zero perfusion. They therefore create low-V/Q/shunt-like exchange unless HPV diverts flow.

The thresholds are engineering parameters anchored to human pressure-dependent recruitment physiology; they are not patient CT opening pressures.

## External primary-source anchors

1. **Asadi AK et al., J Appl Physiol, 2015, PMID 25429099.** Inhaled nitric oxide altered regional blood-flow distribution in healthy human lungs, supporting active HPV as a regional V/Q regulator.
2. **Robin ED et al., 1987, PMID 3545645.** Hypoxic exposure in human heart-lung transplant recipients significantly increased pulmonary artery pressure and pulmonary vascular resistance, showing that the response persists in denervated transplanted lungs.
3. **Crotti S et al., Am J Respir Crit Care Med, 2001, PMID 11435251.** In acute respiratory failure, PEEP/tidal inflation and regional superimposed pressure govern recruitment and collapse; recruitment occurs across a pressure range rather than at one universal opening pressure.
4. **Hammond MD et al., J Appl Physiol, 1986, PMID 3710978.** MIGET demonstrated increased V/Q inequality during exercise and evidence of diffusion limitation at high VO2; retained as a v0.13 regression anchor.

## Numerical release results

From `validation/validation_results_v0.14.json`:

```text
Healthy baseline
recruitment fraction             0.977
HPV equivalent R multiplier      1.082
perfusion redistribution index   0.0127
PaO2                              89.94 mmHg

V/Q mismatch, HPV enabled
PaO2                              68.37 mmHg
redistribution index              0.0319

same V/Q mismatch, HPV disabled
PaO2                              67.40 mmHg
redistribution index              0.0000

Global hypoxia (FiO2 0.13), 20 min
HPV-on PVR multiplier             2.345
HPV-on PAP                        23.25 mmHg
HPV-off PAP                       15.93 mmHg

Dependent derecruitment
recruitment fraction              0.544
PaO2                              75.11 mmHg
A-a                               25.92 mmHg

Same collapsible lung + PEEP state
recruitment fraction              0.998
PaO2                              96.89 mmHg
A-a                                4.14 mmHg
```

The exact gains are model outputs, **not clinical treatment targets**.

## Physical / chemical consistency

The new module does not create or remove blood, oxygen content, ions or exchangeable CO2. In randomized multi-scenario stress checks:

```text
max CO2 mass residual       3.13e-13 mmol
max charge residual         9.83e-09 mEq/L
NaN/Inf                     none
unexpected nonpulmonary termination 0
```

Timestep sensitivity for V/Q mismatch (0.25 vs 0.125 min physiology step):

```text
Delta PaO2                  0.34 mmHg
Delta HPV R multiplier      3.7e-05
Delta recruitment fraction  3.6e-07
```

## Algorithmic status

The regional model uses only six units and two fixed-point redistribution passes, keeping it fast enough for RL. A 60-min V/Q-mismatch zero-action rollout was about 3.4 s in the reference environment.

The public action space stays length 8. PEEP is currently a mechanistic state/scenario input rather than a ninth RL actuator. This avoids pretending that a complete mechanical ventilator has been implemented before airway pressure, lung/chest-wall mechanics and heart-lung pressure interactions exist.

## Tests actually completed during this release

```text
v0.14 pulmonary-control tests          8/8 PASS
v0.14 scientific checks                9/9 PASS
v0.13 pulmonary tests                  8/8 PASS
v0.12 whole-blood gas tests            8/8 PASS
event/replay tests                     6/6 PASS
core test_env groups 0-19             PASS
additional CGM/UQ/acid-base tests      PASS when run as smaller groups
24-h respiratory-acidosis renal test   PASS
24-h reduced-renal-function test       runtime limit reached before completion
```

The final 24-h reduced-renal-function check is therefore **not counted as PASS** for v0.14. The timeout is disclosed as an algorithmic/release limitation.

## Credibility classification

| Domain | v0.14 status |
|---|---|
| V/Q content mixing | mechanistic reduced-order |
| true shunt | mechanistic reduced-order |
| finite O2 diffusion | mechanistic reduced-order |
| HPV regional perfusion control | mechanistic control surrogate, human direction anchored |
| whole-lung HPV/PAP coupling | mechanistic 0D coupling, not fitted to an individual human time course |
| recruitment/derecruitment | pressure-threshold reduced-order, human mechanism anchored |
| PEEP | recruitment-state input only |
| CO2/RBC chemistry | v0.12 conserved reduced-order |
| clinical disease prediction | unsupported |
| ventilator decision support | unsupported |

## Remaining high-priority gaps

1. lung + chest-wall pressure-volume mechanics and compliance distribution;
2. dynamic transpulmonary pressure rather than the present reduced pressure surrogate;
3. overdistension and U-shaped pulmonary vascular resistance versus lung volume;
4. PEEP effects on venous return, RV afterload and cardiac output;
5. continuous/spatial regional geometry and gravity;
6. CT/EIT/MIGET calibration of recruitment and V/Q distributions;
7. surfactant and surface-tension dynamics;
8. disease-specific HPV impairment and parameter estimation.

## Recommended v0.15

Implement a coupled **lung/chest-wall mechanics + pressure-volume hysteresis + overdistension + PEEP-heart interaction** model, and validate it against human CT/EIT pressure-step data before introducing ARDS/COPD disease labels.
