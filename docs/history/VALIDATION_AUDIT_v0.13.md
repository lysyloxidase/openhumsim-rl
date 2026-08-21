# OpenHumSim-RL v0.13 — validation and credibility audit

## Scope

v0.13 addresses a major physiological weakness of v0.12: the previous respiratory path still treated the lung as a single effective alveolar compartment for oxygen. v0.13 introduces a reduced multi-compartment pulmonary exchange model while preserving the v0.12 whole-blood CO2, RBC, acid-base and carbon-conservation machinery.

This release is **not clinically validated** and must not be used for patient care, oxygen prescriptions, ventilator settings, diagnosis, or treatment selection.

## Architecture added

The lung contains six parallel ventilated/perfused units plus an explicit true-shunt pathway. Local V/Q ratios are generated from a smooth log-space distribution around the current whole-lung VA/Q ratio. Perfusion fractions sum to one over ventilated units; ventilation fractions are normalized exactly to total alveolar ventilation.

Oxygen is mixed by **blood O2 content**, not by averaging PO2. This is required because the hemoglobin dissociation curve is nonlinear. Shunted blood bypasses alveolar exchange and is mixed with end-capillary blood at mixed-venous O2 content.

Finite diffusion is represented by an exponential approach from mixed-venous PO2 toward local alveolar PO2 using current pulmonary capillary transit time and a relative diffusing-capacity parameter. Transit time is derived from capillary blood volume divided by pulmonary blood flow.

## External physiological anchors

### V/Q heterogeneity

Wagner, Saltzman and West (1974; PMID 4826323) established the mathematical basis for continuous ventilation/perfusion distributions. Wagner et al. (1974; PMID 4601004) applied the approach in normal human subjects. v0.13 uses these works to justify a distributed rather than single-alveolus architecture.

The v0.13 variable `pulmonary_vq_log_sd` is an **engineering dispersion parameter**, not a claimed numerical reproduction of MIGET LogSDQ.

### Pulmonary capillary transit time and exercise

Warren et al. (1991; PMID 1798377) reported in endurance athletes a mean pulmonary capillary transit time of about 1.05 s at rest and 0.42-0.46 s during exercise, together with widening of the A-a O2 difference. v0.13 reproduces the transit-time scale and the direction of A-a widening. It does not force the published exercise PaO2 of 85 mmHg because that result came from a selected endurance-athlete cohort.

### Diffusion/VQ/shunt separation

Torre-Bueno et al. (1985; PMID 2984169) experimentally separated V/Q inequality, diffusion resistance, post-pulmonary shunt and other contributors to arterial oxygenation in normal humans during exercise/altitude challenges. v0.13 therefore keeps V/Q dispersion, true shunt and diffusion capacity as distinct state variables rather than one generic `lung_efficiency` factor.

## Representative build results

For a nominal baseline seed:

```text
PaO2                         ~89-95 mmHg depending on seeded PaCO2
A-a gradient                 ~5 mmHg
SpO2                         ~97%
capillary transit time       ~0.9 s
Enghoff-like VD/VT           ~0.30-0.32
O2 diffusion equilibration   ~0.99
```

Mechanistic challenges produce distinct phenotypes:

```text
vq_mismatch          PaO2 decreases, A-a widens, wasted ventilation rises
pulmonary_shunt      PaO2 decreases with relative O2 refractoriness
diffusion_limitation equilibration fraction and PaO2 decrease
exercise              capillary transit shortens and A-a widens
```

With the implemented challenge strengths, supplemental oxygen increases PaO2 much more strongly in V/Q mismatch than in the true-shunt scenario. This is used only as a mechanistic check, not a clinical treatment recommendation.

## Physical and numerical verification

- perfusion fractions are normalized;
- ventilation fractions are normalized;
- true shunt is mixed by O2 content;
- current v0.12 CO2 mass ledger remains conserved;
- plasma charge balance remains constrained by the v0.11/v0.12 chemistry;
- pulmonary timestep sensitivity is explicitly checked at 0.25 vs 0.125 min integration steps;
- random multi-scenario stress checks enforce finite observations/rewards and bounded pulmonary diagnostics.

The v0.13 pulmonary model was initially too expensive because it was nested inside every PaCO2 root-solver iteration. The release uses explicit operator splitting: the carbon solver holds current arterial oxygenation while solving PaCO2, then the six-compartment pulmonary model is evaluated once. This materially improves runtime while preserving the release validation gates. That approximation is explicitly part of the model specification.

## Release gates

```text
v0.13 pulmonary unit tests             8 PASS
v0.12 blood-gas regression tests       8 PASS
core/env short regression tests       35 PASS
long renal/acid-base regression tests  2 PASS
event/replay regression tests          6 PASS
v0.13 scientific validation            9/9 PASS
```

The long renal tests are run separately because combining all long simulations in one local process can exceed the execution timeout. This is an infrastructure limit, not a hidden test failure.

## Credibility classification

### Stronger / internally constrained

- explicit true-shunt mixing by blood O2 content;
- normalized parallel V/Q compartments;
- finite capillary-transit/diffusion mechanism;
- preserved whole-blood CO2 conservation and RBC chemistry from v0.12;
- explicit unit metadata for all new policy observations;
- numerical convergence/stress tests.

### Physiologically plausible but reduced-order

- six compartments rather than continuous V/Q distributions;
- fixed compartment perfusion weights before pathology-specific redistribution;
- simplified diffusion time constant instead of explicit DLCO/DLNO membrane/capillary conductances;
- high-V/Q wasted-ventilation term used for dead-space diagnostics;
- exercise capillary recruitment represented by a compact empirical factor.

### Not validated / missing

- no patient-level pulmonary calibration;
- no explicit MIGET inert-gas forward/inverse model;
- no gravity-dependent spatial lung geometry;
- no hypoxic pulmonary vasoconstriction;
- no regional vascular resistance/recruitment-derecruitment;
- no ARDS/COPD/PE disease-specific parameter estimation;
- no explicit regional CO2 blood-content mixing within every V/Q compartment;
- no airway resistance/compliance mechanics or aerosol transport.

## Medical interpretation boundary

`vq_mismatch`, `pulmonary_shunt`, and `diffusion_limitation` are mechanism challenges, not diagnoses. For example, the `pulmonary_shunt` scenario is not an ARDS model, and `vq_mismatch` is not a COPD or pulmonary embolism model.

## Next high-value step

v0.14 should add **hypoxic pulmonary vasoconstriction and dynamic perfusion redistribution/recruitment**, followed by validation against a published MIGET V/Q dataset. Disease labels should only be introduced after the mechanism-level lung can reproduce external V/Q-distribution data.
