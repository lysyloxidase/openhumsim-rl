# OpenHumSim-RL v0.16 — physiological / physical / medical credibility audit

## Scope

v0.16 adds within-breath flow-volume dynamics to the v0.15 lung/chest-wall model.
The goal is to make airway obstruction, expiratory flow limitation, dynamic
hyperinflation and intrinsic PEEP emerge from mechanics rather than from direct
assignments to gas-exchange variables.

## 1. Governing physics

The reduced single-compartment equation is

```text
I dQ/dt + R(Q) Q + E_phase V = Paw + Pmus
```

with V obtained by time-integrating flow. Resistance is treated semi-implicitly;
flow limitation is represented as a separate expiratory constraint pressure.
The unconstrained numerical equation residual is <1e-9 cmH2O in the release gate.

Lanteri et al. (PMID 10423313) explicitly describe the complete single-compartment
equation as containing inertance, resistance and elastance terms. Kessler et al.
(PMID 10942987) studied respiratory-system inertance with physiologic model
parameters in the same order as the v0.16 reference value.

Status: **equation verified; parameterization reduced-order**.

## 2. Dynamic hyperinflation / intrinsic PEEP

Intrinsic PEEP is derived from trapped end-expiratory volume above the static
external-PEEP equilibrium. It is then fed back into end-expiratory transpulmonary
pressure. Higher resistance lengthens the respiratory time constant; shorter
expiratory time produces more trapped volume.

Nominal challenge after 30 min:

```text
baseline                         auto-PEEP ~0.0 cmH2O
airway_obstruction               auto-PEEP ~0.8 cmH2O
tachypnea_airway_obstruction     auto-PEEP ~2.3 cmH2O
```

Dal Vecchio et al. (PMID 2178961) measured PEEPi 2.4 ± 1.6 cmH2O in 18 stable
COPD patients. Haluszka et al. (PMID 2111105) found PEEPi correlated with airway
resistance and obstruction severity.

Status: **direction and order-of-magnitude supported; not a COPD validation**.

## 3. Work and P-V hysteresis

The model integrates:

- resistive dissipation: integral R Q^2 dt;
- positive inspiratory muscle work;
- ventilator work above external PEEP;
- reduced P-V hysteresis area from phase-dependent elastance.

Obstruction increases resistive and muscle work. Pressure-controlled ventilation
transfers modeled inspiratory work from Pmus to Paw. Recruitment/derecruitment as
a contributor to P-V hysteresis is supported by Cheng et al. (PMID 8904012).

Status: **mechanistically plausible diagnostic, not a validated Campbell diagram**.

## 4. Important medical caveats

- `airway_obstruction` is not a diagnosis of COPD or asthma.
- Expiratory muscle activity is not represented; therefore measured PEEPi caused
  by expiratory muscle recruitment is outside model scope.
- No airway tree, equal-pressure-point model, small-airway closure kinetics,
  bronchodilator PK/PD, mucus, or emphysematous heterogeneity.
- No validated ventilator waveform, trigger/cycling algorithm, or patient-ventilator
  synchrony model.
- The action space does not yet expose PEEP or pressure support.

Therefore the environment must not be used for ventilator-setting or treatment
decisions.

## 5. Validation gates

```text
v0.16 respiratory-cycle unit tests        10 PASS
v0.16 scientific checks                  10/10 PASS
v0.15 mechanics                            9 PASS
v0.14 pulmonary                            8 PASS
v0.13 pulmonary                            8 PASS
v0.12 whole-blood gas                      8 PASS
event/replay                               6 PASS
selected whole-environment regression      7 PASS
```

The monolithic historical `test_env.py` remains too slow for the local
available release-test runtime; it is not claimed as a complete pass in this release.

## 6. Algorithmic credibility

The within-breath solver uses a default 10-ms step and is checked against 5 ms.
The scientific gate requires close agreement in VT, auto-PEEP and muscle work.
A representative 30-min whole-body rollout with the new cycle solver completes in
roughly 1–2 seconds in the reference environment.

## 7. Future work

v0.17 should add respiratory-muscle and ventilator timing/interaction:

```text
Pmus neural drive + trigger detection
           ↓
ventilator trigger / rise / cycling
           ↓
patient-ventilator synchrony
           ↓
ineffective efforts / double triggering / delayed cycling
```

Only after this should PEEP/pressure-support controls be exposed as RL actions.
