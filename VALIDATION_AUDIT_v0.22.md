# OpenHumSim-RL v0.22 — energy/carbon conservation audit

## Decision

v0.22 fixes the previously disclosed amount- and coupling-level defects in the
reduced energy, lactate and blood-gas path. The implementation now has explicit
mass/integral invariants and passes its repository regression suite. This is
software verification and mechanistic consistency evidence, not external
clinical validation.

## Corrected defects

1. **Prescribed VCO2 under oxygen failure.** Oxidative CO2 generation is now
   `metabolic RQ × achieved VO2`; unmet VO2 demand cannot continue producing
   fully aerobic CO2. Metabolic RQ is derived from the historical substrate-
   mixture targets and is kept distinct from transient pulmonary RER.
2. **Endpoint-rate carbon integration.** Metabolic production uses the average
   of start and final achieved-VO2-coupled oxidative VCO2. Ventilatory
   elimination is solved with the trapezoidal start/final flux in the same
   scalar pool/content root. All coupled candidates use one immutable ledger
   baseline, so cumulative counters are committed once.
3. **Concentration-only lactate.** `lactate_amount_mmol` is conserved in an
   apparent distribution space. Gross appearance and disposal counters close
   `amount = initial + generated - cleared`; fluid changes alter concentration
   through distribution volume rather than creating or destroying solute.
4. **Ambiguous oxygen debt.** Instantaneous unmet demand retains the historical
   compatibility name `oxygen_debt_ml_min`. A separate monotonic
   `cumulative_oxygen_deficit_ml` records its time integral and is never called
   a repayable debt/EPOC store.
5. **Post-pulmonary Haldane mismatch.** Carbon chemistry formerly solved against
   pre-update oxygenation. The final PaO2/SaO2 and PaCO2/pH/speciation are now
   iterated jointly with achieved VO2/VCO2, pulmonary elimination and venous
   return after exactly one temporal HPV/recruitment advance. The final closure
   residual is explicit.
6. **Fixed-RQ alveolar oxygen equation.** Cellular substrate RQ remains a
   metabolic production quantity. The lung now uses bounded pulmonary RER =
   CO2 elimination / achieved O2 uptake, with a metabolic-RQ fallback when
   either flow is too small to form a stable ratio, in the full FiO2-corrected
   alveolar gas equation.
7. **Reset and transition atomicity.** Challenged reset states now close the
   existing carbon pool against final O2/RER without advancing flux counters.
   Instant interventions and their first continuous substep roll back state and
   private solver phases together on numerical failure.

Lactate remains a monovalent strong anion in the Stewart-Figge SID. Its effect
is not duplicated in UMA, a direct bicarbonate correction or a second CO2
generation term. Pulmonary CO2 elimination can exceed oxidative production by
emptying the existing exchangeable bicarbonate/carbon store.

## Biological anchors and interpretation

- Human 13C-lactate kinetics found steady-state distribution volumes around
  42–45 L and healthy appearance near 13 micromol/kg/min, supporting a
  total-body-water-scale *apparent* pool, not an anatomical compartment
  ([Grip et al. 2020](https://doi.org/10.1186/s13054-020-2753-6)).
- Human exercise tracer studies show simultaneous lactate appearance,
  extraction and oxidation; blood concentration does not uniquely determine
  turnover ([Mazzeo et al. 1986](https://doi.org/10.1152/jappl.1986.60.1.232),
  [Stanley et al. 1986](https://doi.org/10.1152/jappl.1986.60.4.1116)).
- Exercise VCO2 contains oxidative and bicarbonate-store components. The latter
  is redistribution/elimination of existing carbon, not new metabolic carbon
  ([Zhang et al. 1994](https://doi.org/10.1007/BF00392036)).
- VO2/VCO2 cannot identify substrate oxidation during every non-steady state;
  lactate accumulation and gluconeogenesis require corrections
  ([Frayn 1983](https://doi.org/10.1152/jappl.1983.55.2.628)).

These sources constrain structure and order of magnitude. They do not validate
the model's joint trajectories or justify patient-specific inference.

## Executable evidence

`validation/run_validation_v22.py` writes
`validation/validation_results_v0.22.json` and contains 15 checks:

- t=0/scenario lactate-pool initialization;
- dilution without artificial lactate flux;
- exact lactate amount ledger without a second UMA effect;
- achieved-VO2 oxidative VCO2;
- dimensional/monotonic oxygen-deficit integration;
- final Haldane/carbon-pool closure across three FiO2 values;
- challenged reset-time carbon/O2/RER closure with zero ledgers;
- supply-limited final VCO2 and pulmonary-elimination endpoint consistency;
- steady and transient pulmonary-RER semantics;
- interval-average generation and trapezoidal elimination;
- outer-step convergence for CO2, lactate, deficit, PaCO2 and pH;
- exact ordered observation/action hashes and strict benchmark allowlist;
- rejection of invalid RQ/action configurations and explicit state-schema
  rejection of ambiguous legacy lactate payloads.

The five v0.22 regression files contain 34 collected cases. The complete suite
passes 180/180 with Gymnasium installed and warnings promoted to errors.
The validation host interpreter was Python 3.9.6, below the package's declared
Python >=3.10 support floor; a supported-interpreter CI run remains required
before treating this local result as release-platform verification.

### Post-audit supported-interpreter evidence

The operational CI gap recorded above was subsequently closed without
reinterpreting the original 180-test result. GitHub Actions run
[`32410309111`](https://github.com/lysyloxidase/openhumsim-rl/actions/runs/32410309111)
completed successfully for commit `6687c5431ae8640c4291fa400449017f470925b2`:

- full regression suite on Python 3.10, 3.12 and 3.14;
- the v0.22 scientific-integrity gate;
- isolated wheel build, installation and CLI smoke test.

The machine-readable record is `CI_EVIDENCE.json`. This evidence closes the
release-platform execution gap only; it does not add external clinical
validation or change any biological claim.

## Compatibility

- Carbon, PaCO2/pH, lactate and reward trajectories are not numerically
  comparable with v0.21 because the transition kernel changed.
- Observation widths remain clinical 54 / full 138, but unchanged shape does
  not imply checkpoint compatibility. Policies must be retrained/evaluated
  under a versioned protocol.
- Debug `HumanState` JSON gains amount, flux, RQ, deficit and final-closure
  fields. `environment_semantics.state_schema_version` is `0.22`; reward remains
  the explicitly labelled `homeostasis_v0.21` profile.
- Persisted physiology state must use the versioned envelope; v0.21 payloads
  are rejected with a migration explanation. The PPO example writes a v0.22
  checkpoint sidecar containing package/schema/reward/profile/config metadata
  and the exact ordered clinical-observation hash.
- `run_validation_v21.py` is now archival and refuses to run under v0.22.
  Historical JSON artifacts are not overwritten.

## Important unresolved limitations

1. **No protocol-matched external validation.** No independent clinical cohort
   validates coupled VO2/VCO2/lactate/acid-base trajectories.
2. **No full ATP/substrate/redox model.** Exercise and hypoxic lactate sources
   remain reduced empirical functions. Clearance is one lumped first-order
   disappearance pathway from the apparent pool; oxidation, gluconeogenesis,
   redistribution and storage are not separately identified. It must not be
   interpreted as a complete carbon or energy fate ledger.
3. **Apparent one-pool lactate distribution.** Fast central and slower tissue
   exchange are collapsed into a TBW-scaled volume. This is unsuitable for
   interpreting rapid boluses or tissue-specific lactate shuttling.
4. **No EPOC repayment.** Cumulative oxygen deficit is exposure only. A
   repayable store would need independently specified fast/slow recovery
   kinetics and validation.
5. **Residual operator splitting.** Lactate uses previous-substep oxygen deficit,
   and its exercise and hypoxic sources remain additive empirical terms rather
   than a stoichiometric ATP/O2 balance. Refinement tests bound numerical error;
   they do not validate this biological parameterization.
6. **Acid-base scope.** Hemoglobin-bound proton change remains diagnostic and is
   not part of the plasma pH charge root. Whole-blood noncarbonate buffering
   needs a re-derived coupled formulation.
7. **Disease scope.** Sepsis, hepatic failure, seizures, catecholamine-driven
   glycolysis, toxins, thiamine deficiency and mitochondrial disease cannot be
   inferred from this generic lactate source/clearance model.
8. **RL validity.** The clinical interface remains a POMDP; full debug
   observation is not a complete Markov state. The prior/calibration/UQ layers
   still do not establish identifiability or patient-specific posterior validity.

The limitations are retained as explicit model boundaries rather than hidden by
relabeling internal consistency checks as medical validation.
