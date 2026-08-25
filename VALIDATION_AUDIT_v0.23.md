# OpenHumSim-RL v0.23 validation audit

## Decision

The focused v0.23 integrity gate has been executed and passed 10/10 top-level
checks on CPython 3.12.13. This supports the new solver/configuration,
respiratory-mechanics, temporal, reward, measurement and checkpoint-manifest
contracts, transactional step semantics, versioned environment snapshots and
strict observation history.
The full local repository suite also passed 309/309 tests with warnings as
errors. A wheel and source distribution passed `twine check`; the wheel then
passed an isolated core installation, CLI diagnostics and baseline simulation.
The supported-interpreter CI matrix and isolated package job also passed for
the exact clean source commit recorded by the focused gate.

This is internal software verification and mechanistic consistency evidence,
not independent external clinical validation.

## Contract and compatibility

- Package release: `0.23.0`.
- Persisted state schema: `0.22` (unchanged).
- Default debug reward: `latent_research_v0.23`.
- Strict benchmark reward: `observable_benchmark_v0.23`.
- Clinical/full observation widths: 54/138; action width: 8.
- Historical RL checkpoints: incompatible because reward and transition
  semantics changed, irrespective of matching array shapes.

The benchmark environment accepts only the observable reward profile. Debug
runs default to the latent research profile and must not be used as
hidden-state-free policy evidence.

## Executed focused evidence

`validation/run_validation_v23.py` wrote
`validation/validation_results_v0.23.json` only after all checks ran. The file
records execution at `2026-08-25T14:40:14.156704+00:00` on CPython 3.12.13 and
has SHA-256
`21e2bc8f312b2201ba1cc18beca08052442709713bec2b2d8a5ced004267ee6e`.
It identifies 42 relevant source/test files with source fingerprint
`f7b08aea1dca41c3f471192ac29af262f87604f8d0432a7d0486f69ca2793b87`
and verifies that fingerprint did not change while the gate ran.

The ten passing top-level checks were:

1. exact package version `0.23.0`;
2. unchanged state schema and explicit debug/benchmark reward profiles;
3. 26 solver and configuration regression cases;
4. two pressure-assist continuity and total-PEEP plateau cases;
5. nine temporal, reward, measurement, RNG and rollback cases;
6. four transactional-step failure and rollback cases;
7. eight checkpoint-manifest, provenance, normalization and snapshot cases;
8. 34 environment snapshot, state-integrity and deterministic-resume cases;
9. 14 strict observation-history and transparent baseline-harness cases;
10. stability of the complete source fingerprint throughout execution.

The gate promotes warnings to errors for its seven pytest groups. Separately, the
full repository suite passed 309/309 tests locally on CPython 3.12.13 with
warnings as errors. The focused gate does not build the distribution, install
the wheel outside the source tree or exercise the supported Python matrix.

## Interpretation of the changes

### Biological and physical

The pressure-assist pathway no longer switches the cardiovascular effect on at
an infinitesimal positive action. Its continuous pressure fraction removes a
nonphysical zero-to-epsilon jump. Passive plateau pressure is now the sum of
total PEEP and driving pressure, so intrinsic PEEP is not omitted from the
reported static load.

These corrections improve local physical consistency; they do not turn the
reduced respiratory, chest-wall, cardiovascular or gas-exchange modules into a
patient-specific ventilator model.

### Medical

The state-dependent benchmark objective is constructed from public
observations, whereas the debug objective may use latent mechanistic state.
The complete benchmark transition reward also includes the agent's action
cost, elapsed duration and public terminal event. This closes one route by
which a benchmark policy could optimize information unavailable at deployment;
it does not establish that either reward captures clinical benefit or harm.

No protocol-matched cohort, medical-device study or prospective controller
evaluation is bundled. Scenario names denote synthetic model perturbations,
not diagnoses.

### Algorithmic and numerical

Invalid non-finite configurations and structurally inconsistent bounds are
rejected before integration. Acid-base snapshot solutions must satisfy a
bracketed charge root and the configured tolerance. First-substep failures roll
back both intervention cost and transition state, with one terminal event.

Checkpoint loading is fail closed against artifact, environment, reward,
measurement, normalization, source and runtime mismatches. It verifies and
loads the same private snapshot, avoiding a verify-then-reopen race on the
original path. These checks establish provenance consistency, not policy
quality, robustness or safety.

Environment continuation snapshots validate exact state fields, conserved-pool
identities, private solver and measurement runtime, all random streams and the
active horizon. Observation histories accept only the clinical/realistic/
benchmark profile and are bound to the exact base-environment snapshot. The
included evaluation harness compares only transparent no-op and
observation-history rules; it does not train or validate a controller.

## Important unresolved limitations

1. The coupled simulator remains reduced-order and contains empirical control
   laws, bounded engineering parameters and residual operator splitting.
2. Lactate, substrate fate, ATP/redox and oxygen-deficit dynamics are not a
   complete tissue-resolved energy model.
3. Respiratory mechanics, recruitment, airway/chest-wall behavior and
   patient-ventilator interaction are simplified and not device calibrated.
4. The clinical interface is a POMDP; the 138-value debug vector is not claimed
   to be a complete Markov state.
5. Reward functions can omit real-world harms and can still be exploited by an
   agent despite the observability separation.
6. Virtual-subject variability is an engineering prior, not an identified
   clinical population or patient posterior.

## Release completion evidence

- GitHub Actions run `32860834543` passed the full suite on Python 3.10, 3.12
  and 3.14, the focused scientific gate, and the package build/smoke test for
  exact source commit `080e8b82ea050541f13324917d3920b5a8b1807d`.
- The outcome is recorded separately from the historical v0.22 evidence in
  `CI_EVIDENCE.json` and `RELEASE_v0.23.json`.

`RELEASE_v0.23.json` is therefore the final v0.23 software release record. The
scientific and clinical limitations above remain in force.
