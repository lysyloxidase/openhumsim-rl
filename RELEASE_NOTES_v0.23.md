# OpenHumSim-RL v0.23.0

> Research software only. OpenHumSim-RL is not clinically validated and must
> not be used for diagnosis, treatment selection, dosing, alarms, ventilator
> settings or patient-specific prediction.

## Highlights

- Solver and configuration boundaries fail closed for non-finite values,
  inconsistent ranges and unbracketed acid-base roots.
- Pressure-assist coupling is continuous at zero, and passive plateau pressure
  uses total PEEP (external PEEP plus auto-PEEP).
- Debug and benchmark runs use explicit reward contracts:
  `latent_research_v0.23` and `observable_benchmark_v0.23`, respectively.
- Continuous intervention costs are normalized to elapsed model time, sensor
  cadence is independent of agent cadence, and panel dropout is coherent.
- Physiology jitter and measurement uncertainty use separate reproducible child
  random streams.
- PPO checkpoints carry a fail-closed sidecar manifest with the actual scenario,
  profiles, measurement configuration, normalization, source/runtime
  provenance and checkpoint hash. Loading verifies and uses one private
  checkpoint snapshot.
- Versioned JSON environment snapshots preserve the physiological state,
  private solver and measurement runtime, and all random streams atomically.
- The strict clinical/realistic benchmark wrapper provides masked observation
  history, and a deterministic harness compares transparent no-op and
  observation-only rules without claiming learned-policy performance.

## Compatibility

The package version is `0.23.0`; the persisted state schema deliberately
remains `0.22`. The clinical and full observation contracts remain 54 and 138
ordered values, and the action contract remains eight ordered controls.

Reward semantics and the transition kernel changed. Earlier RL checkpoints are
not compatible even when their tensor shapes match. The PPO example writes
`openhumsim_ppo_v023_smoke` and requires the matching manifest.

## Verification status

The focused v0.23 release gate was executed locally on Python 3.12.13 and
passed 10/10 top-level checks, including 97 executed regression cases, exact
version/reward/schema checks and a stable source fingerprint. The generated record is
`validation/validation_results_v0.23.json`.

The full local repository suite passed 309/309 tests on Python 3.12.13 with
warnings treated as errors. GitHub Actions then passed the full suite on Python
3.10, 3.12 and 3.14, the focused scientific gate, and the isolated package
build/smoke test for source commit `080e8b8`. The wheel and source distribution
also passed metadata checks and an isolated wheel-install/CLI smoke test. These
passing checks are internal software and mechanistic consistency evidence; they
are not external clinical validation.

See `VALIDATION_AUDIT_v0.23.md` and `RELEASE_v0.23.json` for the verified scope
and remaining scientific limitations.

OpenHumSim-RL is distributed under the Apache License 2.0. Copyright 2026
lysyloxidase.
