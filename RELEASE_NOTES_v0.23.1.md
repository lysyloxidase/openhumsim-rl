# OpenHumSim-RL v0.23.1

> Research software only. OpenHumSim-RL is not clinically validated and must
> not be used for diagnosis, treatment selection, dosing, alarms, ventilator
> settings or patient-specific prediction.

## Release scope

Version 0.23.1 is a patch release of the v0.23 model and interface family. It
preserves state schema `0.22`, the 54/138 observation contracts, the eight
actions and the `latent_research_v0.23` / `observable_benchmark_v0.23` reward
profiles.

The pulmonary solver now respects every FiO2 value accepted by `HumanConfig`,
including hypoxic challenges below 0.15. Regional carbonate chemistry uses the
configured carbonic-acid pKa and CO2 solubility, and the six-compartment lung
rejects a mismatched number of recruitment thresholds during configuration.

Environment restoration now rejects impossible ABG/chemistry panel schedules
and event counters atomically. `reset()` no longer exposes an already-terminal
initial state as an active episode. The dashboard's decorative latent-state
overlay has been removed while the explicit research-only and synthetic-data
safety labels remain.

## Compatibility

The persisted state and tensor shapes are unchanged. Runtime validation is
stricter, pulmonary behavior changes for non-default carbonate constants and
FiO2 below 0.15, and the training example uses the distinct checkpoint basename
`openhumsim_ppo_v0231_smoke`. Policies and snapshots should be paired with the
exact package version and manifest used to create them.

## Verification

The focused gate passed 11/11 checks with 103 executed regression cases on
CPython 3.12.13. The complete local suite passed 315/315 tests with warnings as
errors. Wheel and source distributions passed metadata checks; the wheel passed
an isolated installation, CLI diagnostics, baseline simulation and the
checkout-only validation guard. GitHub Actions passed the full suite on Python
3.10, 3.12 and 3.14, the scientific gate and package smoke job for source
commit `dac4e3e`; CodeQL passed for the same commit.

See `VALIDATION_AUDIT_v0.23.1.md` and `RELEASE_v0.23.1.json` for the exact
evidence and remaining limitations.

OpenHumSim-RL is distributed under the Apache License 2.0. Copyright 2026
lysyloxidase.
