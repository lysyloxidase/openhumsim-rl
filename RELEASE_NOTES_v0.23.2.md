# OpenHumSim-RL v0.23.2

> Research software only. OpenHumSim-RL is not clinically validated and must
> not be used for diagnosis, treatment selection, dosing, alarms, ventilator
> settings, patient-specific prediction or control of medical equipment.

## Release scope

Version 0.23.2 is a patch release in the v0.23 model and interface family. It
is intended to preserve state schema `0.22`, the 54/138 observation widths, the
eight actions and the `latent_research_v0.23` /
`observable_benchmark_v0.23` reward profiles. Those identifiers describe
interface families; they do not imply that a policy trained against an earlier
transition kernel remains compatible.

### Respiratory and pulmonary behavior

- `pulmonary_positive_pressure_fraction` now controls transmission of an
  existing positive airway-pressure source; it no longer creates an
  inspiratory pressure-control breath by itself. Explicit pressure control,
  action-driven pressure assistance and pressure-support mode remain explicit
  airway-pressure sources.
- A result-only pulmonary oxygen calculation (`apply=False`) evaluates
  recruitment and hypoxic-pulmonary-vasoconstriction kinetics on an isolated
  working state. It no longer advances lung state or overwrites diagnostics,
  including when a positive `dt_min` is supplied.

These changes remove two hidden state/energy effects. They do not turn the
single-compartment respiratory mechanics or six-unit gas-exchange model into a
validated ventilator, patient-specific lung model or clinical recommendation
system.

### Transaction and policy contracts

- Unexpected exceptions during a policy decision, including late integration
  and observation/info construction failures, roll back the complete
  transition so callers do not receive a partially committed step. Controlled
  numerical-failure termination retains its explicit fail-closed pathway.
- Measurement-runtime restoration rejects a fabricated nonzero sample time
  when only the initialization sample has been delivered, and rejects the
  converse inconsistency.
- The environment and observation-history wrapper expose a public,
  policy-facing preprocessing contract. Policy manifests can lock the exact
  masked-history layout, validity mask, base normalization and wrapper length
  instead of relying on private normalization arrays.
- Policy manifests also lock the policy-to-native action mapping. The optional
  symmetric action interface is recorded as componentwise
  `native=max(policy_action,0)` and cannot be confused with the native
  one-sided identity interface solely because tensor shapes match.

### Packaged dashboard and command line

- The dashboard bridge and canonical HTML resource are packaged with the
  library and can be started with `openhumsim dashboard` from an installed
  environment. The former example entry point remains a compatibility shim.
- Release and validation evidence is used only from a matching source checkout;
  an installed wheel reports repository-only evidence as unavailable rather
  than borrowing unrelated files from the current directory.
- Dashboard responses deny framing, and the CLI validates durations, counts,
  seeds and ports as usage errors before starting a simulation.
- Dashboard `step` and `reset` operations roll back the environment and session
  history together if frame/export construction fails after a simulation
  transition, so a retry cannot silently duplicate a bolus or skip a revision.
- CI and release packaging smoke tests are expected to verify the installed
  dashboard command, module entry point and bundled HTML resource.

## Compatibility

Persisted state and public tensor widths are intended to remain unchanged.
Runtime validation, transition rollback and respiratory behavior are stricter,
and history-wrapped policy manifests carry a more complete preprocessing
contract. Policies, checkpoints, snapshots and exported experiments must be
paired with the exact package version and manifest used to create them. A
matching shape alone is not compatibility evidence.

## Verification status

- Focused v0.23.2 integrity gate: **14/14 checks passed**, covering 178
  executed regression cases on clean candidate commit
  `98139c0ac432c3552d03afdfd003bdca4b1b42d8`.
- Machine-readable focused result:
  `validation/validation_results_v0.23.2.json`, SHA-256
  `89024e8a16fcb622b77c65d65f38849fdf16a0a49327975de490a03d4df333d3`.
- Complete repository suite: **370/370 passed** with warnings treated as
  errors on CPython 3.12.13.
- Wheel/source build, Twine metadata check, isolated wheel install, CLI and
  packaged-dashboard smoke checks: **passed**.
- GitHub Actions: Python **3.10, 3.12 and 3.14**, scientific gate and package
  smoke all passed in [run 33369298624](https://github.com/lysyloxidase/openhumsim-rl/actions/runs/33369298624).
- CodeQL passed in [run 33369298701](https://github.com/lysyloxidase/openhumsim-rl/actions/runs/33369298701).
- `RELEASE_v0.23.2.json` is the authoritative machine-readable release record.

Tagged release artifacts include a workflow-generated `SHA256SUMS` file. See
`VALIDATION_AUDIT_v0.23.2.md` for the evidence boundary and remaining
limitations.

OpenHumSim-RL is distributed under the Apache License 2.0. Copyright 2026
lysyloxidase.
