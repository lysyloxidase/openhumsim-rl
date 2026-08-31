# OpenHumSim-RL v0.23.2 validation audit — candidate

## Decision

**PENDING / NOT YET RELEASE EVIDENCE.**

The v0.23.2 focused gate, complete test suite, package build and supported
interpreter CI have not yet been recorded for the final clean candidate. This
document describes what must be checked and how the implemented changes should
be interpreted. It does not claim a passing release, a clinically validated
model or an independently reproduced result.

## Intended contract

- Package release: `0.23.2`.
- Persisted state schema: `0.22` (intended unchanged).
- Debug reward: `latent_research_v0.23` (intended unchanged).
- Benchmark reward: `observable_benchmark_v0.23` (intended unchanged).
- Clinical/full observation widths: 54/138 (intended unchanged).
- Action width: 8 (intended unchanged).
- Historical policy compatibility: not inferred from equal tensor shapes.

Every intended-unchanged item above must be confirmed by the focused gate and
final release manifest. It is not a substitute for executing those checks.

## Focused evidence protocol

`validation/run_validation_v0232.py` is a new patch-specific runner. It refuses
to run against a package version other than `0.23.2` and writes
`validation/validation_results_v0.23.2.json` only after every configured check
has executed. The v0.23.1 runner and result remain frozen historical evidence.

The v0.23.2 gate carries forward the existing checks for:

1. exact package, state-schema and reward-profile contracts;
2. v0.23.1 pulmonary/configuration patch regressions;
3. solver and configuration hardening;
4. respiratory-mechanics continuity and total-PEEP semantics;
5. temporal reward, measurement and RNG behavior;
6. transactional environment steps;
7. policy manifests and checkpoint compatibility;
8. atomic environment snapshots;
9. observation history and deterministic baseline harnesses.

It additionally exercises:

1. positive-pressure source separation and explicit pressure-source behavior;
2. side-effect-free result-only pulmonary evaluation and single-step HPV
   application;
3. current CLI validation and duration contracts;
4. policy-facing observation/preprocessing manifests;
5. policy-facing native and symmetric action mappings;
6. packaged dashboard HTML, compatibility entry point, transactional
   session/HTTP mutation contracts and anti-framing headers.

The source fingerprint covers all Python files below `src/openhumsim_rl`, every
test file selected by the gate, the runner itself, the training example, the
dashboard compatibility example, the packaged and source-checkout dashboard
HTML files, the v0.23 benchmark harness and `pyproject.toml`. It records the Git
commit and dirty-worktree status and verifies that the fingerprint remains
stable while the gate runs. The release-evidence verifier and its tests, CI and
tag workflows, README/dashboard guide, license, notice and citation metadata
are also bound so that an evidence-only commit cannot silently change package
or publication semantics after the clean candidate was tested.

Tests that validate the completed `RELEASE_v0.23.2.json` against the generated
focused result cannot be inputs to the process that creates that result. They
remain mandatory in the complete repository suite after the candidate manifest
has been populated with the actual result hash and counts.

## Biological and physical interpretation

The positive-pressure fraction represents how an existing airway-pressure
source is transmitted through respiratory mechanics. Treating that fraction as
a source previously allowed a PEEP/CPAP-like state flag to add inspiratory
airway pressure and ventilator work while spontaneous muscle effort remained
present. Separating source amplitude from transmission removes that artificial
double drive while retaining explicit pressure control and assistance.

Result-only pulmonary calculations now evolve recruitment and hypoxic vascular
tone on an isolated working state. This permits a hypothetical end-of-interval
estimate without changing the caller's conserved or diagnostic state. Applied
calculations still commit temporal lung state once.

These are internal consistency improvements. The respiratory cycle remains a
reduced single-compartment model, regional exchange uses six idealized units,
and neither subsystem is calibrated to a particular ventilator, lung morphology
or patient.

## Algorithmic and numerical interpretation

A public transition is now atomic for unexpected exceptions across integration
and output construction. Measurement timing restoration binds delivered-result
counters to whether the latest sample is the initialization sample. Policy
manifests describe the values actually supplied through observation-history
wrappers, including padding and validity masks, and the exact policy-to-native
action transformation. The symmetric interface's negative half-space is
explicitly recorded as mapping to zero rather than being treated as a native
negative intervention.

These controls improve retry safety, deterministic restoration and checkpoint
provenance. They do not prove convergence for every custom timestep, eliminate
all operator splitting, establish that the debug vector is Markov or establish
policy performance.

## Product and package interpretation

The local dashboard is now a package resource with a CLI entry point. Its
server keeps measured observations separate from labelled latent diagnostics,
uses session/revision/idempotency guards for mutations and rejects framing. A
failed post-transition frame build restores the environment, revision, history
and request cache as one transaction. A static source-checkout file is only an
offline preview.

The dashboard is research software, not a clinical monitor. Colors, extrema,
scenario names and summaries are model diagnostics, not alarms, reference
intervals, diagnoses or treatment advice. Repository validation and CI metadata
must be reported as unavailable when the matching evidence files are absent.

## Medical interpretation

No protocol-matched cohort, prospective controller study, alarm evaluation,
patient-level calibration or treatment-effect validation is bundled. The
changes must not be interpreted as evidence for ventilator settings, oxygen
prescription, drug or fluid dosing, diagnosis, prognosis or bedside control.

## Remaining limitations

1. The simulator is reduced-order and retains empirical control laws, bounded
   engineering parameters and operator splitting.
2. Energy, lung, cardiovascular, renal and PBPK subsystems are not jointly
   calibrated to an individual or representative clinical population.
3. The clinical interface remains partially observable; the debug vector is
   not claimed to be a complete Markov state.
4. Reward functions can omit real-world harms and can be exploited by a policy.
5. Virtual-subject variability is an engineering prior, not an identified
   population distribution or patient posterior.
6. Internal regression tests and synthetic scenarios are not independent
   external validation.

## Evidence still required

- Focused gate execution, top-level pass/total and executed case count: **TODO**.
- Exact clean Git commit and source fingerprint: **TODO**.
- SHA-256 of `validation/validation_results_v0.23.2.json`: **TODO**.
- Full repository suite count and runtime with warnings as errors: **TODO**.
- Wheel and source-distribution names, hashes, metadata and isolated install:
  **TODO**.
- Supported Python matrix, package smoke job and CodeQL run: **TODO**.
- Final `RELEASE_v0.23.2.json` tied to the same source and evidence: **TODO**.

Until those fields are replaced with actual results, there is no authoritative
v0.23.2 release record.
