# OpenHumSim-RL v0.23.2 validation audit

## Decision

**PASS for the documented internal release gates.**

The focused v0.23.2 gate, complete repository suite, package build and
supported-interpreter CI passed for clean candidate commit
`0677efbc969df8f231f7206b00c6263bd44a6a46`. This is internal verification of
the stated software and mechanistic contracts. It is not clinical validation,
independent reproduction or evidence of treatment safety or efficacy.

## Intended contract

- Package release: `0.23.2`.
- Persisted state schema: `0.22` (confirmed unchanged).
- Debug reward: `latent_research_v0.23` (confirmed unchanged).
- Benchmark reward: `observable_benchmark_v0.23` (confirmed unchanged).
- Clinical/full observation widths: 54/138 (confirmed unchanged).
- Action width: 8 (confirmed unchanged).
- Historical policy compatibility: not inferred from equal tensor shapes.

The exact ordered observation and action hashes are recorded in
`RELEASE_v0.23.2.json`. Equal tensor shapes alone remain insufficient evidence
of policy compatibility.

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

## Recorded evidence

- Focused gate: 14/14 top-level checks and 178 executed regression cases;
  CPython 3.12.13; executed at `2026-08-31T20:37:58.502155+00:00`.
- Candidate commit: `0677efbc969df8f231f7206b00c6263bd44a6a46` with a clean worktree.
- Source fingerprint: 64 files, SHA-256
  `c4c1aa1392d44e100f99163c677f57d180efa955de129dd50474f9ebcba10efe`.
- Focused result: `validation/validation_results_v0.23.2.json`, SHA-256
  `a55ff6563faa711f11ef516e7bc2fa5430df3daa0f6d791deb7023b73a28e855`.
- Complete repository suite: 370/370 passed with warnings as errors in
  788.40 seconds on CPython 3.12.13.
- Local distributions: `openhumsim_rl-0.23.2-py3-none-any.whl` and
  `openhumsim_rl-0.23.2.tar.gz`; build, Twine metadata, isolated installation,
  CLI, dashboard resource and checkout-only validation guard passed. Their
  local SHA-256 values are recorded in `RELEASE_v0.23.2.json`; tagged workflow
  artifacts carry their own authoritative `SHA256SUMS`.
- GitHub Actions [run 33438421715](https://github.com/lysyloxidase/openhumsim-rl/actions/runs/33438421715):
  Python 3.10, 3.12 and 3.14 test jobs, scientific gate and package smoke all
  passed for the candidate commit.
- CodeQL [run 33438421707](https://github.com/lysyloxidase/openhumsim-rl/actions/runs/33438421707): passed for the same candidate commit.
- `RELEASE_v0.23.2.json` binds these results to the release contract.
