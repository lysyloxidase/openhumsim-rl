# OpenHumSim-RL model card

## Model details

| Field | Value |
| --- | --- |
| Project | OpenHumSim-RL |
| Package version | `0.23.0` release |
| State schema | `0.22` |
| Default debug reward | `latent_research_v0.23` |
| Strict benchmark reward | `observable_benchmark_v0.23` |
| Author and maintainer | [lysyloxidase](https://github.com/lysyloxidase) |
| License | [Apache-2.0](../LICENSE) |
| Runtime | Python 3.10 or newer |
| Status | Alpha research software |

OpenHumSim-RL is a deterministic-step, stochastic-observation research
simulator that exposes a reduced-order model of coupled human physiology through
a Gymnasium-compatible environment. The current release focuses on explicit
units, conserved pools, numerical closure, measurement provenance and versioned
RL contracts.

This card describes the software's intended use, evidence and limits. It is not
a claim of clinical validity.

## Intended use

OpenHumSim-RL is intended for:

- research on reinforcement-learning methods in a synthetic physiological
  environment;
- experiments with partial observability, delayed/noisy measurements and
  hidden mechanistic state;
- numerical and mechanistic testing of reduced-order physiological couplings;
- reproducible scenario, ablation, uncertainty and controller comparisons;
- software education and method development using synthetic trajectories.

Results should be reported with the exact package version or commit, state and
reward schemas, observation and measurement profiles, configuration, scenario,
seed and action contract.

## Out-of-scope use

The project does not support:

- diagnosis, screening, prognosis or clinical risk scoring;
- drug, insulin, fluid, oxygen or ventilator dosing recommendations;
- alarms, bedside monitoring or control of medical equipment;
- patient-specific inference, treatment selection or outcome prediction;
- substitution for a validated physiological model, medical device or clinical
  trial;
- claims that an RL policy is safe or effective in humans.

The simulator must not receive patient-identifiable data. The repository does
not bundle individual-level clinical records.

## System overview

The environment couples reduced representations of:

- meal glucose, insulin, glucagon, glycogen and exercise metabolism;
- a closed-loop zero-dimensional cardiovascular circulation;
- respiratory control, within-breath mechanics and patient-ventilator
  interaction;
- multi-compartment pulmonary ventilation/perfusion, shunt, diffusion and gas
  exchange;
- oxygen transport, whole-blood carbon dioxide and acid-base chemistry;
- renal filtration, water balance, electrolytes and osmotic water shifts;
- oxidative energy use, oxygen deficit and an apparent lactate pool;
- a generic oral probe-compound PBPK pathway;
- a configurable clinical-like measurement layer.

The default decision interval is five simulated minutes and the default outer
integration interval is 0.25 minute. Faster internal solvers are used where the
mechanics require them. Both public time steps are configurable, so convergence
must be reassessed when they change materially.

The modules are mechanistic scaffolds with empirical control laws and bounded
engineering parameters. They do not form an anatomically complete human model.

## Inputs and actions

An experiment is defined by a `HumanConfig`, a named scenario, observation,
measurement and information profiles, a reset seed, and a sequence of bounded
actions. The eight normalized action channels are:

1. insulin;
2. oral carbohydrate;
3. exercise intensity;
4. saline;
5. inspired-oxygen control;
6. ventilation pressure assistance;
7. oral water;
8. an oral PBPK probe compound.

Actions are one-directional research controls mapped to configured model
limits. In particular, insulin values are OpenHumSim model units, not clinical
insulin units. Action names must not be interpreted as treatment advice.

## Observations and policy information

The `clinical` observation profile contains 54 normalized values. With the
default `realistic` measurement profile, its channels can include explicit
sampling intervals, analytical delay, noise, dropout and measurement age. The
normalization centers and scales are engineering values, not diagnostic
reference intervals.

The `full` profile contains 138 normalized mechanistic values and requires ideal
measurement mode. It exists for debugging and ablation; it is not claimed to be
a complete Markov state.

Observations are transformed to the bounded interval `[-1, 1]`. Raw units and
measurement provenance are available through the debug interface and dashboard
exports.

The default `info_profile="debug"` exposes hidden state and reward diagnostics.
It can create oracle leakage if passed to a policy. Benchmarks should explicitly
select `info_profile="benchmark"`, whose output is controlled by an allowlist.
Debug runs default to `latent_research_v0.23`; benchmark mode selects and
requires `observable_benchmark_v0.23`. Its state-dependent terms are computed
from public observations; action cost, elapsed duration and the public terminal
event also contribute to the complete transition reward.

The clinical-like interface remains a partially observable Markov decision
process. Observation width alone does not establish policy or checkpoint
compatibility.

## Parameter and data provenance

Model structure and selected scales are anchored to published physiology and
tracer literature documented in the versioned validation reports and external
reference manifests. These sources constrain mechanism, direction and order of
magnitude; they do not validate the joint simulated trajectory.

The correlated virtual-subject layer is an engineering prior. It is not an
identified population distribution or a patient posterior. Optional workflows
can process separately obtained external data under their original access and
redistribution terms, but those data are not distributed with the package.

Synthetic tests and internally generated scenarios provide most executable
verification evidence. Published aggregate checks are not a substitute for an
independent, protocol-matched cohort.

## Evaluation evidence

The v0.23 release adds a focused gate for fail-closed solver and
configuration checks, respiratory-mechanics continuity, total-PEEP plateau
identity, temporal/reward/measurement contracts, checkpoint manifests,
environment snapshots and observation-history contracts. The machine-readable
result is created only by actually running that gate and records a fingerprint
of the source files used by the checks.

Release evidence and its explicit limitations are available in:

- [`validation/validation_results_v0.23.json`](../validation/validation_results_v0.23.json);
- [`VALIDATION_AUDIT_v0.23.md`](../VALIDATION_AUDIT_v0.23.md);
- [`RELEASE_v0.23.json`](../RELEASE_v0.23.json).

The release manifest distinguishes the focused gate, the locally completed
full repository suite and the successful supported-interpreter CI matrix.

This evidence supports software verification and internal mechanistic
consistency only. No independently reviewed, protocol-matched external clinical
validation is bundled.

## Known limitations

### Biological

- There is no complete ATP, substrate-fate, redox or mitochondrial model.
- Lactate uses one apparent total-body-water-scale pool; rapid tissue exchange,
  oxidation, gluconeogenesis and storage are not separately identified.
- Cumulative oxygen deficit is an exposure integral, not a repayable EPOC
  compartment.
- Disease-specific pathways for sepsis, hepatic failure, seizures, toxins,
  thiamine deficiency and mitochondrial disease are absent.
- The generic PBPK probe is not a calibrated representation of a therapeutic
  drug.

### Physical and numerical

- Cardiovascular, organ and gas-exchange geometry is reduced-order rather than
  spatially resolved.
- Residual operator splitting remains in selected pathways; refinement tests
  bound observed numerical differences but do not remove model-form error.
- Ventilator, airway, chest-wall and recruitment mechanics are simplified and
  are not patient-specific device models.
- Hemoglobin-bound proton change remains diagnostic rather than part of a fully
  re-derived whole-blood charge root.

### Medical

- No independent cohort validates coupled oxygen, carbon dioxide, lactate,
  acid-base, metabolic and hemodynamic trajectories.
- Parameters are not calibrated to an individual patient or a representative
  clinical population.
- Scenario labels describe model perturbations, not disease diagnoses.
- Displayed ranges, extrema and summaries are not clinical limits or alarms.

### Algorithmic and statistical

- The clinical-like interface is a POMDP; the debug vector is not guaranteed to
  be Markov.
- Reward design can omit real-world harms and may be exploited by a policy.
- The virtual-subject prior does not establish parameter identifiability,
  calibrated uncertainty or posterior validity.
- Historical policies and checkpoints are incompatible when transition,
  observation, action, schema or reward semantics differ, even if array shapes
  happen to match.
- The bundled no-op and observation-history heuristic are transparent software
  baselines, not trained controllers or evidence of policy quality.
- Reset physiology jitter and realistic measurement uncertainty use separate
  child generators spawned from the reset seed. This removes draw-order
  coupling but does not make one seed a substitute for multi-seed evaluation.

## Foreseeable misuse and mitigations

The most important misuse risk is treating internally consistent output as
medical evidence. The repository mitigates this with explicit research-only
warnings, measurement-versus-model-truth labels, versioned interfaces,
allowlisted benchmark metadata and machine-readable provenance. These controls
do not make the simulator suitable for clinical use.

RL agents can exploit numerical boundaries, reward omissions, hidden-state
leakage or scenario-specific artifacts. Evaluation should therefore include
multiple seeds, held-out configurations, stress scenarios, baseline and
heuristic comparators, action and state traces, and independent safety criteria
outside the training reward.

## Reproducibility

For each reported run, preserve:

- release version and commit SHA;
- state schema, reward profile, ordered observation/action hashes and counts;
- resolved human and measurement configurations;
- scenario, seed, decision and integration time steps;
- observation, measurement and information profiles;
- complete action history and termination status;
- runtime and dependency versions;
- exported experiment manifest and source fingerprint.

Dashboard-backed runs emit `openhumsim.experiment-manifest.v1`. See
[dashboard.md](dashboard.md) for its contents and random-stream semantics.
Long-running programmatic rollouts can also use the environment's versioned
JSON snapshot API. A history-wrapper runtime snapshot is accepted only with the
matching base-environment state, preventing histories from different runs from
being combined silently.

## Citation and license

Use [`CITATION.cff`](../CITATION.cff) and cite the exact release or commit used.
OpenHumSim-RL is authored and maintained by
[lysyloxidase](https://github.com/lysyloxidase) and distributed under the
[Apache License 2.0](../LICENSE).
