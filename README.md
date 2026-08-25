# OpenHumSim-RL

[![CI](https://github.com/lysyloxidase/openhumsim-rl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lysyloxidase/openhumsim-rl/actions/workflows/ci.yml)
[![CodeQL](https://github.com/lysyloxidase/openhumsim-rl/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/lysyloxidase/openhumsim-rl/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-4c1.svg)](LICENSE)

OpenHumSim-RL is a reduced-order, multi-organ physiology simulator for
reproducible mechanistic and reinforcement-learning research. It couples
metabolic, cardiovascular, respiratory, renal, fluid, electrolyte, acid-base,
blood-gas and PBPK dynamics behind a Gymnasium-compatible interface.

> **Research software only.** OpenHumSim-RL is not clinically validated. Do not
> use it for diagnosis, treatment selection, dosing, alarms, ventilator
> settings, patient-specific prediction or control of medical equipment.

**Author:** [lysyloxidase](https://github.com/lysyloxidase)

**Current release candidate:** 0.23.0

## Capabilities

- Coupled whole-body physiology with explicit units, conserved pools and
  diagnostic mass-balance residuals.
- A 54-channel measurement-aware observation profile with sampling intervals,
  analytical delay, noise, dropout and measurement age.
- A 138-value mechanistic debug profile for model inspection and ablation work.
- Eight intervention channels covering insulin, carbohydrate, exercise, saline,
  oxygen, pressure assistance, water and an oral PBPK probe compound.
- Versioned state, observation, action, reward and experiment-manifest
  contracts for reproducible comparisons.
- JSON-safe, fail-closed environment snapshots that preserve physiological,
  solver, measurement and random-generator runtime for exact continuation.
- A masked observation-history wrapper and deterministic transparent-policy
  harness for partial-observability experiments.
- Built-in research scenarios, command-line tools, validation artifacts and a
  local interactive dashboard.

The simulator is intentionally reduced-order. Its variables and scenarios are
model quantities, not clinical reference ranges, recommendations or patient
estimates. See the [model card](docs/MODEL_CARD.md) before interpreting results.

## Installation

OpenHumSim-RL requires Python 3.10 or newer. Install the current source version
in an isolated environment:

```bash
git clone https://github.com/lysyloxidase/openhumsim-rl.git
cd openhumsim-rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gym]"
openhumsim doctor
```

For tests and repository validation, install `.[dev]`. Optional extras `rl`,
`uq`, `torch-rl` and `local` add the corresponding research dependencies.

## Quick start

Use the explicit benchmark profile when an agent must not receive hidden model
state through `info`:

```python
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv

env = HumanHomeostasisEnv(
    observation_profile="clinical",
    measurement_profile="realistic",
    info_profile="benchmark",
)

observation, info = env.reset(seed=42)
action = np.zeros(env.action_space.shape, dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(action)
```

The default `info_profile="debug"` is useful for mechanistic inspection but
contains hidden state and must not be exposed to benchmark policies.
Selecting `info_profile="benchmark"` also selects
`observable_benchmark_v0.23`. Its state-dependent terms use the public
observation; the complete transition reward additionally includes the agent's
action cost, elapsed duration and public termination event.

The CLI provides a deterministic smoke simulation:

```bash
openhumsim demo --scenario baseline --minutes 60 --seed 42
```

Run `openhumsim --help` to inspect the available simulation, measurement,
population and external-data commands.

## Interactive dashboard

Start the local Python bridge from the repository root:

```bash
PYTHONPATH=src python3 examples/dashboard_server.py
```

The server opens a browser on the loopback interface and connects the HTML
workspace to the real Python simulator. It separates measured observations from
explicitly labelled latent model state and exports reproducibility metadata.
See [Dashboard usage and data provenance](docs/dashboard.md) for session,
security, export and random-stream details.

Opening `dashboard/index.html` directly is supported only as an offline UI
preview; simulation steps require the Python bridge.

## Interface contract

| Contract | v0.23 candidate value |
| --- | --- |
| State schema | `0.22` |
| Default debug reward profile | `latent_research_v0.23` |
| Strict benchmark reward profile | `observable_benchmark_v0.23` |
| Clinical observation profile | 54 measurement-aware values |
| Full debug profile | 138 mechanistic values; not claimed to be Markov |
| Action space | 8 bounded, one-directional controls |
| Default measurement profile | `realistic` with clinical observations |
| Strict policy metadata | `info_profile="benchmark"` |
| Default decision / integration step | 5 min / 0.25 min |

Exact ordered observation and action hashes are recorded in
`RELEASE_v0.23.json`. Shape equality alone does not establish checkpoint
compatibility; policies trained on earlier transition kernels must be retrained
and evaluated under a versioned protocol.

## Verification and scientific scope

GitHub Actions is configured to check every change with:

- the full test suite on Python 3.10, 3.12 and 3.14;
- the v0.23 release-candidate integrity gate;
- a source-distribution and wheel build;
- installation and CLI smoke tests outside the source tree.

Run the same checks locally:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q -ra -W error
python validation/run_validation_v23.py
```

These checks establish software behavior, numerical invariants and internal
mechanistic consistency. They are not independent external clinical validation.
The evidence and remaining limitations are documented in:

- [v0.23 candidate validation audit](VALIDATION_AUDIT_v0.23.md);
- [machine-readable focused-gate results](validation/validation_results_v0.23.json);
- [v0.23 release-candidate manifest](RELEASE_v0.23.json);
- [model card and use boundaries](docs/MODEL_CARD.md).

Historical release evidence is retained under `docs/history/` and
`validation/` for reproducibility without turning this README into a changelog.
Release-level changes belong in [GitHub Releases](https://github.com/lysyloxidase/openhumsim-rl/releases).

For a transparent RL smoke comparison with delayed measurements and masked
history, run:

```bash
python validation/rl_benchmark_v0.23.py --output validation/rl_benchmark_v0.23.json
```

This harness compares fixed no-op and observation-only engineering rules. It
does not train a policy or provide evidence of clinical safety or efficacy.

## Contributing and security

Scientific changes should include units, relevant conservation or closure
checks, timestep evidence where appropriate, externally anchored sources and
stress-case regression coverage. See [CONTRIBUTING.md](CONTRIBUTING.md).

Report security vulnerabilities privately according to
[SECURITY.md](SECURITY.md). Never include patient-identifiable data, credentials
or restricted clinical datasets in a public issue.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). Cite the exact
release or commit used so that model, schema and reward semantics remain
identifiable.

Copyright 2026 lysyloxidase. Licensed under the
[Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
