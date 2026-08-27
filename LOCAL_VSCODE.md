# Local development with VS Code

This guide describes the current OpenHumSim-RL v0.23.1 development workflow.
The project requires Python 3.10 or newer and uses
[uv](https://docs.astral.sh/uv/) for local environments.

## Set up the environment

From the repository root:

```bash
uv python install 3.12
uv sync --extra local
uv run openhumsim doctor
```

In VS Code, select `.venv/bin/python` with **Python: Select Interpreter**.
The included launch configurations and tasks run from the repository root.

## Run a simulation

```bash
uv run openhumsim demo --scenario baseline --minutes 60 --seed 42
uv run openhumsim measurement-demo --scenario baseline --minutes 40 --seed 42
```

Use the **OpenHumSim: demo** launch configuration to debug the example
scenario. To inspect the mechanistic state from Python:

```python
from openhumsim_rl import HumanHomeostasisEnv

env = HumanHomeostasisEnv(observation_profile="full")
observation, info = env.reset(seed=42)
```

The default clinical observation is intended for agent-facing experiments.
The full observation exposes internal state for scientific debugging.

## Test and validate

Run the complete test suite and the version-locked scientific integrity gate:

```bash
uv run pytest -q -ra
uv run python validation/run_validation_v23.py
```

The same commands are available as **OpenHumSim: tests** and
**OpenHumSim: scientific validation** in **Tasks: Run Task**. The validation
script can also be started under the debugger with the matching launch
configuration.

The scientific gate checks internal contracts and numerical invariants. It is
not a substitute for external clinical validation.

## Run the dashboard

```bash
uv run python examples/dashboard_server.py
```

Open the local address printed by the server. Stop it with `Ctrl+C`.

## Optional Jaeb evaluation

The Jaeb dataset must be obtained manually under its official terms. Display
the download instructions and then create a participant-level split from the
local archive:

```bash
uv run openhumsim data jaeb-download-instructions
uv run openhumsim data split-jaeb-cgm data/external/CGMND.zip --seed 2019
```

Keep external datasets outside version control.
