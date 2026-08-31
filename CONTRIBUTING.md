# Contributing to OpenHumSim-RL

OpenHumSim-RL welcomes reproducible bug reports, numerical tests, documentation
improvements and carefully scoped mechanistic contributions. It is research
software, not a clinically validated patient model.

## Development setup

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q -ra -W error
python validation/run_validation_v0232.py
```

Submit changes from a focused branch. Keep unrelated edits out of the same pull
request and explain any public API, observation, action, reward or state-schema
change explicitly.

## Scientific and numerical changes

A mechanistic change should include:

- units for every new state, parameter and flux;
- a conservation or closure test where a balance law applies;
- timestep or solver-convergence evidence where integration order matters;
- a source for externally anchored biological parameters or claims;
- a clear distinction between validated behavior and an engineering
  assumption;
- regression tests covering baseline and at least one relevant stress case.

Do not tune an expected value from the same implementation being tested and
present that as independent validation. Historical validation artifacts are
versioned records; add a new artifact instead of silently rewriting an older
release result.

## Safety and data

Do not submit patient-identifiable information, credentials, proprietary
clinical datasets or data whose redistribution terms are unclear. Public bug
reports should use synthetic inputs only. Keep diagnosis, dosing, treatment and
patient-specific outcome claims out of documentation and interfaces unless they
are supported by an independently reviewed clinical validation protocol.

Security vulnerabilities should be reported privately as described in
[SECURITY.md](SECURITY.md), not through a public issue.

Unless explicitly stated otherwise, intentionally submitted contributions are
provided under the repository's Apache License 2.0 terms.
