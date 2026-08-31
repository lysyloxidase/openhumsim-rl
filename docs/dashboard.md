# Dashboard usage and data provenance

The OpenHumSim-RL dashboard is a local research workspace backed by the actual
Python environment. The HTML file provides the interface; physiological
transitions are computed by `HumanHomeostasisEnv` in the local bridge.

> The dashboard is not a clinical monitor. Values, colors, extrema and scenario
> names are model outputs and research diagnostics, not alarms, reference
> intervals or treatment recommendations.

## Start the dashboard

From any working directory after installing OpenHumSim-RL, run:

```bash
openhumsim dashboard
```

The default address is `http://127.0.0.1:8765/`, and the server opens it in the
default browser. Command-line options are available with `--help`:

```bash
openhumsim dashboard --help
```

- `--no-open` starts the bridge without opening a browser.
- `--host ADDRESS` changes the bind address.
- `--port PORT` changes the TCP port.
- `--allowed-host NAME_OR_IP` permits an additional exact HTTP `Host` value and
  can be repeated.

The simulator and dashboard are fully available from an installed wheel.
Version-locked validation and CI evidence remain repository artifacts: outside
the matching source checkout their dashboard fields are marked unavailable
instead of borrowing files from an unrelated current working directory.

The bridge binds to loopback by default. If it is deliberately exposed through
a LAN or DNS name, that name must also be supplied with `--allowed-host`. Host
and port validation reduces DNS-rebinding exposure; it does not turn the local
research server into an internet-facing production service.

All HTML, API and error responses deny framing with both CSP
`frame-ancestors 'none'` and `X-Frame-Options: DENY`, preventing another page
from embedding the local controls for clickjacking.

## Measured observations and latent state

The primary cards use the `clinical + realistic` profile. They display values
from the measurement layer together with sampling time and age. Noise, delay,
sampling intervals and dropout can therefore make them differ from the current
mechanistic state.

Exact internal values appear only in panels explicitly labelled `model truth`
or `Latent model state · debug diagnostics`. They are useful for investigation
but must not be passed to a benchmark policy as if they were observations.

The observation inspector lists all 54 channels in the clinical-like policy
vector with their units, measurement group and current provenance. Displayed
normalization or summaries do not define clinical limits.

## Sessions and mutation safety

Each newly opened browser tab receives an in-memory simulator session. A
duplicated tab can inherit the same session identifier, so each mutating request
also carries the expected run identifier and revision. A stale request is
rejected instead of being silently applied to a newer state.

Step requests use an idempotency UUID. If an HTTP response is lost and the same
request is retried, the bridge returns the recorded result rather than applying
an insulin, carbohydrate, fluid or other intervention twice.

The server keeps at most 32 sessions. A session expires after six idle hours,
and restarting the server clears all sessions. These are in-memory limits, not
persistent storage guarantees.

If an expired or evicted session is detected while the page still holds local
frames, the interface preserves that bounded fragment as a separate recovered
export. It is marked `complete_from_reset: false` before a new run is created,
so partial history is not presented as a complete experiment.

## History, summaries and exports

The bridge retains the complete deduplicated trajectory from reset for the
life of the session. A refresh reloads that server-side history. The chart uses
only the latest 144 points for rendering performance; this window does not
truncate the server-side run history.

The run summary reports observed extrema and their times, changes from the
initial frame, physically summed boluses, time-weighted continuous controls and
cumulative reward components.

Two export formats are available:

- The JSON export contains the full available trajectory, session completeness
  flag, model frames and experiment manifest.
- The tidy CSV export writes one measured variable per row with run, revision,
  simulation time, variable name, value, unit, measurement group, sample time
  and measurement age. It deliberately excludes the hidden mechanistic state.

An export can only be complete while its session history remains available.
Always inspect `complete_from_reset` before treating an export as a full run.

## Experiment manifest

Every server-side history envelope and complete JSON export includes an
`openhumsim.experiment-manifest.v1` object. It records:

- package, environment, schema and reward versions;
- resolved observation, measurement and information profiles;
- the scenario identifier and its binding to executable source;
- the reset seed and random-stream semantics;
- agent, integration and episode time bases;
- resolved human and measurement configurations;
- exact ordered observation and action contracts with SHA-256 hashes;
- the 54-channel observation catalog and units;
- Python and NumPy runtime versions;
- a path-free fingerprint of the executable model sources;
- a canonical self-hash for the manifest.

The source fingerprint contains checkout- or distribution-relative identifiers
and content hashes, not local filesystem paths. Preserve the full JSON export
rather than a screenshot of the abbreviated manifest panel when reproducibility
matters.

## Randomness semantics

For an explicit reset seed, NumPy `SeedSequence.spawn(2)` derives separate
child generators for physiology jitter and realistic measurement noise,
sampling and dropout. Measurement draw counts therefore do not perturb future
physiology draws. Both streams remain reproducibly bound to the declared root
seed.

The action space is seeded separately with `seed + 1`. A single seed still does
not establish robustness or calibrated uncertainty. Comparative studies should
preserve the complete manifest and use multiple declared seeds.

## Offline preview

In a source checkout, opening `dashboard/index.html` redirects to the canonical
packaged HTML resource and is supported as a static UI preview. It uses a
bundled illustrative snapshot and is labelled `offline preview`.
Interventions, reset operations and simulation steps are unavailable because
the physiological model does not run in the browser.

Offline-preview data must not be exported or cited as the output of an executed
OpenHumSim-RL experiment.
