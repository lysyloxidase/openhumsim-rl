"""Local, dependency-free HTTP bridge for the OpenHumSim dashboard.

Run from any environment where OpenHumSim-RL is installed:

    openhumsim dashboard

The server binds to loopback by default and exposes only a curated dashboard
payload. It is a research/debug interface, not a clinical monitoring service.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
import json
from math import isfinite
from pathlib import Path
import platform
from threading import RLock
from time import monotonic
from typing import Any, Callable
from urllib.parse import urlparse, urlsplit
from uuid import UUID, uuid4
import webbrowser

import numpy as np

from . import HumanHomeostasisEnv, __version__
from .env import ACTION_NAMES, CLINICAL_OBSERVATION_NAMES
from .units import OBSERVATION_UNITS


PACKAGE_ROOT = Path(__file__).resolve().parent


def _source_checkout_root() -> Path | None:
    """Locate optional release evidence without requiring a source checkout."""

    candidates = list(PACKAGE_ROOT.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (
            (root / "pyproject.toml").is_file()
            and (root / "src" / "openhumsim_rl").is_dir()
        ):
            return root
    return None


ROOT = _source_checkout_root() or PACKAGE_ROOT
DASHBOARD_HTML = PACKAGE_ROOT / "dashboard" / "index.html"
RELEASE_FILENAME = f"RELEASE_v{__version__}.json"
RELEASE_MANIFEST = ROOT / RELEASE_FILENAME
VALIDATION_RESULTS = (
    ROOT / "validation" / f"validation_results_v{__version__}.json"
)
CI_EVIDENCE = ROOT / "CI_EVIDENCE.json"
MAX_REQUEST_BYTES = 64 * 1024
SESSION_HEADER = "X-OpenHumSim-Session"
DEFAULT_MAX_SESSIONS = 32
DEFAULT_SESSION_TTL_SECONDS = 6 * 60 * 60
EXPERIMENT_MANIFEST_SCHEMA = "openhumsim.experiment-manifest.v1"
RELEASE_MANIFEST_SCHEMA = "openhumsim.release.v1"
VALIDATION_RESULTS_SCHEMA = "openhumsim.validation-results.v1"
CI_EVIDENCE_SCHEMA = "openhumsim.ci-evidence.v1"
ORDERED_NAMES_HASH_FORMAT = "sha256:canonical-json-array:utf-8"
SOURCE_FINGERPRINT_HASH_FORMAT = (
    "sha256:canonical-json-array-of-source-id-and-content-sha256:utf-8"
)

SCENARIOS = (
    {"id": "baseline", "label": "Stan bazowy", "group": "Referencja"},
    {"id": "oral_glucose_75g", "label": "Doustna glukoza 75 g", "group": "Metabolizm"},
    {"id": "fasting", "label": "Głodzenie", "group": "Metabolizm"},
    {"id": "transient_lactic_acidosis", "label": "Przejściowa kwasica mleczanowa", "group": "Metabolizm"},
    {"id": "respiratory_acidosis", "label": "Kwasica oddechowa", "group": "Oddech"},
    {"id": "hypoventilation", "label": "Hipowentylacja", "group": "Oddech"},
    {"id": "vq_mismatch", "label": "Niedopasowanie V/Q", "group": "Płuca"},
    {"id": "pulmonary_shunt", "label": "Przeciek płucny", "group": "Płuca"},
    {"id": "dependent_derecruitment", "label": "Zależna derekrutacja", "group": "Płuca"},
    {"id": "airway_obstruction", "label": "Obturacja dróg oddechowych", "group": "Mechanika"},
    {"id": "dehydrated", "label": "Odwodnienie", "group": "Płyny"},
    {"id": "saline_challenge_30ml_kg", "label": "0,9% NaCl 30 ml/kg", "group": "Płyny"},
    {"id": "reduced_renal_function", "label": "Zmniejszona filtracja", "group": "Nerki"},
    {"id": "hyperkalemia", "label": "Hiperkaliemia", "group": "Elektrolity"},
    {"id": "pbpk_oral_dose", "label": "Doustna sonda PBPK", "group": "PBPK"},
    {"id": "pressure_support_synchronous", "label": "PSV — synchronia", "group": "Wentylator"},
    {"id": "pressure_support_ineffective_trigger", "label": "PSV — nieskuteczny trigger", "group": "Wentylator"},
)
SCENARIO_IDS = frozenset(item["id"] for item in SCENARIOS)

ACTION_CONTROLS = (
    {"name": "insulin", "label": "Insulina", "kind": "bolus", "display_unit": "j.m. modelu", "config_max": "max_insulin_model_units_per_step"},
    {"name": "oral_carbs", "label": "Węglowodany", "kind": "bolus", "display_unit": "g", "config_max": "max_carbs_g_per_step"},
    {"name": "exercise", "label": "Wysiłek", "kind": "continuous", "display_unit": "%", "display_max": 100.0},
    {"name": "saline", "label": "0,9% NaCl", "kind": "bolus", "display_unit": "ml", "config_max": "max_saline_ml_per_step"},
    {"name": "oxygen", "label": "Sterowanie FiO₂", "kind": "continuous", "display_unit": "% zakresu", "display_max": 100.0},
    {"name": "ventilation_pressure_assist", "label": "Asysta ciśnieniowa", "kind": "continuous", "display_unit": "cmH₂O", "config_max": "max_ventilation_pressure_assist_cmH2O"},
    {"name": "oral_water", "label": "Woda doustna", "kind": "bolus", "display_unit": "ml", "config_max": "max_oral_water_ml_per_step"},
    {"name": "oral_probe_compound", "label": "Sonda PBPK", "kind": "bolus", "display_unit": "mg", "config_max": "max_probe_drug_mg_per_step"},
)


class RevisionConflictError(RuntimeError):
    """The caller's run/revision precondition no longer matches the session."""


class SessionNotFoundError(LookupError):
    """The requested dashboard session does not exist in this server process."""


class UnsupportedMediaTypeError(ValueError):
    """A mutating endpoint did not receive application/json."""


class OriginMismatchError(ValueError):
    """An explicit Origin header did not match this HTTP server's origin."""


class RequestIdConflictError(ValueError):
    """A step request id was reused for different request content."""


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r} is not allowed")


def _normalize_request_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("request_id must be a UUID string")
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("request_id must be a UUID string") from exc


def _canonical_host_name(value: str) -> str:
    text = value.strip().lower().rstrip(".")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text or any(character.isspace() for character in text) or any(
        character in text for character in "/@?#"
    ):
        raise ValueError(f"invalid allowed host {value!r}")
    try:
        return ip_address(text).compressed
    except ValueError:
        if ":" in text:
            raise ValueError(f"invalid allowed host {value!r}")
        try:
            return text.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError(f"invalid allowed host {value!r}") from exc


def _parse_json_float(token: str) -> float:
    value = float(token)
    if not isfinite(value):
        raise ValueError(f"non-finite JSON number {token!r} is not allowed")
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _decode_json_object(payload: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _verified_dashboard_artifacts(
    release_path: Path,
    validation_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load release metadata and validation evidence without trusting either early."""

    release = _read_json(release_path)
    if (
        release is None
        or release.get("schema") != RELEASE_MANIFEST_SCHEMA
        or release.get("version") != __version__
    ):
        return None, None

    gate = release.get("focused_integrity_gate")
    if not isinstance(gate, dict):
        return release, None

    declared_path = gate.get("results_path")
    if not isinstance(declared_path, str) or not declared_path:
        return release, None
    relative_path = Path(declared_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return release, None
    try:
        expected_path = (release_path.parent / relative_path).resolve()
        actual_path = validation_path.resolve()
    except OSError:
        return release, None
    if expected_path != actual_path:
        return release, None

    declared_sha256 = gate.get("results_sha256")
    if not (
        isinstance(declared_sha256, str)
        and len(declared_sha256) == 64
        and all(character in "0123456789abcdef" for character in declared_sha256)
    ):
        return release, None
    try:
        validation_bytes = validation_path.read_bytes()
    except OSError:
        return release, None
    actual_sha256 = sha256(validation_bytes).hexdigest()
    if not compare_digest(actual_sha256, declared_sha256):
        return release, None

    validation = _decode_json_object(validation_bytes)
    if (
        validation is None
        or validation.get("schema") != VALIDATION_RESULTS_SCHEMA
        or validation.get("version") != release["version"]
        or validation.get("state_schema_version")
        != release.get("state_schema_version")
    ):
        return release, None

    summary = validation.get("summary")
    checks = validation.get("checks")
    if not isinstance(summary, dict) or not isinstance(checks, list):
        return release, None
    passed = summary.get("passed")
    total = summary.get("total")
    if (
        type(passed) is not int
        or type(total) is not int
        or total <= 0
        or passed < 0
        or passed > total
        or len(checks) != total
    ):
        return release, None
    if any(
        not isinstance(check, dict)
        or type(check.get("passed")) is not bool
        for check in checks
    ):
        return release, None
    if sum(check["passed"] for check in checks) != passed:
        return release, None
    expected_status = "passed" if passed == total else "failed"
    if (
        gate.get("passed") != passed
        or gate.get("total") != total
        or gate.get("status") != expected_status
        or gate.get("executed_at_utc") != validation.get("executed_at_utc")
    ):
        return release, None
    return release, validation


def _verified_ci_evidence(
    path: Path,
    release: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Accept CI evidence only when it identifies the exact clean release source."""

    evidence = _read_json(path)
    if evidence is None or evidence.get("schema") != CI_EVIDENCE_SCHEMA:
        return None
    run = evidence.get("latest_successful_run")
    gate = release.get("focused_integrity_gate") if release is not None else None
    supported_ci = (
        release.get("supported_interpreter_ci")
        if release is not None
        else None
    )
    if (
        not isinstance(run, dict)
        or not isinstance(gate, dict)
        or not isinstance(supported_ci, dict)
    ):
        return None
    versions = run.get("python_versions")
    expected_versions = supported_ci.get("python_versions")
    if (
        run.get("conclusion") != "success"
        or not isinstance(versions, list)
        or not isinstance(expected_versions, list)
        or versions != expected_versions
        or not expected_versions
        or any(not isinstance(item, str) or not item for item in versions)
        or run.get("scientific_validation") != "success"
        or run.get("package_smoke_test") != "success"
        or gate.get("git_worktree_dirty") is not False
        or run.get("commit_sha") != gate.get("git_commit")
    ):
        return None
    return {
        "status": "passed",
        "conclusion": "success",
        "python_versions": list(versions),
        "scientific_validation": "success",
        "package_smoke_test": "success",
        "commit_sha": str(run["commit_sha"]),
        "run_url": run.get("run_url"),
        "completed_at": run.get("completed_at"),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode values exactly once for deterministic contract fingerprints."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _ordered_names_contract(
    names: tuple[str, ...],
    *,
    release_sha256: Any,
) -> dict[str, Any]:
    ordered_names = list(names)
    computed_sha256 = _canonical_sha256(ordered_names)
    declared_sha256 = (
        release_sha256
        if isinstance(release_sha256, str) and release_sha256
        else None
    )
    return {
        "ordered_names": ordered_names,
        "count": len(ordered_names),
        "sha256": computed_sha256,
        "hash_format": ORDERED_NAMES_HASH_FORMAT,
        "release_declared_sha256": declared_sha256,
        "matches_release": (
            computed_sha256 == declared_sha256
            if declared_sha256 is not None
            else None
        ),
    }


def _source_fingerprint() -> dict[str, Any]:
    """Fingerprint executable model sources without exposing host paths."""

    package_sources = sorted(PACKAGE_ROOT.glob("*.py"))
    # Public source identifiers describe repository provenance, not an
    # installation-specific site-packages path. Keeping them canonical also
    # makes wheel and source-checkout experiment manifests comparable.
    source_prefix = "src/openhumsim_rl"
    files = [
        {
            "source_id": f"{source_prefix}/{source.name}",
            "content_sha256": sha256(source.read_bytes()).hexdigest(),
        }
        for source in package_sources
    ]
    release_available = RELEASE_MANIFEST.is_file()
    release_sha256 = (
        sha256(RELEASE_MANIFEST.read_bytes()).hexdigest()
        if release_available
        else None
    )
    return {
        "source_files": files,
        "source_file_count": len(files),
        "sha256": _canonical_sha256(files),
        "hash_format": SOURCE_FINGERPRINT_HASH_FORMAT,
        "release_manifest": {
            "source_id": RELEASE_FILENAME,
            "available": release_available,
            "content_sha256": release_sha256,
        },
    }


def _observation_catalog(
    env: HumanHomeostasisEnv,
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Describe the exact clinical policy vector and measurement provenance."""

    if tuple(env.observation_names) != CLINICAL_OBSERVATION_NAMES:
        raise ValueError("dashboard observation catalog requires the clinical profile")
    measurement = info.get("measurement", {})
    channel_diagnostics = measurement.get("channels", {})
    if not isinstance(channel_diagnostics, dict):
        channel_diagnostics = {}
    group_age_names = {
        "monitor": "monitor_measurement_age_min",
        "abg": "blood_gas_measurement_age_min",
        "chemistry": "chemistry_measurement_age_min",
        "hemodynamic": "hemodynamic_measurement_age_min",
        "cgm": "cgm_measurement_age_min",
    }
    age_names = frozenset(group_age_names.values())
    catalog: list[dict[str, Any]] = []
    for order, name in enumerate(env.observation_names):
        channel = channel_diagnostics.get(name)
        if isinstance(channel, dict) and isinstance(channel.get("group"), str):
            group = str(channel["group"])
            diagnostics = {
                "available": True,
                "scope": "channel",
                "diagnostics_key": name,
                "age_observation_name": group_age_names[group],
            }
        elif name == "sensor_glucose_mg_dl":
            group = "cgm"
            diagnostics = {
                "available": True,
                "scope": "group",
                "diagnostics_key": "cgm",
                "age_observation_name": "cgm_measurement_age_min",
            }
        elif name in age_names:
            group = "measurement_age"
            diagnostics = {
                "available": True,
                "scope": "aggregate_age_observation",
                "diagnostics_key": name,
                "age_observation_name": name,
            }
        else:
            group = "episode_clock"
            diagnostics = {
                "available": False,
                "scope": "derived",
            }
        catalog.append(
            {
                "name": name,
                "order": order,
                "unit": OBSERVATION_UNITS[name],
                "group": group,
                "measurement_diagnostics": diagnostics,
                "normalization": {
                    "transform": "tanh((raw-center)/scale)",
                    "center": _number(env._obs_center[order]),
                    "scale": _number(env._obs_scale[order]),
                    "semantics": "engineering normalization; not a clinical reference range",
                },
            }
        )
    if len(catalog) != 54:
        raise RuntimeError(
            f"dashboard clinical observation catalog has {len(catalog)} entries, expected 54"
        )
    return catalog


def build_experiment_manifest(
    env: HumanHomeostasisEnv,
    info: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Build deterministic, path-free provenance for one dashboard reset."""

    release = _read_json(RELEASE_MANIFEST) or {}
    environment_semantics = info.get("environment_semantics", {})
    if not isinstance(environment_semantics, dict):
        environment_semantics = {}
    source = _source_fingerprint()
    scenario_spec = next(
        (item for item in SCENARIOS if item["id"] == env.active_scenario),
        None,
    )
    if scenario_spec is None:
        raise ValueError(f"unsupported dashboard scenario {env.active_scenario!r}")
    scenario_source = next(
        item
        for item in source["source_files"]
        if item["source_id"] == "src/openhumsim_rl/env.py"
    )
    measurement_config = (
        asdict(env.measurement_model.config)
        if env.measurement_model is not None
        else None
    )
    observation_contract = _ordered_names_contract(
        tuple(env.observation_names),
        release_sha256=release.get("clinical_observation_sha256"),
    )
    action_contract = _ordered_names_contract(
        ACTION_NAMES,
        release_sha256=release.get("action_sha256"),
    )
    manifest: dict[str, Any] = {
        "schema": EXPERIMENT_MANIFEST_SCHEMA,
        "model": {
            "package": "openhumsim_rl",
            "environment": "HumanHomeostasisEnv",
            "version": __version__,
            "research_only": True,
            "clinical_use_supported": False,
        },
        "state": {
            "schema_version": environment_semantics.get(
                "state_schema_version", "unknown"
            ),
            "fully_observed_markov_state": bool(
                environment_semantics.get("fully_observed_markov_state", False)
            ),
            "classification": environment_semantics.get("classification"),
        },
        "reward": {
            "profile": environment_semantics.get("reward_profile", "unknown"),
            "cumulative_semantics": "sum of per-transition environment rewards",
        },
        "profiles": {
            "observation": env.observation_profile,
            "measurement": env.measurement_profile,
            "info": env.info_profile,
        },
        "scenario": {
            "id": scenario_spec["id"],
            "label": scenario_spec["label"],
            "group": scenario_spec["group"],
            "procedural_state_challenge": scenario_spec["id"] != "baseline",
            "clinical_diagnosis_claim": False,
            "semantics": (
                "source-bound procedural initial-state challenge"
                if scenario_spec["id"] != "baseline"
                else "source-bound procedural baseline initialization"
            ),
            "source_binding": {
                "source_id": scenario_source["source_id"],
                "implementation": "HumanHomeostasisEnv._apply_scenario",
                "content_sha256": scenario_source["content_sha256"],
            },
        },
        "randomness": {
            "reset_seed": seed,
            "physiology_and_measurement": {
                "root_seed": seed,
                "derivation": "numpy SeedSequence.spawn(2)",
                "streams": ["physiology", "measurement"],
                "independent_streams": True,
            },
            "action_space": {
                "seed": seed + 1,
                "stream": "separate action-space RNG",
            },
            "semantics": (
                "reset physiology jitter and realistic measurement noise/dropout "
                "use separate child streams spawned from the reset seed; "
                "action-space sampling uses seed+1"
            ),
        },
        "timebase": {
            "agent_step_min": _number(env.config.agent_step_min),
            "integration_step_min": _number(env.config.integration_step_min),
            "episode_minutes": _number(env.config.episode_minutes),
        },
        "config": {
            "human": asdict(env.config),
            "clinical_measurement": measurement_config,
        },
        "interfaces": {
            "observation": observation_contract,
            "action": action_contract,
        },
        "observation_catalog": _observation_catalog(env, info),
        "runtime": {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "numpy_version": np.__version__,
        },
        "source_fingerprint": source,
    }
    # The self-hash covers every manifest field above and deliberately excludes
    # only itself and the sentence that defines the hash format.
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest["manifest_hash_format"] = (
        "sha256:canonical-json-object-without-manifest-sha256-and-hash-format:utf-8"
    )
    json.dumps(manifest, allow_nan=False)
    return manifest


def _number(value: Any) -> float:
    result = float(value)
    if not isfinite(result):
        raise FloatingPointError(f"dashboard payload received non-finite value {value!r}")
    return result


def _pick(state: Any, *names: str) -> dict[str, float]:
    return {name: _number(getattr(state, name)) for name in names}


def _finite_mapping(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    return {str(name): _number(item) for name, item in value.items()}


def _raw_clinical_measurements(env: HumanHomeostasisEnv) -> dict[str, float]:
    """Return physical observations, never the normalized policy vector."""

    values: dict[str, float] = {}
    for name in env.observation_names:
        if name == "time_to_go_fraction":
            value = max(
                0.0,
                1.0 - env.elapsed_minutes / env.config.episode_minutes,
            )
        elif env.measurement_model is not None:
            value = env.measurement_model.measurement_value(name, env.state)
        elif name == "sensor_glucose_mg_dl":
            value = env.state.glucose_mg_dl
        elif name.endswith("_measurement_age_min"):
            value = 0.0
        else:
            value = getattr(env.state, name)
        values[name] = _number(value)
    return values


def dashboard_payload(
    env: HumanHomeostasisEnv,
    info: dict[str, Any],
    *,
    reward: float = 0.0,
    cumulative_reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    seed: int | None = None,
    run_id: str | None = None,
    revision: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Create the stable, curated JSON contract consumed by the dashboard."""

    state = env.state
    mass = info.get("mass_balance", {})
    acid = info.get("acid_base", {})
    energy = info.get("energy_metabolism", {})
    pulmonary = info.get("pulmonary_exchange", {})
    measurement = info.get("measurement", {})

    payload = {
        "schema": "openhumsim.dashboard.v2",
        "model_version": __version__,
        "state_schema_version": info.get("environment_semantics", {}).get(
            "state_schema_version", "unknown"
        ),
        "scenario": env.active_scenario,
        "scenario_warning": info.get("scenario_warning"),
        "seed": seed,
        "run_id": run_id,
        "revision": revision,
        "request_id": request_id,
        "data_profile": {
            "observation": env.observation_profile,
            "measurement": env.measurement_profile,
            "debug_truth_included": True,
            "clinical_values_are_raw": True,
        },
        "time_min": _number(env.elapsed_minutes),
        "step_min": _number(env.config.agent_step_min),
        "episode_minutes": _number(env.config.episode_minutes),
        "reward": _number(reward),
        "cumulative_reward": _number(cumulative_reward),
        "reward_terms": {
            str(name): _number(value)
            for name, value in info.get("reward_terms", {}).items()
        },
        "action": _finite_mapping(info.get("action")),
        "intervention": _finite_mapping(info.get("intervention")),
        "status": {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "needs_reset": bool(env._needs_reset),
            "reason": info.get("termination_reason"),
        },
        "measurements": _raw_clinical_measurements(env),
        "measurement_ages_min": {
            str(name): _number(value)
            for name, value in measurement.get("ages_min", {}).items()
        },
        "measurement_channels": measurement.get("channels", {}),
        "vitals": _pick(
            state,
            "heart_rate_bpm",
            "map_mmHg",
            "systolic_pressure_mmHg",
            "diastolic_pressure_mmHg",
            "cardiac_output_l_min",
            "stroke_volume_ml",
            "pao2_mmHg",
            "paco2_mmHg",
            "spo2_pct",
            "ph_arterial",
            "bicarbonate_mmol_l",
            "respiratory_rate_bpm",
            "tidal_volume_l",
            "glucose_mg_dl",
            "lactate_mmol_l",
            "gfr_ml_min",
        ),
        "electrolytes": _pick(
            state,
            "sodium_mmol_l",
            "potassium_mmol_l",
            "chloride_mmol_l",
            "plasma_osmolality_mOsm_kg",
            "total_body_water_l",
            "ecf_volume_l",
            "icf_volume_l",
            "urine_flow_ml_min",
        ),
        "oxygen": _pick(
            state,
            "vo2_demand_ml_min",
            "vo2_ml_min",
            "oxygen_delivery_ml_min",
            "oxygen_extraction_ratio",
            "oxygen_debt_ml_min",
            "cumulative_oxygen_deficit_ml",
            "arterial_o2_content_ml_dl",
            "mixed_venous_o2_sat_pct",
        ),
        "carbon": _pick(
            state,
            "vco2_ml_min",
            "vco2_elimination_ml_min",
            "exchangeable_co2_pool_mmol",
            "co2_generated_mmol",
            "co2_eliminated_mmol",
            "metabolic_respiratory_quotient",
        ),
        "lactate": _pick(
            state,
            "lactate_amount_mmol",
            "lactate_distribution_volume_l",
            "lactate_production_mmol_min",
            "lactate_clearance_mmol_min",
            "lactate_generated_mmol",
            "lactate_cleared_mmol",
        ),
        "acid_base": {
            "sida_mEq_l": _number(acid.get("sida_mEq_l", state.strong_ion_difference_apparent_mEq_l)),
            "side_mEq_l": _number(acid.get("side_mEq_l", state.strong_ion_difference_effective_mEq_l)),
            "strong_ion_gap_mEq_l": _number(acid.get("strong_ion_gap_mEq_l", state.strong_ion_gap_mEq_l)),
            "albumin_g_dl": _number(state.albumin_g_dl),
            "phosphate_mmol_l": _number(state.phosphate_mmol_l),
        },
        "pulmonary": {
            "aa_gradient_mmHg": _number(pulmonary.get("aa_gradient_mmHg", state.pulmonary_aa_gradient_mmHg)),
            "shunt_fraction": _number(pulmonary.get("shunt_fraction", state.pulmonary_shunt_fraction)),
            "mean_vq_ratio": _number(pulmonary.get("mean_vq_ratio", state.pulmonary_mean_vq_ratio)),
            "low_vq_perfusion_fraction": _number(pulmonary.get("low_vq_perfusion_fraction", state.pulmonary_low_vq_perfusion_fraction)),
            "alveolar_dead_space_fraction": _number(pulmonary.get("alveolar_dead_space_fraction", state.pulmonary_alveolar_dead_space_fraction)),
            "diffusion_equilibration_fraction": _number(pulmonary.get("diffusion_equilibration_fraction", state.pulmonary_diffusion_equilibration_fraction)),
            "recruitment_fraction": _number(state.pulmonary_recruitment_fraction),
            "alveolar_ventilation_l_min": _number(state.alveolar_ventilation_l_min),
            "mechanical_power_j_min": _number(state.pulmonary_mechanical_power_j_min),
            "auto_peep_cmH2O": _number(state.respiratory_cycle_auto_peep_cmH2O),
        },
        "renal": _pick(
            state,
            "gfr_ml_min",
            "urine_flow_ml_min",
            "urine_sodium_mmol_min",
            "urine_potassium_mmol_min",
            "urine_ammonium_mmol_min",
            "urine_titratable_acid_mmol_min",
            "urine_bicarbonate_mmol_min",
            "adh_relative",
            "aldosterone_relative",
        ),
        "pbpk": _pick(
            state,
            "probe_plasma_mg_l",
            "probe_effect_site_mg_l",
            "probe_total_body_mg",
        ),
        "integrity": {
            "co2_mass_balance_error_mmol": _number(mass.get("co2_mass_balance_error_mmol", state.co2_mass_balance_error_mmol)),
            "co2_final_gas_closure_residual_mmol_l": _number(mass.get("co2_final_gas_closure_residual_mmol_l", state.co2_final_gas_closure_residual_mmol_l)),
            "lactate_mass_balance_error_mmol": _number(mass.get("lactate_mass_balance_error_mmol", state.lactate_mass_balance_error_mmol)),
            "water_mass_balance_error_l": _number(mass.get("water_mass_balance_error_l", 0.0)),
            "sodium_mass_balance_error_mmol": _number(mass.get("sodium_mass_balance_error_mmol", 0.0)),
            "chloride_mass_balance_error_mmol": _number(
                mass.get(
                    "chloride_mass_balance_error_mmol",
                    getattr(state, "chloride_mass_balance_error_mmol", 0.0),
                )
            ),
            "potassium_mass_balance_error_mmol": _number(mass.get("potassium_mass_balance_error_mmol", 0.0)),
            "charge_balance_residual_mEq_l": _number(state.charge_balance_residual_mEq_l),
            "cv_blood_volume_error_ml": _number(state.cv_blood_volume_error_ml),
            "fick_residual_ml_min": _number(info.get("oxygen_transport", {}).get("fick_residual_ml_min", 0.0)),
        },
        "energy_summary": {
            "model": energy.get("model", "reduced whole-body energy ledger"),
            "instantaneous_oxygen_deficit_ml_min": _number(
                energy.get("instantaneous_oxygen_deficit_ml_min", state.oxygen_debt_ml_min)
            ),
        },
    }
    # Round-tripping with allow_nan=False is a final contract guard.
    json.dumps(payload, allow_nan=False)
    return payload


def dashboard_meta() -> dict[str, Any]:
    release, validation = _verified_dashboard_artifacts(
        RELEASE_MANIFEST,
        VALIDATION_RESULTS,
    )
    ci_evidence = _verified_ci_evidence(CI_EVIDENCE, release)
    release_available = release is not None
    validation_available = validation is not None
    ci_evidence_available = ci_evidence is not None
    release = release or {}
    validation = validation or {}
    controls = []
    default_env = HumanHomeostasisEnv()
    for index, spec in enumerate(ACTION_CONTROLS):
        item = dict(spec)
        item["index"] = index
        if "display_max" not in item:
            item["display_max"] = _number(
                getattr(default_env.config, str(item["config_max"]))
            )
        item.pop("config_max", None)
        controls.append(item)
    validation_summary = validation.get("summary")
    if not isinstance(validation_summary, dict):
        validation_summary = None
    full_test_suite = release.get("full_test_suite")
    if not isinstance(full_test_suite, dict):
        full_test_suite = None
    supported_python_ci = release.get("supported_interpreter_ci")
    if (
        not isinstance(supported_python_ci, dict)
        or supported_python_ci.get("status") != "passed"
    ):
        supported_python_ci = ci_evidence

    return {
        "schema": "openhumsim.dashboard.meta.v2",
        "model_version": __version__,
        "research_only": True,
        "availability": {
            "release_manifest": release_available,
            "validation_results": validation_available,
            "ci_evidence": ci_evidence_available,
        },
        "scenarios": SCENARIOS,
        "actions": controls,
        "action_names": ACTION_NAMES,
        "validation": validation_summary,
        "full_test_suite": full_test_suite,
        "supported_python_ci": supported_python_ci,
        "observation_contract": {
            "clinical_count": release.get("clinical_observation_count"),
            "full_count": release.get("full_observation_count"),
            "state_schema_version": release.get("state_schema_version"),
            "reward_profile": release.get("reward_profile"),
            "benchmark_reward_profile": release.get(
                "benchmark_reward_profile"
            ),
        },
    }


class DashboardSession:
    """One deterministic simulator session with optimistic concurrency control."""

    def __init__(self, scenario: str = "baseline", seed: int = 42) -> None:
        self.lock = RLock()
        self.env: HumanHomeostasisEnv | None = None
        self.info: dict[str, Any] = {}
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.seed = 42
        self.run_id = ""
        self.revision = 0
        self.last_request_id: str | None = None
        self.frames: list[dict[str, Any]] = []
        self._experiment_manifest: dict[str, Any] = {}
        self._step_results: dict[
            str,
            tuple[tuple[str | None, int | None, tuple[float, ...]], dict[str, Any]],
        ] = {}
        self.reset(scenario, seed)

    def _transaction_snapshot(self) -> dict[str, Any]:
        """Capture every mutable session field outside the environment runtime."""

        return {
            "env": self.env,
            "info": deepcopy(self.info),
            "cumulative_reward": self.cumulative_reward,
            "last_reward": self.last_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "seed": self.seed,
            "run_id": self.run_id,
            "revision": self.revision,
            "last_request_id": self.last_request_id,
            "frames": deepcopy(self.frames),
            "experiment_manifest": deepcopy(self._experiment_manifest),
            "step_results": deepcopy(self._step_results),
        }

    def _restore_transaction_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot captured by :meth:`_transaction_snapshot`."""

        self.env = snapshot["env"]
        self.info = snapshot["info"]
        self.cumulative_reward = snapshot["cumulative_reward"]
        self.last_reward = snapshot["last_reward"]
        self.terminated = snapshot["terminated"]
        self.truncated = snapshot["truncated"]
        self.seed = snapshot["seed"]
        self.run_id = snapshot["run_id"]
        self.revision = snapshot["revision"]
        self.last_request_id = snapshot["last_request_id"]
        self.frames = snapshot["frames"]
        self._experiment_manifest = snapshot["experiment_manifest"]
        self._step_results = snapshot["step_results"]

    def _check_expected(
        self,
        expected_run_id: str | None,
        expected_revision: int | None,
    ) -> None:
        # Direct in-process callers from the original public example may omit
        # preconditions. HTTP endpoints never do: the handler validates both
        # fields before entering the session.
        if expected_run_id is None and expected_revision is None:
            return
        if expected_run_id is None or expected_revision is None:
            raise ValueError(
                "expected_run_id and expected_revision must be supplied together"
            )
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("expected_revision must be a nonnegative integer")
        if expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        if expected_run_id != self.run_id or expected_revision != self.revision:
            raise RevisionConflictError(
                "stale dashboard state: expected "
                f"run_id={expected_run_id!r}, revision={expected_revision}; "
                f"current run_id={self.run_id!r}, revision={self.revision}"
            )

    def _frame(self) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("dashboard session is not initialized")
        return dashboard_payload(
            self.env,
            self.info,
            reward=self.last_reward,
            cumulative_reward=self.cumulative_reward,
            terminated=self.terminated,
            truncated=self.truncated,
            seed=self.seed,
            run_id=self.run_id,
            revision=self.revision,
            request_id=self.last_request_id,
        )

    def reset(
        self,
        scenario: str,
        seed: int,
        *,
        expected_run_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if scenario not in SCENARIO_IDS:
            raise ValueError(f"unsupported dashboard scenario {scenario!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2^32-1]")
        with self.lock:
            self._check_expected(expected_run_id, expected_revision)
            previous_session = self._transaction_snapshot()
            try:
                new_env = HumanHomeostasisEnv(
                    scenario=scenario,
                    observation_profile="clinical",
                    measurement_profile="realistic",
                    info_profile="debug",
                )
                _, new_info = new_env.reset(seed=seed)
                new_manifest = build_experiment_manifest(
                    new_env,
                    new_info,
                    seed=seed,
                )
                self.env = new_env
                self.info = new_info
                self._experiment_manifest = new_manifest
                self.seed = seed
                self.cumulative_reward = 0.0
                self.last_reward = 0.0
                self.terminated = False
                self.truncated = False
                self.run_id = str(uuid4())
                self.revision = 0
                self.last_request_id = None
                self._step_results = {}
                frame = self._frame()
                self.frames = [deepcopy(frame)]
            except Exception:
                self._restore_transaction_snapshot(previous_session)
                raise
            return deepcopy(frame)

    def step(
        self,
        action: list[Any] | dict[str, Any],
        *,
        expected_run_id: str | None = None,
        expected_revision: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if self.env is None:
                raise RuntimeError("dashboard session is not initialized")

            if isinstance(action, dict):
                unknown = sorted(set(action) - set(ACTION_NAMES))
                if unknown:
                    raise ValueError(f"unknown action keys: {unknown}")
                values = [action.get(name, 0.0) for name in ACTION_NAMES]
            elif isinstance(action, list):
                values = action
            else:
                raise TypeError("action must be a list or object")
            if len(values) != len(ACTION_NAMES):
                raise ValueError(f"action must contain {len(ACTION_NAMES)} values")
            if any(isinstance(value, (bool, np.bool_)) for value in values):
                raise ValueError("action values must be numbers, not booleans")
            if any(
                not isinstance(value, (int, float, np.integer, np.floating))
                for value in values
            ):
                raise ValueError("action values must be JSON numbers")
            try:
                vector = np.asarray([float(value) for value in values], dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("action values must be finite numbers") from exc
            if not np.all(np.isfinite(vector)) or np.any(vector < 0.0) or np.any(vector > 1.0):
                raise ValueError("action values must be finite and within [0, 1]")

            normalized_request_id = (
                _normalize_request_id(request_id) if request_id is not None else None
            )
            signature = (
                expected_run_id,
                expected_revision,
                tuple(float(value) for value in vector),
            )
            if normalized_request_id is not None and normalized_request_id in self._step_results:
                stored_signature, stored_frame = self._step_results[normalized_request_id]
                if stored_signature != signature:
                    raise RequestIdConflictError(
                        "request_id was already used for different step content"
                    )
                return deepcopy(stored_frame)

            self._check_expected(expected_run_id, expected_revision)
            if self.env._needs_reset:
                raise RuntimeError("episode has ended; reset the dashboard session")

            environment = self.env
            environment_snapshot = deepcopy(
                environment.to_versioned_snapshot()
            )
            previous_session = self._transaction_snapshot()
            try:
                (
                    _,
                    reward,
                    self.terminated,
                    self.truncated,
                    self.info,
                ) = environment.step(vector)
                self.last_reward = float(reward)
                self.cumulative_reward += self.last_reward
                self.revision += 1
                self.last_request_id = normalized_request_id
                frame = self._frame()
                self.frames.append(deepcopy(frame))
                if normalized_request_id is not None:
                    self._step_results[normalized_request_id] = (
                        signature,
                        deepcopy(frame),
                    )
            except Exception:
                try:
                    environment.restore_versioned_snapshot(
                        environment_snapshot
                    )
                finally:
                    self._restore_transaction_snapshot(previous_session)
                raise
            return deepcopy(frame)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.env is None or not self.frames:
                raise RuntimeError("dashboard session is not initialized")
            return deepcopy(self.frames[-1])

    def manifest_snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.env is None or not self._experiment_manifest:
                raise RuntimeError("dashboard session is not initialized")
            return deepcopy(self._experiment_manifest)

    def history_envelope(
        self,
        *,
        include_manifest: bool = True,
    ) -> dict[str, Any]:
        with self.lock:
            if self.env is None or not self.frames:
                raise RuntimeError("dashboard session is not initialized")
            envelope = {
                "schema": "openhumsim.dashboard.history.v2",
                "complete_from_reset": True,
                "run_id": self.run_id,
                "revision": self.revision,
                "frames": deepcopy(self.frames),
            }
            if include_manifest:
                envelope["experiment_manifest"] = deepcopy(
                    self._experiment_manifest
                )
            json.dumps(envelope, allow_nan=False)
            return envelope


class SessionStore:
    """Bounded, expiring registry of in-memory dashboard sessions."""

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        if not isfinite(float(ttl_seconds)) or float(ttl_seconds) <= 0.0:
            raise ValueError("ttl_seconds must be positive and finite")
        self.lock = RLock()
        self.max_sessions = max_sessions
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._sessions: OrderedDict[
            str, tuple[DashboardSession, float]
        ] = OrderedDict()

    def _purge_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, (_, last_access) in self._sessions.items()
            if now - last_access >= self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def create(
        self,
        scenario: str = "baseline",
        seed: int = 42,
    ) -> tuple[str, DashboardSession]:
        session = DashboardSession(scenario, seed)
        session_id = str(uuid4())
        with self.lock:
            now = self._clock()
            self._purge_expired(now)
            while len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[session_id] = (session, now)
        return session_id, session

    def get(self, session_id: str | None) -> DashboardSession:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"missing required {SESSION_HEADER} header")
        normalized = session_id.strip()
        with self.lock:
            now = self._clock()
            self._purge_expired(now)
            entry = self._sessions.pop(normalized, None)
            if entry is None:
                raise SessionNotFoundError(
                    "dashboard session was not found or has expired"
                )
            session, _ = entry
            self._sessions[normalized] = (session, now)
            return session


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "OpenHumSimDashboard/1"
    store = SessionStore()
    allowed_hosts: frozenset[str] = frozenset()

    def _headers(
        self,
        status: HTTPStatus,
        content_type: str,
        length: int,
        *,
        session_id: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        if session_id is not None:
            self.send_header(SESSION_HEADER, session_id)
        self.end_headers()

    def _json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
        *,
        session_id: str | None = None,
    ) -> None:
        body = json.dumps(
            payload, allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self._headers(
            status,
            "application/json; charset=utf-8",
            len(body),
            session_id=session_id,
        )
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The state transition may already be committed; request_id makes a
            # client retry safe, so a disconnected response needs no traceback.
            return

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message, "status": int(status)}, status)

    def _check_host(self) -> tuple[str, int]:
        raw_host = self.headers.get("Host")
        if not isinstance(raw_host, str) or not raw_host:
            raise OriginMismatchError("Host header is required")
        parsed = urlsplit(f"//{raw_host}")
        if (
            parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise OriginMismatchError("Host header is malformed")
        try:
            hostname = _canonical_host_name(parsed.hostname)
            port = parsed.port
        except ValueError as exc:
            raise OriginMismatchError("Host header is malformed") from exc
        expected_port = int(self.server.server_address[1])
        effective_port = 80 if port is None else port
        if effective_port != expected_port:
            raise OriginMismatchError("Host port does not match this dashboard server")

        bound_host = _canonical_host_name(str(self.server.server_address[0]))
        local_host = _canonical_host_name(str(self.connection.getsockname()[0]))
        allowed = set(self.allowed_hosts)
        allowed.add(bound_host)
        if bound_host in {"0.0.0.0", "::"}:
            allowed.add(local_host)
        numeric_hosts = []
        for candidate in allowed | {local_host}:
            try:
                numeric_hosts.append(ip_address(candidate))
            except ValueError:
                continue
        if any(address.is_loopback for address in numeric_hosts):
            allowed.update({"127.0.0.1", "::1", "localhost"})
        if hostname not in allowed:
            raise OriginMismatchError(
                "Host is not allowed; use --allowed-host for an explicit LAN/DNS name"
            )
        return hostname, effective_port

    def _check_origin(self) -> None:
        hostname, port = self._check_host()
        origin = self.headers.get("Origin")
        if origin is None:
            return
        parsed = urlparse(origin)
        try:
            origin_hostname = (
                _canonical_host_name(parsed.hostname) if parsed.hostname else None
            )
            origin_port = parsed.port
        except ValueError as exc:
            raise OriginMismatchError("Origin is malformed") from exc
        if (
            parsed.scheme.lower() != "http"
            or origin_hostname != hostname
            or (80 if origin_port is None else origin_port) != port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise OriginMismatchError(
                "Origin must exactly match the allowed dashboard Host and port"
            )

    def _body(self) -> dict[str, Any]:
        if self.headers.get_content_type().lower() != "application/json":
            raise UnsupportedMediaTypeError(
                "Content-Type must be application/json"
            )
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        if not raw:
            raise ValueError("request body must contain a JSON object")
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
        if not isinstance(value, dict):
            raise TypeError("JSON body must be an object")
        return value

    @staticmethod
    def _validate_fields(
        body: dict[str, Any],
        *,
        allowed: set[str],
        required: set[str] | None = None,
    ) -> None:
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise ValueError(f"unknown top-level JSON keys: {unknown}")
        missing = sorted((required or set()) - set(body))
        if missing:
            raise ValueError(f"missing required JSON keys: {missing}")

    @staticmethod
    def _expected(body: dict[str, Any]) -> tuple[str, int]:
        run_id = body["expected_run_id"]
        revision = body["expected_revision"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("expected_run_id must be a nonempty string")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        return run_id, revision

    def _session(self) -> tuple[str, DashboardSession]:
        session_id = self.headers.get(SESSION_HEADER)
        normalized = session_id.strip() if isinstance(session_id, str) else ""
        return normalized, self.store.get(session_id)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        try:
            self._check_origin()
            if path in {"/", "/index.html"}:
                body = DASHBOARD_HTML.read_bytes()
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
                self.wfile.write(body)
            elif path == "/api/meta":
                self._json(dashboard_meta())
            elif path == "/api/state":
                session_id, session = self._session()
                self._json(
                    session.snapshot(),
                    session_id=session_id,
                )
            elif path == "/api/history":
                session_id, session = self._session()
                self._json(
                    session.history_envelope(),
                    session_id=session_id,
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except OriginMismatchError as exc:
            self._error(HTTPStatus.FORBIDDEN, str(exc))
        except SessionNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValueError, TypeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except (OSError, RuntimeError, FloatingPointError) as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path not in {"/api/session", "/api/reset", "/api/step"}:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            self._check_origin()
            body = self._body()
            if path == "/api/session":
                self._validate_fields(
                    body,
                    allowed={"scenario", "seed"},
                )
                scenario = body.get("scenario", "baseline")
                seed = body.get("seed", 42)
                if not isinstance(scenario, str):
                    raise ValueError("scenario must be a string")
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError("seed must be an integer")
                session_id, session = self.store.create(scenario, seed)
                self._json(
                    {
                        "schema": "openhumsim.dashboard.session.v1",
                        "session_id": session_id,
                        "experiment_manifest": session.manifest_snapshot(),
                        "frame": session.snapshot(),
                        # The manifest is a sibling in the session envelope, so
                        # it is not repeated inside its nested history object.
                        "history": session.history_envelope(
                            include_manifest=False
                        ),
                    },
                    HTTPStatus.CREATED,
                    session_id=session_id,
                )
            elif path == "/api/reset":
                self._validate_fields(
                    body,
                    allowed={
                        "scenario", "seed", "expected_run_id", "expected_revision"
                    },
                    required={
                        "scenario", "seed", "expected_run_id", "expected_revision"
                    },
                )
                scenario = body["scenario"]
                seed = body["seed"]
                if not isinstance(scenario, str):
                    raise ValueError("scenario must be a string")
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError("seed must be an integer")
                expected_run_id, expected_revision = self._expected(body)
                session_id, session = self._session()
                frame = session.reset(
                    scenario,
                    seed,
                    expected_run_id=expected_run_id,
                    expected_revision=expected_revision,
                )
                self._json(frame, session_id=session_id)
            elif path == "/api/step":
                self._validate_fields(
                    body,
                    allowed={
                        "action", "expected_run_id", "expected_revision", "request_id"
                    },
                    required={
                        "action", "expected_run_id", "expected_revision", "request_id"
                    },
                )
                expected_run_id, expected_revision = self._expected(body)
                request_id = _normalize_request_id(body["request_id"])
                session_id, session = self._session()
                frame = session.step(
                    body["action"],
                    expected_run_id=expected_run_id,
                    expected_revision=expected_revision,
                    request_id=request_id,
                )
                self._json(frame, session_id=session_id)
        except UnsupportedMediaTypeError as exc:
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, str(exc))
        except OriginMismatchError as exc:
            self._error(HTTPStatus.FORBIDDEN, str(exc))
        except SessionNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except RevisionConflictError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
            OverflowError,
        ) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except FloatingPointError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} — {fmt % args}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local OpenHumSim dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: loopback)")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="NAME_OR_IP",
        help="additional exact HTTP Host name allowed for LAN/DNS access (repeatable)",
    )
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser.parse_args(argv)


def serve_dashboard(
    *,
    host: str = "127.0.0.1",
    allowed_hosts: tuple[str, ...] = (),
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    """Serve the packaged dashboard until interrupted."""

    if not 1 <= port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if not DASHBOARD_HTML.is_file():
        raise SystemExit(f"dashboard file not found: {DASHBOARD_HTML}")

    try:
        DashboardHandler.allowed_hosts = frozenset(
            _canonical_host_name(value)
            for value in [host, *allowed_hosts]
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    DashboardHandler.store = SessionStore()
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    url = f"http://{display_host}:{port}/"
    print(f"OpenHumSim dashboard: {url}")
    print("Research use only. Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return serve_dashboard(
        host=args.host,
        allowed_hosts=tuple(args.allowed_host),
        port=args.port,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    raise SystemExit(main())
