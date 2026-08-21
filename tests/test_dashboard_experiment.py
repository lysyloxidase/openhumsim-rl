from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from typing import Any, Iterator

import pytest

from examples.dashboard_server import (
    DashboardHandler,
    DashboardSession,
    EXPERIMENT_MANIFEST_SCHEMA,
    SESSION_HEADER,
    SessionStore,
    _canonical_sha256,
)
from openhumsim_rl import ClinicalMeasurementConfig, HumanConfig, __version__
from openhumsim_rl.env import ACTION_NAMES, CLINICAL_OBSERVATION_NAMES
from openhumsim_rl.units import OBSERVATION_UNITS


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "RELEASE_v0.22.json"


@pytest.fixture
def dashboard_manifest_http() -> Iterator[tuple[str, int]]:
    old_store = DashboardHandler.store
    old_allowed_hosts = DashboardHandler.allowed_hosts
    DashboardHandler.store = SessionStore()
    DashboardHandler.allowed_hosts = frozenset({"127.0.0.1", "localhost"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        DashboardHandler.store = old_store
        DashboardHandler.allowed_hosts = old_allowed_hosts


def _http_json(
    server: tuple[str, int],
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    connection = HTTPConnection(*server, timeout=10.0)
    headers = {"Content-Type": "application/json"}
    if session_id is not None:
        headers[SESSION_HEADER] = session_id
    body = None if payload is None else json.dumps(payload, allow_nan=False)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        decoded = json.loads(response.read().decode("utf-8"))
        return (
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            decoded,
        )
    finally:
        connection.close()


def _manifest_without_self_hash(manifest: dict[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    value.pop("manifest_sha256")
    value.pop("manifest_hash_format")
    return value


def test_manifest_is_deterministic_resolved_and_json_finite() -> None:
    first = DashboardSession("baseline", 42).manifest_snapshot()
    second = DashboardSession("baseline", 42).manifest_snapshot()

    assert first == second
    assert first["schema"] == EXPERIMENT_MANIFEST_SCHEMA
    assert __version__ == "0.22.0"
    assert first["model"] == {
        "package": "openhumsim_rl",
        "environment": "HumanHomeostasisEnv",
        "version": __version__,
        "research_only": True,
        "clinical_use_supported": False,
    }
    assert first["state"]["schema_version"] == "0.22"
    assert first["reward"]["profile"] == "homeostasis_v0.21"
    assert first["profiles"] == {
        "observation": "clinical",
        "measurement": "realistic",
        "info": "debug",
    }
    assert first["timebase"] == {
        "agent_step_min": 5.0,
        "integration_step_min": 0.25,
        "episode_minutes": 720.0,
    }
    assert first["config"]["human"] == asdict(HumanConfig())
    assert first["config"]["clinical_measurement"] == asdict(
        ClinicalMeasurementConfig()
    )
    assert first["runtime"]["python"]["implementation"]
    assert first["runtime"]["python"]["version"]
    assert first["runtime"]["numpy_version"]
    assert first["manifest_sha256"] == _canonical_sha256(
        _manifest_without_self_hash(first)
    )
    json.dumps(first, allow_nan=False)


def test_manifest_locks_exact_release_interfaces_and_54_item_catalog() -> None:
    manifest = DashboardSession("baseline", 42).manifest_snapshot()
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    observation = manifest["interfaces"]["observation"]
    action = manifest["interfaces"]["action"]

    assert observation["ordered_names"] == list(CLINICAL_OBSERVATION_NAMES)
    assert observation["count"] == release["clinical_observation_count"] == 54
    assert observation["sha256"] == release["clinical_observation_sha256"]
    assert observation["release_declared_sha256"] == observation["sha256"]
    assert observation["matches_release"] is True
    assert action["ordered_names"] == list(ACTION_NAMES)
    assert action["count"] == release["action_count"] == 8
    assert action["sha256"] == release["action_sha256"]
    assert action["release_declared_sha256"] == action["sha256"]
    assert action["matches_release"] is True

    catalog = manifest["observation_catalog"]
    assert len(catalog) == 54
    assert [item["order"] for item in catalog] == list(range(54))
    assert [item["name"] for item in catalog] == list(
        CLINICAL_OBSERVATION_NAMES
    )
    assert [item["unit"] for item in catalog] == [
        OBSERVATION_UNITS[name] for name in CLINICAL_OBSERVATION_NAMES
    ]
    assert all(item["normalization"]["scale"] > 0.0 for item in catalog)
    scopes = [item["measurement_diagnostics"]["scope"] for item in catalog]
    assert scopes.count("channel") == 47
    assert scopes.count("group") == 1
    assert scopes.count("aggregate_age_observation") == 5
    assert scopes.count("derived") == 1
    json.dumps(catalog, allow_nan=False)


def test_manifest_records_shared_rng_and_source_bound_scenario_semantics() -> None:
    baseline = DashboardSession("baseline", 19).manifest_snapshot()
    challenge = DashboardSession("airway_obstruction", 19).manifest_snapshot()
    other_seed = DashboardSession("baseline", 20).manifest_snapshot()

    randomness = baseline["randomness"]
    assert randomness["reset_seed"] == 19
    assert randomness["physiology_and_measurement"] == {
        "seed": 19,
        "stream": "shared numpy Generator owned by the environment",
        "independent_streams": False,
    }
    assert randomness["action_space"]["seed"] == 20
    assert baseline["scenario"]["procedural_state_challenge"] is False
    assert challenge["scenario"]["id"] == "airway_obstruction"
    assert challenge["scenario"]["label"] == "Obturacja dróg oddechowych"
    assert challenge["scenario"]["group"] == "Mechanika"
    assert challenge["scenario"]["procedural_state_challenge"] is True
    assert challenge["scenario"]["clinical_diagnosis_claim"] is False
    assert challenge["scenario"]["source_binding"]["source_id"] == (
        "src/openhumsim_rl/env.py"
    )
    assert challenge["scenario"]["source_binding"]["implementation"] == (
        "HumanHomeostasisEnv._apply_scenario"
    )
    assert baseline["manifest_sha256"] != challenge["manifest_sha256"]
    assert baseline["manifest_sha256"] != other_seed["manifest_sha256"]


def test_source_fingerprint_is_recomputable_and_contains_no_host_paths_or_secrets() -> None:
    manifest = DashboardSession("baseline", 42).manifest_snapshot()
    source = manifest["source_fingerprint"]
    files = source["source_files"]

    assert source["source_file_count"] == len(files) >= 2
    assert files == sorted(files, key=lambda item: item["source_id"])
    assert source["sha256"] == _canonical_sha256(files)
    for item in files:
        source_id = item["source_id"]
        assert not Path(source_id).is_absolute()
        assert ".." not in Path(source_id).parts
        assert item["content_sha256"] == sha256(
            (ROOT / source_id).read_bytes()
        ).hexdigest()

    release_lock = source["release_manifest"]
    assert release_lock["source_id"] == "RELEASE_v0.22.json"
    assert release_lock["available"] is True
    assert release_lock["content_sha256"] == sha256(RELEASE.read_bytes()).hexdigest()

    serialized = json.dumps(manifest, allow_nan=False, sort_keys=True)
    assert str(ROOT) not in serialized
    assert str(Path.home()) not in serialized
    lowered = serialized.lower()
    for forbidden in ("password", "api_key", "access_token", "private_key"):
        assert forbidden not in lowered


def test_http_session_and_history_expose_one_manifest_outside_frames(
    dashboard_manifest_http: tuple[str, int],
) -> None:
    status, headers, created = _http_json(
        dashboard_manifest_http,
        "POST",
        "/api/session",
        payload={"scenario": "fasting", "seed": 81},
    )
    assert status == 201
    session_id = headers[SESSION_HEADER.lower()]
    manifest = created["experiment_manifest"]
    assert manifest["schema"] == EXPERIMENT_MANIFEST_SCHEMA
    assert "experiment_manifest" not in created["frame"]
    assert "experiment_manifest" not in created["history"]
    assert all("experiment_manifest" not in frame for frame in created["history"]["frames"])

    status, _, history = _http_json(
        dashboard_manifest_http,
        "GET",
        "/api/history",
        session_id=session_id,
    )
    assert status == 200
    assert history["experiment_manifest"] == manifest
    assert all("experiment_manifest" not in frame for frame in history["frames"])
    json.dumps(created, allow_nan=False)
    json.dumps(history, allow_nan=False)
