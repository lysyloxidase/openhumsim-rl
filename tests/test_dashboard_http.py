from __future__ import annotations

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from typing import Any, Iterator
from uuid import UUID, uuid4

import pytest

import examples.dashboard_server as dashboard_server
from examples.dashboard_server import (
    MAX_REQUEST_BYTES,
    SESSION_HEADER,
    DashboardHandler,
    SessionNotFoundError,
    SessionStore,
)


_NO_BODY = object()


@pytest.fixture
def dashboard_http() -> Iterator[tuple[str, int, str]]:
    class TestDashboardHandler(DashboardHandler):
        store = SessionStore()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), TestDashboardHandler)
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield str(host), int(port), f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def _request(
    server: tuple[str, int, str],
    method: str,
    path: str,
    *,
    payload: object = _NO_BODY,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    host, port, _ = server
    request_headers = dict(headers or {})
    if payload is not _NO_BODY:
        if raw_body is not None:
            raise AssertionError("payload and raw_body are mutually exclusive")
        raw_body = json.dumps(payload, allow_nan=False).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    connection = HTTPConnection(host, port, timeout=10.0)
    try:
        connection.request(method, path, body=raw_body, headers=request_headers)
        response = connection.getresponse()
        response_body = response.read()
        decoded = json.loads(response_body.decode("utf-8"))
        return (
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            decoded,
        )
    finally:
        connection.close()


def _create_session(
    server: tuple[str, int, str],
    *,
    scenario: str = "baseline",
    seed: int = 42,
) -> tuple[str, dict[str, Any]]:
    status, headers, body = _request(
        server,
        "POST",
        "/api/session",
        payload={"scenario": scenario, "seed": seed},
    )
    assert status == 201
    session_id = body["session_id"]
    UUID(session_id)
    assert headers[SESSION_HEADER.lower()] == session_id
    assert body["schema"] == "openhumsim.dashboard.session.v1"
    assert body["frame"]["schema"] == "openhumsim.dashboard.v2"
    assert body["history"] == {
        "schema": "openhumsim.dashboard.history.v2",
        "complete_from_reset": True,
        "run_id": body["frame"]["run_id"],
        "revision": 0,
        "frames": [body["frame"]],
    }
    return session_id, body


def _step(
    server: tuple[str, int, str],
    session_id: str,
    frame: dict[str, Any],
    action: dict[str, float] | None = None,
    request_id: str | None = None,
) -> tuple[int, dict[str, Any]]:
    status, _, body = _request(
        server,
        "POST",
        "/api/step",
        payload={
            "action": action or {},
            "expected_run_id": frame["run_id"],
            "expected_revision": frame["revision"],
            "request_id": request_id or str(uuid4()),
        },
        headers={SESSION_HEADER: session_id},
    )
    return status, body


def test_http_sessions_are_isolated(dashboard_http: tuple[str, int, str]) -> None:
    first_id, first = _create_session(dashboard_http, seed=11)
    second_id, second = _create_session(dashboard_http, seed=22)

    assert first_id != second_id
    assert first["frame"]["run_id"] != second["frame"]["run_id"]

    status, stepped = _step(
        dashboard_http,
        first_id,
        first["frame"],
        {"exercise": 0.2},
    )
    assert status == 200
    assert stepped["revision"] == 1
    assert stepped["time_min"] == 5.0

    first_status, _, first_state = _request(
        dashboard_http,
        "GET",
        "/api/state",
        headers={SESSION_HEADER: first_id},
    )
    second_status, _, second_state = _request(
        dashboard_http,
        "GET",
        "/api/state",
        headers={SESSION_HEADER: second_id},
    )
    assert first_status == second_status == 200
    assert first_state["revision"] == 1
    assert second_state["revision"] == 0
    assert second_state["time_min"] == 0.0
    assert second_state == second["frame"]


def test_stale_step_and_reset_are_conflicts(
    dashboard_http: tuple[str, int, str],
) -> None:
    session_id, created = _create_session(dashboard_http)
    initial = created["frame"]

    status, current = _step(dashboard_http, session_id, initial)
    assert status == 200
    assert current["revision"] == 1

    stale_status, stale_error = _step(dashboard_http, session_id, initial)
    assert stale_status == 409
    assert stale_error["status"] == 409
    assert "stale dashboard state" in stale_error["error"]

    history_status, _, unchanged_history = _request(
        dashboard_http,
        "GET",
        "/api/history",
        headers={SESSION_HEADER: session_id},
    )
    assert history_status == 200
    assert unchanged_history["revision"] == 1
    assert len(unchanged_history["frames"]) == 2

    stale_reset_status, _, _ = _request(
        dashboard_http,
        "POST",
        "/api/reset",
        payload={
            "scenario": "baseline",
            "seed": 77,
            "expected_run_id": initial["run_id"],
            "expected_revision": initial["revision"],
        },
        headers={SESSION_HEADER: session_id},
    )
    assert stale_reset_status == 409

    reset_status, _, reset = _request(
        dashboard_http,
        "POST",
        "/api/reset",
        payload={
            "scenario": "baseline",
            "seed": 77,
            "expected_run_id": current["run_id"],
            "expected_revision": current["revision"],
        },
        headers={SESSION_HEADER: session_id},
    )
    assert reset_status == 200
    assert reset["run_id"] != current["run_id"]
    assert reset["revision"] == 0
    assert reset["seed"] == 77

    history_status, _, history = _request(
        dashboard_http,
        "GET",
        "/api/history",
        headers={SESSION_HEADER: session_id},
    )
    assert history_status == 200
    assert history["complete_from_reset"] is True
    assert history["run_id"] == reset["run_id"]
    assert history["revision"] == 0
    assert history["frames"] == [reset]


def test_history_is_complete_from_t0_and_json_finite(
    dashboard_http: tuple[str, int, str],
) -> None:
    session_id, created = _create_session(dashboard_http)
    status, first = _step(
        dashboard_http,
        session_id,
        created["frame"],
        {"oral_water": 0.1},
    )
    assert status == 200
    status, second = _step(
        dashboard_http,
        session_id,
        first,
        {"oxygen": 0.25},
    )
    assert status == 200

    status, headers, history = _request(
        dashboard_http,
        "GET",
        "/api/history",
        headers={SESSION_HEADER: session_id},
    )
    assert status == 200
    assert headers[SESSION_HEADER.lower()] == session_id
    assert history["schema"] == "openhumsim.dashboard.history.v2"
    assert history["complete_from_reset"] is True
    assert history["run_id"] == second["run_id"]
    assert history["revision"] == 2
    assert [frame["revision"] for frame in history["frames"]] == [0, 1, 2]
    assert [frame["time_min"] for frame in history["frames"]] == [0.0, 5.0, 10.0]
    assert {frame["run_id"] for frame in history["frames"]} == {second["run_id"]}
    assert {frame["seed"] for frame in history["frames"]} == {42}
    assert history["frames"][0]["action"] is None
    assert history["frames"][1]["action"]["oral_water"] == pytest.approx(0.1)
    assert history["frames"][2]["action"]["oxygen"] == pytest.approx(0.25)
    assert history["frames"][1]["intervention"] is not None
    assert history["frames"][2]["intervention"] is not None
    assert "chloride_mass_balance_error_mmol" in second["integrity"]
    json.dumps(history, allow_nan=False)


def test_step_request_id_is_idempotent(
    dashboard_http: tuple[str, int, str],
) -> None:
    session_id, created = _create_session(dashboard_http)
    request_id = str(uuid4())

    status, first = _step(
        dashboard_http,
        session_id,
        created["frame"],
        {"oral_water": 0.25},
        request_id=request_id,
    )
    assert status == 200
    assert first["request_id"] == request_id

    retry_status, retry = _step(
        dashboard_http,
        session_id,
        created["frame"],
        {"oral_water": 0.25},
        request_id=request_id,
    )
    assert retry_status == 200
    assert retry == first

    changed_status, changed = _step(
        dashboard_http,
        session_id,
        created["frame"],
        {"oral_water": 0.5},
        request_id=request_id,
    )
    assert changed_status == 400
    assert "request_id was already used" in changed["error"]

    _, _, history = _request(
        dashboard_http,
        "GET",
        "/api/history",
        headers={SESSION_HEADER: session_id},
    )
    assert len(history["frames"]) == 2
    assert history["revision"] == 1


def test_session_store_is_lru_bounded_and_expires_idle_sessions() -> None:
    now = [0.0]
    store = SessionStore(max_sessions=2, ttl_seconds=10.0, clock=lambda: now[0])
    first_id, first = store.create(seed=1)
    second_id, _ = store.create(seed=2)
    now[0] = 1.0
    assert store.get(first_id) is first
    now[0] = 2.0
    third_id, _ = store.create(seed=3)

    with pytest.raises(SessionNotFoundError):
        store.get(second_id)
    assert store.get(first_id) is first
    now[0] = 13.0
    with pytest.raises(SessionNotFoundError, match="expired"):
        store.get(first_id)
    with pytest.raises(SessionNotFoundError, match="expired"):
        store.get(third_id)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_are_rejected(
    dashboard_http: tuple[str, int, str],
    constant: str,
) -> None:
    status, _, body = _request(
        dashboard_http,
        "POST",
        "/api/session",
        raw_body=f'{{"seed":{constant}}}'.encode("ascii"),
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert body["status"] == 400
    assert "non-finite JSON constant" in body["error"]


def test_http_input_contract_is_strict(
    dashboard_http: tuple[str, int, str],
) -> None:
    _, _, origin = dashboard_http
    _, port, _ = dashboard_http

    status, _, body = _request(
        dashboard_http,
        "GET",
        "/api/meta",
        headers={
            "Host": f"evil.example:{port}",
            "Origin": f"http://evil.example:{port}",
        },
    )
    assert status == 403
    assert "Host is not allowed" in body["error"]

    status, _, _ = _request(
        dashboard_http,
        "GET",
        "/api/meta",
        headers={
            "Host": f"localhost:{port}",
            "Origin": f"http://localhost:{port}",
        },
    )
    assert status == 200

    status, _, _ = _request(
        dashboard_http,
        "POST",
        "/api/session",
        raw_body=b"{}",
    )
    assert status == 415

    status, _, body = _request(
        dashboard_http,
        "POST",
        "/api/session",
        payload={"unknown": True},
    )
    assert status == 400
    assert "unknown top-level JSON keys" in body["error"]

    status, _, _ = _request(
        dashboard_http,
        "POST",
        "/api/session",
        payload={},
        headers={"Origin": "http://example.invalid"},
    )
    assert status == 403

    status, _, _ = _request(
        dashboard_http,
        "GET",
        "/api/meta",
        headers={"Origin": "https://example.invalid"},
    )
    assert status == 403

    status, _, _ = _request(
        dashboard_http,
        "POST",
        "/api/session",
        payload={},
        headers={"Origin": origin},
    )
    assert status == 201

    status, _, _ = _request(
        dashboard_http,
        "POST",
        "/api/session",
        raw_body=b" " * (MAX_REQUEST_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )
    assert status == 400

    status, _, _ = _request(dashboard_http, "GET", "/api/state")
    assert status == 400

    status, _, body = _request(
        dashboard_http,
        "POST",
        "/api/session",
        raw_body=b'{"seed":1e999}',
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert "non-finite JSON number" in body["error"]


def test_step_requires_preconditions_and_known_top_level_keys(
    dashboard_http: tuple[str, int, str],
) -> None:
    session_id, created = _create_session(dashboard_http)

    status, _, body = _request(
        dashboard_http,
        "POST",
        "/api/step",
        payload={"action": {}},
        headers={SESSION_HEADER: session_id},
    )
    assert status == 400
    assert "missing required JSON keys" in body["error"]

    status, _, body = _request(
        dashboard_http,
        "POST",
        "/api/step",
        payload={
            "action": {},
            "expected_run_id": created["frame"]["run_id"],
            "expected_revision": 0,
            "request_id": str(uuid4()),
            "extra": None,
        },
        headers={SESSION_HEADER: session_id},
    )
    assert status == 400
    assert "unknown top-level JSON keys" in body["error"]

    status, _, body = _request(
        dashboard_http,
        "POST",
        "/api/step",
        payload={
            "action": {"exercise": "0.2"},
            "expected_run_id": created["frame"]["run_id"],
            "expected_revision": 0,
            "request_id": str(uuid4()),
        },
        headers={SESSION_HEADER: session_id},
    )
    assert status == 400
    assert "action values must be JSON numbers" in body["error"]


def test_meta_missing_artifacts_are_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        dashboard_server,
        "RELEASE_MANIFEST",
        tmp_path / "missing-release.json",
    )
    monkeypatch.setattr(
        dashboard_server,
        "VALIDATION_RESULTS",
        tmp_path / "missing-validation.json",
    )
    monkeypatch.setattr(
        dashboard_server,
        "CI_EVIDENCE",
        tmp_path / "missing-ci-evidence.json",
    )

    meta = dashboard_server.dashboard_meta()

    assert meta["schema"] == "openhumsim.dashboard.meta.v2"
    assert meta["availability"] == {
        "release_manifest": False,
        "validation_results": False,
        "ci_evidence": False,
    }
    assert meta["validation"] is None
    assert meta["full_test_suite"] is None
    assert meta["supported_python_ci"] is None
    assert meta["observation_contract"] == {
        "clinical_count": None,
        "full_count": None,
        "state_schema_version": None,
        "reward_profile": None,
    }
