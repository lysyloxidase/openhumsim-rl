from __future__ import annotations

import json
from pathlib import Path

import pytest

from openhumsim_rl import cli
from openhumsim_rl import dashboard_server


def test_validate_outside_source_checkout_fails_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installed_module = tmp_path / "site-packages" / "openhumsim_rl" / "cli.py"
    monkeypatch.setattr(cli, "__file__", str(installed_module))
    monkeypatch.chdir(tmp_path)

    result = cli.main(["validate", "--scientific-only"])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "requires a source checkout" in captured.err
    assert "tests/ and validation/" in captured.err
    assert "github.com/lysyloxidase/openhumsim-rl" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["demo", "--minutes", "-1"],
        ["demo", "--minutes", "nan"],
        ["demo", "--minutes", "inf"],
        ["measurement-demo", "--minutes", "0"],
        ["cgm-demo", "--step-min", "0"],
        ["cgm-demo", "--lag-min", "-0.5"],
        ["population-demo", "--n", "0"],
        ["population-demo", "--n", "-1"],
        ["demo", "--seed", "-1"],
        ["dashboard", "--port", "0"],
    ],
)
def test_invalid_numeric_cli_arguments_are_usage_errors_without_tracebacks(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["demo", "measurement-demo"])
def test_unknown_scenario_is_an_actionable_usage_error(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [command, "--scenario", "not-a-scenario", "--minutes", "5"]
        )

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "Unknown scenario 'not-a-scenario'" in captured.err
    assert "choose from" in captured.err
    assert "Traceback" not in captured.err


def test_run_demo_rejects_invalid_minutes_for_direct_callers() -> None:
    with pytest.raises(ValueError, match="minutes must be finite"):
        cli.run_demo("baseline", minutes=-1.0, seed=42)


def test_demo_commands_honor_a_non_multiple_requested_duration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    direct = cli.run_demo("baseline", minutes=6.0, seed=42)
    assert direct["requested_minutes"] == 6.0
    assert direct["simulated_minutes"] == pytest.approx(6.0)

    assert cli.main(
        ["measurement-demo", "--scenario", "baseline", "--minutes", "6", "--seed", "42"]
    ) == 0
    measured = json.loads(capsys.readouterr().out)
    assert measured["time_min"] == pytest.approx(6.0)


def test_cli_seed_validator_preserves_large_nonnegative_seeds() -> None:
    large_seed = 2**64 + 17

    assert cli._seed(str(large_seed)) == large_seed


def test_dashboard_command_delegates_validated_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_serve_dashboard(**kwargs: object) -> int:
        received.update(kwargs)
        return 17

    monkeypatch.setattr(
        dashboard_server,
        "serve_dashboard",
        fake_serve_dashboard,
    )

    result = cli.main(
        [
            "dashboard",
            "--host",
            "127.0.0.1",
            "--allowed-host",
            "dashboard.local",
            "--port",
            "9876",
            "--no-open",
        ]
    )

    assert result == 17
    assert received == {
        "host": "127.0.0.1",
        "allowed_hosts": ("dashboard.local",),
        "port": 9876,
        "open_browser": False,
    }
