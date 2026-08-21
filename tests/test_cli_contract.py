from __future__ import annotations

from pathlib import Path

import pytest

from openhumsim_rl import cli


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
