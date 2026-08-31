"""Compatibility entry point for the packaged dashboard server.

New code should use ``openhumsim dashboard`` or import
``openhumsim_rl.dashboard_server``. The module alias preserves existing
imports and monkeypatch-based integrations from source checkouts.
"""

from __future__ import annotations

import sys

from openhumsim_rl import dashboard_server as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
