from __future__ import annotations


def require_exact_version(expected: str) -> None:
    """Refuse to overwrite a historical artifact with a different model."""
    from openhumsim_rl import __version__

    if __version__ != expected:
        raise RuntimeError(
            f"This is an archival {expected} script, but openhumsim-rl "
            f"{__version__} is loaded. Check out/install the exact historical "
            "release before reproducing this artifact; refusing to mislabel or "
            "overwrite it with outputs from a different model."
        )
