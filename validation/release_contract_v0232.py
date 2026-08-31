"""Side-effect-free source and check contract for the v0.23.2 release gate."""

from __future__ import annotations

from pathlib import Path


VERSION = "0.23.2"
RESULTS_RELATIVE_PATH = "validation/validation_results_v0.23.2.json"
STATE_SCHEMA_VERSION = "0.22"
DEBUG_REWARD_PROFILE = "latent_research_v0.23"
BENCHMARK_REWARD_PROFILE = "observable_benchmark_v0.23"
CLINICAL_OBSERVATION_COUNT = 54
CLINICAL_OBSERVATION_SHA256 = (
    "56770d5ea4d5ed4f81f98042bb4dcba7d0e40bfc73d109e5bdc3c2c5f5647de8"
)
FULL_OBSERVATION_COUNT = 138
FULL_OBSERVATION_SHA256 = (
    "cf544ac7d1fdae6cf7b52e4320ec091409094ed43fd7bce90f4d496a854f813a"
)
ACTION_COUNT = 8
ACTION_SHA256 = (
    "9bc31bce6639bed396a5406518695c988e0d4f5d8740e893cec9ecfdfb985dfb"
)

# A contract may use several pytest targets while remaining one top-level gate
# check. Keep this data free of imports from the simulator so the release
# verifier can inspect it without executing the gate.
PYTEST_CONTRACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "patch_release_regressions_v0231",
        ("tests/test_patch_regressions_v0231.py",),
    ),
    (
        "solver_and_config_regressions",
        ("tests/test_solver_config_hardening.py",),
    ),
    (
        "mechanics_continuity_and_total_peep",
        ("tests/test_mechanics_continuity_v023.py",),
    ),
    (
        "temporal_reward_and_measurement_contracts",
        ("tests/test_temporal_contract_v023.py",),
    ),
    (
        "transactional_step_contracts",
        ("tests/test_step_transaction_v023.py",),
    ),
    (
        "policy_manifest_and_checkpoint_contracts",
        ("tests/test_policy_manifest_v023.py",),
    ),
    (
        "environment_snapshot_contracts",
        ("tests/test_environment_snapshot_v023.py",),
    ),
    (
        "observation_history_and_baseline_harness_contracts",
        ("tests/test_history_wrapper_v023.py",),
    ),
    (
        "biophysics_and_result_only_pulmonary_contracts",
        (
            "tests/test_biophysics_regressions.py",
            (
                "tests/test_physics_regressions_v021.py::"
                "test_hpv_fixed_point_advances_kinetics_only_once"
            ),
        ),
    ),
    (
        "cli_policy_metadata_and_packaged_dashboard_contracts",
        (
            "tests/test_cli_contract.py",
            "tests/test_config_manifest_v022.py",
            "tests/test_dashboard_http.py",
            (
                "tests/test_dashboard.py::"
                "test_dashboard_is_self_contained_and_has_research_boundary"
            ),
            (
                "tests/test_dashboard.py::"
                "test_packaged_dashboard_is_canonical_and_legacy_import_is_compatible"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_session_exposes_measurements_separately_from_debug_truth"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_step_uses_real_environment_and_validates_actions"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_step_rolls_back_after_post_transition_frame_failure"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_reset_rolls_back_after_frame_failure"
            ),
            (
                "tests/test_dashboard.py::"
                "test_dashboard_documentation_targets_existing_files"
            ),
        ),
    ),
    (
        "release_evidence_verifier_contracts",
        ("tests/test_release_evidence.py",),
    ),
)

EXPLICIT_SOURCE_RELATIVE_PATHS: tuple[str, ...] = (
    "validation/run_validation_v0232.py",
    "validation/historical_version_guard.py",
    "validation/release_contract_v0232.py",
    "validation/verify_release_evidence.py",
    "examples/train_ppo.py",
    "examples/dashboard_server.py",
    "src/openhumsim_rl/dashboard/index.html",
    "dashboard/index.html",
    "validation/rl_benchmark_v0.23.py",
    "README.md",
    "docs/dashboard.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    "pyproject.toml",
)

EXPECTED_CHECK_NAMES: tuple[str, ...] = (
    "exact_release_version",
    "state_schema_and_reward_profiles",
    *(name for name, _ in PYTEST_CONTRACTS),
    "source_snapshot_stable_during_gate",
)

EVIDENCE_ONLY_PATHS: tuple[str, ...] = (
    "CI_EVIDENCE.json",
    "RELEASE_NOTES_v0.23.2.md",
    "RELEASE_v0.23.2.json",
    "VALIDATION_AUDIT_v0.23.2.md",
    RESULTS_RELATIVE_PATH,
)


def pytest_source_relative_paths() -> tuple[str, ...]:
    """Return the sorted unique files referenced by pytest targets."""

    return tuple(
        sorted(
            {
                target.split("::", maxsplit=1)[0]
                for _, targets in PYTEST_CONTRACTS
                for target in targets
            }
        )
    )


def release_source_paths(root: Path) -> tuple[Path, ...]:
    """Resolve the complete v0.23.2 source-fingerprint subject set."""

    paths = list((root / "src" / "openhumsim_rl").glob("**/*.py"))
    paths.extend(root / path for path in pytest_source_relative_paths())
    paths.extend(root / path for path in EXPLICIT_SOURCE_RELATIVE_PATHS)
    return tuple(
        sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())
    )
