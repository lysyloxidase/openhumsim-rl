from __future__ import annotations

import numpy as np

from openhumsim_rl import HumanHomeostasisEnv
from openhumsim_rl.measurement import ClinicalMeasurementConfig, ClinicalMeasurementModel
from openhumsim_rl.population import (
    DEFAULT_PARAMETER_SPECS,
    LockedCohortManifest,
    correlated_latin_hypercube,
    sample_virtual_cohort,
)

ZERO = np.zeros(8, dtype=np.float32)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def test_realistic_measurements_are_seed_reproducible_but_not_truth_identical():
    a = HumanHomeostasisEnv(measurement_profile="realistic")
    b = HumanHomeostasisEnv(measurement_profile="realistic")
    oa, ia = a.reset(seed=91)
    ob, ib = b.reset(seed=91)
    assert np.array_equal(oa, ob)
    oa, *_ = a.step(ZERO)
    ob, *_ = b.step(ZERO)
    assert np.array_equal(oa, ob)

    ideal = HumanHomeostasisEnv(measurement_profile="ideal")
    oi, _ = ideal.reset(seed=91)
    assert not np.array_equal(oa, oi)


def test_abg_is_sampled_and_delivered_with_delay_not_ground_truth_leakage():
    env = HumanHomeostasisEnv(measurement_profile="realistic")
    env.reset(seed=12)
    m = env.measurement_model
    assert m is not None
    initial = m.measurement_value("paco2_mmHg", env.state)
    env.state.paco2_mmHg = 80.0

    for minute in (5, 10, 15, 20, 25, 30, 35):
        m.advance(env.state, time_min=float(minute), dt_min=5.0, rng=env.np_random)
    assert m.measurement_value("paco2_mmHg", env.state) == initial
    assert m.group_ages()["blood_gas_measurement_age_min"] >= 35.0

    m.advance(env.state, time_min=40.0, dt_min=5.0, rng=env.np_random)
    reported = m.measurement_value("paco2_mmHg", env.state)
    assert reported > 70.0
    # Sample occurred at t=30 and became visible at the next 5-min decision after
    # the configured 7-min analytical delay, so its information age is 10 min.
    assert abs(m.group_ages()["blood_gas_measurement_age_min"] - 10.0) < 1e-12


def test_monitor_dropout_holds_last_value_and_increases_age():
    cfg = ClinicalMeasurementConfig(monitor_dropout_probability=1.0, cgm_dropout_probability=1.0)
    env = HumanHomeostasisEnv(measurement_profile="realistic", measurement_config=cfg)
    env.reset(seed=4)
    m = env.measurement_model
    assert m is not None
    old = m.measurement_value("heart_rate_bpm", env.state)
    env.state.heart_rate_bpm = 160.0
    m.advance(env.state, time_min=5.0, dt_min=5.0, rng=env.np_random)
    assert m.measurement_value("heart_rate_bpm", env.state) == old
    assert m.group_ages()["monitor_measurement_age_min"] >= 5.0
    assert m.diagnostics()["groups"]["monitor"]["dropped"] > 0


def test_cgm_channel_has_interstitial_lag_and_age_metadata():
    cfg = ClinicalMeasurementConfig(cgm_dropout_probability=0.0, cgm_relative_noise_sd=0.0)
    env = HumanHomeostasisEnv(measurement_profile="realistic", measurement_config=cfg)
    env.reset(seed=5)
    m = env.measurement_model
    assert m is not None
    g0 = m.measurement_value("sensor_glucose_mg_dl", env.state)
    env.state.glucose_mg_dl = 200.0
    m.advance(env.state, time_min=5.0, dt_min=5.0, rng=env.np_random)
    g1 = m.measurement_value("sensor_glucose_mg_dl", env.state)
    assert g0 < g1 < 200.0
    assert m.group_ages()["cgm_measurement_age_min"] == 0.0


def test_correlated_lhs_preserves_marginals_and_requested_rank_direction():
    n = 512
    U = correlated_latin_hypercube(n, seed=123)
    assert U.shape == (n, len(DEFAULT_PARAMETER_SPECS))
    assert np.all((U > 0.0) & (U < 1.0))

    names = [s.name for s in DEFAULT_PARAMETER_SPECS]
    def col(name):
        return U[:, names.index(name)]

    assert _spearman(col("body_weight_kg"), col("tbw_fraction")) < -0.25
    assert _spearman(col("tbw_fraction"), col("ecf_fraction")) > 0.35
    assert _spearman(col("body_weight_kg"), col("blood_volume_ml_per_kg")) < -0.10

    # Exact LHS property: one point in each 1/n marginal stratum.
    for j in range(U.shape[1]):
        strata = np.floor(U[:, j] * n).astype(int)
        assert len(np.unique(strata)) == n


def test_virtual_cohort_is_correlated_by_default_and_reproducible():
    a = sample_virtual_cohort(64, seed=99)
    b = sample_virtual_cohort(64, seed=99)
    assert [x.latent for x in a] == [x.latent for x in b]
    assert all(x.config.total_body_water_baseline_l > x.config.ecf_volume_baseline_l for x in a)


def test_locked_validation_manifest_is_disjoint_and_tamper_evident():
    ids = [f"SUBJ-{i:03d}" for i in range(30)]
    manifest = LockedCohortManifest.create(ids, "external-cohort-A", seed=2020, dataset_fingerprint="sha256:abc")
    manifest.assert_no_leakage()
    assert manifest.verify_lock()
    assert not (set(manifest.calibration_subject_ids) & set(manifest.validation_subject_ids))

    tampered = LockedCohortManifest(
        manifest.dataset_name,
        manifest.dataset_fingerprint,
        manifest.calibration_subject_ids,
        manifest.validation_subject_ids[:-1] + ("SUBJ-999",),
        manifest.split_seed,
        manifest.validation_lock_sha256,
    )
    assert not tampered.verify_lock()


def test_full_profile_stays_ideal_ground_truth_debug_state():
    full = HumanHomeostasisEnv(observation_profile="full")
    _, info = full.reset(seed=7)
    assert info["measurement_profile"] == "ideal"
    try:
        HumanHomeostasisEnv(observation_profile="full", measurement_profile="realistic")
    except ValueError:
        pass
    else:
        raise AssertionError("full ground-truth profile must reject realistic measurement mode")


def test_benchmark_info_profile_hides_mechanistic_truth():
    env = HumanHomeostasisEnv(info_profile="benchmark")
    _, info = env.reset(seed=8)
    assert set(info) == {
        "observation_names", "observation_profile", "measurement_profile",
        "info_profile", "action_names", "gymnasium_installed",
        "environment_semantics",
    }
    assert {
        "state", "blood_gas", "blood_gas_carbon", "pbpk", "measurement",
        "reward_terms", "scenario", "time_min",
    }.isdisjoint(info)
    assert info["measurement_profile"] == "realistic"
