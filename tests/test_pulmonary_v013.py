from dataclasses import replace
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv

ZERO = np.zeros(8, dtype=np.float32)


def _reset(scenario="baseline", seed=13, config=None):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    _, info = env.reset(seed=seed)
    return env, info


def test_v013_baseline_gas_exchange_is_physically_plausible():
    _, info = _reset()
    s = info["state"]
    assert 85.0 <= s["pao2_mmHg"] <= 105.0
    assert 94.0 <= s["spo2_pct"] <= 100.0
    assert 0.0 <= s["pulmonary_aa_gradient_mmHg"] <= 15.0
    assert 0.20 <= s["pulmonary_enghoff_dead_space_fraction"] <= 0.45
    assert 0.90 <= s["pulmonary_diffusion_equilibration_fraction"] <= 1.0


def test_v013_vq_mismatch_lowers_pao2_and_widens_aa_gradient():
    _, base = _reset("baseline", seed=20)
    _, bad = _reset("vq_mismatch", seed=20)
    assert bad["state"]["pao2_mmHg"] < base["state"]["pao2_mmHg"] - 15.0
    assert bad["state"]["pulmonary_aa_gradient_mmHg"] > base["state"]["pulmonary_aa_gradient_mmHg"] + 15.0
    assert bad["state"]["pulmonary_enghoff_dead_space_fraction"] > base["state"]["pulmonary_enghoff_dead_space_fraction"]


def test_v013_true_shunt_is_less_oxygen_responsive_than_vq_mismatch():
    gains = {}
    for scenario in ("vq_mismatch", "pulmonary_shunt"):
        room = HumanHomeostasisEnv(scenario=scenario); room.reset(seed=21)
        oxy = HumanHomeostasisEnv(scenario=scenario); oxy.reset(seed=21)
        a = ZERO.copy(); a[4] = 1.0
        _, _, _, _, ir = room.step(ZERO)
        _, _, _, _, io = oxy.step(a)
        gains[scenario] = io["state"]["pao2_mmHg"] - ir["state"]["pao2_mmHg"]
    assert gains["vq_mismatch"] > 2.0 * gains["pulmonary_shunt"]


def test_v013_diffusion_limitation_reduces_equilibration_and_pao2():
    _, base = _reset("baseline", seed=22)
    _, diff = _reset("diffusion_limitation", seed=22)
    assert diff["state"]["pulmonary_diffusion_equilibration_fraction"] < base["state"]["pulmonary_diffusion_equilibration_fraction"] - 0.10
    assert diff["state"]["pao2_mmHg"] < base["state"]["pao2_mmHg"] - 8.0


def test_v013_exercise_shortens_capillary_transit_and_widens_aa():
    env, rest = _reset("baseline", seed=23)
    a = ZERO.copy(); a[2] = 1.0
    ex = None
    for _ in range(6):
        _, _, terminated, _, ex = env.step(a)
        assert not terminated
    assert ex["state"]["pulmonary_capillary_transit_time_s"] < rest["state"]["pulmonary_capillary_transit_time_s"]
    assert ex["state"]["pulmonary_aa_gradient_mmHg"] > rest["state"]["pulmonary_aa_gradient_mmHg"]


def test_v013_shunt_monotonically_reduces_o2_content():
    values = []
    for shunt in (0.0, 0.05, 0.15, 0.30):
        c = replace(HumanConfig(), pulmonary_baseline_shunt_fraction=shunt)
        _, info = _reset("baseline", seed=24, config=c)
        values.append(info["state"]["arterial_o2_content_ml_dl"])
    assert all(a > b for a, b in zip(values, values[1:]))


def test_v013_pulmonary_timestep_convergence():
    a = HumanHomeostasisEnv(config=replace(HumanConfig(), integration_step_min=0.25), scenario="vq_mismatch")
    b = HumanHomeostasisEnv(config=replace(HumanConfig(), integration_step_min=0.125), scenario="vq_mismatch")
    a.reset(seed=25); b.reset(seed=25)
    ia = ib = None
    for _ in range(4):
        _, _, ta, _, ia = a.step(ZERO)
        _, _, tb, _, ib = b.step(ZERO)
        assert not ta and not tb
    assert abs(ia["state"]["pao2_mmHg"] - ib["state"]["pao2_mmHg"]) < 2.0
    assert abs(ia["state"]["paco2_mmHg"] - ib["state"]["paco2_mmHg"]) < 0.75


def test_v013_stress_no_nan_and_vq_diagnostics_bounded():
    rng = np.random.default_rng(26)
    scenarios = ["baseline", "vq_mismatch", "pulmonary_shunt", "diffusion_limitation"]
    for seed in range(4):
        env = HumanHomeostasisEnv(scenario=scenarios[seed])
        env.reset(seed=seed)
        for _ in range(8):
            action = rng.uniform(0.0, 0.35, size=8).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            s = info["state"]
            assert 0.0 <= s["pulmonary_shunt_fraction"] < 0.8
            assert 0.0 <= s["pulmonary_enghoff_dead_space_fraction"] <= 0.95
            assert 0.0 <= s["pulmonary_diffusion_equilibration_fraction"] <= 1.0
            if terminated or truncated:
                break
