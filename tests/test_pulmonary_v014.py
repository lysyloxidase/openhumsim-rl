from dataclasses import replace
import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv

ZERO = np.zeros(8, dtype=np.float32)


def _state(scenario="baseline", seed=140, config=None):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    _, info = env.reset(seed=seed)
    return env, info["state"]


def test_v014_healthy_lung_is_mostly_recruited_with_low_hpv_burden():
    _, s = _state()
    assert 0.90 <= s["pulmonary_recruitment_fraction"] <= 1.0
    assert 1.0 <= s["pulmonary_hpv_resistance_multiplier"] < 1.20
    assert s["pulmonary_perfusion_redistribution_index"] < 0.05


def test_v014_hpv_improves_vq_mismatch_by_diverting_perfusion():
    # HPV is a dynamic vascular response. Compare after several outer steps, not
    # immediately after reset before regional tone has equilibrated.
    eh, _ = _state("vq_mismatch", seed=141)
    eo, _ = _state("hpv_disabled_vq_mismatch", seed=141)
    hpv = off = None
    for _ in range(4):
        _, _, th, _, ih = eh.step(ZERO)
        _, _, to, _, io = eo.step(ZERO)
        assert not th and not to
        hpv, off = ih["state"], io["state"]
    assert hpv["pulmonary_perfusion_redistribution_index"] > off["pulmonary_perfusion_redistribution_index"] + 0.015
    assert hpv["pao2_mmHg"] > off["pao2_mmHg"] + 2.0


def test_v014_global_hypoxia_raises_pulmonary_resistance_and_pap():
    # This is a mechanism test below the environment's normal safety boundary.
    # Disable only the two oxygen termination thresholds so HPV can evolve; the
    # production environment must still terminate this severe challenge early.
    common = replace(
        HumanConfig(),
        baseline_fio2=0.13,
        spo2_min_terminate=0.0,
        pao2_min_terminate=0.0,
    )
    on_cfg = replace(common, pulmonary_hpv_baseline_function_fraction=1.0)
    off_cfg = replace(common, pulmonary_hpv_baseline_function_fraction=0.0)
    on = HumanHomeostasisEnv(config=on_cfg)
    off = HumanHomeostasisEnv(config=off_cfg)
    on.reset(seed=142); off.reset(seed=142)
    ion = ioff = None
    for _ in range(4):
        _, _, ton, _, ion = on.step(ZERO)
        _, _, toff, _, ioff = off.step(ZERO)
        assert not ton and not toff
    assert ion["state"]["pulmonary_hpv_resistance_multiplier"] > 1.5
    assert ion["state"]["pulmonary_artery_pressure_mmHg"] > ioff["state"]["pulmonary_artery_pressure_mmHg"] + 3.0


def test_v014_dependent_derecruitment_reduces_aerated_fraction_and_oxygenation():
    _, base = _state("baseline", seed=143)
    _, col = _state("dependent_derecruitment", seed=143)
    assert col["pulmonary_recruitment_fraction"] < base["pulmonary_recruitment_fraction"] - 0.25
    assert col["pao2_mmHg"] < base["pao2_mmHg"] - 12.0
    assert col["pulmonary_aa_gradient_mmHg"] > base["pulmonary_aa_gradient_mmHg"] + 12.0


def test_v014_peep_recruits_same_collapsible_lung_and_improves_gas_exchange():
    _, low = _state("dependent_derecruitment", seed=144)
    _, peep = _state("recruitment_peep", seed=144)
    assert peep["pulmonary_recruitment_fraction"] > low["pulmonary_recruitment_fraction"] + 0.25
    assert peep["pao2_mmHg"] > low["pao2_mmHg"] + 12.0
    assert peep["pulmonary_aa_gradient_mmHg"] < low["pulmonary_aa_gradient_mmHg"] - 10.0


def test_v014_recruitment_is_dynamic_after_peep_change():
    env, s0 = _state("dependent_derecruitment", seed=145)
    r0 = s0["pulmonary_recruitment_fraction"]
    env.state.pulmonary_peep_cmH2O = env.config.pulmonary_recruitment_peep_cmH2O
    info = None
    for _ in range(2):
        _, _, term, _, info = env.step(ZERO)
        assert not term
    assert info["state"]["pulmonary_recruitment_fraction"] > r0 + 0.20


def test_v014_hpv_timestep_convergence():
    c = HumanConfig(cv_internal_step_s=0.04)
    a = HumanHomeostasisEnv(config=replace(c, integration_step_min=0.25), scenario="vq_mismatch")
    b = HumanHomeostasisEnv(config=replace(c, integration_step_min=0.125), scenario="vq_mismatch")
    a.reset(seed=146); b.reset(seed=146)
    ia = ib = None
    for _ in range(4):
        _, _, ta, _, ia = a.step(ZERO)
        _, _, tb, _, ib = b.step(ZERO)
        assert not ta and not tb
    assert abs(ia["state"]["pao2_mmHg"] - ib["state"]["pao2_mmHg"]) < 2.0
    assert abs(ia["state"]["pulmonary_hpv_resistance_multiplier"] - ib["state"]["pulmonary_hpv_resistance_multiplier"]) < 0.08


def test_v014_random_stress_bounded_and_finite():
    rng = np.random.default_rng(147)
    for sc in ["baseline", "vq_mismatch", "dependent_derecruitment", "recruitment_peep"]:
        env = HumanHomeostasisEnv(scenario=sc)
        env.reset(seed=147)
        for _ in range(8):
            action = rng.uniform(0.0, 0.25, size=8).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            assert np.all(np.isfinite(obs)) and np.isfinite(reward)
            s = info["state"]
            assert 0.0 <= s["pulmonary_recruitment_fraction"] <= 1.0
            assert 1.0 <= s["pulmonary_hpv_resistance_multiplier"] <= 3.0
            assert 0.0 <= s["pulmonary_perfusion_redistribution_index"] <= 1.0
            if term or trunc:
                break
