import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.acid_base import PhysicochemicalAcidBaseModel
from openhumsim_rl.blood_gas import WholeBloodGasChemistryModel
from openhumsim_rl.physiology import HumanState

ZERO = np.zeros(8, dtype=np.float32)


def standard_snapshot():
    c = HumanConfig()
    s = HumanState()
    s.paco2_mmHg = 40.0
    s.pao2_mmHg = 100.0
    s.spo2_pct = 97.2
    ab = PhysicochemicalAcidBaseModel(c)
    ab.initialize_state(s)
    d = ab.snapshot_for_pco2(s, 40.0)
    bg = WholeBloodGasChemistryModel(c, ab)
    wb = bg.snapshot(
        plasma_ph=d.ph,
        pco2_mmHg=40.0,
        oxygen_sat_fraction=0.972,
        plasma_chloride_mmol_l=s.chloride_mmol_l,
    )
    return c, s, d, bg, wb


def test_v012_standard_whole_blood_co2_matches_oneill_reference_scale():
    c, _, _, _, wb = standard_snapshot()
    ml_per_dl = wb.total_co2_mmol_l_blood * c.co2_gas_molar_volume_l_per_mol_stpd / 10.0
    assert abs(ml_per_dl - 44.6) < 0.8


def test_v012_funder_wieth_donnan_and_rbc_ph_are_physiologic():
    _, _, d, _, wb = standard_snapshot()
    expected_rcl = 3.319 - 0.359 * d.ph
    expected_rh = 3.094 - 0.335 * d.ph
    assert abs(wb.donnan_rcl - expected_rcl) < 1e-12
    assert abs(wb.donnan_rh - expected_rh) < 1e-12
    assert 7.15 <= wb.rbc_ph <= 7.23


def test_v012_carbamino_reference_and_haldane_direction():
    _, _, _, bg, _ = standard_snapshot()
    assert abs(bg.carbamino_fraction(40.0, 0.972) - 0.131) < 1e-12
    oxy = bg.snapshot(plasma_ph=7.40, pco2_mmHg=46.0, oxygen_sat_fraction=0.97, plasma_chloride_mmol_l=103.0)
    deoxy = bg.snapshot(plasma_ph=7.40, pco2_mmHg=46.0, oxygen_sat_fraction=0.75, plasma_chloride_mmol_l=103.0)
    assert deoxy.carbamino_co2_mmol_l_blood > oxy.carbamino_co2_mmol_l_blood
    assert deoxy.total_co2_mmol_l_blood > oxy.total_co2_mmol_l_blood


def test_v012_arteriovenous_co2_and_chloride_shift_close_locally():
    env = HumanHomeostasisEnv(scenario="baseline")
    _, info = env.reset(seed=42)
    s = info["state"]
    assert s["mixed_venous_pco2_mmHg"] > s["paco2_mmHg"]
    assert s["mixed_venous_ph"] < s["ph_arterial"]
    assert s["mixed_venous_total_co2_mmol_l_blood"] > s["arterial_total_co2_mmol_l_blood"]
    assert 0.5 <= s["chloride_shift_plasma_mmol_l"] <= 3.0
    free_rbc_drop = s["rbc_chloride_mmol_l"] - s["mixed_venous_rbc_chloride_mmol_l"]
    assert 1.0 <= free_rbc_drop <= 3.0
    assert s["mixed_venous_hb_bound_chloride_gain_mmol_l_rbc"] > 0.0
    assert abs(s["chloride_shift_balance_residual_mmol_l_blood"]) < 1e-10
    assert abs(s["co2_fick_content_residual_mmol_l"]) < 1e-6


def test_v012_exchangeable_carbon_ledger_conserves_mass():
    env = HumanHomeostasisEnv(scenario="baseline")
    env.reset(seed=7)
    for _ in range(24):
        _, _, terminated, truncated, info = env.step(ZERO)
        assert not terminated
        if truncated:
            break
    assert abs(info["mass_balance"]["co2_mass_balance_error_mmol"]) < 1e-8
    assert abs(info["state"]["co2_content_solver_residual_mmol_l"]) < 1e-5


def test_v012_ventilatory_support_removes_carbon_and_lowers_paco2():
    no = HumanHomeostasisEnv(scenario="respiratory_acidosis")
    yes = HumanHomeostasisEnv(scenario="respiratory_acidosis")
    no.reset(seed=8); yes.reset(seed=8)
    support = ZERO.copy(); support[5] = 0.25
    for _ in range(6):
        _, _, tn, _, ino = no.step(ZERO)
        _, _, ty, _, iyes = yes.step(support)
        assert not tn and not ty
    assert iyes["state"]["paco2_mmHg"] < ino["state"]["paco2_mmHg"]
    assert iyes["state"]["exchangeable_co2_pool_mmol"] < ino["state"]["exchangeable_co2_pool_mmol"]
    assert iyes["state"]["ph_arterial"] > ino["state"]["ph_arterial"]


def test_v012_exercise_preserves_fick_co2_identity_and_increases_haldane_gain():
    rest = HumanHomeostasisEnv(); ex = HumanHomeostasisEnv()
    _, ir = rest.reset(seed=9); ex.reset(seed=9)
    a = ZERO.copy(); a[2] = 0.7
    ie = None
    for _ in range(4):
        _, _, t, _, ie = ex.step(a)
        assert not t
    assert abs(ie["state"]["co2_fick_content_residual_mmol_l"]) < 1e-6
    assert ie["state"]["mixed_venous_pco2_mmHg"] > ie["state"]["paco2_mmHg"]
    assert ie["state"]["chloride_shift_plasma_mmol_l"] > ir["state"]["chloride_shift_plasma_mmol_l"]
    assert ie["state"]["haldane_co2_content_gain_mmol_l"] > ir["state"]["haldane_co2_content_gain_mmol_l"]


def test_v012_carbon_solver_timestep_convergence():
    from dataclasses import replace
    c0 = HumanConfig(cv_internal_step_s=0.04)
    a = HumanHomeostasisEnv(config=replace(c0, integration_step_min=0.25), scenario="baseline")
    b = HumanHomeostasisEnv(config=replace(c0, integration_step_min=0.125), scenario="baseline")
    a.reset(seed=10); b.reset(seed=10)
    ia = ib = None
    for _ in range(12):
        _, _, ta, _, ia = a.step(ZERO)
        _, _, tb, _, ib = b.step(ZERO)
        assert not ta and not tb
    assert abs(ia["state"]["paco2_mmHg"] - ib["state"]["paco2_mmHg"]) < 0.5
    assert abs(ia["state"]["ph_arterial"] - ib["state"]["ph_arterial"]) < 0.005
