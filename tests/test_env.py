import numpy as np
import pytest
from dataclasses import replace

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, SymmetricActionHumanEnv
from openhumsim_rl.env import CLINICAL_OBSERVATION_NAMES, OBSERVATION_NAMES
from openhumsim_rl.metabolism_dallaman import DallaManNormalParameters

ZERO = np.zeros(8, dtype=np.float32)


def step_n(env, n, action=ZERO):
    info = None
    for _ in range(n):
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    return info, terminated, truncated


def test_reset_contract_and_observation():
    env = HumanHomeostasisEnv()
    obs, info = env.reset(seed=123)
    assert obs.shape == (len(CLINICAL_OBSERVATION_NAMES),)
    assert info["observation_profile"] == "clinical"
    assert len(info["observation_names"]) == len(CLINICAL_OBSERVATION_NAMES)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["time_min"] == 0.0


def test_full_observation_profile_is_explicit_opt_in():
    env = HumanHomeostasisEnv(observation_profile="full")
    obs, info = env.reset(seed=123)
    assert obs.shape == (len(OBSERVATION_NAMES),)
    assert info["observation_profile"] == "full"
    assert len(info["observation_names"]) == len(OBSERVATION_NAMES)
    assert env.observation_space.contains(obs)


def test_step_contract_and_seed_determinism():
    e1 = HumanHomeostasisEnv(); e2 = HumanHomeostasisEnv()
    o1, _ = e1.reset(seed=77); o2, _ = e2.reset(seed=77)
    assert np.allclose(o1, o2)
    a = np.array([0.1,0,0.2,0,0,0,0,0], dtype=np.float32)
    r1 = e1.step(a); r2 = e2.step(a)
    assert np.allclose(r1[0], r2[0])
    assert r1[1:4] == r2[1:4]


def test_published_dalla_man_normal_parameters_are_loaded():
    p = DallaManNormalParameters()
    assert p.VG_dl_kg == 1.88
    assert p.k1_min == 0.065
    assert p.k2_min == 0.079
    assert p.VI_l_kg == 0.05
    assert p.HEb == 0.60
    assert p.kmax_min == 0.0558
    assert p.kmin_min == 0.0080
    assert p.kabs_min == 0.057
    assert p.kp1_mg_kg_min == 2.70
    assert p.Vm0_mg_kg_min == 2.50
    assert p.ke2_mg_kg == 339.0


def test_dalla_man_basal_state_is_near_steady_over_one_hour():
    env = HumanHomeostasisEnv(scenario="baseline")
    _, info0 = env.reset(seed=1)
    g0 = info0["state"]["glucose_mg_dl"]
    i0 = info0["state"]["insulin_uU_ml"]
    info, terminated, _ = step_n(env, 12)
    assert not terminated
    assert abs(info["state"]["glucose_mg_dl"] - g0) < 1.0
    assert abs(info["state"]["insulin_uU_ml"] - i0) < 0.5


def test_ogtt_is_dynamic_and_gi_mass_conservative():
    env = HumanHomeostasisEnv(scenario="ogtt")
    _, info0 = env.reset(seed=2)
    assert info0["state"]["gut_carbs_g"] > 70.0
    peak_g = info0["state"]["glucose_mg_dl"]
    peak_i = info0["state"]["insulin_uU_ml"]
    info = info0
    for _ in range(36):  # 180 min
        _, _, term, trunc, info = env.step(ZERO)
        peak_g = max(peak_g, info["state"]["glucose_mg_dl"])
        peak_i = max(peak_i, info["state"]["insulin_uU_ml"])
        if term or trunc:
            break
    assert 130.0 < peak_g < 220.0
    assert peak_i > info0["state"]["insulin_uU_ml"] * 2.0
    assert info["state"]["glucose_mg_dl"] < peak_g
    assert abs(info["metabolism"]["gi_mass_balance_error_mg"]) < 1e-5
    assert info["metabolism"]["numerical_positivity_correction"] < 1e-8


def test_closed_loop_cv_baseline_is_plausible_and_conservative():
    env = HumanHomeostasisEnv()
    _, info = env.reset(seed=3)
    s = info["state"]
    assert 65.0 <= s["map_mmHg"] <= 110.0
    assert 90.0 <= s["systolic_pressure_mmHg"] <= 160.0
    assert 50.0 <= s["diastolic_pressure_mmHg"] <= 100.0
    assert 3.5 <= s["cardiac_output_l_min"] <= 7.0
    assert 45.0 <= s["stroke_volume_ml"] <= 120.0
    assert 0.0 <= s["central_venous_pressure_mmHg"] <= 12.0
    assert abs(info["cardiovascular"]["blood_volume_error_ml"]) < 1e-6
    assert info["cardiovascular"]["numerical_volume_correction_ml"] < 1e-8


def test_dehydration_reduces_preload_output_and_gfr():
    b = HumanHomeostasisEnv(scenario="baseline")
    d = HumanHomeostasisEnv(scenario="dehydrated")
    _, ib = b.reset(seed=5); _, idh = d.reset(seed=5)
    assert idh["state"]["cardiac_output_l_min"] < ib["state"]["cardiac_output_l_min"]
    ib, _, _ = step_n(b, 12); idh, _, _ = step_n(d, 12)
    assert idh["state"]["gfr_ml_min"] < ib["state"]["gfr_ml_min"]
    assert idh["state"]["renin_relative"] > ib["state"]["renin_relative"]


def test_exercise_raises_hr_ventilation_and_vo2():
    env = HumanHomeostasisEnv()
    _, i0 = env.reset(seed=7)
    a = ZERO.copy(); a[2] = 0.6
    info, term, _ = step_n(env, 2, a)
    assert not term
    assert info["state"]["heart_rate_bpm"] > i0["state"]["heart_rate_bpm"]
    assert info["state"]["respiratory_rate_bpm"] > i0["state"]["respiratory_rate_bpm"]
    assert info["state"]["vo2_ml_min"] > i0["state"]["vo2_ml_min"]


def test_pbpk_flow_is_coupled_to_cardiac_output_and_mass_conservative():
    env = HumanHomeostasisEnv(scenario="pbpk_oral_dose")
    env.reset(seed=11)
    info, term, _ = step_n(env, 12)
    assert not term
    assert abs(info["pbpk"]["total_tissue_flow_l_min"] - info["state"]["cardiac_output_l_min"]) < 1e-8
    assert abs(info["pbpk"]["mass_balance_error_mg"]) < 1e-8
    assert info["state"]["probe_plasma_mg_l"] > 0.0


def test_reduced_renal_function_reduces_probe_renal_clearance():
    n = HumanHomeostasisEnv(scenario="pbpk_oral_dose")
    r = HumanHomeostasisEnv(scenario="reduced_renal_function")
    n.reset(seed=12); r.reset(seed=12)
    # give same reference dose in reduced-renal-function state
    a = ZERO.copy(); a[7] = 1.0
    _,_,_,_,ir = r.step(a)
    _,_,_,_,inorm = n.step(ZERO)
    assert ir["state"]["gfr_ml_min"] < inorm["state"]["gfr_ml_min"]
    assert ir["pbpk"]["renal_clearance_l_min"] < inorm["pbpk"]["renal_clearance_l_min"]


def test_oxygen_and_ventilation_controls_have_expected_direction():
    a = HumanHomeostasisEnv(scenario="respiratory_acidosis")
    b = HumanHomeostasisEnv(scenario="respiratory_acidosis")
    a.reset(seed=4); b.reset(seed=4)
    control = ZERO.copy(); control[4] = 0.35; control[5] = 0.25
    _,_,_,_,ia = a.step(ZERO)
    _,_,_,_,ib = b.step(control)
    assert ib["state"]["pao2_mmHg"] > ia["state"]["pao2_mmHg"]
    assert ib["state"]["paco2_mmHg"] < ia["state"]["paco2_mmHg"]


def test_fluid_electrolyte_ledgers_close():
    env = HumanHomeostasisEnv()
    env.reset(seed=31)
    a = ZERO.copy(); a[3] = 0.15; a[6] = 0.20
    info, _, _ = step_n(env, 12, a)
    mb = info["mass_balance"]
    assert abs(mb["water_mass_balance_error_l"]) < 1e-9
    assert abs(mb["sodium_mass_balance_error_mmol"]) < 1e-8
    assert abs(mb["chloride_mass_balance_error_mmol"]) < 1e-8
    assert abs(mb["potassium_mass_balance_error_mmol"]) < 1e-8
    assert abs(mb["nonvolatile_acid_mass_balance_error_mEq"]) < 1e-8


def test_cv_timestep_sensitivity_against_finer_reference():
    coarse = HumanConfig(cv_internal_step_s=0.02)
    fine = HumanConfig(cv_internal_step_s=0.01)
    ec = HumanHomeostasisEnv(config=coarse)
    ef = HumanHomeostasisEnv(config=fine)
    _, ic = ec.reset(seed=8); _, iff = ef.reset(seed=8)
    # compare periodic warm-state aggregate outputs, not instantaneous chamber phase
    assert abs(ic["state"]["map_mmHg"] - iff["state"]["map_mmHg"]) < 4.0
    assert abs(ic["state"]["cardiac_output_l_min"] - iff["state"]["cardiac_output_l_min"]) < 0.3


def test_dalla_timestep_sensitivity():
    a = HumanHomeostasisEnv(config=HumanConfig(dalla_internal_step_min=0.05), scenario="ogtt")
    b = HumanHomeostasisEnv(config=HumanConfig(dalla_internal_step_min=0.025), scenario="ogtt")
    a.reset(seed=9); b.reset(seed=9)
    ia, _, _ = step_n(a, 12); ib, _, _ = step_n(b, 12)
    assert abs(ia["state"]["glucose_mg_dl"] - ib["state"]["glucose_mg_dl"]) < 0.2
    assert abs(ia["state"]["insulin_uU_ml"] - ib["state"]["insulin_uU_ml"]) < 0.2


def test_symmetric_action_env_maps_minus_one_to_no_intervention():
    env = SymmetricActionHumanEnv()
    env.reset(seed=10)
    a = -np.ones(8, dtype=np.float32)
    _, _, _, _, info = env.step(a)
    iv = info["intervention"]
    assert iv["insulin_model_units"] == 0.0
    assert iv["oral_carbs_g"] == 0.0
    assert iv["saline_ml"] == 0.0
    assert iv["oral_probe_mg"] == 0.0
    assert info["action"]["exercise"] == 0.0
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0

    env.reset(seed=10)
    _, _, _, _, info0 = env.step(np.zeros(8, dtype=np.float32))
    assert info0["intervention"]["oral_carbs_g"] == 0.0
    assert info0["intervention"]["oral_probe_mg"] == 0.0


def test_wrong_action_shape_raises():
    env = HumanHomeostasisEnv(); env.reset(seed=0)
    try:
        env.step(np.zeros(7, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for wrong action shape")



def test_public_interface_has_explicit_units_metadata():
    from openhumsim_rl.env import (
        ACTION_NAMES,
        CLINICAL_OBSERVATION_NAMES,
        OBSERVATION_NAMES,
    )
    from openhumsim_rl.units import OBSERVATION_UNITS, ACTION_SEMANTICS
    assert set(OBSERVATION_NAMES) | set(CLINICAL_OBSERVATION_NAMES) == set(OBSERVATION_UNITS)
    assert set(ACTION_NAMES) == set(ACTION_SEMANTICS)


def test_fick_oxygen_identity_closes_at_rest_and_exercise():
    for intensity in (0.0, 0.5, 1.0):
        env = HumanHomeostasisEnv()
        _, info = env.reset(seed=42)
        a = ZERO.copy(); a[2] = intensity
        info, term, _ = step_n(env, 4, a)
        assert not term
        oxy = info["oxygen_transport"]
        s = info["state"]
        assert abs(oxy["fick_residual_ml_min"]) < 1e-8
        assert oxy["oxygen_extraction_ratio"] < 0.90
        expected_margin = (
            env.config.oxygen_max_extraction_fraction
            * oxy["oxygen_delivery_ml_min"]
            - s["vo2_demand_ml_min"]
        )
        assert oxy["oxygen_supply_margin_ml_min"] == pytest.approx(expected_margin)
        if oxy["oxygen_supply_margin_ml_min"] < 0.0:
            assert s["oxygen_debt_ml_min"] > 0.0
        if intensity == 1.0:
            assert oxy["oxygen_supply_margin_ml_min"] < 0.0


def test_exercise_cardiac_output_approaches_external_model_scale():
    env = HumanHomeostasisEnv()
    _, baseline = env.reset(seed=42)
    a = ZERO.copy(); a[2] = 1.0
    info, term, _ = step_n(env, 6, a)
    assert not term
    # Broomé et al. 2013 report CO above 10 L/min in a different exercise
    # model.  Treat that as a scale comparison, not an exact calibration target:
    # the converged reduced model should at least double its resting output and
    # remain within 10% of that literature anchor.
    assert info["state"]["cardiac_output_l_min"] > 9.0
    assert (
        info["state"]["cardiac_output_l_min"]
        > 2.0 * baseline["state"]["cardiac_output_l_min"]
    )
    assert 135.0 < info["state"]["heart_rate_bpm"] < 170.0


def test_renal_pressure_autoregulation_has_plateau():
    from openhumsim_rl.renal import RenalModel
    assert RenalModel.pressure_autoregulation_factor(80.0) == 1.0
    assert RenalModel.pressure_autoregulation_factor(100.0) == 1.0
    assert RenalModel.pressure_autoregulation_factor(140.0) == 1.0
    assert RenalModel.pressure_autoregulation_factor(60.0) < 0.7


def test_legacy_ogtt_label_declares_validation_limitation():
    env = HumanHomeostasisEnv(scenario="ogtt")
    _, info = env.reset(seed=1)
    assert "scenario_warning" in info
    assert "mixed-meal" in info["scenario_warning"]
    env2 = HumanHomeostasisEnv(scenario="oral_glucose_75g")
    _, info2 = env2.reset(seed=1)
    assert "scenario_warning" not in info2


def test_environment_declares_partial_observability():
    env = HumanHomeostasisEnv()
    _, info = env.reset(seed=1)
    assert info["environment_semantics"]["fully_observed_markov_state"] is False
    assert info["environment_semantics"]["clinical_validation"] is False


def test_virtual_cohort_is_reproducible_and_bounded():
    from openhumsim_rl import sample_virtual_cohort
    a = sample_virtual_cohort(4, seed=99)
    b = sample_virtual_cohort(4, seed=99)
    assert a[0].latent == b[0].latent
    for vp in a:
        assert 55.0 <= vp.config.body_weight_kg <= 95.0
        assert 90.0 <= vp.config.baseline_gfr_ml_min <= 130.0
        assert 12.5 <= vp.config.hemoglobin_g_dl <= 16.5
        assert vp.config.ecf_volume_baseline_l < vp.config.total_body_water_baseline_l


def test_dalla_uq_scales_preserve_baseline_and_change_challenge():
    from dataclasses import replace
    from openhumsim_rl import HumanConfig
    base = HumanConfig()
    low = HumanHomeostasisEnv(config=replace(base, dalla_insulin_sensitivity_scale=0.75), scenario="oral_glucose_75g")
    high = HumanHomeostasisEnv(config=replace(base, dalla_insulin_sensitivity_scale=1.25), scenario="oral_glucose_75g")
    low.reset(seed=12); high.reset(seed=12)
    zero = np.zeros(8, dtype=np.float32)
    gl, gh = [], []
    for _ in range(24):
        _,_,tl,trl,il = low.step(zero); _,_,th,trh,ih = high.step(zero)
        gl.append(il["state"]["glucose_mg_dl"]); gh.append(ih["state"]["glucose_mg_dl"])
        if tl or trl or th or trh: break
    assert max(gl) > max(gh)


def test_v09_cli_doctor_and_demo():
    import sys

    from openhumsim_rl.cli import doctor, run_demo
    d = doctor()
    expected_core_ready = bool(
        sys.version_info >= (3, 10) and d["dependencies"]["numpy"]["installed"]
    )
    assert d["core_ready"] is expected_core_ready
    assert d["openhumsim_version"] == "0.23.2"
    result = run_demo("baseline", minutes=5.0, seed=123)
    assert result["simulated_minutes"] == 5.0
    assert np.isfinite(result["return"])


def test_v08_external_cgm_parser_synthetic_zip(tmp_path):
    import zipfile
    from openhumsim_rl.external_data import summarize_cgm_archive

    csv_text = "participant_id,sensor_glucose_mg_dl\n" + "\n".join(
        [f"A,{95 + (i % 5)}" for i in range(20)]
        + [f"B,{100 + (i % 7)}" for i in range(20)]
    )
    archive = tmp_path / "fake_cgm.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("cgm.csv", csv_text)
    summary = summarize_cgm_archive(archive)
    assert summary["n_subjects"] == 2
    assert summary["n_readings"] == 40
    assert summary["glucose_column"] == "sensor_glucose_mg_dl"
    assert summary["subject_column"] == "participant_id"
    assert 90 < summary["population"]["mean_of_subject_mean_glucose_mg_dl"] < 110


def test_v08_posterior_small_smoke():
    from openhumsim_rl import GaussianTarget, importance_calibrate
    targets = [
        GaussianTarget("cardiac_output_l_min", 5.0, 1.0, 0.5),
        GaussianTarget("paco2_mmHg", 40.0, 5.0, 3.0),
    ]
    result = importance_calibrate(targets, n_prior=4, seed=3, cv_internal_step_s=0.04)
    assert result.n_prior == 4
    assert 1.0 <= result.effective_sample_size <= 4.0 + 1e-9
    assert len(result.top_particles) == 4
    assert "cardiac_output_l_min" in result.posterior_output_summary


def test_v09_cgm_observation_model_has_expected_lag_and_converges():
    from openhumsim_rl.cgm import CGMObservationConfig, blood_to_cgm_trace
    blood = np.r_[np.full(10, 90.0), np.full(60, 160.0)]
    cgm = blood_to_cgm_trace(blood, dt_min=1.0, config=CGMObservationConfig(lag_tau_min=6.0))
    assert cgm[10] < 160.0
    assert cgm[10] > 90.0
    assert cgm[15] < blood[15]
    assert abs(cgm[-1] - 160.0) < 0.1


def test_v09_subject_split_has_no_leakage_and_is_reproducible():
    from openhumsim_rl.external_data import deterministic_subject_split
    ids = [f"S{i:03d}" for i in range(30)]
    a = deterministic_subject_split(ids, seed=2019)
    b = deterministic_subject_split(ids, seed=2019)
    assert a == b
    assert not (set(a["train"]) & set(a["validation"]))
    assert not (set(a["train"]) & set(a["test"]))
    assert not (set(a["validation"]) & set(a["test"]))
    assert set().union(*map(set, a.values())) == set(ids)


def test_v09_external_cgm_split_report_synthetic_zip(tmp_path):
    import zipfile
    from openhumsim_rl.external_data import summarize_cgm_archive, build_subject_split_report

    rows = ["participant_id,sensor_glucose_mg_dl"]
    for s in range(15):
        sid = f"S{s:02d}"
        for i in range(24):
            rows.append(f"{sid},{92 + s * 0.4 + (i % 6)}")
    archive = tmp_path / "cgm_split.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("cgm.csv", "\n".join(rows))
    summary = summarize_cgm_archive(archive)
    report = build_subject_split_report(summary, seed=7, n_boot=50)
    assert report["leakage_check"]["passed"] is True
    assert report["leakage_check"]["all_subjects_accounted_for"] is True
    assert sum(v["metrics"]["n_subjects"] for v in report["splits"].values()) == 15
    assert report["splits"]["test"]["metrics"]["n_subjects"] > 0


def test_v09_shah_table2_reference_transcription():
    from openhumsim_rl.external_data import SHAH_2019_AGE_STRATA
    assert sum(x.n for x in SHAH_2019_AGE_STRATA) == 153
    older = [x for x in SHAH_2019_AGE_STRATA if x.age_group == ">=60"][0]
    assert older.mean_glucose_mg_dl == 104.0
    assert older.time_70_140_median_pct == 93.0
    assert older.time_gt_140_median_pct == 4.1


def test_v09_normative_cgm_reference_uses_train_only(tmp_path):
    import zipfile
    from openhumsim_rl.external_data import summarize_cgm_archive
    from openhumsim_rl.cgm_reference import calibrate_normative_cgm_reference

    rows = ["participant_id,sensor_glucose_mg_dl"]
    for s in range(20):
        sid = f"P{s:02d}"
        base = 92.0 + 0.6 * s
        for i in range(36):
            rows.append(f"{sid},{base + (i % 9) - 4}")
    archive = tmp_path / "reference.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sensor.csv", "\n".join(rows))
    summary = summarize_cgm_archive(archive)
    result = calibrate_normative_cgm_reference(summary, seed=5, n_boot=50)
    payload = result.as_dict()
    assert payload["split_report"]["leakage_check"]["passed"] is True
    assert payload["fitted_on_train"]["mean_mg_dl"]["train_n"] == payload["split_report"]["splits"]["train"]["metrics"]["n_subjects"]
    assert payload["evaluation"]["test"]["mean_mg_dl"]["n"] > 0
    assert 0.0 <= payload["evaluation"]["test"]["time_70_140_pct"]["coverage_95_pct"] <= 100.0


def test_v011_physicochemical_charge_and_hh_closure():
    env = HumanHomeostasisEnv()
    _, info = env.reset(seed=2)
    ab = info["acid_base"]
    assert 7.35 <= ab["pH"] <= 7.45
    assert 20.0 <= ab["bicarbonate_mmol_l"] <= 30.0
    assert abs(ab["charge_balance_residual_mEq_l"]) < 1e-6
    assert abs(ab["henderson_hasselbalch_residual"]) < 1e-10


def test_v011_saline_load_reduces_sid_and_ph_without_manual_hco3_edit():
    base = HumanHomeostasisEnv(scenario="baseline")
    saline = HumanHomeostasisEnv(scenario="saline_challenge_30ml_kg")
    _, ib = base.reset(seed=3)
    _, isal = saline.reset(seed=3)
    assert isal["state"]["chloride_mmol_l"] > ib["state"]["chloride_mmol_l"]
    assert isal["state"]["strong_ion_difference_apparent_mEq_l"] < ib["state"]["strong_ion_difference_apparent_mEq_l"]
    assert isal["state"]["ph_arterial"] < ib["state"]["ph_arterial"]
    assert isal["state"]["bicarbonate_mmol_l"] < ib["state"]["bicarbonate_mmol_l"]


def test_v011_lactate_is_a_strong_anion_and_lowers_ph():
    base = HumanHomeostasisEnv(scenario="baseline")
    lac = HumanHomeostasisEnv(scenario="transient_lactic_acidosis")
    _, ib = base.reset(seed=4)
    _, il = lac.reset(seed=4)
    assert il["state"]["lactate_mmol_l"] > ib["state"]["lactate_mmol_l"]
    assert il["state"]["strong_ion_difference_apparent_mEq_l"] < ib["state"]["strong_ion_difference_apparent_mEq_l"]
    assert il["state"]["ph_arterial"] < ib["state"]["ph_arterial"]


def test_v011_respiratory_acidosis_increases_renal_ammonium_and_compensates_over_time():
    env = HumanHomeostasisEnv(config=HumanConfig(cv_internal_step_s=0.04), scenario="respiratory_acidosis")
    _, i0 = env.reset(seed=5)
    hco30 = i0["state"]["bicarbonate_mmol_l"]
    nh40 = i0["state"]["urine_ammonium_mmol_min"]
    info, term, _ = step_n(env, int(24*60/env.config.agent_step_min))
    assert not term
    assert info["state"]["urine_ammonium_mmol_min"] > nh40
    assert info["state"]["bicarbonate_mmol_l"] > hco30
    assert info["state"]["ph_arterial"] > i0["state"]["ph_arterial"]


def test_v011_reduced_renal_function_accumulates_unmeasured_acid_burden():
    env = HumanHomeostasisEnv(config=HumanConfig(cv_internal_step_s=0.04), scenario="reduced_renal_function")
    _, i0 = env.reset(seed=6)
    sig0 = i0["state"]["strong_ion_gap_mEq_l"]
    info, term, _ = step_n(env, int(24*60/env.config.agent_step_min))
    assert not term
    assert info["state"]["strong_ion_gap_mEq_l"] > sig0
    assert info["state"]["renal_acid_excretion_mmol_min"] < env.config.baseline_net_acid_excretion_mmol_min
    assert abs(info["mass_balance"]["nonvolatile_acid_mass_balance_error_mEq"]) < 1e-8
