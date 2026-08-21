from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import csv
import zipfile

import numpy as np

from openhumsim_rl.event_replay import (
    DUBOSE_REFERENCE,
    archive_schema_report,
    calibrate_aggregate_event_profile,
    event_metrics_from_archive,
    extract_events_from_archive,
    fit_mechanistic_event_profile,
    infer_effective_meal_carbs,
    summarize_event_metrics,
)
from openhumsim_rl.external_data import jaeb_download_instructions, download_jaeb_healthy_cgm


def _write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _synthetic_archive(tmp_path: Path) -> Path:
    # Four participants so train/validation/test event splitting is possible.
    cgm_rows = []
    diary_rows = []
    day = datetime(2020, 1, 2)
    for sid_i in range(1, 5):
        sid = str(sid_i)
        # 24 h CGM every 5 min. Add modest event-shaped excursions.
        meals = [day.replace(hour=8), day.replace(hour=12), day.replace(hour=17)]
        ex_start = day.replace(hour=15)
        for m in range(0, 24 * 60, 5):
            t = day + timedelta(minutes=m)
            g = 93.0 + 2.0 * np.sin(2 * np.pi * m / 1440.0)
            for mt in meals:
                dt = (t - mt).total_seconds() / 60.0
                if 0 <= dt <= 240:
                    g += 36.0 * np.exp(-0.5 * ((dt - 95.0) / 45.0) ** 2)
            exdt = (t - ex_start).total_seconds() / 60.0
            if 0 <= exdt <= 45:
                g -= 14.0 * (exdt / 45.0)
            cgm_rows.append({"Participant ID": sid, "DateTime": t.strftime("%Y-%m-%d %H:%M:%S"), "Sensor Glucose (mg/dL)": f"{g:.3f}"})
        diary_rows.append({
            "Participant ID": sid,
            "Date": day.strftime("%Y-%m-%d"),
            "Breakfast Time": "08:00",
            "Lunch Time": "12:00",
            "Dinner Time": "17:00",
            "Exercise Start": "15:00",
            "Exercise Duration": "45",
            "Exercise Type": "Aerobic",
            "Sleep Time": "23:00",
            "Wake Time": "06:30",
        })
    cgm = tmp_path / "CGM.csv"
    diary = tmp_path / "DailyLog.csv"
    _write_csv(cgm, ["Participant ID", "DateTime", "Sensor Glucose (mg/dL)"], cgm_rows)
    _write_csv(diary, ["Participant ID", "Date", "Breakfast Time", "Lunch Time", "Dinner Time", "Exercise Start", "Exercise Duration", "Exercise Type", "Sleep Time", "Wake Time"], diary_rows)
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cgm, cgm.name)
        zf.write(diary, diary.name)
    return archive


def test_schema_and_event_autodetection(tmp_path):
    archive = _synthetic_archive(tmp_path)
    report = archive_schema_report(archive)
    assert len(report["tables"]) == 2
    events = extract_events_from_archive(archive)
    assert events["counts"]["meal"] == 12
    assert events["counts"]["exercise"] == 4
    assert events["counts"]["sleep"] == 4
    assert events["counts"]["wake"] == 4


def test_dubose_protocol_alignment_and_summary(tmp_path):
    archive = _synthetic_archive(tmp_path)
    payload = event_metrics_from_archive(archive)
    # 3 main meals x 4 subjects, and one exercise x 4 subjects.
    assert payload["n_meal_metrics"] == 12
    assert payload["n_exercise_metrics"] == 4
    summary = summarize_event_metrics(payload)
    assert 120.0 < summary["meal"]["peak_mg_dl"]["mean"] < 145.0
    assert summary["exercise"]["change_mg_dl"]["mean"] < 0.0


def test_published_aggregate_replay_calibration_matches_reference():
    result = calibrate_aggregate_event_profile()
    meal = result["high_fidelity"]["meal"]
    exercise = result["high_fidelity"]["exercise"]
    assert abs(meal["peak_mg_dl"] - DUBOSE_REFERENCE.meal_peak_mean_mg_dl) < 2.0
    assert abs(meal["time_to_peak_min"] - DUBOSE_REFERENCE.meal_time_to_peak_mean_min) <= 5.0
    assert abs(exercise["change_mg_dl"] - DUBOSE_REFERENCE.exercise_change_mean_mg_dl) < 2.0
    assert result["objective_mean_squared_z"] < 0.02


def test_latent_meal_carbohydrate_inference_recovers_near_reference():
    result = infer_effective_meal_carbs(93.0, 130.0, 97.0)
    assert 25.0 <= result["effective_carbs_g"] <= 50.0
    assert abs(result["predicted_peak_mg_dl"] - 130.0) < 5.0


def test_train_only_event_fit_has_no_participant_leakage(tmp_path):
    archive = _synthetic_archive(tmp_path)
    metrics = event_metrics_from_archive(archive)
    result = fit_mechanistic_event_profile(metrics, seed=2020)
    assert result["leakage_check"]["passed"] is True
    assert set(result["split_subject_ids"]) == {"train", "validation", "test"}
    assert result["evaluation"]["train"]["error"]["mean_squared_z"] is not None


def test_jaeb_download_requires_manual_terms_acceptance(tmp_path):
    instructions = jaeb_download_instructions()
    assert "public.jaeb.org" in instructions["dataset_page"]
    try:
        download_jaeb_healthy_cgm(tmp_path / "CGMND.zip")
    except PermissionError as exc:
        assert "does not bypass" in str(exc)
    else:
        raise AssertionError("Expected manual-download guard")
