from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.10.0")

from datetime import datetime, timedelta
from pathlib import Path
import csv
import json
import tempfile
import zipfile

import numpy as np

from openhumsim_rl import __version__
from openhumsim_rl.event_replay import (
    DUBOSE_REFERENCE,
    archive_schema_report,
    calibrate_aggregate_event_profile,
    event_metrics_from_archive,
    extract_events_from_archive,
    fit_mechanistic_event_profile,
)
from openhumsim_rl.external_data import JAEB_DATASET_PAGE_URL, jaeb_download_instructions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.10.json"
CAL_OUT = ROOT / "validation" / "published_event_calibration_v0.10.json"
REF_OUT = ROOT / "validation" / "external_reference_dubose2020_v0.10.json"


def check(name, value, passed, tier, note=""):
    return {"name": name, "value": value, "pass": bool(passed), "tier": tier, "note": note}


def _write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fixture_archive(folder: Path) -> Path:
    cgm_rows, diary_rows = [], []
    day = datetime(2020, 1, 2)
    for sid_i in range(1, 6):
        sid = str(sid_i)
        meals = [day.replace(hour=8), day.replace(hour=12), day.replace(hour=17)]
        ex_start = day.replace(hour=15)
        for minute in range(0, 24 * 60, 5):
            t = day + timedelta(minutes=minute)
            g = 93.0 + 1.5 * np.sin(2 * np.pi * minute / 1440.0)
            for mt in meals:
                dt = (t - mt).total_seconds() / 60.0
                if 0 <= dt <= 240:
                    g += 36.0 * np.exp(-0.5 * ((dt - 95.0) / 45.0) ** 2)
            exdt = (t - ex_start).total_seconds() / 60.0
            if 0 <= exdt <= 45:
                g -= 14.0 * exdt / 45.0
            cgm_rows.append({"Participant ID": sid, "DateTime": t.strftime("%Y-%m-%d %H:%M:%S"), "Sensor Glucose (mg/dL)": f"{g:.3f}"})
        diary_rows.append({
            "Participant ID": sid, "Date": day.strftime("%Y-%m-%d"),
            "Breakfast Time": "08:00", "Lunch Time": "12:00", "Dinner Time": "17:00",
            "Exercise Start": "15:00", "Exercise Duration": "45", "Exercise Type": "Aerobic",
            "Sleep Time": "23:00", "Wake Time": "06:30",
        })
    cgm = folder / "CGM.csv"
    diary = folder / "DailyLog.csv"
    _write_csv(cgm, ["Participant ID", "DateTime", "Sensor Glucose (mg/dL)"], cgm_rows)
    _write_csv(diary, ["Participant ID", "Date", "Breakfast Time", "Lunch Time", "Dinner Time", "Exercise Start", "Exercise Duration", "Exercise Type", "Sleep Time", "Wake Time"], diary_rows)
    archive = folder / "fixture.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cgm, cgm.name)
        zf.write(diary, diary.name)
    return archive


def main():
    checks = []
    prior = json.loads((ROOT / "validation/validation_results_v0.9.json").read_text())
    checks.append(check("v0.9 frozen scientific regression", prior["summary"], prior["summary"]["failed"] == 0, "regression"))
    checks.append(check("release version", __version__, __version__ == "0.10.0", "software-verification"))

    r = DUBOSE_REFERENCE
    ref_ok = (
        r.n_participants == 153 and r.exercise_sessions == 451 and
        r.exercise_baseline_mean_mg_dl == 99.0 and r.exercise_change_mean_mg_dl == -15.0 and
        r.meal_participants == 56 and r.meal_events == 306 and
        r.meal_premeal_mean_mg_dl == 93.0 and r.meal_peak_mean_mg_dl == 130.0 and
        r.meal_time_to_peak_mean_min == 97.0 and r.meal_excursion_mean_mg_dl == 37.0
    )
    checks.append(check("DuBose event-reference transcription", r.as_dict(), ref_ok, "external-human-reference"))

    calibration = calibrate_aggregate_event_profile()
    CAL_OUT.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    meal, ex = calibration["high_fidelity"]["meal"], calibration["high_fidelity"]["exercise"]
    fit_ok = (
        abs(meal["peak_mg_dl"] - r.meal_peak_mean_mg_dl) < 2.0 and
        abs(meal["time_to_peak_min"] - r.meal_time_to_peak_mean_min) <= 5.0 and
        abs(ex["change_mg_dl"] - r.exercise_change_mean_mg_dl) < 2.0 and
        calibration["objective_mean_squared_z"] < 0.02
    )
    checks.append(check("published event-level replay calibration", calibration["high_fidelity"], fit_ok, "mechanistic-calibration",
                        "Latent replay inputs fit aggregate data; not unique physiological estimates because meal quantity/intensity were not recorded."))

    instructions = jaeb_download_instructions()
    terms_ok = "public.jaeb.org" in instructions["dataset_page"] and len(instructions["steps"]) >= 4
    checks.append(check("Jaeb terms-respecting manual download workflow", instructions, terms_ok, "data-governance"))

    with tempfile.TemporaryDirectory() as d:
        archive = _fixture_archive(Path(d))
        schema = archive_schema_report(archive)
        events = extract_events_from_archive(archive)
        metrics = event_metrics_from_archive(archive)
        mech = fit_mechanistic_event_profile(metrics, seed=2020)
        schema_ok = len(schema["tables"]) == 2 and events["counts"].get("meal") == 15 and events["counts"].get("exercise") == 5
        protocol_ok = metrics["n_meal_metrics"] == 15 and metrics["n_exercise_metrics"] == 5
        split_ok = mech["leakage_check"]["passed"] and all(mech["split_subject_ids"][k] for k in ("train", "validation", "test"))
        checks.append(check("schema-adaptive diary/CGM parser", {"counts": events["counts"], "tables": schema["tables"]}, schema_ok, "data-engineering-verification"))
        checks.append(check("DuBose inclusion-rule event alignment", {"meals": metrics["n_meal_metrics"], "exercise": metrics["n_exercise_metrics"], "exclusions": metrics["exclusions"]}, protocol_ok, "analysis-verification"))
        checks.append(check("participant-level held-out mechanistic event fit", {"profile": mech["profile"], "leakage": mech["leakage_check"], "errors": {k: v["error"] for k, v in mech["evaluation"].items()}}, split_ok, "algorithm-verification"))

    raw_archives = list((ROOT / "data" / "external").glob("*.zip")) if (ROOT / "data" / "external").exists() else []
    checks.append(check("human raw dataset not bundled", [p.name for p in raw_archives], len(raw_archives) == 0, "data-governance"))

    reference_payload = {
        "version": "0.10.0",
        "paper": r.citation,
        "jaeb_official_dataset_page": JAEB_DATASET_PAGE_URL,
        "reference": r.as_dict(),
        "study_limitations_relevant_to_model": [
            "Meal/snack start times were logged but carbohydrate/fat quantity was not captured.",
            "Exercise logs provide time/type and reported duration, not metabolic workload or VO2.",
            "Diary omissions are possible in a free-living unsupervised study.",
            "CGM measures interstitial sensor glucose rather than venous plasma glucose.",
        ],
    }
    REF_OUT.write_text(json.dumps(reference_payload, indent=2), encoding="utf-8")

    result = {
        "version": "0.10.0",
        "summary": {"passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks), "total": len(checks)},
        "checks": checks,
        "credibility_classification": {
            "event_parser": "verified on long/wide synthetic schemas; real Jaeb schema must still be inspected from a local archive obtained under the dataset terms",
            "event_inclusion_rules": "implements the published DuBose meal/exercise analysis rules used for event-level comparison",
            "aggregate_meal_replay": "matches published mean peak/excursion/time-to-peak after fitting latent effective meal input; not independent validation",
            "aggregate_exercise_replay": "matches published mean glucose drop after fitting the reduced-order exercise extension; not a workload-calibrated exercise model",
            "participant_holdout_pipeline": "implemented and leakage-tested on synthetic event fixture; real-data run pending local Jaeb download",
            "sleep": "timestamps ingested but no causal circadian/sleep physiology is fitted yet",
            "clinical_decision_use": "not supported",
        },
        "remaining_requirement": (
            "A local Jaeb CGMND.zip obtained under the dataset terms is required to inspect the real schema and execute participant-level "
            "held-out event fitting. Meal quantity is absent by study design, so v0.11 should add hierarchical latent-input "
            "inference and a protocol with known nutrient composition rather than pretending carbs are observed."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["tier"], c["name"])


if __name__ == "__main__":
    main()
